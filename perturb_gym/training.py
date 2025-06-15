"""Copyright (C) 2025  GlaxoSmithKline plc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Module used to train perturbation models.
"""

import base64
import json
import os
import shutil
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from random import random
from typing import Dict

import fire
import torch
import yaml

import perturb_lib as plib
from perturb_gym.configs.access import load_training_configs
from perturb_gym.configs.base import DataConfig, EnvironmentConfig, ModelConfig, TrainingConfig
from perturb_gym.evaluation import evaluate_model
from perturb_gym.paths import DEFAULT_RESULTS_DIRNAME, TRAINING_LOG_FILENAME
from perturb_gym.utils import get_user_confirmation, hash_training_config_excluding_seed


def train_from_args(training_config: TrainingConfig | Dict, results_dir: Path | str):
    """Train a single perturbation model given a training configuration.

    Args:
        training_config: training config object.
        results_dir: where to store the results of training.
    """
    # convert to TrainingConfig if not in that form already
    if not isinstance(training_config, TrainingConfig):
        training_config = TrainingConfig(
            environment_config=EnvironmentConfig(**training_config["environment_config"]),
            data_config=DataConfig(**training_config["data_config"]),
            model_config=ModelConfig(**training_config["model_config"]),
        )

    # resolve the results directory
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        raise ValueError(f"{results_dir} does not exit!")

    # fix random seed
    seed = training_config.environment_config.seed
    plib.set_all_seeds(seed if seed is not None else plib.get_seed())

    # start tracking memory consumption
    tracemalloc.start()

    # get train/val/test data
    traindata, valdata, testdata = training_config.get_train_val_test_data()

    # model naming and save directory preparation
    plib.logger.info(f"Loading and training model with id={training_config.model_config.model_id}")
    unique_model_name = (
        f"{training_config.model_config.model_id}_{hash_training_config_excluding_seed(training_config)}"
    )
    model_dir = results_dir / unique_model_name / f"seed_{seed}"
    model_dir.mkdir(exist_ok=True, parents=True)

    # model loading
    model = plib.load_model(training_config.model_config.model_id, training_config.model_config.model_args)

    # set up model training logging if specified and applicable
    if hasattr(model, "add_logger"):
        model.add_logger(model_dir, "learning_curves")

    # optional model compilation
    if training_config.model_config.torch_compile and isinstance(model, torch.nn.Module):
        plib.logger.info("Model compilation in the max-autotune mode..")
        model = torch.compile(model, "max-autotune")  # type: ignore[call-overload]

    # model training
    start_time = time.time()
    plib.logger.info("Model training..")
    model.fit(traindata=traindata, valdata=valdata)
    training_time = time.time() - start_time
    plib.logger.info("Model training done!")
    tracemalloc.start()  # stop tracking memory consumption
    _, peak_memory = tracemalloc.get_traced_memory()

    # save the model if specified
    if training_config.model_config.save_model_after_training:
        plib.logger.info("Saving trained model...")
        plib.save_trained_model(model, model_dir / "model.pt", training_config.model_config.model_args)

    # logging the training configuration
    plib.logger.info("Saving model training log...")
    model_training_log = {
        "data_config": dict(training_config.data_config),
        "seed": training_config.environment_config.seed,
        "model_id": training_config.model_config.model_id,
        "model_args": training_config.model_config.model_args,
        "torch_compile": training_config.model_config.torch_compile,
        "training_time": f"{training_time / 60.0:.2f} min",
        "execution_datetime": datetime.now().strftime("%A, %B %d, %Y %I:%M:%S %p"),
        "peak_CPU-RAM_consumption": f"{peak_memory / (1024 * 1024):.1f} MB",
    }

    # extra log for pytorch models
    if isinstance(model, torch.nn.Module):
        if torch.cuda.is_available():
            device = "gpu"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        model_training_log["device"] = device
        if device == "gpu":
            model_training_log["gpu name"] = torch.cuda.get_device_properties(0).name
            model_training_log["max memory allocated per gpu"] = (
                f"{torch.cuda.max_memory_allocated(0) / (1024 ** 2):.1f} MB"
            )

    with open(model_dir / TRAINING_LOG_FILENAME, "w") as f:
        yaml.safe_dump(model_training_log, f, sort_keys=False)

    # model evaluation
    plib.logger.info("Running post-training evaluation...")
    evaluate_model(model, (traindata, valdata, testdata), model_dir)

    # stop tracking memory consumption
    tracemalloc.stop()
    plib.logger.info("All done!")


def train_from_b64_encoded_args(training_config: str, results_dir: Path | str):
    """Train a single perturbation model given a base64 encoded training configuration."""
    training_config_deserialized = json.loads(base64.b64decode(training_config))
    train_from_args(training_config_deserialized, results_dir)


def train_from_config_file(
    config_file_id_or_path: str,
    use_slurm: bool = False,
    experiment_probability: float = 1.0,
    slurm_args: str = "--mem=50G --time=12:00:00 --nodes=1 --partition=gpu --gres=gpu:1",
    results_dir: Path | str | None = None,
):
    """Train one or more perturbation models as specified in the configuration file.

    Args:
        config_file_id_or_path: Identifier of the config file within perturb-gym or custom path to one.
        use_slurm: Whether to run jobs in parallel on slurm.
        experiment_probability: probability of running each experiment. We stochastically decide whether to run the
        experiment based on this probability, essentially implementing random search to replace expensive grid search.
        slurm_args: If slurm is used, the arguments for the 'sbatch' call.
        results_dir: Where the results of training are to be stored.
    """
    if experiment_probability > 1.0 or experiment_probability < 0.0:
        raise ValueError("Experiment probability must be in [0,1]")

    # resolve the results directory
    results_dir = plib.get_path_to_cache() / DEFAULT_RESULTS_DIRNAME if results_dir is None else Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if config_file_id_or_path.endswith(".yaml"):
        config_results_dir = results_dir / Path(config_file_id_or_path).stem
    else:
        config_results_dir = results_dir / config_file_id_or_path

    if config_results_dir.is_dir() and any(config_results_dir.iterdir()):
        if not get_user_confirmation(f"This will delete all files in '{config_results_dir}'. Continue? (yes/no)"):
            return
        shutil.rmtree(config_results_dir)
    config_results_dir.mkdir(exist_ok=True)

    # loop over training configurations training different perturbation models
    for training_config in load_training_configs(config_file_id_or_path):
        if random() > experiment_probability:
            continue
        if not use_slurm:  # run locally
            train_from_args(training_config, config_results_dir)
        else:  # send a job to slurm
            # serialize training_config
            training_config_serialized = base64.b64encode(json.dumps(training_config).encode("utf-8")).decode("utf-8")

            cli_command = f"{sys.executable} -m {__package__}.{Path(__file__).stem} train_from_b64_encoded_args"
            cli_command_with_args = (
                f"{cli_command} "
                f'--training_config "{training_config_serialized}" '
                f'--results_dir "{str(config_results_dir)}"'
            )
            model_args = training_config["model_config"]["model_args"]
            ncpus = 2 if ("num_workers" not in model_args) else (model_args["num_workers"] + 2)
            slurm_args += f" --cpus-per-task={ncpus}"
            slurm_command_with_args = f"sbatch {slurm_args} --wrap '{cli_command_with_args}'"
            os.system(slurm_command_with_args)


if __name__ == "__main__":
    fire.Fire()
