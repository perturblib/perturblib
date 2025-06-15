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

Interface module for all the training config-related operations.
"""

from pathlib import Path
from typing import List

import yaml

from perturb_gym.configs.base import DataConfig, EnvironmentConfig, ModelConfig, TrainingConfig
from perturb_gym.utils import dict_product, list_yaml_modules_from_directory

_collection_dir = Path(__file__).parent / "collection"


def list_config_file_ids() -> List[str]:
    """Get IDs of existing configuration files.

    Returns:
        A list of training configuration identifiers that exist in Perturb-gym.
    """
    return sorted(list_yaml_modules_from_directory(_collection_dir.resolve()))


def _verify_that_config_exists(config_id: str):
    if config_id not in list_config_file_ids():
        raise ValueError(f"Unavailable config {config_id}: chose one of {list_config_file_ids()}")


def load_training_configs(config_file_id_or_path: str) -> List[TrainingConfig]:
    """Load configs specified in the corresponding config file.

    Args:
        config_file_id_or_path: Identifier of the config file within perturb-gym or custom path to one.

    Returns:
        List of training configs.
    """
    if not config_file_id_or_path.endswith(".yaml"):
        _verify_that_config_exists(config_file_id_or_path)
        path_to_config_file = _collection_dir / f"{config_file_id_or_path}.yaml"
    else:
        path_to_config_file = Path(config_file_id_or_path)
    with open(path_to_config_file, "r") as file:
        config_file_dict = yaml.safe_load(file)
        # resolve grid search of model arguments
        model_configs_expended = []
        for model_config in config_file_dict["model_configs"]:
            model_config_copy = model_config.copy()
            model_config_copy["model_args"] = dict_product(model_config["model_args"])
            model_configs_expended.extend(dict_product(model_config_copy))
        config_file_dict["model_configs"] = model_configs_expended
        # get a list of configurations
        config_list = dict_product(config_file_dict)
        # create training configs
        training_configs = []
        for config in config_list:
            environment_config = EnvironmentConfig(**config["environment_configs"])
            data_config = DataConfig(**config["data_configs"])
            model_config = ModelConfig(**config["model_configs"])
            training_configs.append(TrainingConfig(environment_config, data_config, model_config))
        return training_configs
