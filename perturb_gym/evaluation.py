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

Module used for evaluating trained perturbation models.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, cast

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import yaml

import perturb_lib as plib
from perturb_gym.configs.base import DataConfig
from perturb_gym.paths import EVALUATION_LOG_FILENAME, TRAINING_LOG_FILENAME
from perturb_gym.utils import get_same_models_with_different_seed
from perturb_lib import PlibData, Vocabulary
from perturb_lib.models.collection.lpm import LPM

# main metrics to consider during results processing and figure creation
# dictionary keys represent evaluator ids, dictionary values represent the directionality -- True means lower is better
EVALUATION_METRICS = {
    # where lower is better
    "RMSE_test": {"lower_is_better": True, "select_best_on": "RMSE_val"},
    "RMSE_val": {"lower_is_better": True, "select_best_on": "RMSE_val"},
    # "RMSEtop20DE_test": {"lower_is_better": True, "select_best_on": "RMSEtop20DE_val"},
    "MAE_test": {"lower_is_better": True, "select_best_on": "MAE_val"},
    "MAE_val": {"lower_is_better": True, "select_best_on": "MAE_val"},
    # where higher is better
    "R2_test": {"lower_is_better": False, "select_best_on": "R2_val"},
    "R2_val": {"lower_is_better": False, "select_best_on": "R2_val"},
    # "Pearson_test": {"lower_is_better": False, "select_best_on": "Pearson_val"},
    # "Pearson_val": {"lower_is_better": False, "select_best_on": "Pearson_val"},
    # "PearsonDelta_test": {"lower_is_better": False, "select_best_on": "PearsonDelta_val"},
    # "PearsonDeltaTop20DE_test": {"lower_is_better": False, "select_best_on": "PearsonDeltaTop20DE_val"},
    # "R2Delta_test": {"lower_is_better": False, "select_best_on": "R2Delta_val"},
}
INDEX_KEYS = ["target_contexts", "preprocessing_type"]


def evaluate_model(
    model: plib.ModelMixin,
    data: Tuple[PlibData, PlibData | None, PlibData | None],
    model_dir: Path | str | None = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate given model with respect to all the metrics defined within perturb-lib, and with respect to given data.

    Args:
        model: Model to be evaluated.
        data: Contains training, validation, and test data in the form of a tuple.
        model_dir: If specified, this is the directory where the results should be saved.
    """
    _, valdata, testdata = data  # for the moment, we ignore training data evaluation..
    data_dict = {}

    if testdata is not None:
        data_dict["test"] = testdata

    if valdata is not None:
        data_dict["val"] = valdata

    results: Dict[str, Dict[str, float]] = {}
    for evaluator_id in plib.list_evaluators():
        evaluator = plib.load_evaluator(evaluator_id)
        results[evaluator_id] = {}
        for data_split_key, data_to_evaluate_on in data_dict.items():
            predictions = model.predict(data_to_evaluate_on.subset_columnwise(["context", "perturbation", "readout"]))
            results[evaluator_id][data_split_key] = round(
                float(evaluator.evaluate(predictions, data_to_evaluate_on)) + 1e-10, 4
            )

    if model_dir is not None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        path_to_results = model_dir / EVALUATION_LOG_FILENAME
        with open(path_to_results, "w") as f:
            yaml.safe_dump(results, f, sort_keys=False)
        plib.logger.info(f"Saving {results}")

    return results


def evaluate_all_trained_models(root_dir: Path | str):
    """Run 'evaluate_model' on each saved model found in the specified directory or any subdirectory.

    Args:
        root_dir: path to the root dir where trained models are searched for, including subdirectories.
    """
    root_dir = Path(root_dir)

    if not root_dir.is_dir():
        raise RuntimeError(f"Not a valid directory: {root_dir}")

    list_of_paths_to_models = list(root_dir.rglob("*.pt"))
    for path_to_model in list_of_paths_to_models:
        path_to_model_dir = path_to_model.parent

        with open(path_to_model_dir / TRAINING_LOG_FILENAME, "r") as f:
            training_log = yaml.safe_load(f)
            data_config = DataConfig(**training_log["data_config"])
            traindata, valdata, testdata = data_config.get_train_val_test_data()
            model = plib.load_trained_model(path_to_model)
            evaluate_model(model, (traindata, valdata, testdata), path_to_model_dir)


def make_figs(path_to_results: Path | str, save_dir: Path | str, evaluation_metrics: Dict | None = None):
    """Make figures given results in the form of .csv table."""
    # initialization
    path_to_results = Path(path_to_results)
    save_dir = Path(save_dir)
    results = pd.read_csv(path_to_results)
    evaluation_metrics = EVALUATION_METRICS if evaluation_metrics is None else evaluation_metrics
    evaluation_metrics = {key: val for key, val in evaluation_metrics.items() if key in set(results.columns)}
    sns.set(style="whitegrid")
    # Use palette that highlights the first model in the list
    muted_palette = sns.color_palette("light:r")
    muted_palette[0] = "#1f77b4"
    # font size
    sns.set_context(context="paper", font_scale=1.7)

    # simple trick to ensure different preprocessing types go into different figures
    results.target_contexts = results.target_contexts + "_" + results.preprocessing_type

    # prepare matplotlib figures
    target_contexts_figs = {}
    all_target_contexts = list(set(results.target_contexts))
    for target_contexts in all_target_contexts:
        fig, axes = plt.subplots(nrows=1, ncols=len(evaluation_metrics), figsize=(18, 6))
        target_contexts_figs[target_contexts] = (fig, axes)
    # generate performance plots
    for idx, (metric_key, metric_desc) in enumerate(evaluation_metrics.items()):
        table = results[INDEX_KEYS + ["model_id", "seed", metric_key, metric_desc["select_best_on"]]]
        table = table.T.drop_duplicates().T  # remove duplicate columns (when metric_key==metric_desc["select_best_on"])
        if metric_desc["lower_is_better"]:
            table = table.loc[table.groupby(INDEX_KEYS + ["model_id", "seed"])[metric_desc["select_best_on"]].idxmin()]
        else:
            table = table.loc[table.groupby(INDEX_KEYS + ["model_id", "seed"])[metric_desc["select_best_on"]].idxmax()]
        table.set_index("target_contexts", inplace=True)
        for target_contexts in all_target_contexts:
            fig, axes = target_contexts_figs[target_contexts]
            subtable = table.loc[target_contexts]
            if isinstance(subtable, pd.Series):
                continue
            subtable = subtable.sort_values(by=metric_desc["select_best_on"], ascending=metric_desc["lower_is_better"])
            sns.barplot(data=subtable, x="model_id", y=metric_key, ax=axes[idx])  # , palette=muted_palette)
            sns.stripplot(data=subtable, x="model_id", y=metric_key, color="black", size=8, jitter=True, ax=axes[idx])
            axes[idx].set_xlabel("")
            axes[idx].set_ylabel(axes[idx].get_ylabel().replace("_test", ""))
    # edit and save plots
    for target_contexts in all_target_contexts:
        fig, axes = target_contexts_figs[target_contexts]
        fig.suptitle(f"{target_contexts}")
        fig.tight_layout()
        fig.savefig(save_dir / f"{target_contexts}.png")


def process_results(root_dir: Path | str):
    """Process training and evaluation logs in a meaningful way.

    Args:
        root_dir: path to the root dir where logs are searched for, including subdirectories.
    """
    root_dir = Path(root_dir)

    if not root_dir.is_dir():
        raise RuntimeError(f"Not a valid directory: {root_dir}")

    entries = []

    for subdir in [x for x in root_dir.rglob("*") if x.is_dir()]:
        training_log_file = subdir / TRAINING_LOG_FILENAME
        evaluation_log_file = subdir / EVALUATION_LOG_FILENAME

        if training_log_file.exists() and evaluation_log_file.exists():
            with open(training_log_file, "r") as f:
                training_log = yaml.safe_load(f)
            with open(evaluation_log_file, "r") as f:
                evaluation_log = yaml.safe_load(f)

            data_config = DataConfig(**training_log["data_config"])

            if data_config.val_and_test_perturbations_selected_from is not None:
                target_contexts = data_config.val_and_test_perturbations_selected_from
            elif data_config.val_perturbations_selected_from is not None:
                target_contexts = data_config.val_perturbations_selected_from
            else:
                target_contexts = ""

            if target_contexts != "":
                entry = {}
                for evaluator_id in evaluation_log.keys():
                    for data_split, value in evaluation_log[evaluator_id].items():
                        entry[f"{evaluator_id}_{data_split}"] = value
                entry.update(
                    {
                        "training_contexts": data_config.training_contexts,
                        "target_contexts": target_contexts,
                        "preprocessing_type": data_config.preprocessing_type,
                        "seed": training_log["seed"],
                        "model_id": training_log["model_id"],
                        "model_args": training_log["model_args"],
                        "training_time": training_log["training_time"],
                        "peak_CPU-RAM_consumption": training_log["peak_CPU-RAM_consumption"],
                    }
                )
                entries.append(entry)

    # create a results table
    results = pd.DataFrame(entries)
    results = results.dropna(axis=1)
    sel_evaluation_metrics = {key: val for key, val in EVALUATION_METRICS.items() if key in set(results.columns)}

    if not sel_evaluation_metrics:
        raise ValueError("No valid evaluation keys were found in the results table.")

    if not results.empty:
        # sort (for readability) and then save the table
        results.set_index(keys=INDEX_KEYS, inplace=True)
        cols_to_sort_by = list(sel_evaluation_metrics.keys())
        ascending_list = [v["lower_is_better"] for k, v in sel_evaluation_metrics.items()]
        results = results.sort_values(by=cols_to_sort_by, ascending=ascending_list).sort_index()
        path_to_results = root_dir / "results.csv"
        results.to_csv(path_to_results, index=True)

        # make figures from the results table
        figs_dir = root_dir / "figs"
        figs_dir.mkdir(exist_ok=True)
        make_figs(path_to_results, figs_dir, sel_evaluation_metrics)


def extract_information_from_trained_lpm(path_to_model: Path | str, extract_embeddings: bool = False):
    """Load trained model and extract necessary information."""
    # create path to results
    results_dir = plib.get_path_to_cache() / "trained_model_info"
    prediction_results_dir = results_dir / "predictions"
    results_dir.mkdir(exist_ok=True)
    prediction_results_dir.mkdir(exist_ok=True)

    # load trained models
    path_to_model = Path(path_to_model)
    all_models_paths = get_same_models_with_different_seed(path_to_model)
    plib.logger.info("Loading trained models..")
    trained_models = cast(list[LPM], [plib.load_trained_model(path) for path in all_models_paths])
    assert all(isinstance(model, LPM) for model in trained_models), "Trained model must be LPM!"

    # check vocabularies
    plib.logger.info("Checking vocabularies...")
    vocabulary = cast(Vocabulary, trained_models[0].vocab)
    for model in trained_models:
        assert model.vocab is not None, "Trained model is missing vocabulary!"
        assert model.vocab == vocabulary, "Trained models have different vocabularies!"

    plib.logger.warning("Removing perturbations considered to be less informative in an adhoc manner..")
    vocabulary.perturb_vocab = vocabulary.perturb_vocab.filter(
        pl.col("symbol").str.contains("CRISPR")
        | (
            # Keep only if:
            # - The first character is a-z or A-Z
            # - AND the sliced part (except last char) has no uppercase letters
            pl.col("symbol").str.slice(0, 1).str.contains(r"[a-zA-Z]")
            & ~pl.col("symbol").str.slice(0, pl.col("symbol").str.len_chars() - 1).str.contains(r"[A-Z]")
        )
    )
    plib.logger.warning("Subsampling to 10000 perturbations..")
    vocabulary.perturb_vocab = vocabulary.perturb_vocab.sample(
        n=min(10000, len(vocabulary.perturb_vocab)),
        with_replacement=False,
        shuffle=True,
        seed=42,
    )

    plib.logger.info("Saving vocabulary information")
    context_info = pl.DataFrame(
        {
            "context": vocabulary.context_vocab["symbol"],
            "source": ["CMAP2020"] * vocabulary.context_vocab.height,
            "description": ["placeholder"] * vocabulary.context_vocab.height,
        }
    )
    context_info = context_info.with_columns(pl.col("context").map_elements(plib.describe_context).alias("description"))
    perturbation_info = pl.DataFrame(
        {
            "perturbation": vocabulary.perturb_vocab["symbol"],
            "metadata": ["placeholder"] * vocabulary.perturb_vocab.height,
            "perturbation_type": ["placeholder"] * vocabulary.perturb_vocab.height,
            "perturbation_target": ["placeholder"] * vocabulary.perturb_vocab.height,
        }
    )

    def map_name_to_perturb_type(name):
        return "Control" if "Control" in name else "CRISPR-KO" if "CRISPR" in name else "Compound"

    perturbation_info = perturbation_info.with_columns(
        pl.col("perturbation").map_elements(map_name_to_perturb_type).alias("perturbation_type")
    )
    readout_info = pl.DataFrame(
        {
            "readout": vocabulary.readout_vocab["symbol"],
            "metadata": ["placeholder"] * vocabulary.readout_vocab.height,
            "readout_type": ["Transcriptome"] * vocabulary.readout_vocab.height,
            "readout_target": ["placeholder"] * vocabulary.readout_vocab.height,
        }
    )
    context_info.write_csv(results_dir / "context_info.csv")
    perturbation_info.write_csv(results_dir / "perturbation_info.csv")
    readout_info.write_csv(results_dir / "readout_info.csv")

    if extract_embeddings:
        # Use the first model to extract embeddings
        trained_model = trained_models[0]

        max_length = max(vocabulary.context_vocab.height, vocabulary.perturb_vocab.height)
        placeholder_context = vocabulary.context_vocab["symbol"][0]
        placeholder_perturbation = vocabulary.perturb_vocab["symbol"][0]
        placeholder_readout = vocabulary.readout_vocab["symbol"][0]

        # Create dataframe with symbols
        input_data = pl.DataFrame(
            data={
                "context": vocabulary.context_vocab["symbol"].extend_constant(
                    placeholder_context, n=max_length - vocabulary.context_vocab.height
                ),
                "perturbation": vocabulary.perturb_vocab["symbol"].extend_constant(
                    placeholder_perturbation, n=max_length - vocabulary.perturb_vocab.height
                ),
                "readout": [placeholder_readout] * max_length,
            }
        )

        plib.logger.info("Extracting context and perturbation embeddings..")
        context_embeddings = {}
        perturbation_embeddings = {}
        for data_batch in input_data.iter_slices(n_rows=1000):
            context_embedding, perturbation_embedding, _ = trained_model.embed(data_batch)
            for i in range(len(data_batch)):
                context = data_batch["context"][i]
                perturbation = data_batch["perturbation"][i]
                if context not in context_embeddings:
                    context_embeddings[context] = list(context_embedding[i].flatten().cpu().numpy().astype(float))
                if perturbation not in perturbation_embeddings:
                    perturbation_embeddings[perturbation] = list(
                        perturbation_embedding[i].flatten().cpu().numpy().astype(float)
                    )

        with open(results_dir / "context_embeddings.json", "w") as f:
            json.dump(context_embeddings, f)
        with open(results_dir / "perturbation_embeddings.json", "w") as f:
            json.dump(perturbation_embeddings, f)

        # To ensure we don't use these later by accident and to save space
        del trained_model, context_embeddings, perturbation_embeddings

    c, p, r = len(vocabulary.context_vocab), len(vocabulary.perturb_vocab), len(vocabulary.readout_vocab)
    plib.logger.info(f"Contexts {c}; Perturbations {p}; Readouts {r}")
    plib.logger.info(f"Creating a DataFrame of {c * p * r} (context, perturbation, readout) combinations..")
    combinations_df = (
        vocabulary.context_vocab.select(context=pl.col("symbol"))
        .join(vocabulary.perturb_vocab.select(perturbation=pl.col("symbol")), how="cross")
        .join(vocabulary.readout_vocab.select(readout=pl.col("symbol")), how="cross")
    )
    combinations_pdata = plib.InMemoryPlibData(data=combinations_df)

    all_predictions: list[np.ndarray] = []
    for i, trained_model in enumerate(trained_models):
        plib.logger.info(f"Making predictions with model {i}..")
        all_predictions.append(trained_model.predict(combinations_pdata).reshape(-1))

    all_predictions_np = np.vstack(all_predictions)
    del all_predictions

    mean_pred = np.mean(all_predictions_np, axis=0)
    std_pred: np.ndarray | None = None
    if all_predictions_np.shape[0] > 1:
        std_pred = np.std(all_predictions_np, ddof=1, axis=0)
    del all_predictions_np

    combinations_df = combinations_df.with_columns(
        pl.Series("prediction", mean_pred),
        pl.Series("std_dev", std_pred) if std_pred is not None else pl.lit(None).alias("std_dev"),
    )

    # Iterate over unique contexts and save each subset as a CSV file
    for partition_key, context_results in combinations_df.partition_by("context", as_dict=True).items():
        context = cast(str, partition_key[0])
        true_values = plib.load_plibdata(context)[:]
        context_results = context_results.join(true_values, on=["context", "perturbation", "readout"], how="left")
        context_results = context_results.rename({"value": "truth"})
        context_results.write_csv(prediction_results_dir / f"{context}.csv")
        plib.logger.info(f"Results for context {context} saved in {results_dir}")

    plib.logger.info("All results saved.")


def assess_transfer_learning(train_contexts: str | list[str], target_context: str):
    """Assess transfer learning capabilities, from train contexts to target context."""

    def get_symbols(data: plib.OnDiskPlibData):
        vocabulary = plib.Vocabulary.initialize_from_data(data)
        context_symbols = set(vocabulary.context_vocab["symbol"].to_list())
        perturb_symbols = set(vocabulary.perturb_vocab["symbol"].to_list())
        readout_symbols = set(vocabulary.readout_vocab["symbol"].to_list())
        return context_symbols, perturb_symbols, readout_symbols

    plib.logger.info("Loading training and target data..")
    train_data = plib.load_plibdata(train_contexts, plibdata_type=plib.OnDiskPlibData)
    target_data = plib.load_plibdata(target_context, plibdata_type=plib.OnDiskPlibData)

    plib.logger.info("Extracting symbols..")
    train_context_symbols, train_perturb_symbols, train_readout_symbols = get_symbols(train_data)  # type: ignore
    target_context_symbols, target_perturb_symbols, target_readout_symbols = get_symbols(target_data)  # type: ignore

    plib.logger.info("Performing 3-fold split..")
    target_train_data, target_val_data = plib.split_plibdata_2fold(target_data, list(target_context_symbols))

    plib.logger.info("Extracting symbols..")
    _, target_train_perturb_symbols, target_train_readout_symbols = get_symbols(target_train_data)  # type: ignore
    _, target_val_perturb_symbols, target_val_readout_symbols = get_symbols(target_val_data)  # type: ignore

    plib.logger.info("")
    plib.logger.info("Perturbation symbol statistics:")
    plib.logger.info("-------------------------------")
    plib.logger.info(f"Training contexts contains: {len(train_perturb_symbols)} perturbations")
    plib.logger.info(f"Target contexts contains: {len(target_perturb_symbols)} perturbations")
    plib.logger.info(
        f"Total overlap is: {len(train_perturb_symbols & target_perturb_symbols)}/{len(target_perturb_symbols)} perturbations"
    )
    plib.logger.info(
        f"Train overlap is: {len(train_perturb_symbols & target_train_perturb_symbols)}/{len(target_train_perturb_symbols)} perturbations"
    )
    plib.logger.info(
        f"Val overlap is: {len(train_perturb_symbols & target_val_perturb_symbols)}/{len(target_val_perturb_symbols)} perturbations"
    )

    plib.logger.info("Readout symbol statistics:")
    plib.logger.info("-------------------------------")
    plib.logger.info(f"Training contexts contains: {len(train_readout_symbols)} readouts")
    plib.logger.info(f"Target contexts contains: {len(target_readout_symbols)} readouts")
    plib.logger.info(f"Total overlap is: {len(train_readout_symbols & target_readout_symbols)} readouts")


if __name__ == "__main__":
    fire.Fire()
