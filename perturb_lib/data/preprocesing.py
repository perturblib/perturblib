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

AnnData preprocessing routines.
"""

from typing import Callable, Dict, Literal

import pandas as pd
from scanpy import AnnData
from scanpy.preprocessing import highly_variable_genes, log1p, normalize_total

from perturb_lib.environment import logger


def _replica_aggregation(adata: AnnData, func: Literal["mean", "median"]) -> AnnData:
    logger.info(f"{func}-aggregating data..")

    # aggregate obs (obs) DataFrame and X matrix along with it
    obs_cols_to_keep = ["perturbation", "context"] + [x for x in adata.obs.columns.tolist() if "split" in x]
    obs_and_x = pd.concat([adata.obs[obs_cols_to_keep], pd.DataFrame(adata.X, index=adata.obs_names)], axis=1)
    if func == "mean":
        obs_and_x_grouped = obs_and_x.groupby(obs_cols_to_keep).mean()
    elif func == "median":
        obs_and_x_grouped = obs_and_x.groupby(obs_cols_to_keep).median()
    else:
        raise ValueError(f"Unrecognized aggregation function {func}")
    obs_and_x_grouped = obs_and_x_grouped.reset_index()
    obs_and_x_grouped.index = obs_and_x_grouped.index.astype(str)
    aggregated_obs = obs_and_x_grouped[obs_cols_to_keep]
    aggregated_X = obs_and_x_grouped.drop(obs_cols_to_keep, axis=1).values

    # aggregate var DataFrame and X matrix along with it
    var_cols_to_keep = ["readout"]
    var_and_x = pd.concat(
        [adata.var[var_cols_to_keep], pd.DataFrame(aggregated_X.transpose(), index=adata.var_names)], axis=1
    )
    if func == "mean":
        var_and_x_grouped = var_and_x.groupby(var_cols_to_keep).mean()
    elif func == "median":
        var_and_x_grouped = var_and_x.groupby(var_cols_to_keep).median()
    else:
        raise ValueError(f"Unrecognized aggregation function {func}")
    var_and_x_grouped = var_and_x_grouped.reset_index()
    var_and_x_grouped.index = var_and_x_grouped.index.astype(str)
    aggregated_var = var_and_x_grouped[var_cols_to_keep]
    aggregated_X = var_and_x_grouped.drop(var_cols_to_keep, axis=1).values
    aggregated_X = aggregated_X.transpose()

    return AnnData(aggregated_X, aggregated_obs, aggregated_var)


def raw(adata: AnnData) -> AnnData:
    """No preprocessing."""
    return adata


def median_aggregation(adata: AnnData) -> AnnData:
    """Replicas (if exist) are median aggregated."""
    return _replica_aggregation(adata, "median")


def mean_aggregation(adata: AnnData) -> AnnData:
    """Replicas (if exist) are mean aggregated."""
    return _replica_aggregation(adata, "mean")


def standard_transcriptomics_preprocessing(adata: AnnData) -> AnnData:
    """Preprocessing typically applied in the related literature. Requires gene expression data to be given in the
    raw/count format. Normalization of total counts is followed by log operation and then filtering of highly variable
    genes.
    """
    logger.info("Performing standard preprocessing (normalization->log->variable-readout-selection)..")
    adata_copy = adata.copy()
    normalize_total(adata_copy)
    log1p(adata_copy)
    highly_variable_genes(adata_copy, n_top_genes=5000, subset=True)
    return mean_aggregation(adata_copy)


preprocessors: Dict[str, Callable[[AnnData], AnnData]] = {
    "raw": raw,
    "median_aggregation": median_aggregation,
    "mean_aggregation": mean_aggregation,
    "standard_transcriptomics_preprocessing": standard_transcriptomics_preprocessing,
}

PreprocessingType = Literal[
    "raw",
    "median_aggregation",
    "mean_aggregation",
    "standard_preprocessing",
]

DEFAULT_PREPROCESSING_TYPE: PreprocessingType = "mean_aggregation"
