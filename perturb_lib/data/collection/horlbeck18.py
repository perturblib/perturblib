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

Loading and processing of perturb-seq (Horlbeck et al. 2018) data.
"""

from typing import Literal

import numpy as np
import pandas as pd

from perturb_lib._utils import download_extract_zip_file, download_file
from perturb_lib.data import ControlSymbol
from perturb_lib.data.access import create_and_register_context
from perturb_lib.environment import get_path_to_cache, logger


def _get_raw_horlbeck18_data(
    data_url: str = "https://ars.els-cdn.com/content/image/1-s2.0-S0092867418307359-mmc3.zip",
    gene_names_url: str = "https://ars.els-cdn.com/content/image/1-s2.0-S0092867418307359-mmc2.xlsx",
) -> pd.DataFrame:
    # Get the data in the original format
    logger.info("Loading Horlbeck18 data which is originally given in .txt tabular format")
    path_to_raw_dataset = get_path_to_cache() / "raw_datasets" / "Horlbeck18"
    path_to_raw_dataset.mkdir(exist_ok=True, parents=True)
    download_extract_zip_file(data_url, path_to_raw_dataset / "Horlbeck18", path_to_raw_dataset)
    data_df = pd.read_csv(path_to_raw_dataset / "cell_10260_mmc3.txt", sep="\t")
    data_df.set_index(data_df.columns[0], inplace=True)
    data_df.index.name = "Perturbations"
    gene_names_file = path_to_raw_dataset / "gene_name_info.xlsx"
    download_file(gene_names_url, gene_names_file)
    gene_names_df = pd.read_excel(gene_names_file)
    gene_names_df["gene name"] = gene_names_df["gene name"].replace("NEGATIVE", ControlSymbol)
    name_map = dict(zip(gene_names_df["sgRNA ID"], gene_names_df["gene name"]))
    for original, new_name in name_map.items():
        data_df.index = data_df.index.str.replace(original, new_name, regex=False)
    data_df.index = data_df.index.str.replace("++", "+", regex=False)
    data_df.index = data_df.index.str.replace(f"{ControlSymbol}+{ControlSymbol}", ControlSymbol, regex=False)
    return data_df


def _get_horlbeck18_anndata(cell_line: Literal["K562", "Jurkat"], z_normalize: bool = True):
    df = _get_raw_horlbeck18_data()
    df = df.filter(like=f"{cell_line}")
    var = pd.DataFrame({"readout_type": "Viability", "readout_target": df.head(2).agg("_".join).unique()})
    df = df[4:].astype(np.float32)
    obs = pd.DataFrame({"perturbation_type": "CRISPRi", "perturbation_target": df.index})
    X = df.values
    X = X.reshape(X.shape[0], -1, 2).mean(axis=2)
    # Z-score normalization
    if z_normalize:
        mean_vals = X.mean(axis=0)
        std_vals = X.std(axis=0)
        normalized_X = (X - mean_vals) / std_vals
    return normalized_X, obs, var


for cell_line in ["K562", "Jurkat"]:
    create_and_register_context(
        model_system="HumanCellLine",
        model_system_id=cell_line,
        technology_info="GrowthScreen",
        data_source_info="Horlbeck18",
        batch_info=None,
        full_context_description=f"Human cell line {cell_line} prepared as described in "
        f"`2022 Horlbeck et al <https://www.sciencedirect.com/science/article/pii/S0092867418307359>`_",
        anndata_fn=_get_horlbeck18_anndata,
        cell_line=cell_line,
    )
