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

Loading and processing of perturb-seq (Replogle et al. 2022) data.
"""

import numpy as np
import scanpy as sc

from perturb_lib._utils import download_file
from perturb_lib.data import ControlSymbol
from perturb_lib.data.access import create_and_register_context
from perturb_lib.environment import get_path_to_cache, logger


def _get_replogle_dataset(cell_line: str):
    url = {
        "K562": "https://plus.figshare.com/ndownloader/files/35773075",
        "RPE1": "https://plus.figshare.com/ndownloader/files/35775554",
        "K562_GWPS": "https://plus.figshare.com/ndownloader/files/35774440",
    }[cell_line]

    # Get the data in the original format
    dataset_name = f"replogle_{cell_line}"
    logger.info(f"Loading dataset '{dataset_name}' which is originally given in AnnData format..")
    path_to_raw_dataset = get_path_to_cache() / "raw_datasets" / "Replogle22"
    path_to_raw_dataset.mkdir(exist_ok=True, parents=True)
    adata_path = path_to_raw_dataset / f"{dataset_name}.h5ad"
    download_file(url, save_path=adata_path)
    adata = sc.read_h5ad(adata_path)

    # Cast the data to standardized AnnData format
    logger.debug("Renaming meta-data and outdated gene names..")
    adata.obs.gene = adata.obs.gene.str.replace(r"non-targeting", ControlSymbol, regex=True)
    adata.obs = adata.obs.rename(columns={"gene": "perturbation_target", "gem_group": "batch"})
    adata.obs["perturbation_type"] = "CRISPRi"
    adata.obs = adata.obs[["perturbation_type", "perturbation_target", "batch"]]
    adata.var = adata.var.rename(columns={"gene_name": "readout_target"})
    adata.var["readout_type"] = "Transcriptome"
    adata.var = adata.var[["readout_type", "readout_target"]]

    # Type conversion
    logger.debug("Type conversion..")
    adata.obs.index = adata.obs.index.astype(str)
    adata.var.index = adata.var.index.astype(str)

    infinite_indices = np.where(~np.isfinite(adata.X))
    if len(infinite_indices[0]) > 0:
        logger.debug(f"Zero-imputation of {len(infinite_indices[0])} values..")
        adata.X[infinite_indices] = 0.0

    # Normalization
    logger.debug("Data has been gem-group z-normalized already..")

    return adata.X, adata.obs, adata.var


for cell_line in ["K562", "RPE1"]:
    create_and_register_context(
        model_system="HumanCellLine",
        model_system_id=cell_line,
        technology_info="10xChromium3-scRNA-seq",
        data_source_info="Replogle22",
        batch_info=None,
        full_context_description=f"Human cell line {cell_line} prepared for perturbation analysis of essential genes as described in"
        f"`2022 Replogle et al. <https://pubmed.ncbi.nlm.nih.gov/35688146>`_.",
        anndata_fn=_get_replogle_dataset,
        cell_line=cell_line,
    )
