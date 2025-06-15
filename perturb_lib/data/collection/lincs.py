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

Loading and processing of LINCS (CMap) data.
"""

import warnings
from typing import Literal

import numpy as np
import pandas as pd
from cmapPy.pandasGEXpress.parse import parse

from perturb_lib._utils import download_file
from perturb_lib.data import ControlSymbol
from perturb_lib.data.access import create_and_register_context
from perturb_lib.environment import get_path_to_cache, logger

# URLs were found here: https://clue.io/releases/data-dashboard
CRISPR_URL = "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level5/level5_beta_trt_xpr_n142901x12328.gctx"
CMP_URL = "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level5/level5_beta_trt_cp_n720216x12328.gctx"
path_to_raw_dataset = get_path_to_cache() / "raw_datasets" / "LINCS"
apply_aggressive_filtering = False


# fmt: off
# 19 cell lines from CRISPR studies that have at least 50 samples after basic filtering (though they have >500 samples)
crispr_cell_lines = ['U251MG', 'ES2', 'A549', 'A375', 'BICR6', 'AGS', 'HT29', 'HS944T', 'IPC298', 'KYSE30', 'KELLY',
                     'DANG', 'SNU761', 'SNGM', 'MCF7', 'YAPC', 'LCLC103H', 'PC3', 'HCC1806']

# 83 cell lines from drug studies that have at least 50 samples after basic filtering
compound_cell_lines = ['PC3', 'MCF7', 'A375', 'HA1E', 'A549', 'HT29', 'MCF10A', 'HELA', 'VCAP', 'HCC515', 'HEPG2',
                       'YAPC', 'MDAMB231', 'NPC', 'THP1', 'XC.L10', 'HEK293', 'ASC', 'XC.R10', 'JURKAT', 'HUVEC',
                       'SKL', 'HCT116', 'HAP1', 'SKBR3', 'BT20', 'U2OS', 'HS578T', 'HME1', 'CD34', 'XC.P934', 'XC.P935',
                       'XC.P933', 'XC.P092', 'XC.P091', 'XC.P908', 'K562', 'XC.P904', 'WA09', 'XC.P930', 'HFL1', 'NL20',
                       'XC.P915', 'XC.P026', 'XC.P909', 'LNCAP', 'XC.P031', 'XC.P914', 'SKMEL5', 'XC.P910', 'XC.P905',
                       'XC.P911', 'XC.P912', 'KMS34', 'XC.P033', 'XC.P932', 'P1A82', 'TMD8', 'XC.P906', 'XC.P936',
                       '1HAE', 'XC.P931', 'XC.P901', 'IMR90', 'MINO', 'WI38', 'XC.P922', 'BJAB', 'OCILY3', 'NALM6',
                       'XC.P907', 'OCILY19', 'SKB', 'HBL1', 'HUES3', 'SHSY5Y', 'OCILY10', 'HPTEC', 'MNEU', 'SKNSH',
                       'PHH', 'NKDBA', 'HUH7']

dosages = ['10uM']
# fmt: on


def _get_lincs(pert_type: Literal["CRISPR-KO", "Compounds"], cell_line: str):
    """Get AnnData elements for LINCS data based on specified arguments."""
    if pert_type == "CRISPR-KO":
        filename, data_url = "crispr.gctx", CRISPR_URL
    elif pert_type == "Compounds":
        filename, data_url = "compounds.gctx", CMP_URL
    else:
        raise ValueError("Unknown perturbation type")

    # get data matrix
    logger.info(f"Loading Lincs 2020 data ({pert_type}, Level 5, Phase II) [cell line = {cell_line}] ...")
    path_to_raw_dataset.mkdir(exist_ok=True, parents=True)
    data_path = path_to_raw_dataset / filename
    download_file(data_url, save_path=data_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # silence annoying warning due to obsolete cmapPy library
        data = parse(str(data_path)).data_df.transpose()

    # create AnnData obs object by extracting meta data
    meta_url = "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/siginfo_beta.txt"
    meta_path = path_to_raw_dataset / "siginfo.txt"
    download_file(meta_url, save_path=meta_path)
    siginfo_df = pd.read_csv(meta_path, delimiter="\t", index_col="sig_id", low_memory=False)
    obs = siginfo_df.loc[data.index]
    if pert_type == "CRISPR-KO":
        obs = obs.rename(columns={"cmap_name": "perturbation_target", "cell_iname": "cell_line"})
        obs.loc[obs.pert_type == "ctl_untrt", "perturbation_target"] = ControlSymbol
        obs["perturbation_type"] = "CRISPR-KO"
        obs = obs[["perturbation_type", "perturbation_target", "is_hiq", "cell_line", "cc_q75", "pct_self_rank_q25"]]
    else:
        obs = obs.rename(
            columns={
                "cmap_name": "perturbation",
                "pert_idose": "perturbation_dose",
                "pert_itime": "time",
                "cell_iname": "cell_line",
            }
        )
        obs = obs[["perturbation", "is_hiq", "cell_line", "perturbation_dose", "time", "cc_q75", "pct_self_rank_q25"]]
        obs["perturbation_dose"] = obs["perturbation_dose"].str.replace(r"\s", "", regex=True)  # remove whitespace
        obs["time"] = obs["time"].str.replace(r"\s", "", regex=True)  # remove whitespace

    # create AnnData var object
    geneinfo_url = "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/geneinfo_beta.txt"
    geneinfo_path = path_to_raw_dataset / "geneinfo.txt"
    download_file(geneinfo_url, save_path=geneinfo_path)
    geneinfo = pd.read_csv(geneinfo_path, delimiter="\t", index_col="gene_id")
    geneinfo.index = geneinfo.index.map(str)
    data.columns = data.columns.to_series().replace(geneinfo.gene_symbol)
    landmark_genes = list(geneinfo[geneinfo.feature_space == "landmark"].gene_symbol)  # Drop inferred gene expression
    data = data[landmark_genes]
    # cleanup
    var = pd.DataFrame(list(data.columns), columns=["readout_target"])
    var["readout_type"] = "Transcriptome"
    var = var[["readout_type", "readout_target"]]

    # observations filtering
    to_keep = np.ones(len(obs), dtype=bool)
    if apply_aggressive_filtering:
        to_keep = to_keep & (obs["cc_q75"] >= 0.2) & (obs["pct_self_rank_q25"] <= 0.05)
    # remove nans, duplicates, and low quality samples, and integrate dosage and time information for compounds
    if pert_type == "CRISPR-KO":
        to_keep = (
            to_keep
            & ~obs.duplicated()
            & obs.perturbation_target.notna()
            & (obs.is_hiq == 1)
            & (obs.cell_line == cell_line)
        )
        obs = obs[["perturbation_type", "perturbation_target"]]
    else:
        to_keep = (
            to_keep
            & ~obs.duplicated()
            & obs.perturbation.notna()
            & (obs.is_hiq == 1)
            & (obs.cell_line == cell_line)
            & obs.perturbation_dose.notna()
            & obs.perturbation_dose.isin(dosages)
            & (obs.time == "24h")
        )
        obs.perturbation = obs.perturbation + "-" + obs.perturbation_dose
        obs = obs[["perturbation"]]

    data = data[to_keep]
    obs = obs[to_keep]

    obs.index = obs.index.astype(str)
    var.index = var.index.astype(str)

    return data.to_numpy(), obs, var


for cell_line in crispr_cell_lines:
    create_and_register_context(
        model_system="HumanCellLine",
        model_system_id=cell_line,
        technology_info="L1000-RNA-seq",
        data_source_info="LINCS-CMap2020_XPR",
        batch_info=None,
        full_context_description=f"Human cell line {cell_line} prepared for CRISPR-KO perturbations as described in the"
        f" `LINCS project website <https://clue.io/data/CMap2020#LINCS2020>`_. Bulk RNA sequencing was performed using "
        f"L1000 assay.",
        anndata_fn=_get_lincs,
        pert_type="CRISPR-KO",
        cell_line=cell_line,
    )

for cell_line in compound_cell_lines:
    create_and_register_context(
        model_system="HumanCellLine",
        model_system_id=cell_line,
        technology_info="L1000-RNA-seq",
        data_source_info="LINCS-CMap2020_CMP",
        batch_info=None,
        full_context_description=f"Human cell line {cell_line} prepared for compound perturbations as described in the "
        f"`LINCS project website <https://clue.io/data/CMap2020#LINCS2020>`_. Bulk RNA sequencing was performed using "
        f"L1000 assay.",
        anndata_fn=_get_lincs,
        pert_type="Compounds",
        cell_line=cell_line,
    )
