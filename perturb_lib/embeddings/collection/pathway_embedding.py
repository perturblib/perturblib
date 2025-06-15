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

Embeddings generated from the pathway databases.
"""

import zipfile
from pathlib import Path
from typing import Dict, List

import polars as pl
from polars import selectors as cs

from perturb_lib._utils import download_file
from perturb_lib.embeddings.access import register_embedding
from perturb_lib.environment import get_path_to_cache


def _get_embedding_from_pathway_dict(pathway_dict: dict[str, list[str]]) -> pl.DataFrame:
    # Create a long dataframe representing gene-pathway pairs
    long_dataframes: list[pl.DataFrame] = []
    for pathway, genes in pathway_dict.items():
        df = pl.DataFrame(data=genes, schema=["gene"]).with_columns(pathway=pl.lit(pathway))
        long_dataframes.append(df)
    long_df: pl.DataFrame = pl.concat(long_dataframes).with_columns(value=pl.lit(1.0))
    del long_dataframes

    embedding = long_df.pivot(on="pathway", index="gene", values="value").rename({"gene": "index"})
    embedding = embedding.sort(by="index")
    embedding = embedding.fill_null(0.0)

    return embedding


@register_embedding
class ReactomePathway(pl.DataFrame):
    """`REACTOME pathway database <https://reactome.org>`_

    Institution of corresponding author:
        OICR, OHSU, EMBL-EBI and NYULMC
    Vector length:
        2593
    Data used to build the embedding:
        Reactome Pathway database <https://reactome.org/>
    """

    DIMENSIONALITY = 200

    def __init__(self):
        url = "https://reactome.org/download/current/ReactomePathways.gmt.zip"
        file_path = get_path_to_cache() / "raw_embeddings" / "ReactomePathways.gmt.zip"
        download_file(url, file_path)
        pathway_dict = self._get_reactome_pathway_gene_dict(file_path)

        # Create embedding with a 1.0 for each gene-pathway pair
        embedding = _get_embedding_from_pathway_dict(pathway_dict)

        # Drop all columns except the first self.DIMENSIONALITY + 1 columns
        embedding = embedding.drop(~cs.by_index(range(0, self.DIMENSIONALITY + 1)))

        super().__init__(data=embedding)

    @staticmethod
    def _get_reactome_pathway_gene_dict(file_path: Path) -> Dict[str, List[str]]:
        """Extracts the zip file to pathway: genes dictionary"""
        all_pathways = {}
        with zipfile.ZipFile(file_path) as zp:
            with zp.open(zp.namelist()[0]) as fp:
                for line in fp:
                    tokens = line.decode("utf-8").rstrip().split("\t")
                    pathway_name, members = str(tokens[0]), tokens[2:]
                    all_pathways[pathway_name] = members
        # sort pathways by occurence
        all_pathways = {
            pathway: genes
            for pathway, genes in sorted(all_pathways.items(), key=lambda item: len(item[1]), reverse=True)
        }
        return all_pathways


@register_embedding
class Replogle22Pathway(pl.DataFrame):
    """`2022 Replogle et al <https://www.cell.com/cell/pdf/S0092-8674(22)00597-9.pdf>`_

    Paper:
        "Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq"
    Institution of corresponding author:
        MSKCC, MIT
    Vector length:
        8
    Data used to build the embedding:
        Known pathways for different genes
    Method used to build the embedding:
        One Hot Encoding of pathways
    """

    def __init__(self):
        pathway_dict = self._get_replogle_pathway_gene_dict()
        embedding = _get_embedding_from_pathway_dict(pathway_dict)

        super().__init__(data=embedding)

    @staticmethod
    def _get_replogle_pathway_gene_dict() -> Dict:
        EXOSOME = [
            "ZC3H3",
            "ZFC3H1",
            "CAMTA2",
            "DHX29",
            "DIS3",
            "EXOSC1",
            "EXOSC2",
            "EXOSC3",
            "EXOSC4",
            "EXOSC5",
            "EXOSC6",
            "EXOSC7",
            "EXOSC8",
            "EXOSC9",
            "MBNL1",
            "PABPN1",
            "PIBF1",
            "MTREX",
            "ST20-MTHFS",
            "THAP2",
        ]
        SPLICEOSOME = [
            "ZMAT2",
            "CLNS1A",
            "DDX20",
            "DDX41",
            "DDX46",
            "ECD",
            "GEMIN4",
            "GEMIN5",
            "GEMIN6",
            "GEMIN8",
            "INTS3",
            "INTS4",
            "INTS9",
            "ICE1",
            "LSM2",
            "LSM3",
            "LSM5",
            "LSM6",
            "LSM7",
            "MMP17",
            "PHAX",
            "PRPF4",
            "PRPF6",
            "SART3",
            "SF3A2",
            "SMN2",
            "SNAPC1",
            "SNAPC3",
            "SNRPD3",
            "SNRPG",
            "TIPARP",
            "TTC27",
            "TXNL4A",
            "USPL1",
        ]
        MEDIATOR_COMPLEX = [
            "ZDHHC7",
            "ADAM10",
            "EPS8L1",
            "FAM136A",
            "POGLUT3",
            "MED10",
            "MED11",
            "MED12",
            "MED14",
            "MED17",
            "MED18",
            "MED19",
            "MED1",
            "MED20",
            "MED21",
            "MED22",
            "MED28",
            "MED29",
            "MED30",
            "MED6",
            "MED7",
            "MED8",
            "MED9",
            "SUPT6H",
            "BRIX1",
            "TMX2",
        ]
        NUCLEOTIDE_EXCISION_REPAIR = [
            "C1QBP",
            "CCNH",
            "ERCC2",
            "ERCC3",
            "GPN1",
            "GPN3",
            "GTF2E1",
            "GTF2E2",
            "GTF2H1",
            "GTF2H4",
            "MNAT1",
            "NUMA1",
            "PDRG1",
            "PFDN2",
            "POLR2B",
            "POLR2F",
            "POLR2G",
            "RPAP1",
            "RPAP2",
            "RPAP3",
            "TANGO6",
            "TMEM161B",
            "UXT",
        ]
        S40_RIBOSOMAL_UNIT = [
            "ZCCHC9",
            "ZNF236",
            "C1orf131",
            "ZNF84",
            "ZNHIT6",
            "CCDC59",
            "AATF",
            "CPEB1",
            "DDX10",
            "DDX18",
            "DDX21",
            "DDX47",
            "DDX52",
            "DHX33",
            "DHX37",
            "DIMT1",
            "DKC1",
            "DNTTIP2",
            "ESF1",
            "FBL",
            "FBXL14",
            "FCF1",
            "GLB1",
            "HOXA3",
            "IMP4",
            "IMPA2",
            "KRI1",
            "KRR1",
            "LTV1",
            "MPHOSPH10",
            "MRM1",
            "NAF1",
            "NOB1",
            "NOC4L",
            "NOL6",
            "NOP10",
            "PDCD11",
            "ABT1",
            "PNO1",
            "POP1",
            "POP4",
            "POP5",
            "PSMG4",
            "PWP2",
            "RCL1",
            "RIOK1",
            "RIOK2",
            "RNF31",
            "RPP14",
            "RPP30",
            "RPP40",
            "RPS10-NUDT3",
            "RPS10",
            "RPS11",
            "RPS12",
            "RPS13",
            "RPS15A",
            "RPS18",
            "RPS19BP1",
            "RPS19",
            "RPS21",
            "RPS23",
            "RPS24",
            "RPS27A",
            "RPS27",
            "RPS28",
            "RPS29",
            "RPS2",
            "RPS3A",
            "RPS3",
            "RPS4X",
            "RPS5",
            "RPS6",
            "RPS7",
            "RPS9",
            "RPSA",
            "RRP12",
            "RRP7A",
            "RRP9",
            "SDR39U1",
            "SRFBP1",
            "TBL3",
            "TRMT112",
            "TSR1",
            "TSR2",
            "BYSL",
            "C12orf45",
            "USP36",
            "UTP11",
            "UTP20",
            "UTP23",
            "UTP6",
            "BUD23",
            "WDR36",
            "WDR3",
            "WDR46",
            "AAR2",
        ]
        S39_RIBOSOMAL_UNIT = [
            "AARS2",
            "DHX30",
            "GFM1",
            "HMGB3",
            "MALSU1",
            "MRPL10",
            "MRPL11",
            "MRPL13",
            "MRPL14",
            "MRPL16",
            "MRPL17",
            "MRPL18",
            "MRPL19",
            "MRPL22",
            "MRPL23",
            "MRPL24",
            "MRPL27",
            "MRPL2",
            "MRPL33",
            "MRPL35",
            "MRPL36",
            "MRPL37",
            "MRPL38",
            "MRPL39",
            "MRPL3",
            "MRPL41",
            "MRPL42",
            "MRPL43",
            "MRPL44",
            "MRPL4",
            "MRPL50",
            "MRPL51",
            "MRPL53",
            "MRPL55",
            "MRPL9",
            "MRPS18A",
            "MRPS30",
            "NARS2",
            "PTCD1",
            "RPUSD4",
            "TARS2",
            "VARS2",
            "YARS2",
        ]
        S60_RIBOSOMAL_UNIT = [
            "CARF",
            "CCDC86",
            "DDX24",
            "DDX51",
            "DDX56",
            "EIF6",
            "ABCF1",
            "GNL2",
            "LSG1",
            "MAK16",
            "MDN1",
            "MYBBP1A",
            "NIP7",
            "NLE1",
            "NOL8",
            "NOP16",
            "NVL",
            "PES1",
            "PPAN",
            "RBM28",
            "RPL10A",
            "RPL10",
            "RPL11",
            "RPL13",
            "RPL14",
            "RPL17",
            "RPL19",
            "RPL21",
            "RPL23A",
            "RPL23",
            "RPL24",
            "RPL26",
            "RPL27A",
            "RPL30",
            "RPL31",
            "RPL32",
            "RPL34",
            "RPL36",
            "RPL37A",
            "RPL37",
            "RPL38",
            "RPL4",
            "RPL5",
            "RPL6",
            "RPL7",
            "RPL8",
            "RPL9",
            "RRS1",
            "RSL1D1",
            "SDAD1",
            "BOP1",
            "TEX10",
            "WDR12",
        ]
        MT_PROTEIN_TRANSLOCATION = [
            "AARS",
            "CHCHD4",
            "DNAJA3",
            "DNAJC19",
            "EIF2B1",
            "EIF2B2",
            "EIF2B3",
            "EIF2B4",
            "EIF2B5",
            "FARSA",
            "FARSB",
            "GFER",
            "GRPEL1",
            "HARS",
            "HSPA9",
            "HSPD1",
            "HSPE1",
            "IARS2",
            "LARS",
            "LETM1",
            "NARS",
            "OXA1L",
            "PGS1",
            "PHB2",
            "PHB",
            "PMPCA",
            "PMPCB",
            "ATP5F1A",
            "ATP5F1B",
            "ATP5PD",
            "QARS",
            "RARS",
            "SAMM50",
            "PRELID3B",
            "TARS",
            "TIMM23B",
            "TIMM44",
            "TOMM22",
            "TTC1",
            "VARS",
        ]

        pathway_dict = {
            "exosome": EXOSOME,
            "spliceosome": SPLICEOSOME,
            "mediator_complex": MEDIATOR_COMPLEX,
            "nucleotide_excision_repair": NUCLEOTIDE_EXCISION_REPAIR,
            "s40_ribosomal_unit": S40_RIBOSOMAL_UNIT,
            "s39_ribosomal_unit": S39_RIBOSOMAL_UNIT,
            "s60_ribosomal_unit": S60_RIBOSOMAL_UNIT,
            "mt_protein_translocation": MT_PROTEIN_TRANSLOCATION,
        }

        return pathway_dict
