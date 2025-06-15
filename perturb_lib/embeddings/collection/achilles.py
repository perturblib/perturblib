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

Embeddings extracted from the Achilles project.
"""

import polars as pl

from perturb_lib._utils import download_file
from perturb_lib.environment import get_path_to_cache


# Need to review whether to use this context. Note this is the new version, not the retracted 2020Q4 version
class Achilles(pl.DataFrame):
    """`2015 Broad Institute, Depmap <https://depmap.org>`_

    Project:
        Achilles
    Paper:
        "Extracting Biological Insights from the Project Achilles Genome-Scale CRISPR Screens in Cancer Cell Lines"
    Institution of corresponding author:
        Broad Institute
    Source:
        https://depmap.org/portal/download/ 2022Q2 Achilles_gene_effect.csv
    Vector length:
        957
    Data used to build the embedding:
        Achilles context
    Method used to build the embedding:
        No method was used but raw data
    """

    def __init__(self):
        url = "https://ndownloader.figshare.com/files/34989889"
        csv_file_path = get_path_to_cache() / "raw_embeddings" / "achilles_gene_effect_2022Q2.csv"
        download_file(url, csv_file_path)

        # CSV has Columns: genes in format "HUGO (Entrez)", Rows: cell lines
        embedding = pl.read_csv(csv_file_path)

        # Transpose genes into rows
        embedding = embedding.transpose(include_header=True, header_name="index", column_names="DepMap_ID")
        embedding = embedding.sort(by="index")

        # Drop the " (Entrez)" part of the gene names
        embedding = embedding.with_columns(index=pl.col("index").str.replace_all(" \\(\\d+\\)", ""))
        super().__init__(data=embedding)
