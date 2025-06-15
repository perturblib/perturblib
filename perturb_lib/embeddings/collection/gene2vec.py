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

Gene2vec embeddings from the corresponding paper.
"""

import polars as pl

from perturb_lib._utils import download_file
from perturb_lib.embeddings.access import register_embedding
from perturb_lib.environment import get_path_to_cache


@register_embedding
class Gene2vec(pl.DataFrame):
    """`2019 Du et al <https://github.com/jingcheng-du/Gene2vec>`_

    Paper:
        "Gene2vec: distributed representation of genes based on co-expression"
    Institution of corresponding author:
        The University of Texas School of Biomedical Informatics
    Vector length:
        200
    Data used to build the embedding:
        Gene co-expression data from 985 datasets scraped from GEO
    Method used to build the embedding:
        Optimization-based graph representation learning
    """

    def __init__(self):
        url = "https://github.com/jingcheng-du/Gene2vec/raw/7617af7/pre_trained_emb/gene2vec_dim_200_iter_9_w2v.txt"
        csv_path = get_path_to_cache() / "raw_embeddings" / "gene2vec_dim_200_iter_9_w2v.tsv"
        download_file(url, csv_path)
        embedding = pl.read_csv(
            csv_path,
            skip_rows=1,
            has_header=False,
            separator=" ",
            columns=list(range(201)),  # Load columns [0, 200] (201 cols). Column 201 is all nulls so it's skipped
            new_columns=["index", *[f"gene2vec_{i:03d}" for i in range(200)]],  # First column is 'index'
        )
        embedding = embedding.sort(by="index")
        super().__init__(data=embedding)
