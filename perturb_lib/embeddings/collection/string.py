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

Embeddings extracted from the STRING database.
"""

import polars as pl

from perturb_lib._utils import download_file
from perturb_lib.embeddings.access import register_embedding
from perturb_lib.environment import get_path_to_cache


@register_embedding
class STRING(pl.DataFrame):
    """`2016 Cho et al <https://cb.csail.mit.edu/cb/mashup>`_

    Paper:
        "Compact Integration of Multi-Network Topology for Functional Analysis of Genes"
    Authors:
        Hyunghoon Cho, Bonnie Berger, Jian Peng
    Institution of corresponding author:
        MIT, Illinois
    Source:
        https://cb.csail.mit.edu/cb/mashup/
    Vector length:
        800
    Data used to build the embedding:
        String database
    Method used to build the embedding:
        Optimization-based graph representation learning
    """

    def __init__(self):
        genes_url = "https://groups.csail.mit.edu/cb/mashup/vectors/string_human_genes.txt"
        embedding_url = "https://groups.csail.mit.edu/cb/mashup/vectors/string_human_mashup_vectors_d800.txt"
        genes_path = get_path_to_cache() / "raw_embeddings" / "string_human_genes.tsv"
        embedding_path = get_path_to_cache() / "raw_embeddings" / "string_human_mashup_vectors_d800.tsv"

        download_file(genes_url, genes_path)
        download_file(embedding_url, embedding_path)
        gene_column = pl.read_csv(genes_path, has_header=False, separator="\t", new_columns=["index"])
        embedding = pl.read_csv(
            embedding_path, has_header=False, separator="\t", new_columns=[f"STRING_{i:03d}" for i in range(800)]
        )
        embedding = pl.concat([gene_column, embedding], how="horizontal")
        embedding = embedding.sort(by="index")

        super().__init__(data=embedding)
