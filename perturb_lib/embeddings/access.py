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

Interface module for all the predefined embeddings-related operations.
"""

import polars as pl
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from perturb_lib.data import ControlSymbol
from perturb_lib.embeddings._utils import embedding_format_verification
from perturb_lib.environment import logger
from perturb_lib.utils import update_symbols

embedding_catalogue: dict[str, type[pl.DataFrame]] = {}


def list_embeddings() -> list[str]:
    """Get IDs of registered embeddings.

    Returns:
        A list of embedding identifiers that are registered in Perturb-lib.
    """
    return sorted(list(embedding_catalogue.keys()))


def _verify_that_embedding_exists(embedding_id: str):
    if embedding_id not in list_embeddings():
        raise ValueError(f"Unavailable embedding {embedding_id}: chose one of {list_embeddings()}")


def describe_embedding(embedding_id: str) -> str | None:
    """Describe specified embedding.

    Args:
        embedding_id: Identifier of the embedding.

    Returns:
        The description of the embedding given as a string.

    Raises:
        ValueError: If given embedding identifier is not recognized.
    """
    _verify_that_embedding_exists(embedding_id)
    return embedding_catalogue[embedding_id].__doc__


def load_embedding(
    embedding_id: str, symbol_prefix: str = "", num_pca_dims: int | None = None, hgnc_renaming: bool = False
) -> pl.DataFrame:
    """Load specified embedding as ``DataFrame``.

    NOTE: pl.DataFrame contain a column named 'index' that holds the symbols but it is not an actual index
    as polars DataFrames does not have the concept of an index.

    Args:
        embedding_id:  ID of the embedding.
        symbol_prefix: prefix to add on all symbols in the vocabulary.
        num_pca_dims: If not None, reduce the embedding to this number of dimensions via PCA.

    Returns:
        The corresponding embedding map.

    Raises:
        ValueError: when the given embedding identifier is not registered.
    """
    _verify_that_embedding_exists(embedding_id)

    embedding: pl.DataFrame = embedding_catalogue[embedding_id]()
    if hgnc_renaming:
        embedding = update_symbols(embedding)

    embedding_ldf: pl.LazyFrame = embedding.lazy()
    # Cast columns to Float32 and add prefix to column 'index' (excluding ControlSymbol)
    embedding_ldf = embedding_ldf.with_columns(
        pl.when(pl.col("index") == ControlSymbol)
        .then(pl.lit(ControlSymbol))  # no prefix needed for ControlSymbol
        .otherwise(symbol_prefix + pl.col("index"))
        .alias("index"),
        pl.all().exclude("index").cast(pl.Float32),
    )

    # Set all columns except 'index' to 0.0 when index == ControlSymbol
    embedding_ldf = embedding_ldf.with_columns(
        pl.when(pl.col("index") == ControlSymbol).then(0.0).otherwise(pl.all().exclude("index")).name.keep(),
    )

    # Split the ControlSymbol row from the rest of the embedding to prepend it to the top
    embedding_ldf_ctrl = embedding_ldf.filter(pl.col("index") == ControlSymbol)
    embedding_ldf_not_ctrl = embedding_ldf.filter(pl.col("index") != ControlSymbol)
    embedding_ldf = pl.concat([embedding_ldf_ctrl, embedding_ldf_not_ctrl], how="vertical")

    # Execute all operations
    embedding = embedding_ldf.collect()

    # reduce the dimensionality of embedding if specified
    if num_pca_dims is not None:
        logger.info(f"Reducing embedding to {num_pca_dims} dimensions using PCA.")
        # note that we need to fix the random state to always produce equal embedding (important for some PCA solvers)
        dim_red_pipeline = Pipeline([("scaler", StandardScaler()), ("pca", PCA(num_pca_dims, random_state=13))])

        index_col = embedding.select("index")
        old_data_matrix = embedding.drop("index")

        new_data_matrix: NDArray = dim_red_pipeline.fit_transform(old_data_matrix).astype("float32")
        new_column_names = ["PCA_" + c for c in old_data_matrix.columns[:num_pca_dims]]

        embedding = pl.concat(
            [index_col, pl.DataFrame(data=new_data_matrix, schema=new_column_names)], how="horizontal"
        )

        # ControlSymbol should remain 0
        embedding = embedding.with_columns(
            pl.when(pl.col("index") == ControlSymbol).then(0.0).otherwise(pl.all().exclude("index")).name.keep(),
        )

    embedding_format_verification(embedding)
    return embedding


def encode_embedding(embedding: pl.DataFrame, vocab: pl.DataFrame) -> pl.DataFrame:
    """Encode an embedding with a given vocabulary.

    Args:
        embedding: the embedding to be encoded.
        vocab: the vocabulary to be used.

    Returns:
        encoded embedding.
    """
    try:
        embedding = embedding.with_columns(index=pl.col("index").replace_strict(vocab["symbol"], vocab["code"]))
    except pl.exceptions.InvalidOperationError:
        raise RuntimeError("Certain symbols in embedding do not have a match in the provided vocabulary.")

    embedding = embedding.with_columns(pl.all().exclude("index").cast(pl.Float32))
    embedding = embedding.sort(by="index")
    embedding_format_verification(embedding)

    return embedding


def register_embedding(embedding_class: type[pl.DataFrame]):
    """Register new embedding to the collection.

    Example::

            import polars as pl
            import perturb_lib as plib


            @plib.register_embedding
            class CoolEmbedding(pl.DataFrame):
                def __init__(self):
                    super().__init__(data={"feature": [1.1, 2.2]}, index=["PSMA1", "STAT1"])

    Args:
        embedding_class: embedding class to register
    Raises:
        ValueError: If embedding class with the same name exists already.
    """
    context = embedding_class.__name__
    if context in list_embeddings() and (
        getattr(embedding_class, "__module__", "") != getattr(embedding_catalogue[context], "__module__", "")
        or embedding_class.__qualname__ != embedding_catalogue[context].__qualname__
    ):
        raise ValueError(f"Existing id {context} already registered for a different dataset!")
    embedding_catalogue[context] = embedding_class
    return embedding_class
