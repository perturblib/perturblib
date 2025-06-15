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
"""

import polars as pl


def embedding_format_verification(embedding: pl.DataFrame):
    schema = embedding.schema
    cols_excl_index = [col for col in schema.keys() if col != "index"]
    if not all(schema[col] == pl.Float32 for col in cols_excl_index):
        raise ValueError("Non-floating-point column in embedding")
    if embedding.select(
        pl.any_horizontal(pl.all().exclude("index").is_null() | pl.all().exclude("index").is_nan()).any()
    ).item():
        raise ValueError("Null or NaN value in embedding")
    if embedding.select(pl.col("index").is_null().any()).item():
        raise ValueError("Null or NaN value in embedding index")
    if not embedding["index"].is_unique().any():
        raise ValueError("Non-unique index in embedding")
