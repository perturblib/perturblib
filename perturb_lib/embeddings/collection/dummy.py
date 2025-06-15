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

Artificially generated dummy embeddings.
"""

import numpy as np
import polars as pl

from perturb_lib.data import ControlSymbol
from perturb_lib.embeddings.access import register_embedding


@register_embedding
class DummyEmbedding(pl.DataFrame):
    """Dummy embedding for testing purposes"""

    NUM_DIMENSIONS = 50
    NUM_SYMBOLS = 40

    def __init__(self):
        embedding_data_np = np.random.RandomState(42).random(size=(self.NUM_SYMBOLS, self.NUM_DIMENSIONS))
        embedding_data_np[0, :] = 0  # Set ctrl to 0
        embedding_data = pl.from_numpy(embedding_data_np, schema=[f"dim{i}" for i in range(self.NUM_DIMENSIONS)])

        index_col = pl.Series("index", [ControlSymbol] + [f"SMBL{i}" for i in range(1, self.NUM_SYMBOLS)])

        embedding = pl.concat([index_col.to_frame(), embedding_data], how="horizontal")
        super().__init__(data=embedding)
