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

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scanpy import AnnData

import perturb_lib as plib


def test_context_registration():
    @plib.register_context
    class DummyContext(AnnData):
        def __init__(self):
            super().__init__(
                X=np.zeros((4, 5), dtype=np.float32),
                obs={"perturbation": [plib.ControlSymbol, "P1", "P2", "P3"]},
                var={"readout": ["R1", "R2", "R3", "R4", "R5"]},
            )

    context = DummyContext.__name__

    dummy_anndata = plib.load_anndata(context_id=context)
    assert isinstance(dummy_anndata, DummyContext)


def test_embedding_registration():
    @plib.register_embedding
    class CoolEmbedding(pl.DataFrame):
        def __init__(self):
            super().__init__(data={"index": [plib.ControlSymbol, "STAT1"], "feature": [1.1, 2.2]})

    embedding_id = CoolEmbedding.__name__

    dummy_embedding = plib.load_embedding(embedding_id=embedding_id)

    # A limitation of the current implementation is that the loaded embedding is not an instance of the registered class
    assert isinstance(dummy_embedding, pl.DataFrame)


def test_model_registration():
    @plib.register_model
    class CoolModel(plib.ModelMixin):
        def fit(self, traindata: plib.PlibData, valdata: plib.PlibData | None = None):
            pass

        def predict(self, data_x: plib.PlibData, batch_size: int | None = None) -> NDArray:
            return np.zeros(len(data_x))

    model_id = CoolModel.__name__

    model = plib.load_model(model_id)
    assert isinstance(model, CoolModel)


def test_evaluator_registration():
    @plib.register_evaluator
    class CoolEvaluator(plib.PlibEvaluatorMixin):
        def evaluate(self, predictions, true_values, context_adata=None) -> float:
            return 0.0

    evaluator_id = CoolEvaluator.__name__
    evaluator = plib.load_evaluator(evaluator_id)
    assert isinstance(evaluator, CoolEvaluator)
