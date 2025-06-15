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

from typing import cast

import numpy as np
import polars as pl
import pytest

import perturb_lib as plib


@pytest.mark.parametrize("model_name", ["GlobalMean", "ReadoutMean", "NoPerturb", "Catboost", "LPM"])
@pytest.mark.parametrize("plib_data", ["in_memory_plib_data", "on_disk_plib_data"])
def test_models_can_fit_predict(model_name: str, plib_data: plib.PlibData, request):
    plib_data = request.getfixturevalue(plib_data)

    model_to_args: dict[str, dict] = {
        "LPM": {
            "optimizer_name": "AdamW",
            "learning_rate": 0.001,
            "learning_rate_decay": 0.999,
            "num_layers": 1,
            "hidden_dim": 200,
            "batch_size": 10000,
            "embedding_dim": 10,
            "lightning_trainer_pars": {
                "max_epochs": 1,
                "logger": False,
                "accelerator": "cpu",
                "enable_checkpointing": False,
            },
        },
        "Catboost": {"embedding_id": "DummyEmbedding"},
    }

    model_args = model_to_args.get(model_name, {})
    model: plib.ModelMixin = plib.load_model(model_name, model_args)

    traindata, valdata, testdata = plib.split_plibdata_3fold(plib_data, "DummyData")
    testdata_x = testdata.subset_columnwise(["context", "perturbation", "readout"])

    model.fit(traindata, valdata)
    predictions = model.predict(testdata_x)
    assert isinstance(predictions, np.ndarray)


def test_model_operations():
    plib.logger.debug("Running model operations test...")
    context = "DummyData"
    data = plib.load_plibdata(context, plibdata_type=plib.InMemoryPlibData)
    traindata, valdata, _ = plib.split_plibdata_3fold(data, context)
    traindata_x = traindata.subset_columnwise(["context", "perturbation", "readout"])
    mean_value = cast(float, pl.concat([traindata[:]])["value"].mean())
    plib.logger.debug(f"Mean value across dataset: {mean_value}")

    plib.logger.debug(f"Available models: {plib.list_models()}")

    plib.logger.debug("Test GlobalMean")
    global_mean = plib.load_model("GlobalMean")
    global_mean.fit(traindata, valdata)
    predictions = global_mean.predict(traindata_x)
    plib.logger.debug(f"GlobalMean predictions: {predictions}")
    assert (predictions - mean_value < 0.0001).all(), "GlobalMean gone bad!"

    plib.logger.debug("Test pytorch model")
    mlp = plib.load_model(
        "LPM",
        model_args={
            "optimizer_name": "AdamW",
            "learning_rate": 0.001,
            "learning_rate_decay": 0.999,
            "num_layers": 1,
            "hidden_dim": 200,
            "batch_size": 10000,
            "embedding_dim": 10,
            "lightning_trainer_pars": {
                "max_epochs": 1,
                "logger": False,
                "accelerator": "cpu",
                "enable_checkpointing": False,
            },
        },
    )
    mlp.fit(traindata, valdata)
    predictions = mlp.predict(traindata_x)
    plib.logger.debug(f"Predictions: {predictions}")

    plib.logger.debug("Test successfully executed!")
