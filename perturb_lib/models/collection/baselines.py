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

Different perturbation model baselines.
"""

from abc import ABCMeta

import numpy as np
import polars as pl
from numpy.typing import NDArray
from sklearn.dummy import DummyRegressor

from perturb_lib._utils import try_import
from perturb_lib.data import ControlSymbol
from perturb_lib.data.plibdata import PlibData
from perturb_lib.environment import get_path_to_cache, get_seed, logger
from perturb_lib.models.access import register_model
from perturb_lib.models.base import ModelMixin, SklearnModel
from perturb_lib.utils import inherit_docstring


@register_model
@inherit_docstring
class GlobalMean(ModelMixin):
    """Computes mean value from the training data and then uses it to make predictions.

    Note:
        Current implementation does not scale to datasets larger than RAM.
    """

    def __init__(self):
        self.dummy_regressor = DummyRegressor(strategy="mean")

    def fit(self, traindata: PlibData[pl.DataFrame], valdata: PlibData[pl.DataFrame] | None = None):  # noqa: D102
        # valdata is ignored completely since it's not needed
        traindata_df = traindata[:]
        y = traindata_df["value"]
        self.dummy_regressor.fit(np.zeros_like(y), y)  # X is ignored so we can put whatever

    def predict(self, data_x: PlibData[pl.DataFrame], batch_size: int | None = None) -> NDArray:  # noqa: D102
        return self.dummy_regressor.predict(np.zeros(len(data_x)))


@register_model
class ReadoutMean(ModelMixin):
    """Predicts mean readout value without taking perturbation information into account.

    Note:
        Current implementation does not scale to datasets larger than RAM.
    """

    def __init__(self):
        self.context_readout2mean = None

    def fit(self, traindata: PlibData[pl.DataFrame], valdata: PlibData[pl.DataFrame] | None = None):  # noqa: D102
        # valdata is ignored completely since it's not needed
        traindata_df: pl.DataFrame = traindata[:]
        self.context_readout2mean = traindata_df.group_by(["context", "readout"], maintain_order=True).agg(
            value=pl.col("value").mean()
        )

    def predict(self, data_x: PlibData[pl.DataFrame], batch_size: int | None = None) -> NDArray:  # noqa: D102
        if self.context_readout2mean is None:
            raise RuntimeError("One must fit the model before making predictions!")
        data_x_df: pl.DataFrame = data_x[:].drop("perturbation", "value", strict=False)  # Ok if 'value' was missing
        result_df = data_x_df.join(
            self.context_readout2mean, on=["context", "readout"], how="left", maintain_order="left"
        ).with_columns(value=pl.col("value").fill_null(self.context_readout2mean["value"].mean()))
        return result_df["value"].to_numpy()


@register_model
class NoPerturb(ModelMixin):
    """Replaces any perturbation symbols with a no-perturbation symbol.

    Note:
        Current implementation does not scale to datasets larger than RAM.
    """

    def __init__(self):
        self.context_readout2value = None

    def fit(self, traindata: PlibData[pl.DataFrame], valdata: PlibData[pl.DataFrame] | None = None):  # noqa: D102
        # valdata is ignored completely since it's not needed
        traindata_df: pl.DataFrame = traindata[:]
        self.context_readout2value = traindata_df.filter(pl.col("perturbation") == ControlSymbol)
        self.context_readout2value = self.context_readout2value.drop("perturbation")

    def predict(self, data_x: PlibData[pl.DataFrame], batch_size: int | None = None) -> NDArray:  # noqa: D102
        if self.context_readout2value is None:
            raise RuntimeError("One must fit the model before making predictions!")
        data_x_df: pl.DataFrame = data_x[:].drop("perturbation", "value", strict=False)  # Ok if 'value' was missing
        data_x_df = data_x_df.join(
            self.context_readout2value, on=["context", "readout"], how="left", maintain_order="left"
        )
        return data_x_df["value"].to_numpy()


@register_model
class Catboost(SklearnModel, metaclass=ABCMeta):
    """CatBoostRegressor used on top of predefined embeddings."""

    def __init__(
        self,
        embedding_id="ReactomePathway",
        symbol_prefix: str | None = None,
        **model_kwargs,
    ):
        try_import("catboost")
        from catboost import CatBoostRegressor

        model_kwargs["random_seed"] = get_seed()
        model_kwargs["train_dir"] = get_path_to_cache()

        super().__init__(
            embedding_id=embedding_id,
            symbol_prefix=symbol_prefix,
            sklearn_model_type=CatBoostRegressor,
            num_pca_dims=20,
            **model_kwargs,
        )

    def fit(self, traindata: PlibData[pl.DataFrame], valdata: PlibData[pl.DataFrame] | None = None):  # noqa: D102
        logger.info("Loading training data into RAM..")
        traindata_df: pl.DataFrame = traindata[:]
        self._initialize_vocabularies_and_embeddings(traindata_df)

        logger.info(f"Embedding training data of size {len(traindata_df)}..")
        logger.info("Creating X matrix")
        x = self.embed(traindata_df)
        logger.info("Creating y vector")
        y = traindata_df["value"].to_numpy()

        if valdata is not None:
            logger.info("Loading validation data into RAM..")
            valdata_df: pl.DataFrame = valdata[:]
            logger.info(f"Embedding validation data of size {len(valdata_df)}..")
            x_val = self.embed(valdata_df)
            y_val = valdata_df["value"].to_numpy()
            eval_set = (x_val, y_val)
        else:
            eval_set = None

        logger.info(f"Fitting {type(self.sklearn_model).__name__} on data of shape {x.shape}..")
        self.sklearn_model.fit(x, y, eval_set=eval_set)
        logger.info("Model fitting completed")
