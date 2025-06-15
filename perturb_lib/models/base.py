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

Mixins and base classes for the models.
"""

from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import overload

import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from torch import nn as nn

from perturb_lib.data.access import Vocabulary, encode_data
from perturb_lib.data.plibdata import InMemoryPlibData, OnDiskPlibData, PlibData
from perturb_lib.embeddings.access import encode_embedding, load_embedding
from perturb_lib.environment import logger
from perturb_lib.models._utils import OneHotEmbedding, numeric_series2tensor
from perturb_lib.utils import inherit_docstring


class ModelMixin(metaclass=ABCMeta):
    """Mixin for perturbation models."""

    @abstractmethod
    def fit(self, traindata: PlibData[pl.DataFrame], valdata: PlibData[pl.DataFrame] | None = None):
        """Model fitting.

        Args:
            traindata: Training data.
            valdata: Validation data.
        """

    @abstractmethod
    def predict(self, data_x: PlibData[pl.DataFrame], batch_size: int | None = None) -> NDArray:
        """Predict values for the given data.

        Args:
            data_x: Data without labels i.e. without the "values" column.
            batch_size: Batch size for prediction. Some models might not support this functionality

        Returns:
            Value predictions.
        """

    def save(self, path_to_model: Path, model_pars: dict):
        """Args:
        path_to_model: Path where the model should be saved.
        model_pars: Model parameters.
        """
        model_id = type(self).__name__
        if isinstance(self, nn.Module):
            torch.save((model_id, model_pars, self.state_dict()), path_to_model)
        else:
            torch.save(self, path_to_model, pickle_protocol=4)

    def load_state(self, model_state):
        """Args:
        model_state: Recovering the state of the model.
        """
        if isinstance(self, nn.Module):
            self.load_state_dict(model_state)


@overload
def to_tensor_dict(enc_data: pl.DataFrame) -> dict[str, torch.Tensor]: ...


@overload
def to_tensor_dict(enc_data: InMemoryPlibData[pl.DataFrame]) -> InMemoryPlibData[dict[str, torch.Tensor]]: ...


@overload
def to_tensor_dict(enc_data: OnDiskPlibData[pl.DataFrame]) -> OnDiskPlibData[dict[str, torch.Tensor]]: ...


@overload
def to_tensor_dict(enc_data: PlibData[pl.DataFrame]) -> PlibData[dict[str, torch.Tensor]]: ...


def to_tensor_dict(enc_data):
    """Convert a Polars DataFrame to a dictionary of PyTorch tensors.

    Args:
        enc_data: Polars DataFrame or PlibData with already encoded data.

    Returns:
        Dictionary of PyTorch tensors.
    """
    if isinstance(enc_data, PlibData):
        return enc_data.apply_transform(to_tensor_dict)

    result: dict[str, torch.Tensor] = {}

    if "context" in enc_data.columns:
        contexts_tensor = numeric_series2tensor(enc_data["context"])
        result["context"] = contexts_tensor

    if "readout" in enc_data.columns:
        readouts_tensor = numeric_series2tensor(enc_data["readout"])
        result["readout"] = readouts_tensor

    if "perturbation" in enc_data.columns:
        # convert perturbations to two tensors: flat indices and offsets
        lazy_enc_data = enc_data.lazy()
        lazy_perturbation_inds = lazy_enc_data.select(pl.col("perturbation").list.explode().alias("perturbation_flat"))
        lazy_perturbations_offsets = lazy_enc_data.select(
            pl.col("perturbation")
            .list.len()
            .shift(n=1, fill_value=0)
            .cum_sum()
            .alias("perturbation_offset")
            .cast(pl.Int64)
        )
        perturbation_inds, perturbation_offsets = pl.collect_all([lazy_perturbation_inds, lazy_perturbations_offsets])
        result["perturbation_flat"] = numeric_series2tensor(perturbation_inds["perturbation_flat"])
        result["perturbation_offset"] = numeric_series2tensor(perturbation_offsets["perturbation_offset"])

    if "value" in enc_data.columns:
        result["value"] = numeric_series2tensor(enc_data["value"])

    return result


def embed_tensor_dict(
    tensor_dict: dict[str, torch.Tensor],
    context_embedding_layer: nn.Embedding | OneHotEmbedding,
    perturb_embedding_layer: nn.EmbeddingBag,
    readout_embedding_layer: nn.Embedding,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Embed a dictionary of PyTorch tensors using specified embeddings."""
    # embed context, perturbations and readouts
    embedded_contexts: torch.Tensor = context_embedding_layer(tensor_dict["context"])
    embedded_perturbs: torch.Tensor = perturb_embedding_layer(
        tensor_dict["perturbation_flat"], tensor_dict["perturbation_offset"]
    )
    embedded_readouts: torch.Tensor = readout_embedding_layer(tensor_dict["readout"])

    return embedded_contexts, embedded_perturbs, embedded_readouts


def embed(
    batch: pl.DataFrame,
    vocab: Vocabulary | None,
    context_embedding_layer: nn.Embedding | OneHotEmbedding,
    perturb_embedding_layer: nn.EmbeddingBag,
    readout_embedding_layer: nn.Embedding,
):
    """Embed a ``DataFrame`` batch using specified vocabulary and specified embeddings.

    Args:
        batch: batch of data given as ``pl.DataFrame``
        vocab: vocabulary to encode the batch with before embedding it
        context_embedding_layer: context embedding look-up table
        perturb_embedding_layer: perturbation embedding look-up table
        readout_embedding_layer: readout embedding look-up table

    Returns:
        Tuple of embeddings.
    """
    encode_first = True
    if (
        batch["context"].dtype.is_integer()
        and batch["perturbation"].dtype.is_nested()
        and batch["readout"].dtype.is_integer()
    ):
        encode_first = False  # Already encoded

    if encode_first:
        if vocab is None:
            raise ValueError("Vocabulary must be provided to encode data.")
        enc_batch = encode_data(data=batch, vocab=vocab)
    else:
        enc_batch = batch

    tensor_dict = to_tensor_dict(enc_batch)

    # ensure tensors are on the same device as embedding layers
    device = readout_embedding_layer.weight.device
    tensor_dict = {k: v.to(device, non_blocking=True) for k, v in tensor_dict.items()}

    return embed_tensor_dict(tensor_dict, context_embedding_layer, perturb_embedding_layer, readout_embedding_layer)


@inherit_docstring
class SklearnModel(ModelMixin, metaclass=ABCMeta):
    """Base class for sklearn-style models.

    Args:
        sklearn_model_type: the class that admits basic sklearn interface.
        embedding_id: the ID of the embedding to use for perturbations.
        symbol_prefix: prefix to add on perturbation vocabulary keys
        perturb_num_pca_dims: optionally apply PCA to perturbation embeddings.
        **sklearn_model_kwargs: parameters to use to initialize sklearn model.
    """

    def __init__(
        self,
        sklearn_model_type: type,
        embedding_id: str,
        symbol_prefix: str | None = None,
        num_pca_dims: int | None = None,
        **sklearn_model_kwargs,
    ):
        super().__init__()
        self.embedding_id = embedding_id
        self.sklearn_model = sklearn_model_type(**sklearn_model_kwargs)
        self.symbol_prefix = symbol_prefix if symbol_prefix is not None else ""
        self.num_pca_dims = num_pca_dims

        # vocabulary, to be initialized upon "fit"
        self.vocab: Vocabulary | None = None

        # embeddings, to be initialized upon "fit"
        self.context_embedding_layer: OneHotEmbedding | None = None
        self.perturb_embedding_layer: nn.EmbeddingBag | None = None
        self.readout_embedding_layer: nn.Embedding | None = None

    def _initialize_vocabularies_and_embeddings(self, traindata: pl.DataFrame):
        # set up vocabularies based on given data
        self.vocab = Vocabulary.initialize_from_data(traindata)

        # load perturbation embedding and adapt vocabulary to use its symbols
        embedding = load_embedding(self.embedding_id, self.symbol_prefix, self.num_pca_dims)
        context_symbols = self.vocab.context_vocab["symbol"].to_list()
        perturb_symbols = embedding["index"].to_list()
        readout_symbols = self.vocab.readout_vocab["symbol"].to_list()
        self.vocab = Vocabulary.initialize_from_symbols(context_symbols, perturb_symbols, readout_symbols)

        encoded_embedding = encode_embedding(embedding=embedding, vocab=self.vocab.perturb_vocab)
        self.perturb_embedding_layer = nn.EmbeddingBag.from_pretrained(
            torch.from_numpy(encoded_embedding.select(pl.all().exclude("index")).to_numpy(order="c")), freeze=True
        )
        # context embeddings are one hot encodings
        self.context_embedding_layer = OneHotEmbedding(len(self.vocab.context_vocab))
        # readout embeddings are simply random values that help distinguish different symbols
        readout_weights = np.random.RandomState(13).uniform(1, -1, size=(len(self.vocab.readout_vocab), 20))
        self.readout_embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(readout_weights), freeze=True)

    def embed(self, batch: pl.DataFrame) -> NDArray:  # noqa: D102
        if (
            self.context_embedding_layer is None
            or self.perturb_embedding_layer is None
            or self.readout_embedding_layer is None
        ):
            raise ValueError("Embedding layers not initialized.")

        embedded_contexts, embedded_perturbs, embedded_readouts = embed(
            batch=batch,
            vocab=self.vocab,
            context_embedding_layer=self.context_embedding_layer,
            perturb_embedding_layer=self.perturb_embedding_layer,
            readout_embedding_layer=self.readout_embedding_layer,
        )
        return torch.cat((embedded_contexts, embedded_perturbs, embedded_readouts), dim=1).numpy()

    def fit(self, traindata: PlibData, valdata: PlibData | None = None):  # noqa: D102
        # by default validation data is not used for early stopping, but simply integrated into training data
        traindata_df = traindata[:] if valdata is None else pl.concat([traindata[:], valdata[:]])

        self._initialize_vocabularies_and_embeddings(traindata_df)

        x = self.embed(traindata_df)
        y = traindata_df["value"].to_numpy()

        logger.debug(f"Fitting {type(self.sklearn_model).__name__} on data of shape {x.shape}..")
        self.sklearn_model.fit(x, y)
        logger.debug("Model fitting completed")

    def predict(self, data_x: PlibData[pl.DataFrame], batch_size: int | None = None) -> NDArray:
        """Predict values for the given data.

        Args:
            data_x: Data without labels i.e. without the "values" column.
            batch_size: Not supported for Sklearn models.

        Returns:
            Value predictions.
        """
        x = self.embed(data_x[:])
        return self.sklearn_model.predict(x)
