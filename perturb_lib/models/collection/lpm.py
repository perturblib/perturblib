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

Large perturbation model implementation.
"""

import string
from abc import ABCMeta
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl
import pytorch_lightning as pyl
import torch
from numpy.random import RandomState
from numpy.typing import NDArray
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn as nn
from tqdm import tqdm

from perturb_lib.data.access import Vocabulary, encode_data
from perturb_lib.data.plibdata import PlibData
from perturb_lib.environment import get_path_to_cache, logger
from perturb_lib.models._utils import LPMProfiler
from perturb_lib.models.access import register_model
from perturb_lib.models.base import ModelMixin, embed, embed_tensor_dict, to_tensor_dict
from perturb_lib.utils import inherit_docstring


@inherit_docstring
@register_model
class LPM(ModelMixin, pyl.LightningModule, metaclass=ABCMeta):
    """Large perturbation model.

    Args:
        embedding_dim: Dimensionality of all embedding layers.
        optimizer_name: Name of pytorch optimizer to use.
        learning_rate: Learning rate.
        learning_rate_decay: Exponential learning rate decay.
        num_layers: Depth of the MLP.
        hidden_dim: Number of units in the hidden nodes.
        batch_size: Size of batches during training.
        embedding_aggregation_mode: Defines how to aggregate embeddings.
        num_workers: Number of workers to use during data loading.
        pin_memory: Whether to pin the memory.
        early_stopping_patience: Patience for early stopping in case validation set is given.
        lightning_trainer_pars: Parameters for pytorch-lightning.
    """

    def __init__(
        self,
        embedding_dim: int,
        optimizer_name: str,
        learning_rate: float,
        learning_rate_decay: float,
        num_layers: int,
        hidden_dim: int,
        batch_size: int,
        embedding_aggregation_mode: Literal["sum", "mean", "max"] = "mean",
        dropout: float = 0.0,
        num_workers: int = 0,
        pin_memory: bool = True,
        early_stopping_patience: int = 0,
        profiler: bool = False,
        lightning_trainer_pars: dict | None = None,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.early_stopping_patience = early_stopping_patience
        self.optimizer_name = optimizer_name
        self.embedding_aggregation_mode = embedding_aggregation_mode
        self.lightning_trainer_pars = {} if (lightning_trainer_pars is None) else lightning_trainer_pars
        self.loss = nn.MSELoss(reduction="none")
        self.default_root_dir = get_path_to_cache()

        # vocabulary, to be initialized upon "fit"
        self.vocab: Vocabulary | None = None

        # actual embeddings, to be initialized upon "fit"
        self.context_embedding_layer: nn.Embedding | None = None
        self.perturb_embedding_layer: nn.EmbeddingBag | None = None
        self.readout_embedding_layer: nn.Embedding | None = None

        # prediction neural network
        self.predictor = self.build_predictor()

        # PL-related
        self.training_step_outputs: list[torch.Tensor] = []
        self.validation_step_outputs: list[torch.Tensor] = []
        self.throughput_outputs: list[float] = []
        self.validation_step_per_context_outputs: dict[str, list[torch.Tensor]] = defaultdict(list)
        self.lightning_trainer_pars["default_root_dir"] = self.default_root_dir
        self.save_hyperparameters(ignore="lightning_trainer_pars")
        self.model_checkpoints_path = self.default_root_dir / "checkpoints"
        if profiler:
            self.lightning_trainer_pars["profiler"] = LPMProfiler()
        self.ckpt_filename: str | None
        if early_stopping_patience > 0:
            self.ckpt_filename = "".join(RandomState(None).choice(list(string.ascii_lowercase)) for _ in range(10))
        else:
            self.ckpt_filename = None

    def state_dict(self, *args, **kwargs):  # noqa: D102
        state = super().state_dict(*args, **kwargs)
        # ensure that the vocabulary is a part of model's state, so it gets serialized and saved when required
        if self.vocab is not None:
            state["context_symbols"] = self.vocab.context_vocab["symbol"].to_list()
            state["perturb_symbols"] = self.vocab.perturb_vocab["symbol"].to_list()
            state["readout_symbols"] = self.vocab.readout_vocab["symbol"].to_list()
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):  # noqa: D102
        state_dict = cast(dict[str, Any], state_dict)  # to avoid MyPy complaining about "pop" operation
        if "context_symbols" in state_dict:
            self.vocab = Vocabulary.initialize_from_symbols(
                context_symbols=state_dict["context_symbols"],
                perturb_symbols=state_dict["perturb_symbols"],
                readout_symbols=state_dict["readout_symbols"],
            )
            state_dict.pop("context_symbols")
            state_dict.pop("perturb_symbols")
            state_dict.pop("readout_symbols")
        if "context_embedding_layer.weight" in state_dict:
            self.context_embedding_layer = nn.Embedding.from_pretrained(state_dict["context_embedding_layer.weight"])
        if "perturb_embedding_layer.weight" in state_dict:
            self.perturb_embedding_layer = nn.EmbeddingBag.from_pretrained(state_dict["perturb_embedding_layer.weight"])
        if "readout_embedding_layer.weight" in state_dict:
            self.readout_embedding_layer = nn.Embedding.from_pretrained(state_dict["readout_embedding_layer.weight"])
        super().load_state_dict(state_dict, strict, assign)

    def _initialize_vocabularies_and_embeddings(self, data: PlibData):
        self.vocab = Vocabulary.initialize_from_data(data)

        self.context_embedding_layer = nn.Embedding(len(self.vocab.context_vocab), self.embedding_dim)
        self.perturb_embedding_layer = nn.EmbeddingBag(
            len(self.vocab.perturb_vocab), self.embedding_dim, mode=self.embedding_aggregation_mode
        )
        self.readout_embedding_layer = nn.Embedding(len(self.vocab.readout_vocab), self.embedding_dim)

    def embed(self, batch: pl.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # noqa: D102
        # This method is not used during training, but we keep it for convenience
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

        return embedded_contexts, embedded_perturbs, embedded_readouts

    def embed_tensor_dict(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # noqa: D102
        if (
            self.context_embedding_layer is None
            or self.perturb_embedding_layer is None
            or self.readout_embedding_layer is None
        ):
            raise ValueError("Embedding layers not initialized.")

        embedded_contexts, embedded_perturbs, embedded_readouts = embed_tensor_dict(
            tensor_dict=batch,
            context_embedding_layer=self.context_embedding_layer,
            perturb_embedding_layer=self.perturb_embedding_layer,
            readout_embedding_layer=self.readout_embedding_layer,
        )

        return embedded_contexts, embedded_perturbs, embedded_readouts

    def fit(self, traindata: PlibData[pl.DataFrame], valdata: PlibData[pl.DataFrame] | None = None):  # noqa: D102
        self.train()

        self._initialize_vocabularies_and_embeddings(traindata)
        assert self.vocab is not None

        traindata_tensors = to_tensor_dict(encode_data(traindata, self.vocab))
        train_loader = traindata_tensors.get_data_loader(
            self.batch_size, self.num_workers, self.pin_memory, shuffle=True
        )
        if valdata is not None:
            valdata_tensors = to_tensor_dict(encode_data(valdata, self.vocab))
            val_loader = valdata_tensors.get_data_loader(None, self.num_workers, self.pin_memory, shuffle=False)
        else:
            val_loader = None

        trainer = pyl.Trainer(**self.lightning_trainer_pars)
        logger.info(f"Fitting {self.__class__.__name__}..")
        trainer.fit(model=self, train_dataloaders=train_loader, val_dataloaders=val_loader)
        if len(self.throughput_outputs) > 1:
            avg_thr = sum(self.throughput_outputs[1:]) / len(self.throughput_outputs[1:])
            logger.info(f"Average throughput: {int(avg_thr)} samples/sec == {int(avg_thr / self.batch_size)} it/sec")
        logger.info("Model fitting completed")

    def configure_callbacks(self):  # noqa: D102
        cblist: list[Callback] = []
        if self.early_stopping_patience > 0:
            # add early stopping callback
            cblist.append(EarlyStopping("Validation RMSE", patience=self.early_stopping_patience, verbose=True))
            # add model checkpointing callback
            logger.info(f"Temporary file for checkpoints is {self.ckpt_filename}.ckpt")
            cblist.append(ModelCheckpoint(self.model_checkpoints_path, self.ckpt_filename, monitor="Validation RMSE"))
        return cblist

    def add_logger(self, output_dir: Path, log_name: str):  # noqa: D102
        self.lightning_trainer_pars["logger"] = TensorBoardLogger(save_dir=output_dir, name=log_name)

    @torch.no_grad()
    def predict(self, data_x: PlibData[pl.DataFrame], batch_size: int | None = 100_000) -> NDArray:  # noqa: D102
        if self.vocab is None:
            raise ValueError("Model not fitted yet.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()

        batch_size = min(len(data_x), batch_size) if batch_size is not None else None
        data_x_tensors = to_tensor_dict(encode_data(data_x, self.vocab))
        data_loader = data_x_tensors.get_data_loader(batch_size, self.num_workers, self.pin_memory, shuffle=False)

        predictions_list: list[torch.Tensor] = []
        for batch in data_loader:
            batch_device = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            predictions_list.append(self(batch_device))

        return torch.cat(predictions_list).cpu().detach().numpy().flatten()

    @staticmethod
    def _init_weights(module):  # noqa: D102
        if isinstance(module, (nn.Linear, nn.Embedding, nn.EmbeddingBag)):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("relu"))

    def build_predictor(self) -> nn.Module:
        """Where neural network architecture is instantiated."""
        input_dim = 3 * self.embedding_dim
        neural_network = nn.Sequential()
        for i in range(self.num_layers):
            neural_network.append(nn.Linear(self.hidden_dim if i > 0 else input_dim, self.hidden_dim))
            neural_network.append(nn.ReLU())
            neural_network.append(nn.Dropout(self.dropout))
        neural_network.append(nn.Linear(self.hidden_dim if self.num_layers > 0 else input_dim, 1))
        neural_network.apply(self._init_weights)
        return neural_network

    # -----------------------------------------------------
    # PyTorch Lightning-related methods
    # -----------------------------------------------------

    def configure_optimizers(self):  # noqa: D102
        optimizer_dict = {
            "Adam": torch.optim.Adam(self.parameters(), lr=self.learning_rate),
            "AdamW": torch.optim.AdamW(self.parameters(), lr=self.learning_rate),
            "Adagrad": torch.optim.Adagrad(self.parameters(), lr=self.learning_rate),
            "Adadelta": torch.optim.Adadelta(self.parameters(), lr=self.learning_rate),
            "Adamax": torch.optim.Adamax(self.parameters(), lr=self.learning_rate),
            "SGD": torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9),
            "RMSprop": torch.optim.RMSprop(self.parameters(), lr=self.learning_rate),
        }
        if self.optimizer_name not in optimizer_dict.keys():
            ValueError(f"Unrecognized optimizer {self.optimizer_name}")
        optimizer = optimizer_dict[self.optimizer_name]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.learning_rate_decay)
        return [optimizer], [scheduler]

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:  # noqa: D102
        return self.predictor(torch.cat(self.embed_tensor_dict(batch), dim=1))

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int):  # noqa: D102
        pred: torch.Tensor = self(batch)
        unreduced_loss: torch.Tensor = self.loss(pred, batch["value"].unsqueeze(-1)).squeeze()
        self.training_step_outputs.append(unreduced_loss.detach())  # detach() to not keep the graph alive
        return unreduced_loss.mean()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int):  # noqa: D102
        pred = self(batch)
        unreduced_loss = self.loss(pred, batch["value"].unsqueeze(-1)).squeeze()
        self.validation_step_outputs.append(unreduced_loss)

        assert self.vocab is not None
        context_codes = torch.unique(batch["context"])
        # NOTE: this will cause a GPU-CPU synchronization, but can't be avoided since the vocabulary is on the CPU
        context_codes_series = pl.from_numpy(context_codes.cpu().numpy(), schema=["context"])["context"]
        context_symbol_series = context_codes_series.replace_strict(
            self.vocab.context_vocab["code"], self.vocab.context_vocab["symbol"]
        )

        for context_code, context_symbol in zip(context_codes_series, context_symbol_series):
            context_mask = batch["context"] == context_code
            unreduced_context_loss = unreduced_loss[context_mask]
            self.validation_step_per_context_outputs[context_symbol].append(unreduced_context_loss)

    @staticmethod
    def _reduce_outputs(outputs):
        return "RMSE", torch.cat(outputs).mean().sqrt()

    def on_train_epoch_end(self):  # noqa: D102
        metric_label, metric_result = self._reduce_outputs(self.training_step_outputs)
        self.log(f"Training {metric_label}", metric_result)
        self.training_step_outputs.clear()
        # update throughput log if applicable
        progress_bar_callback = self.trainer.progress_bar_callback
        if isinstance(progress_bar_callback, TQDMProgressBar):
            train_pbar = progress_bar_callback.train_progress_bar
            if isinstance(train_pbar, tqdm):
                samples_processed_in_epoch = train_pbar.format_dict["n"]
                time_to_process_epoch = train_pbar.format_dict["elapsed"]
                self.throughput_outputs.append((self.batch_size * samples_processed_in_epoch) / time_to_process_epoch)

    def on_validation_epoch_end(self):  # noqa: D102
        metric_label, metric_result = self._reduce_outputs(self.validation_step_outputs)
        self.log(f"Validation {metric_label}", metric_result)
        for context in self.validation_step_per_context_outputs.keys():
            metric_label, metric_result = self._reduce_outputs(self.validation_step_per_context_outputs[context])
            self.log(f"Validation {metric_label} {context}", metric_result)
        self.validation_step_outputs.clear()
        self.validation_step_per_context_outputs.clear()

    def on_fit_end(self):  # noqa: D102
        # at the end of training
        logger.info("Cleaning up...")
        # if applicable, load the best model, the one that minimizes validation loss
        if self.ckpt_filename is not None:
            path_to_checkpoint = self.model_checkpoints_path / (self.ckpt_filename + ".ckpt")
            best_model = LPM.load_from_checkpoint(path_to_checkpoint, **self.hparams)
            self.load_state_dict(best_model.state_dict())
            path_to_checkpoint.unlink()
        # detach the trainer from the model
        self.lightning_trainer_pars = {}
        self.trainer = None  # type: ignore
