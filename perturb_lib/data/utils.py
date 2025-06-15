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

Data-related utility functions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from typing import Generic, Literal, Protocol, cast, runtime_checkable

import pandas as pd
import polars as pl
import scanpy as sc
import torch.utils.data
from typing_extensions import TypeVar

from perturb_lib._utils import select_random_subset
from perturb_lib.data import ControlSymbol
from perturb_lib.environment import get_seed, logger
from perturb_lib.utils import get_rank_info

VALDATA_PORTION = 0.15
TESTDATA_PORTION = 0.15
SEED = 13

# We have to use the older TypeVar syntax because the new syntax causes issues when pickling during multiprocessing
TCInT = TypeVar("TCInT")  # TransformChain input type
TCOutT = TypeVar("TCOutT")  # TransformChain output type
TCNewOutT = TypeVar("TCNewOutT")  # TransformChain new output type
SBOutT = TypeVar("SBOutT", default=pl.DataFrame)  # ShuffleBuffer output type


ModelSystemType = Literal[
    "HumanCellLine",
    "hiPSC-derivedGlutamatergicNeurons",
    "hiPSC-derivedAstrocytes",
    "PBMC-derivedBCells",
    "PBMC-derivedTCells",
    "PBMC-derivedNKCells",
    "PBMC-derivedMyeloidCells",
]

TechnologyType = Literal[
    "10xChromium3-scRNA-seq",
    "10xChromium5-scRNA-seq",
    "Mosaic-scRNA-seq",
    "GrowthScreen",
    "L1000-RNA-seq",
]


def anndata_format_verification(adata: sc.AnnData) -> None:
    """Basic AnnData format verification."""
    if "readout" not in adata.var.columns:
        raise ValueError("Badly formatted AnnData: 'readout' is missing in 'var'.")
    if "perturbation" not in adata.obs.columns:
        raise ValueError("Badly formatted AnnData: 'perturbation' is missing in 'obs'.")


def _get_random_split(perturbations: Iterable[str]) -> tuple[set[str], set[str], set[str]]:
    control_exists = ControlSymbol in set(perturbations)
    all_perturbations_excluding_control = sorted(set(perturbations).difference({ControlSymbol}))
    num_of_val_perturbations = int(len(all_perturbations_excluding_control) * VALDATA_PORTION)
    num_of_test_perturbations = int(len(all_perturbations_excluding_control) * TESTDATA_PORTION)
    val_perturbations = select_random_subset(all_perturbations_excluding_control, num_of_val_perturbations, seed=SEED)
    remaining_perturbations = sorted(set(all_perturbations_excluding_control) - set(val_perturbations))
    test_perturbations = select_random_subset(remaining_perturbations, num_of_test_perturbations, seed=SEED)
    train_perturbations = set(remaining_perturbations).difference(test_perturbations)
    if control_exists:
        train_perturbations.add(ControlSymbol)
    return train_perturbations, val_perturbations, test_perturbations


def add_train_test_val_splits(adata: sc.AnnData):
    """Adding train/test/validation splits to the given AnnData object."""
    logger.info("Adding train/val/test splits..")
    trainset, valset, testset = _get_random_split(adata.obs.perturbation)
    split_vector = pd.Series([""] * len(adata.obs), index=adata.obs.perturbation)
    split_vector.loc[list(trainset)] = "train"
    split_vector.loc[list(valset)] = "val"
    split_vector.loc[list(testset)] = "test"
    assert "" not in set(split_vector), "Some samples have not been assigned to train/val/test."
    adata.obs["split"] = split_vector.values


class TransformChain(Generic[TCInT, TCOutT]):
    """Chain of functions that can be applied to a single input.

    This class support static type-checking. For example:

    def int_to_str(x: int) -> str:
        return str(x)

    chain: TransformChain[int, int] = TransformChain()  # type of 'chain' is TransformChain[int, int]
    chain2 = chain.append(int_to_str)  # type of 'chain2' correctly inferred as TransformChain[int, str]
    """

    def __init__(self):
        self.callables: list[Callable] = []

    def append(self, function: Callable[[TCOutT], TCNewOutT]) -> TransformChain[TCInT, TCNewOutT]:
        """Append a function to the chain."""
        chain = TransformChain[TCInT, TCNewOutT]()
        chain.callables = self.callables + [function]
        return chain

    def __call__(self, input_: TCInT) -> TCOutT:
        """Apply the chain of functions to the input."""
        for function in self.callables:
            input_ = function(input_)

        # Since we don't keep track of the precise types of the functions in the chain, we need to cast the output
        return cast(TCOutT, input_)


@runtime_checkable
class ShardedIterableDataset(Protocol):
    """Protocol for sharded iterable datasets."""

    def _iterate_shards(
        self,
        seed: int | None,
        start_shard_idx: int | None = None,
        end_shard_idx: int | None = None,
        shard_step_size: int = 1,
    ) -> Iterator[pl.DataFrame]:
        """Iterate over shards of data.

        Args:
            seed: Shuffle shard if a seed is provided. Otherwise, do not shuffle.
            start_shard_idx: Start iterating from this shard index.
            end_shard_idx: Stop iterating before reaching this shard index.
            shard_step_size: Step size for iterating over shards

        Returns: Iterator over shards of data.

        """

    def _num_shards(self) -> int:
        """Get the number of shards in the dataset."""

    def __len__(self) -> int:
        """Get the number of samples in the dataset"""


class ShuffleBuffer(torch.utils.data.IterableDataset[SBOutT], Generic[SBOutT]):
    """Buffer chunks of data and yield them as batches while optionally shuffling the buffer.

    Args:
        dataset: ShardedIterableDataset to sample from.
        batch_size: Size of the batches to yield. If batch_size = None, no batching is done and the full shard is returned
        refill_threshold: Maximum number of samples to keep in the buffer before consolidating and shuffling.
            If batch_size is None, this parameter is ignored.
        shuffle: Whether to shuffle the buffer.
        transforms: Chain of transformations to apply to each batch.
    """

    def __init__(
        self,
        dataset: ShardedIterableDataset,
        batch_size: int | None,
        refill_threshold: int = 1_000_000,
        shuffle: bool = True,
        transforms: TransformChain[pl.DataFrame, SBOutT] | None = None,
    ):
        if batch_size is not None:
            if batch_size <= 0:
                raise ValueError("Batch size must be positive.")

            if shuffle is False:
                refill_threshold = batch_size

            if refill_threshold < batch_size:
                raise ValueError("Buffer refill threshold must be at least equal to batch_size.")
        else:
            # By setting refill_threshold to 0, we ensure that the buffer is immediately returned after every shard
            refill_threshold = 0

        self.dataset = dataset
        self.batch_size = batch_size
        self.refill_threshold = refill_threshold
        self.shuffle = shuffle
        self.transforms: TransformChain[pl.DataFrame, SBOutT] = (
            transforms if transforms is not None else TransformChain()
        )

        self._iter_counter = 0
        self._start_shard_idx: int | None = None
        self._stop_shard_idx: int | None = None
        self._shard_step_size = 1

        self._buffer: list[pl.DataFrame] = []

    @property
    def _num_samples_in_buffer(self) -> int:
        return sum(len(shard) for shard in self._buffer)

    def _get_seed_for_current_iteration(self) -> int | None:
        if self.shuffle:
            return get_seed() + self._iter_counter

        return None

    def _add_shard(self, shard: pl.DataFrame):
        if self.shuffle and (len(shard) > self.refill_threshold):
            logger.warning(
                f"Shard of length {len(shard)} exceeds buffer refill threshold of {self.refill_threshold}. "
                "This might result in a buffer with only one shard and low shuffling efficiency."
            )
        self._buffer.append(shard)

    def _consolidate_and_shuffle_buffer(self):
        self._buffer = [pl.concat(self._buffer, how="vertical", rechunk=True)]
        if self.shuffle:
            self._buffer[0] = self._buffer[0].sample(
                fraction=1, with_replacement=False, shuffle=True, seed=self._get_seed_for_current_iteration()
            )

    def _pop_batch(self) -> SBOutT:
        # Assume that buffer is of length 1
        if self.batch_size is not None:
            batch = self._buffer[0].head(self.batch_size)
            self._buffer[0] = self._buffer[0].tail(-self.batch_size)
        else:
            # Automatic batching is disabled, therefore return the full shard
            batch = self._buffer[0]
            self._buffer = []
        return self.transforms(batch)

    def __len__(self):
        if self.batch_size is None:
            # noinspection PyProtectedMember
            return self.dataset._num_shards()
        else:
            return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        self._iter_counter += 1
        self._buffer = []

        # noinspection PyProtectedMember
        for shard in self.dataset._iterate_shards(
            seed=self._get_seed_for_current_iteration(),
            start_shard_idx=self._start_shard_idx,
            end_shard_idx=self._stop_shard_idx,
            shard_step_size=self._shard_step_size,
        ):
            self._add_shard(shard)

            if self._num_samples_in_buffer > self.refill_threshold:
                # Once we have enough samples, concatenate all dataframes and shuffle them if needed
                self._consolidate_and_shuffle_buffer()  # After this step, buffer is a list with 1 shuffled dataframe only
                # Then pop batches until we fall below the threshold
                while self._num_samples_in_buffer > self.refill_threshold:
                    yield self._pop_batch()

        # Return the remaining samples one batch at a time
        if self._num_samples_in_buffer > 0:
            self._consolidate_and_shuffle_buffer()
            while self._num_samples_in_buffer > 0:
                yield self._pop_batch()

    @staticmethod
    def worker_init_fn(worker_id: int):
        """Worker initialization function for shuffle buffer to be used with PyTorch DataLoader.

        Ensures that each worker processes a different subset of shards.
        """
        rank_info = get_rank_info()
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is not None, "This function should only be called from a worker process!"

        worker_id = worker_info.id
        num_workers = worker_info.num_workers
        global_worker_id = rank_info.rank * num_workers + worker_id
        global_num_workers = rank_info.world_size * num_workers

        buffer: ShuffleBuffer = cast(ShuffleBuffer, worker_info.dataset)

        buffer._start_shard_idx = global_worker_id
        buffer._stop_shard_idx = None  # Iterate till the end
        buffer._shard_step_size = global_num_workers  # Each worker processes a different subset of shards
