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

Native perturb-lib data formats.
"""

from __future__ import annotations

import copy
from abc import abstractmethod
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Generic, Self, cast

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchData
from typing_extensions import TypeVar

from perturb_lib.data.utils import ShuffleBuffer, TransformChain
from perturb_lib.environment import get_path_to_cache, get_seed, logger
from perturb_lib.utils import inherit_docstring

OutT = TypeVar("OutT", default=pl.DataFrame)
NewOutT = TypeVar("NewOutT")


class PlibData(TorchData[OutT], Generic[OutT]):
    """Data structure for hosting perturb-lib data.

    Args:
        data: data to initialize the class with.
        data_sources: if data is None, data_sources can be used to specify the names of the data sources.
    """

    def __init__(
        self, data=None, data_sources: str | list[str] | None = None, path_to_data_sources: Path | None = None
    ):
        super().__init__()
        if path_to_data_sources is None:
            path_to_data_sources = get_path_to_cache() / "plibdata"
        if data is None and data_sources is None:
            raise ValueError("Either 'data' or 'data_sources' need to be given")
        if data is not None:
            self._data: pl.DataFrame = data
        elif data_sources is not None:
            if isinstance(data_sources, str):
                data_sources = [data_sources]
            self._data = self.init_from_files(path_to_data_sources, data_sources)

        self._transforms: TransformChain[pl.DataFrame, OutT] = TransformChain()

    @abstractmethod
    def init_from_files(self, path_to_data_sources: Path, data_sources: list[str]) -> pl.DataFrame:
        """Initializes PlibData from multiple files."""

    def apply_transform(self, transform: Callable[[OutT], NewOutT]) -> PlibData[NewOutT]:
        """Apply a transformation to the data."""
        new_chain = self._transforms.append(transform)
        # Shallow copy instead of creating another object using type(self)(data=self._data) to avoid forgetting some
        # of the state e.g. OnDiskPlibData has `self._columns` as part of its state
        new_plibdata = cast(PlibData[NewOutT], copy.copy(self))
        new_plibdata._transforms = new_chain
        return new_plibdata

    @abstractmethod
    def __getitem__(self, index) -> OutT:
        """Get item specified by index."""

    @abstractmethod
    def __len__(self) -> int:
        """Get number of samples."""

    @property
    @abstractmethod
    def columns(self) -> list[str]:
        """The list of column names."""
        ...

    @abstractmethod
    def subset_columnwise(self, columns: list[str]) -> Self:
        """Select a subset of existing columns.

        Args:
            columns: The names of columns to keep
        """
        ...

    @property
    @abstractmethod
    def dtypes(self) -> dict:
        """Dictionary of data types."""

    @abstractmethod
    def get_data_loader(
        self, batch_size: int | None, num_workers: int = 0, pin_memory: bool = False, shuffle: bool = False
    ) -> TorchDataLoader[OutT]:
        """Fetch a torch-style data loader for batch sampling.

        Args:
            batch_size: The size of a batch to fetch in each iteration.
            num_workers: Number of pytorch workers.
            pin_memory: If true, Copy Tensors into device pinned memory before returning them.
            shuffle: If false, samples will be sampled sequentially to form batches. If true, samples will be shuffled.

        Returns:
            an instance of ``torch.utils.data.DataLoader``
        """


@inherit_docstring
class InMemoryPlibData(PlibData[OutT], Generic[OutT]):
    """In-memory variant of ``PlibData``. Implemented using ``polars`` backend."""

    _hidden_columns = ["split"]

    def init_from_files(self, path_to_data_sources: Path, data_sources: list[str]) -> pl.DataFrame:  # noqa: D102
        source_shards: list[pl.LazyFrame] = []
        for source in data_sources:
            # Load and add "split" column to all shards
            if len(list((path_to_data_sources / source).glob("shard_*.parquet"))) == 0:
                continue
            shards = pl.scan_parquet(path_to_data_sources / source / "shard_*.parquet", include_file_paths="shard_path")
            shards = shards.with_columns(pl.col("shard_path").str.extract(r"^.*(shard_.*\.parquet)$", 1))
            metadata = pl.scan_parquet(path_to_data_sources / source / "metadata.parquet").select("shard_path", "split")
            metadata = metadata.with_columns(pl.col("shard_path").str.extract(r"^.*(shard_.*\.parquet)$", 1))
            shards = shards.join(metadata, on="shard_path").drop("shard_path")
            source_shards.append(shards)

        full_df = pl.concat(source_shards, how="vertical", rechunk=True)
        return full_df.collect()

    def apply_transform(self, transform: Callable[[OutT], NewOutT]) -> InMemoryPlibData[NewOutT]:
        """Apply a transformation to the data."""
        # We override just to narrow the return type for static-checkers
        return cast(InMemoryPlibData[NewOutT], super().apply_transform(transform))

    def __getitem__(self, index) -> OutT:
        if getattr(self, "_data_pd", None) is None:
            self._data_pd: pd.DataFrame = self._data.to_pandas(use_pyarrow_extension_array=True)

        df: pd.DataFrame = self._data_pd.iloc[index].drop(self._hidden_columns, errors="ignore", axis=1)
        return self._transforms(pl.from_pandas(df))

    def __iter__(self):
        for df in self._data.drop(self._hidden_columns, strict=False).iter_slices(n_rows=1):
            yield self._transforms(df)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def columns(self) -> list[str]:  # noqa: D102
        columns = [col for col in self._data.columns if col not in self._hidden_columns]
        return columns

    def subset_columnwise(self, columns: list[str]) -> Self:  # noqa: D102
        if not set(columns) <= set(self.columns):
            raise ValueError("When sub-setting InMemoryPlibData column-wise, newly specified columns must exist!")

        columns.extend(self._hidden_columns)

        new_plibdata = copy.copy(self)
        new_plibdata._data = self._data[columns]
        return new_plibdata

    @property
    def dtypes(self) -> dict:  # noqa: D102
        schema = {k: v for k, v in self._data.schema.items()}
        return schema

    @staticmethod
    def _collate_fn(batch):
        return batch[0]

    def get_data_loader(
        self, batch_size: int | None, num_workers: int = 0, pin_memory: bool = False, shuffle: bool = False
    ) -> TorchDataLoader[OutT]:
        """Fetch a torch-style data loader for batch sampling.

        Args:
            batch_size: The size of a batch to fetch in each iteration.
            num_workers: Number of pytorch workers.
            pin_memory: If true, Copy Tensors into device pinned memory before returning them.
            shuffle: If false, samples will be sampled sequentially to form batches. If true, samples will be shuffled.

        Returns:
            an instance of ``torch.utils.data.DataLoader``
        """
        if batch_size is None:
            # We don't want to raise an exception here, since we want the user to be able to pass 'None' without
            # knowing if the loader is in-memory or on-disk. We will just set a default value.
            logger.warning("batch_size=None is not supported for InMemoryPlibData. Setting batch_size=5000")
            batch_size = 5000

        generator = torch.Generator()
        generator.manual_seed(get_seed())
        sampler = RandomSampler(self, generator=generator) if shuffle else SequentialSampler(self)

        return TorchDataLoader(
            dataset=self,
            sampler=BatchSampler(sampler, batch_size=batch_size, drop_last=False),
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )


@inherit_docstring
class OnDiskPlibData(PlibData[OutT], Generic[OutT]):
    """Class for handling on-disk data. Implemented using ``pytables`` backend.
    . +====================+
    . |   OnDiskPlibData   |
    . |                    | --- __getitem__ ----->
    . |                    | --- __iter__ ------->
    . |                    |
    . +====================+
    .           |
    .           v
    .     _iterate_shards (internal)
    .           |
    .           v
    . +-------------------+
    . |   ShuffleBuffer   |
    . +-------------------+
    .           |
    .        __iter__ (produce batches)
    .           |
    .           v
    . +-------------------+
    . |     DataLoader    |
    . +-------------------+
    """

    def __init__(
        self,
        data=None,
        data_sources: str | None = None,
        path_to_data_sources: Path | None = None,
        columns: list[str] | None = None,
    ):
        super().__init__(data, data_sources, path_to_data_sources)
        if columns:
            self._columns = columns
        else:
            first_shard_path = self._data.head(1)["shard_path"].item()
            # Read schema without reading whole file
            self._columns = list(pl.read_parquet_schema(first_shard_path).keys())

    def init_from_files(self, path_to_data_sources: Path, data_sources: list[str]) -> pl.DataFrame:  # noqa: D102
        # Note that it's important that shards should always have the same order after concatenation for reproducibility
        dataframes = [pl.scan_parquet(path_to_data_sources / source / "metadata.parquet") for source in data_sources]
        metadata = pl.concat(dataframes, how="vertical", rechunk=True)
        metadata = metadata.with_columns(shard_path=pl.lit(str(path_to_data_sources) + "/") + pl.col("shard_path"))
        return metadata.collect()

    def _load_shard(self, shard_number: int) -> pl.DataFrame:
        shard_metadata: dict = self._data.row(shard_number, named=True)
        shard = pl.read_parquet(shard_metadata["shard_path"], columns=self.columns)

        return shard

    def apply_transform(self, transform: Callable[[OutT], NewOutT]) -> OnDiskPlibData[NewOutT]:
        """Apply a transformation to the data."""
        # We override just to narrow the return type for static-checkers
        return cast(OnDiskPlibData[NewOutT], super().apply_transform(transform))

    def __getitem__(self, index) -> OutT:
        """Get item specified by index.

        Since data is sharded, we need to identify the correct shards to load the data from:
        |----------| |----------|  |----------|  |----------|
        | shard 1  | | shard 2  |  | shard 3  |  | shard 4  |
        |----------| |----------|  |----------|  |----------|
             ^                                        ^
           start                                    stop
        In addition, we need to potentially discard some data from the first and last shard if
        start and stop are not aligned with the shard boundaries.
        """
        if isinstance(index, slice):
            start, stop = index.start, index.stop
        elif isinstance(index, int):
            start, stop = index, index + 1
        elif isinstance(index, list) or isinstance(index, np.ndarray):
            if not index:  # empty sequence
                start, stop = 0, 0
            else:  # non-empty sequence
                if isinstance(index, list):
                    index = np.array(index)
                if not np.array_equal(index[:-1] + 1, index[1:]):
                    raise ValueError("Indexing list must be in a stepwise format!")
                start, stop = index[0], index[-1] + 1
        else:
            raise ValueError(f"Index of {type(index)} not supported.")

        if start is None:
            start = 0

        if stop is None:
            stop = len(self)

        # Use size information to only read the necessary shards
        cumsum: pl.Series = self._data["size"].cum_sum()
        start_shard = cumsum.search_sorted(start, side="right")
        stop_shard = cumsum.search_sorted(stop, side="left") + 1
        n_shards = stop_shard - start_shard

        start_offset = cumsum[start_shard - 1] if start_shard > 0 else 0
        new_start = start - start_offset
        new_stop = stop - start_offset
        shards_to_use = self._data["shard_path"].slice(start_shard, n_shards).to_list()

        shards = pl.scan_parquet(shards_to_use).select(self.columns)
        return self._transforms(shards.slice(new_start, new_stop - new_start).collect())

    def _iterate_shards(
        self,
        seed: int | None,
        start_shard_idx: int | None = None,
        end_shard_idx: int | None = None,
        shard_step_size: int = 1,
    ) -> Iterator[pl.DataFrame]:
        shard_numbers = np.arange(self._num_shards(), dtype=np.int_)
        if seed is not None:
            np.random.RandomState(seed).shuffle(shard_numbers)

        if start_shard_idx is None:
            start_shard_idx = 0
        if end_shard_idx is None:
            end_shard_idx = self._num_shards()

        for shard_number in shard_numbers[start_shard_idx:end_shard_idx:shard_step_size]:
            yield self._load_shard(shard_number)

    def _num_shards(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[OutT]:
        """Iterate over the dataset row-by-row. This is a user-facing method, and it is not used by torch DataLoader."""
        for shard in self._iterate_shards(seed=None, start_shard_idx=None, end_shard_idx=None, shard_step_size=1):
            for df in shard.iter_slices(n_rows=1):
                yield self._transforms(df)

    def __len__(self) -> int:
        """Get number of samples."""
        size = self._data.select(pl.col("size").sum()).item()
        return cast(int, size)

    @property
    def columns(self) -> list[str]:
        """The list of column names."""
        return self._columns

    def subset_columnwise(self, columns: list[str]) -> Self:  # noqa: D102
        if not set(columns) <= set(self.columns):
            raise ValueError("When sub-setting OnDiskPlibData column-wise, newly specified columns must exist!")

        new_plibdata = copy.copy(self)
        new_plibdata._columns = columns
        return new_plibdata

    @property
    def dtypes(self) -> dict:  # noqa: D102
        df = pl.from_pandas(self._data.select("context", start=0, stop=0, columns=self.columns))
        return dict(cast(dict, df.schema))

    def get_data_loader(
        self, batch_size: int | None, num_workers: int = 0, pin_memory: bool = False, shuffle: bool = False
    ) -> TorchDataLoader[OutT]:
        """Fetch a torch-style data loader for batch sampling.

        Args:
            batch_size: The size of a batch to fetch in each iteration. If None, return shards directly
            num_workers: Number of pytorch workers.
            pin_memory: If true, Copy Tensors into device pinned memory before returning them.
            shuffle: If false, samples will be sampled sequentially to form batches. If true, samples will be shuffled.

        Returns:
            an instance of ``torch.utils.data.DataLoader``
        """
        if shuffle is False and batch_size is not None and num_workers > 1:
            logger.warning(
                "Using more than 1 worker with shuffle=False and a specified batch_size "
                "is not supported. Setting num_workers=1."
            )
            num_workers = 1

        batched_dataset = ShuffleBuffer(self, batch_size=batch_size, shuffle=shuffle, transforms=self._transforms)

        return TorchDataLoader(
            dataset=batched_dataset,
            batch_size=None,  # Disable automatic batching
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=True if num_workers > 0 else False,
            worker_init_fn=ShuffleBuffer.worker_init_fn,
            multiprocessing_context="spawn" if num_workers > 0 else None,
        )
