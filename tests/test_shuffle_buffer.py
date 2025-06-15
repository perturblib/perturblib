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

from typing import Iterator

import numpy as np
import polars as pl
import polars.testing
import pytest

from perturb_lib.data.utils import ShuffleBuffer


class DummyShardedIterableData:
    def __init__(self, n_shards: int):
        self.n_shards = n_shards
        self.shard_size = 100

    def _num_shards(self):
        return self.n_shards

    def __len__(self):
        return self.n_shards * self.shard_size

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
            shard = np.repeat(
                np.arange(start=shard_number * self.shard_size, stop=(shard_number + 1) * self.shard_size).reshape(
                    -1, 1
                ),
                repeats=4,
                axis=1,
            )
            yield pl.DataFrame(shard, schema=["a", "b", "c", "d"])


@pytest.mark.parametrize("shuffle", [False, True])
def test_shuffle_buffer_returns_all_data(shuffle: bool):
    dataset = DummyShardedIterableData(n_shards=20)
    all_data = pl.concat([shard for shard in dataset._iterate_shards(seed=None)])

    buffer = ShuffleBuffer(dataset, batch_size=20, refill_threshold=500, shuffle=False)
    batched_data = pl.concat([batch for batch in buffer])

    if shuffle:
        # Sort before comparing
        batched_data = batched_data.sort(by=pl.all())

    pl.testing.assert_frame_equal(all_data, batched_data)


def test_shuffle_buffer_shuffles_data():
    dataset = DummyShardedIterableData(n_shards=20)
    all_data = pl.concat([shard for shard in dataset._iterate_shards(seed=None)])

    buffer = ShuffleBuffer(dataset, batch_size=20, refill_threshold=500, shuffle=True)
    batched_data = pl.concat([batch for batch in buffer])

    pl.testing.assert_frame_not_equal(all_data, batched_data)


def test_shuffle_buffer_deterministic():
    dataset = DummyShardedIterableData(n_shards=20)

    buffer = ShuffleBuffer(dataset, batch_size=20, refill_threshold=500, shuffle=True)
    batched_data = pl.concat([batch for batch in buffer])

    buffer = ShuffleBuffer(dataset, batch_size=20, refill_threshold=500, shuffle=True)
    batched_data_2 = pl.concat([batch for batch in buffer])

    pl.testing.assert_frame_equal(batched_data, batched_data_2)


def test_shuffle_buffer_shuffles_different_in_each_iter():
    dataset = DummyShardedIterableData(n_shards=20)

    buffer = ShuffleBuffer(dataset, batch_size=20, refill_threshold=500, shuffle=True)
    batched_data = pl.concat([batch for batch in buffer])
    batched_data_2 = pl.concat([batch for batch in buffer])

    pl.testing.assert_frame_not_equal(batched_data, batched_data_2)
