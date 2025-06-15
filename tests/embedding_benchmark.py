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

import cProfile
import io
import pstats
from pstats import SortKey

import fire
from torch import nn

import perturb_lib as plib
from perturb_lib import Vocabulary
from perturb_lib.data.plibdata import InMemoryPlibData
from perturb_lib.models.collection.lpm import embed


def profile_func(n_runs: int, func, *args, **kwargs):
    assert n_runs > 0

    s = io.StringIO()
    total_stats = pstats.Stats(stream=s)

    for index in range(n_runs):
        pr = cProfile.Profile()

        pr.enable()
        _ = func(*args, **kwargs)
        pr.disable()

        total_stats.add(pr)

        plib.logger.info(
            f"profiling iteration: {index}, function name: {func.__name__}, time up to now: {total_stats.total_tt: .3f}s"  # type: ignore
        )

    # Average stats
    total_stats.total_calls /= n_runs  # type: ignore
    total_stats.prim_calls /= n_runs  # type: ignore
    total_stats.total_tt /= n_runs  # type: ignore
    for func, source in total_stats.stats.items():  # type: ignore
        cc, nc, tt, ct, callers = source
        total_stats.stats[func] = (cc / n_runs, nc / n_runs, tt / n_runs, ct / n_runs, callers)  # type: ignore

    sortby = SortKey.CUMULATIVE
    total_stats.sort_stats(sortby)
    total_stats.print_stats()
    print(s.getvalue())


def run_embedding_benchmark(context_id: str, batch_size: int = 200000):
    plib_data = plib.load_plibdata(context_id, plibdata_type=InMemoryPlibData)

    if batch_size > len(plib_data):
        plib.logger.warning(
            f"Max batch size is larger than the number of samples in the dataset. Setting batch size to {len(plib_data)}."
        )
        batch_size = len(plib_data)

    # Create vocab
    vocab = Vocabulary.initialize_from_data(plib_data)

    # Create embedding layers
    embedding_dim = 1024
    context_embedding_layer = nn.Embedding(len(vocab.context_vocab), embedding_dim)
    perturb_embedding_layer = nn.EmbeddingBag(len(vocab.perturb_vocab), embedding_dim, mode="mean")
    readout_embedding_layer = nn.Embedding(len(vocab.readout_vocab), embedding_dim)

    plib.logger.info(f"Testing with batch size: {batch_size}")
    batch = plib_data.get_data_loader(batch_size=batch_size, shuffle=True).__iter__().__next__()

    profile_func(
        n_runs=10,
        func=embed,
        batch=batch,
        vocab=vocab,
        context_embedding_layer=context_embedding_layer,
        perturb_embedding_layer=perturb_embedding_layer,
        readout_embedding_layer=readout_embedding_layer,
    )


if __name__ == "__main__":
    fire.Fire(run_embedding_benchmark)
