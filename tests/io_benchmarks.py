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

import time
from platform import platform
from typing import Iterator

import fire
import polars as pl

import perturb_gym.utils
import perturb_lib as plib
from perturb_lib import OnDiskPlibData
from perturb_lib.data.plibdata import InMemoryPlibData


def _iterate_through_data(iterator: Iterator):
    num_samples = 0
    for batch_idx, batch in enumerate(iterator):
        num_samples += len(batch)
    return num_samples


def run_io_benchmarks(context_id: str):
    plib.logger.info("\nRunning IO benchmark")
    benchmark_io_pars_dict = {"batch_size": [4096, 1024, 256], "num_workers": [0, 1, 2, 4]}
    benchmark_io_pars_list = perturb_gym.utils.dict_product(benchmark_io_pars_dict)

    all_results = []

    for idx, pars in enumerate(benchmark_io_pars_list):
        for plib_data_type in [InMemoryPlibData, OnDiskPlibData]:
            plib.logger.info(f"Running benchmark with parameters: {pars} for {plib_data_type.__name__}")

            plib_data = plib.load_plibdata(context_id, plibdata_type=plib_data_type)
            dataloader = plib_data.get_data_loader(**pars, shuffle=True)

            start = time.time()
            iterator = iter(dataloader)
            end = time.time()
            time_to_create_iterator = end - start

            start = time.time()
            num_samples_all = _iterate_through_data(iterator)
            end = time.time()
            time_to_iterate_through_data = end - start

            del dataloader

            start = time.time()
            plib_data = plib.split_plibdata_2fold(plib_data, context_id)[0]
            end = time.time()
            time_to_split_data = end - start

            dataloader = plib_data.get_data_loader(**pars, shuffle=True)
            iterator = iter(dataloader)
            start = time.time()
            num_samples_split = _iterate_through_data(iterator)
            end = time.time()
            time_to_iterate_through_data_after_split = end - start

            result = pars.copy()
            result["create_iterator_time"] = time_to_create_iterator
            result["samples/s (excl. create)"] = num_samples_all / time_to_iterate_through_data
            result["plib_data_type"] = str(plib_data_type.__name__)
            result["split_time"] = time_to_split_data
            result["samples/s (excl. create, after split)"] = (
                num_samples_split / time_to_iterate_through_data_after_split
            )
            all_results.append(result)

            print(flush=True)
            del iterator, dataloader, plib_data

    results = pl.DataFrame(all_results).select(
        [
            "plib_data_type",
            "batch_size",
            "num_workers",
            "create_iterator_time",
            "samples/s (excl. create)",
            "split_time",
            "samples/s (excl. create, after split)",
        ]
    )
    results = results.sort(by="samples/s (excl. create)")

    with pl.Config(
        tbl_rows=-1,
        tbl_cell_numeric_alignment="RIGHT",
        thousands_separator=",",
        float_precision=3,
    ):
        print(results)

    benchmark_folder = plib.get_path_to_cache() / "results" / "benchmarks"
    benchmark_folder.mkdir(parents=True, exist_ok=True)
    results.write_csv(benchmark_folder / f"benchmark_io_results_{context_id}_{platform()}.csv")
    plib.logger.info("Results of IO benchmark saved.")


if __name__ == "__main__":
    fire.Fire(run_io_benchmarks)
