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

from collections.abc import Sequence

import numpy as np
import polars as pl
import polars.testing
import pytest

import perturb_lib as plib


def assert_all_frames_equal(frames: Sequence[pl.DataFrame]):
    for i in range(1, len(frames)):
        pl.testing.assert_frame_equal(frames[0], frames[i])


def get_splits(context: str):
    adata = plib.load_anndata(context)
    return pl.from_pandas(adata.obs[["perturbation", "split"]]).rename({"perturbation": "index"})


def convert_to_numpy_dummy(data: pl.DataFrame):
    # Dummy transformation just for testing
    data_np = data.with_columns(
        pl.col("context").cast(pl.Categorical).to_physical(),
        pl.col("perturbation").cast(pl.Categorical).to_physical(),
        pl.col("readout").cast(pl.Categorical).to_physical(),
    ).to_numpy()
    return data_np


@pytest.mark.parametrize("test_row", [10, slice(1, 10)])
def test_dataset_consistency_by_index(list_of_plibdata, test_row: int | slice):
    results = [data[test_row] for data in list_of_plibdata]
    assert_all_frames_equal(results)


def test_dataset_consistency_by_iter(list_of_plibdata):
    results: list[pl.DataFrame] = []
    for plib_data in list_of_plibdata:
        all_data: pl.DataFrame = pl.concat([item for item in plib_data], how="vertical")
        results.append(all_data)

    assert_all_frames_equal(results)


@pytest.mark.parametrize("context", ["DummyData", None])
def test_dataset_consistency_after_split(list_of_plibdata, context: str):
    results = []
    for data in list_of_plibdata:
        trainset, _, _ = plib.split_plibdata_3fold(data, context)
        results.append(trainset)
    test_dataset_consistency_by_iter(results)


@pytest.mark.parametrize("plib_data", ["in_memory_plib_data", "on_disk_plib_data"])
@pytest.mark.parametrize("split_fn", [plib.split_plibdata_2fold, plib.split_plibdata_3fold])
def test_dataset_no_overlap_after_split(plib_data: plib.PlibData, split_fn, request):
    plib_data = request.getfixturevalue(plib_data)

    context = "DummyData"
    expected_all_data = plib_data[:].sort(by=pl.all())

    plibdata_objs = split_fn(plib_data, context)
    all_data = pl.concat([plibdata_obj[:] for plibdata_obj in plibdata_objs]).sort(by=pl.all())

    pl.testing.assert_frame_equal(expected_all_data, all_data)


@pytest.mark.parametrize("plib_data", ["in_memory_plib_data", "on_disk_plib_data"])
@pytest.mark.parametrize("split_fn", [plib.split_plibdata_2fold, plib.split_plibdata_3fold])
def test_dataset_no_perturbation_overlap_after_split(plib_data: plib.PlibData, split_fn, request):
    plib_data = request.getfixturevalue(plib_data)

    context = "DummyData"
    dataframes = [plib_data_split[:] for plib_data_split in split_fn(plib_data, context)]

    ctx_pert_tuples_per_split = [set(zip(df["context"], df["perturbation"])) for df in dataframes]
    for i in range(0, len(ctx_pert_tuples_per_split)):
        for j in range(i + 1, len(ctx_pert_tuples_per_split)):
            assert ctx_pert_tuples_per_split[i].isdisjoint(ctx_pert_tuples_per_split[j])


@pytest.mark.parametrize("plib_data", ["in_memory_plib_data", "on_disk_plib_data"])
@pytest.mark.parametrize("split_fn", [plib.split_plibdata_2fold, plib.split_plibdata_3fold])
def test_dataset_contains_correct_perturbations_after_split(plib_data: plib.PlibData, split_fn, request):
    plib_data = request.getfixturevalue(plib_data)

    context = "DummyData"
    split_df = get_splits(context)

    dataframes = [plib_data_split[:] for plib_data_split in split_fn(plib_data, context)]
    dataframe_names = ["train", "val", "test"]

    for df_name, df in zip(dataframe_names, dataframes):
        expected_ctx_pert_tuples = set(
            (context, perturbation) for perturbation in split_df.filter(pl.col("split") == df_name)["index"]
        )
        if df_name == "train":
            if len(dataframes) == 2:
                # If we have a 2-fold split, the test perturbations in split_df should be used as train perturbations
                expected_ctx_pert_tuples = expected_ctx_pert_tuples.union(
                    set((context, perturbation) for perturbation in split_df.filter(pl.col("split") == "test")["index"])
                )
            # For train dataset, check only the perturbations of the given context we are using as a split context.
            # The remaining contexts can have any perturbations.
            df_filtered = df.filter(pl.col("context") == context)
            actual_ctx_pert_tuples = set(zip(df_filtered["context"], df_filtered["perturbation"]))
        else:
            actual_ctx_pert_tuples = set(zip(df["context"], df["perturbation"]))

        assert actual_ctx_pert_tuples == expected_ctx_pert_tuples


@pytest.mark.parametrize("plib_data", ["in_memory_plib_data", "on_disk_plib_data"])
def test_dataset_returns_correct_type(plib_data: plib.PlibData, request):
    plib_data = request.getfixturevalue(plib_data)
    plib_data_encoded = plib_data.apply_transform(convert_to_numpy_dummy)
    assert isinstance(plib_data_encoded[0], np.ndarray)


@pytest.mark.parametrize("plib_data", ["in_memory_plib_data", "on_disk_plib_data"])
@pytest.mark.parametrize("batch_size", [20])
@pytest.mark.parametrize("pin_memory", [False, True])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("shuffle", [False, True])
def test_dataloader_returns_all_data(
    plib_data: plib.PlibData, num_workers: int, pin_memory: bool, batch_size: int, shuffle: bool, request
):
    plib_data = request.getfixturevalue(plib_data)

    dataloader = plib_data.get_data_loader(
        num_workers=num_workers, pin_memory=pin_memory, batch_size=batch_size, shuffle=shuffle
    )
    all_data = plib_data[:].sort(by=pl.all())
    dataloader_data = pl.concat([batch for batch in dataloader]).sort(by=pl.all())

    pl.testing.assert_frame_equal(all_data, dataloader_data)


@pytest.mark.parametrize("plib_data", ["in_memory_plib_data", "on_disk_plib_data"])
@pytest.mark.parametrize("batch_size", [None, 20])
@pytest.mark.parametrize("num_workers", [0, 1, 2])
def test_dataloader_returns_correct_order(plib_data: plib.PlibData, num_workers: int, batch_size: int, request):
    plib_data = request.getfixturevalue(plib_data)

    dataloader = plib_data.get_data_loader(num_workers=num_workers, batch_size=batch_size, shuffle=False)
    all_data = plib_data[:]
    dataloader_data = pl.concat([batch for batch in dataloader])

    pl.testing.assert_frame_equal(all_data, dataloader_data)


@pytest.mark.parametrize("plib_data", ["in_memory_plib_data", "on_disk_plib_data"])
@pytest.mark.parametrize("batch_size", [5, 7, 17, 20])
@pytest.mark.parametrize("num_workers", [0, 4])
def test_dataloader_iterates_consistently(plib_data: str, num_workers: int, batch_size: int):
    # Don't use the fixtures for this since the fixtures are loaded once per test. Manually create a second
    # dataloader to better simulate the real training scenario.
    context = ["DummyData", "DummyDataLongStrings"]
    plibtype_dict = {"in_memory_plib_data": plib.InMemoryPlibData, "on_disk_plib_data": plib.OnDiskPlibData}
    plibtype = plibtype_dict[plib_data]

    plib_data_obj = plib.load_plibdata(context, plibdata_type=plibtype)
    train, _ = plib.split_plibdata_2fold(plib_data_obj, None)
    dataloader = train.get_data_loader(num_workers=num_workers, batch_size=batch_size, shuffle=True)
    dataloader_data = pl.concat([batch for batch in dataloader], how="vertical")

    plib_data_obj = plib.load_plibdata(context, plibdata_type=plibtype)
    train, _ = plib.split_plibdata_2fold(plib_data_obj, None)
    dataloader = train.get_data_loader(num_workers=num_workers, batch_size=batch_size, shuffle=True)
    dataloader_data_2 = pl.concat([batch for batch in dataloader], how="vertical")

    pl.testing.assert_frame_equal(dataloader_data, dataloader_data_2)
