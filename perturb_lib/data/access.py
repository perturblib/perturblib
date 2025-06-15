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

Interface module for all the data-related operations.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Literal, Optional, Self, Type, cast

import numpy as np
import polars as pl
from scanpy import AnnData

from perturb_lib.data import ControlSymbol
from perturb_lib.data.plibdata import InMemoryPlibData, OnDiskPlibData, PlibData
from perturb_lib.data.preprocesing import DEFAULT_PREPROCESSING_TYPE, PreprocessingType, preprocessors
from perturb_lib.data.utils import (
    ModelSystemType,
    TechnologyType,
    add_train_test_val_splits,
    anndata_format_verification,
)
from perturb_lib.environment import get_path_to_cache, logger
from perturb_lib.utils import update_symbols

context_catalogue: dict[str, type[AnnData]] = {}

SHARDSIZE = 200_000
PDATA_FORMAT_VERSION = 1  # Increase this number if the data format changes


class Vocabulary:
    """Vocabulary class for managing symbols of contexts, perturbations, and readouts."""

    def __init__(self, context_vocab: pl.DataFrame, perturb_vocab: pl.DataFrame, readout_vocab: pl.DataFrame):
        self._verify_vocabulary(context_vocab, "context_vocab")
        self._verify_vocabulary(perturb_vocab, "perturb_vocab")
        self._verify_vocabulary(readout_vocab, "readout_vocab")

        self.context_vocab: pl.DataFrame = context_vocab
        self.perturb_vocab: pl.DataFrame = perturb_vocab
        self.readout_vocab: pl.DataFrame = readout_vocab

        ctrl_symbol_expr = pl.col("symbol") == ControlSymbol
        self.perturb_control_code = self.perturb_vocab.row(by_predicate=ctrl_symbol_expr, named=True)["code"]

    @classmethod
    def initialize_from_symbols(
        cls: type[Self], context_symbols: list[str], perturb_symbols: list[str], readout_symbols: list[str]
    ) -> Self:
        """Initialize vocabulary using symbols given in the arguments."""
        # ensure existence of control symbol, and make sure its placed at the beginning
        if ControlSymbol in perturb_symbols:
            perturb_symbols.remove(ControlSymbol)
        perturb_symbols = [ControlSymbol] + perturb_symbols

        context_vocab = pl.DataFrame(data={"symbol": context_symbols, "code": np.arange(len(context_symbols))})
        perturb_vocab = pl.DataFrame(data={"symbol": perturb_symbols, "code": np.arange(len(perturb_symbols))})
        readout_vocab = pl.DataFrame(data={"symbol": readout_symbols, "code": np.arange(len(readout_symbols))})

        return cls(context_vocab, perturb_vocab, readout_vocab)

    @classmethod
    def initialize_from_data(cls: type[Self], data: AnnData | PlibData | pl.DataFrame) -> Self:
        """Initialize vocabulary using symbols given in the data."""
        if isinstance(data, AnnData):
            context_symbols = sorted(data.obs.context.unique())
            perturb_symbols = sorted(pl.Series(data.obs.perturbation).str.split("+").explode().unique())
            readout_symbols = sorted(data.var.readout.unique())
        elif isinstance(data, pl.DataFrame):
            context_symbols = sorted(data["context"].unique())
            perturb_symbols = sorted(data["perturbation"].str.split("+").explode().unique())
            readout_symbols = sorted(data["readout"].unique())
        elif isinstance(data, InMemoryPlibData):
            context_symbols = sorted(data._data["context"].unique())
            perturb_symbols = sorted(data._data["perturbation"].str.split("+").explode().unique())
            readout_symbols = sorted(data._data["readout"].unique())
        elif isinstance(data, OnDiskPlibData):
            context_symbols = sorted(data._data["context"].unique())
            perturb_symbols = sorted(data._data["perturbations"].explode().unique())
            readout_symbols = sorted(data._data["readouts"].explode().unique())
        else:
            raise ValueError("Wrong data format!")

        return cls.initialize_from_symbols(context_symbols, perturb_symbols, readout_symbols)

    @staticmethod
    def _verify_vocabulary(df: pl.DataFrame, name: str):
        if "symbol" not in df.columns:
            raise ValueError(f"Vocabulary component {name} must contain a column named 'symbol'.")
        if "code" not in df.columns:
            raise ValueError("Vocabulary component {name} must contain a column named 'code'.")
        if not df["symbol"].is_unique().all():
            raise ValueError(f"Vocabulary component {name} must contain unique symbols.")

    def __eq__(self, other) -> bool:
        """Check if two vocabularies are equal."""
        if not isinstance(other, Vocabulary):
            return NotImplemented

        return (
            self.context_vocab.equals(other.context_vocab)
            and self.perturb_vocab.equals(other.perturb_vocab)
            and self.readout_vocab.equals(other.readout_vocab)
        )


def list_contexts() -> list[str]:
    """List registered contexts.

    Returns:
        A list of context identifiers that are registered in perturb_lib.
    """
    return sorted(list(context_catalogue.keys()))


def _verify_that_context_exists(context_id: str):
    if context_id not in list_contexts():
        raise ValueError(f"Unavailable context {context_id}!")


def describe_context(context_id: str) -> Optional[str]:
    """Describe specified context.

    Args:
        context_id: Identifier of the context.

    Returns:
        The description of the context given as a string.

    Raises:
        ValueError: If given context is not registered.
    """
    _verify_that_context_exists(context_id)
    return context_catalogue[context_id].__doc__


def load_anndata(context_id: str, hgnc_renaming: bool = False) -> AnnData:
    """Load data from given context as ``AnnData``.

    Args:
        context_id:  Identifier of the context.
        hgnc_renaming: Whether to apply HNC renaming.

    Returns:
        AnnData object of the corresponding context.

    Raises:
        ValueError: If given context is not registered.
    """
    _verify_that_context_exists(context_id)
    adata = context_catalogue[context_id]()
    if hgnc_renaming:
        adata = update_symbols(adata)
    perturbation_columns = [x for x in adata.obs.columns if "perturbation" in x]
    adata.obs["perturbation"] = adata.obs[perturbation_columns[0]]
    for column in perturbation_columns[1:]:
        adata.obs["perturbation"] = adata.obs["perturbation"] + "_" + adata.obs[column]
    adata.obs["context"] = context_id

    if "perturbation_type" in set(adata.obs.columns):
        pert_type = list(set(adata.obs.perturbation_type))[0]
        adata.obs["perturbation"] = adata.obs["perturbation"].str.replace("+", f"+{pert_type}_", regex=False)
        adata.obs["perturbation"] = adata.obs["perturbation"].str.replace(f"{pert_type}_{ControlSymbol}", ControlSymbol)
    readout_columns = [x for x in adata.var.columns if "readout" in x]
    adata.var["readout"] = adata.var[readout_columns].agg(lambda row: "_".join(row), axis=1)

    if "split" not in adata.obs.columns:
        add_train_test_val_splits(adata)

    adata.obs.reset_index(inplace=True)
    adata.var.set_index("readout", inplace=True)
    adata.var.reset_index(inplace=True)

    # for some reason, AnnData requires index to be of type str
    adata.var.index = adata.var.index.astype(str)
    adata.obs.index = adata.obs.index.astype(str)
    anndata_format_verification(adata)
    return adata


def _get_module_hash_for_context(context: str) -> str:
    """Get hash of the module where the context is defined."""
    func = context_catalogue[context]
    source_file = inspect.getsourcefile(func)

    if source_file is None:
        raise ValueError(f"Could not find source file for context {context}")

    source_code = Path(source_file).read_text()
    return hashlib.md5(source_code.encode()).hexdigest()


def _sanitize_context_name(context: str) -> str:
    """Sanitize context name by removing special characters."""
    sanitized_name = re.sub(r'[ \'\]\[<>:"/\\|?*-]', "", context)
    return sanitized_name


def _get_plib_data_source_name(context: str, preprocessing_type: PreprocessingType) -> str:
    """Get name of the plibdata source."""
    return f"{_sanitize_context_name(context)}-{preprocessing_type}"


def _process_context_and_generate_metadata(
    context: str,
    preprocessing_type: PreprocessingType = DEFAULT_PREPROCESSING_TYPE,
) -> pl.DataFrame:
    plibdata_dir = get_path_to_cache() / "plibdata"
    path_to_context = plibdata_dir / _get_plib_data_source_name(context, preprocessing_type)

    adata = load_anndata(context)
    adata = preprocessors[preprocessing_type](adata)

    logger.info("Casting to standardized DataFrame format..")
    ldf = pl.DataFrame(
        data=adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray(),
        schema=adata.var.readout.tolist(),
    ).lazy()
    ldf = ldf.with_columns(
        pl.Series("context", adata.obs.context.values).cast(pl.String),
        pl.Series("perturbation", adata.obs.perturbation.values).cast(pl.String),
        pl.Series("split", adata.obs.split.values).cast(pl.String),
    )
    ldf = ldf.unpivot(index=["context", "perturbation", "split"], variable_name="readout")

    # Generate shard id per split
    ldf = ldf.with_columns(shard_id_per_split=pl.arange(pl.len()).over("split") // SHARDSIZE)

    # Generate global shard id
    ldf = ldf.with_row_index().with_columns(shard_id=pl.col("index").first().over("split", "shard_id_per_split"))
    ldf = ldf.with_columns(
        shard_id=(pl.col("shard_id").rank("dense") - 1)
    )  # -1 since rank generates numbers starting from 1
    ldf = ldf.select("context", "perturbation", "readout", "value", "split", "shard_id")
    df = ldf.collect()

    metadata = df.group_by("shard_id", maintain_order=True).agg(
        pl.col("context").first(),
        pl.col("split").first(),
        pl.len().alias("size"),
        pl.col("perturbation").str.split("+").list.explode().unique().alias("perturbations"),
        pl.col("readout").unique().alias("readouts"),
    )
    metadata = metadata.select(
        pl.format(
            "{}/shard_{}.parquet",
            pl.lit(path_to_context.name),
            pl.col("shard_id").cast(pl.String).str.zfill(6),
        ).alias("shard_path"),
        pl.col("size"),
        pl.col("split"),
        pl.col("context"),
        pl.col("perturbations"),
        pl.col("readouts"),
    )

    for group_id, shard_df in df.group_by("shard_id"):
        shard_id = group_id[0]
        shard_path = path_to_context / f"shard_{shard_id:06d}.parquet"
        shard_df.drop("split", "shard_id").write_parquet(shard_path)

    return metadata


def _process_and_cache_context(
    context: str,
    preprocessing_type: PreprocessingType = DEFAULT_PREPROCESSING_TYPE,
):
    plibdata_dir = get_path_to_cache() / "plibdata"
    path_to_context = plibdata_dir / _get_plib_data_source_name(context, preprocessing_type)

    info_file = path_to_context / "info.json"
    metadata_file = path_to_context / "metadata.parquet"

    context_module_hash = _get_module_hash_for_context(context)

    # Some sanity checks for the cache
    expected_info_dict = {
        "PDATA_FORMAT_VERSION": PDATA_FORMAT_VERSION,
        "CONTEXT_MODULE_HASH": context_module_hash,
        "SHARDSIZE": SHARDSIZE,
    }
    try:
        info = json.loads(info_file.read_text())
        should_cache = info != expected_info_dict
    except (FileNotFoundError, json.JSONDecodeError):
        should_cache = True

    if not should_cache:
        try:
            expected_n_shards = len(pl.read_parquet(metadata_file))
            current_n_shards = len(list(path_to_context.glob("shard_*.parquet")))
            should_cache = current_n_shards != expected_n_shards
        except (FileNotFoundError, ValueError):
            should_cache = True

    if should_cache:
        shutil.rmtree(path_to_context, ignore_errors=True)
        path_to_context.mkdir(exist_ok=True, parents=True)
        metadata = _process_context_and_generate_metadata(context, preprocessing_type)
        metadata.write_parquet(metadata_file)
        info_file.write_text(json.dumps(expected_info_dict))


def _unpack_context_ids(contexts_ids: str | Sequence[str]) -> list[str]:
    if isinstance(contexts_ids, str):
        contexts_ids = [contexts_ids]

    expanded_contexts_ids = []
    for context_id in contexts_ids:
        if context_id not in context_catalogue:
            # check if context_id is a substring that matches actual context ids
            context_id_list = [c for c in list_contexts() if context_id in c]
            if not context_id_list:
                # trigger an exception
                _verify_that_context_exists(context_id)
            else:
                # add all contexts that matched the substring
                expanded_contexts_ids.extend(context_id_list)
        else:
            expanded_contexts_ids.append(context_id)

    return sorted(list(set(expanded_contexts_ids)))


def load_plibdata(
    contexts_ids: str | Sequence[str],
    preprocessing_type: PreprocessingType = DEFAULT_PREPROCESSING_TYPE,
    plibdata_type: type[PlibData] = InMemoryPlibData,
) -> PlibData:
    """Load data from given context(s) as ``PlibData``.

    Note:
        Triggers data caching.

    Args:
        contexts_ids: Either a list of context identifiers that will be stacked together, or a single context id. If
        no match is found for an identifier, all identifiers that contain that substring will be included.
        preprocessing_type: The type of preprocessing to apply.
        plibdata_type: The type of ``PlibData`` to use.

    Returns:
        An instance of a ``PlibData`` that has columns [context, perturbation, readout, value].
    """
    contexts_ids = _unpack_context_ids(contexts_ids)

    plibdata_dir = get_path_to_cache() / "plibdata"

    for context in contexts_ids:
        _process_and_cache_context(context, preprocessing_type)

    data_sources = [_get_plib_data_source_name(context, preprocessing_type) for context in contexts_ids]
    return plibdata_type(data_sources=data_sources, path_to_data_sources=plibdata_dir)


def _split_plibdata[PlibDataT: PlibData](
    pdata: PlibDataT,
    context_ids: str | Sequence[str] | None,
    split_name: Literal["train", "val", "test"],
) -> tuple[PlibDataT, PlibDataT]:
    """Split PlibData into two parts

    Args:
        pdata: PlibData to split.
        context_ids: Contexts from which validation perturbations should be selected. If None, use all contexts.
        split_name: The name of the split for the second return argument

    Returns: Two ``PlibData`` instances. The second instance will contain all samples belonging to the given
    ``split_name`` while the first instance will contain all remaining samples.

    """
    split_expr = pl.col("split") == split_name
    if context_ids is not None:
        context_ids = _unpack_context_ids(context_ids)
        split_expr = split_expr & (pl.col("context").is_in(context_ids))

    # noinspection PyProtectedMember
    data_or_metadata = pdata._data
    split_data = data_or_metadata.filter(split_expr)
    remaining_data = data_or_metadata.filter(~split_expr)

    split_pdata = type(pdata)(data=split_data)
    remaining_pdata = type(pdata)(data=remaining_data)

    return remaining_pdata, split_pdata


def split_plibdata_2fold[PlibDataT: PlibData](
    pdata: PlibDataT, context_ids: str | Sequence[str] | None
) -> tuple[PlibDataT, PlibDataT]:
    """Split data to training and validation.

    Args:
        pdata: An instance of ``PlibData`` to split.
        context_ids: Contexts from which validation perturbations should be selected. If None, a portion of validation
        data from each context will be taken.

    Returns:
        Three instances of ``PlibData``: training and validation.
    """
    logger.debug("Splitting plibdata instance into train and validation.")
    train_pdata, val_pdata = _split_plibdata(pdata=pdata, context_ids=context_ids, split_name="val")
    return train_pdata, val_pdata


def split_plibdata_3fold[PlibDataT: PlibData](
    pdata: PlibDataT, context_ids: str | Sequence[str] | None
) -> tuple[PlibDataT, PlibDataT, PlibDataT]:
    """Split data to training, validation, and test.

    Args:
        pdata: An instance of ``PlibData`` to split.
        context_ids: Contexts from which test and validation perturbations should be selected.

    Returns:
        Three instances of ``PlibData``: training, validation, and test.
    """
    logger.debug("Splitting plibdata instance into train, val, and test")
    remaining_pdata, test_pdata = _split_plibdata(pdata=pdata, context_ids=context_ids, split_name="test")
    train_pdata, val_pdata = _split_plibdata(pdata=remaining_pdata, context_ids=context_ids, split_name="val")
    return train_pdata, val_pdata, test_pdata


def _encode_polars_df(data: pl.DataFrame, vocab: Vocabulary) -> pl.DataFrame:
    """Encode data using given vocabulary.

    Args:
        data: data to encode
        vocab: vocabulary object

    Returns:
        encoded data.
    """
    data = data.with_columns(
        context=pl.col("context").replace_strict(vocab.context_vocab["symbol"], vocab.context_vocab["code"]),
        perturbation=pl.col("perturbation")
        .str.split("+")
        .list.eval(
            pl.element().replace_strict(
                vocab.perturb_vocab["symbol"], vocab.perturb_vocab["code"], default=vocab.perturb_control_code
            )
        ),
        readout=pl.col("readout").replace_strict(vocab.readout_vocab["symbol"], vocab.readout_vocab["code"]),
    )
    return data


class _PolarsDFEncoder:
    """Encode data using a given vocabulary."""

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        return _encode_polars_df(data, self.vocab)


def encode_data[
    T: (
        pl.DataFrame,
        PlibData[pl.DataFrame],
        InMemoryPlibData[pl.DataFrame],
        OnDiskPlibData[pl.DataFrame],
        AnnData,
    )
](data: T, vocab: Vocabulary) -> T:
    """Encode data using given vocabulary.

    Args:
        data: data to encode
        vocab: vocabulary object

    Returns:
        encoded data.
    """
    if isinstance(data, AnnData):
        data_copy = data.copy()  # to ensure we don't modify the original data object
        obs_df: pl.DataFrame = pl.from_pandas(data_copy.obs)
        var_df: pl.DataFrame = pl.from_pandas(data_copy.var)

        obs_df = obs_df.with_columns(
            context=pl.col("context").replace_strict(vocab.context_vocab["symbol"], vocab.context_vocab["code"]),
            perturbation=pl.col("perturbation")
            .str.split("+")
            .list.eval(
                pl.element().replace_strict(
                    vocab.perturb_vocab["symbol"], vocab.perturb_vocab["code"], default=vocab.perturb_control_code
                )
            ),
        )
        var_df = var_df.with_columns(
            readout=pl.col("readout").replace_strict(vocab.readout_vocab["symbol"], vocab.readout_vocab["code"])
        )

        data_copy.obs = obs_df.to_pandas()
        data_copy.var = var_df.to_pandas()
        data = data_copy
    elif isinstance(data, PlibData):
        data = data.apply_transform(_PolarsDFEncoder(vocab))
    elif isinstance(data, pl.DataFrame):
        data = _encode_polars_df(data, vocab)
    else:
        raise ValueError("Wrong data format!")

    return data


def register_context(context_class: type[AnnData]):
    """Register new context to the collection.

    Example::

            import perturb_lib as plib
            import scanpy as sc
            import numpy as np


            @plib.register_context
            class CoolPerturbationScreen(AnnData):
                def __init__(self):
                    super().__init__(
                        X=np.zeros((3, 5), dtype=np.float32),
                        obs={"perturbation": ["P1", "P2", "P3"]},
                        var={"readout": ["R1", "R2", "R3", "R4", "R5"]},
                    )

    Args:
        context_class: context class to register
    Raises:
        ValueError: If a context class with the same name exists already.
    """
    context = context_class.__uncut_name__ if hasattr(context_class, "__uncut_name__") else context_class.__name__
    if context in list_contexts() and (
        getattr(context_class, "__module__", "") != getattr(context_catalogue[context], "__module__", "")
        or context_class.__qualname__ != context_catalogue[context].__qualname__
    ):
        raise ValueError(f"Existing id {context} already registered for a different context!")
    context_catalogue[context] = context_class
    return context_class


def create_and_register_context(
    model_system: ModelSystemType,
    model_system_id: str | None,
    technology_info: TechnologyType,
    data_source_info: str,
    batch_info: str | None,
    full_context_description: str,
    anndata_fn: Callable,
    **anndata_fn_kwargs,
) -> Type[AnnData]:
    """Context creation factory.

    The created context class will be inserted in the namespace of the module where the anndata_fn is defined.

    Args:
        model_system: One of the pre-specified descriptions of the model system.
        model_system_id: Community-recognized identifier of the model system, e.g. K562 for a cell line.
        technology_info: Information about technology used to create the data.
        data_source_info: Identifier of the data source.
        batch_info: Either None or some description of a batch.
        full_context_description: Detailed description of the context.
        anndata_fn: Function used to instantiate constructor arguments for creation of the context object.
        anndata_fn_kwargs: Kwargs for anndata_fn.
    """
    model_system_id = f"_{model_system_id}" if model_system_id is not None else ""
    batch_info = f"_{batch_info}" if batch_info is not None else ""
    uncut_name = f"{model_system}{model_system_id}_{technology_info}_{data_source_info}{batch_info}"
    class_name = _sanitize_context_name(uncut_name)
    source_module_name: str = anndata_fn.__module__
    class_dict = {
        "__doc__": full_context_description,
        "__init__": lambda self: super(context, self).__init__(*anndata_fn(**anndata_fn_kwargs)),  # type: ignore
        "__uncut_name__": uncut_name,
        "__module__": source_module_name,
    }
    context: Type[AnnData] = cast(Type[AnnData], type(class_name, (AnnData,), class_dict))
    setattr(sys.modules[context.__module__], context.__name__, context)  # Insert the class into the module's namespace
    register_context(context)
    return context
