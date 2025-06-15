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

Library-level utility functions.
"""

from collections import defaultdict
from dataclasses import dataclass

import polars as pl
import torch.distributed
import torch.utils.data
from scanpy import AnnData

from perturb_lib._utils import download_file
from perturb_lib.environment import get_path_to_cache, logger


def get_hgnc_name_updates():
    """Mapping of outdated symbols and aliases to the latest HGNC symbol.
    Columns:
        <index>: potentially outdated gene symbol
        rename_to: the best candidate to rename to (same as <index> if no rename is needed)
        is_ambiguous: True if the new symbol couldn't be unambiguously chosen
        is_protein_coding: whether rename_to's locus type is "gene with protein product"

    Renaming uses these rules:
    1. If `symbol` is already in HGNC: don't rename it (`rename_to = symbol`)
    2. If `symbol` is in HGNC with a different case: rename to the correct case
    3. Otherwise, if `symbol` is in another symbol's `prev_symbol`: rename to that other symbol
    4. Otherwise, if `symbol` is in another symbol's `aliases`: rename to that other symbol

    Citation:
    HGNC Database, HUGO Gene Nomenclature Committee (HGNC), European Molecular Biology Laboratory,
    European Bioinformatics Institute (EMBL-EBI), Wellcome Genome Campus, Hinxton, Cambridge CB10 1SD, United Kingdom
    www.genenames.org
    """
    url = "https://ftp.ebi.ac.uk/pub/databases/genenames/out_of_date_hgnc/archive/monthly/tsv/hgnc_complete_set_2022-07-01.txt"
    download_path = get_path_to_cache() / "hgnc_complete_set.tsv"
    csv_path = get_path_to_cache() / "hgnc_name_updates.csv"

    if not csv_path.exists():
        download_file(url, download_path)
        hgnc = pl.read_csv(
            download_path,
            separator="\t",
            columns=[
                "hgnc_id",
                "symbol",
                "name",
                "alias_symbol",
                "alias_name",
                "prev_symbol",
                "prev_name",
                "locus_type",
            ],
        )

        protein_coding_symbols = hgnc.select("symbol", "locus_type").filter(
            pl.col("locus_type") == "gene with protein product"
        )["symbol"]

        # Make inverse prev_symbol lookup
        next_symbols = defaultdict(list)
        prev_symbols_df = hgnc.select("symbol", "prev_symbol").filter(pl.col("prev_symbol").is_not_null())
        for row in prev_symbols_df.rows(named=True):
            symbol = row["symbol"]
            prev_symbols = row["prev_symbol"]
            for prev_symbol in prev_symbols.split("|"):
                next_symbols[prev_symbol].append(symbol)

        # Make inverse aliases lookup
        alias_ofs = defaultdict(list)
        alias_symbols_df = hgnc.select("symbol", "alias_symbol").filter(pl.col("alias_symbol").is_not_null())
        for row in alias_symbols_df.rows(named=True):
            symbol = row["symbol"]
            aliases = row["alias_symbol"]
            for alias in aliases.split("|"):
                alias_ofs[alias].append(symbol)

        # Add an alias to fix incorrectly upper-cased symbols (e.g. "C11ORF1" should be "C11orf1").
        # Currently only genes containing "orf" (="open reading frame") seem to contain lower-case letters, and they
        # can be renamed unambiguously. If HGNC changes cause these aliases to become ambiguous, it may be necessary to
        # rethink this solution, e.g. by only doing this rename on datasets that consistently use the incorrect case.
        # (?-i) forces case-sensitive matching and therefore detects genes that contain lower-case letters
        mixed_case_symbols: pl.Series = hgnc.filter(pl.col("symbol").str.contains("(?-i)[a-z]"))["symbol"]
        if hgnc["symbol"].is_in(mixed_case_symbols.str.to_uppercase()).any():
            raise AssertionError("HGNC contains ambiguous mixed-case genes")
        for symbol in mixed_case_symbols.to_list():
            alias_ofs[symbol.upper()].append(symbol)

        hgnc_index = set(hgnc["symbol"].to_list())
        all_symbols = hgnc_index.union(next_symbols.keys(), alias_ofs.keys())

        mappings = []
        for symbol in sorted(all_symbols):
            next_candidates = next_symbols.get(symbol, None)
            alias_candidates = alias_ofs.get(symbol, None)
            if symbol in hgnc_index:
                rename_to = symbol
                is_ambiguous = False
            elif next_candidates is not None:
                rename_to = next_candidates[0]
                is_ambiguous = len(next_candidates) > 1
            else:
                assert alias_candidates is not None
                rename_to = alias_candidates[0]
                is_ambiguous = len(alias_candidates) > 1

            mappings.append(
                {
                    "symbol": symbol,
                    "rename_to": rename_to,
                    "is_ambiguous": is_ambiguous,
                    "is_protein_coding": rename_to in protein_coding_symbols,
                }
            )

        df = pl.DataFrame(mappings)
        df.write_csv(csv_path)

    ret_df = pl.read_csv(csv_path)
    ret_df = ret_df.filter((pl.col("symbol") != "") & pl.col("symbol").is_not_null())
    return ret_df


def update_symbols[T: (AnnData, pl.DataFrame)](data: T) -> T:
    """Using HGNC mapping to update symbols."""
    logger.debug("Updating outdated symbol names based on HGNC..")
    hgnc_name_updates: pl.DataFrame = get_hgnc_name_updates()
    hgnc_map = dict(zip(hgnc_name_updates["symbol"], hgnc_name_updates["rename_to"]))
    # data contains an experiment
    if isinstance(data, AnnData):
        for df in [data.obs, data.var]:
            for col in df.columns:
                df[col] = df[col].map(lambda x: hgnc_map.get(x, x))
    # data contains an embedding
    elif isinstance(data, pl.DataFrame):
        data = data.with_columns(index=pl.col("index").replace(hgnc_map))
        data = data.unique(subset="index")
    else:
        raise ValueError("Wrong data type!")
    return data


def get_ensembl_ids_to_hgnc_name_map() -> pl.DataFrame:
    """Mapping of Ensembl IDs the latest HGNC symbol.

    Returned DataFrame has the following columns:
        <index>: Ensemble ID
        rename_to: the best candidate to rename to
        is_ambiguous: True if the new symbol couldn't be unambiguously chosen

    Citation:
    HGNC Database, HUGO Gene Nomenclature Committee (HGNC), European Molecular Biology Laboratory,
    European Bioinformatics Institute (EMBL-EBI), Wellcome Genome Campus, Hinxton, Cambridge CB10 1SD, United Kingdom
    www.genenames.org
    """
    url = "https://ftp.ebi.ac.uk/pub/databases/genenames/out_of_date_hgnc/archive/monthly/tsv/hgnc_complete_set_2022-07-01.txt"
    download_path = get_path_to_cache() / "hgnc_complete_set.tsv"

    csv_path = get_path_to_cache() / "ensembl_to_hgnc_map.csv"

    if not csv_path.exists():
        download_file(url, download_path)
        hgnc = pl.read_csv(download_path, separator="\t")

        # Make ensemble to HGNC map
        ensemble_to_hgnc = hgnc.select("symbol", "ensembl_gene_id").filter(pl.col("ensembl_gene_id").is_not_null())
        ensemble_to_hgnc = ensemble_to_hgnc.with_columns(has_duplicates=pl.col("ensembl_gene_id").is_duplicated())
        ensemble_to_hgnc = ensemble_to_hgnc.unique(subset=["ensembl_gene_id"], keep="first", maintain_order=True)

        mappings = ensemble_to_hgnc.rename(
            {"symbol": "rename_to", "ensembl_gene_id": "symbol", "has_duplicates": "is_ambiguous"}
        ).select("symbol", "rename_to", "is_ambiguous")

        mappings.write_csv(csv_path)

    ret_df = pl.read_csv(csv_path)
    # drop the rows that contain empty cells
    ret_df = ret_df.filter((pl.col("symbol") != "") & pl.col("symbol").is_not_null())
    return ret_df


def inherit_docstring(cls):
    """Decorator to enable docstring inheritance."""
    for name, method in cls.__dict__.items():
        if callable(method) and not method.__doc__:
            base_method = getattr(super(cls, cls), name, None)
            if base_method and base_method.__doc__:
                method.__doc__ = base_method.__doc__
    return cls


@dataclass
class RankInfo:
    """Dataclass storing information about the rank of the current process."""

    rank: int = 0
    world_size: int = 1


def get_rank_info(group: torch.distributed.ProcessGroup | None = None) -> RankInfo:
    """Get node rank info.

    Args:
        group: ProcessGroup used for determining distributed rank. If None,
            `torch.distributed.group.WORLD` will be used.

    Returns:
        `RankInfo` with information on the node and total number of nodes.
    """
    rank: int = 0
    world_size: int = 1

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        group = group or torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        world_size = torch.distributed.get_world_size(group=group)

    return RankInfo(rank=rank, world_size=world_size)
