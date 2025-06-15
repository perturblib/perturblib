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

Module responsible for environmental details and configurations, on the library level.
"""

import logging
import os
import random
from pathlib import Path

import appdirs
import numpy as np
import polars as pl
import torch

# perturb_lib logger
logger = logging.getLogger("perturb_lib")
_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
_handler = logging.StreamHandler()
_handler.setFormatter(_formatter)
logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def set_path_to_cache(path_to_cache: str | Path):
    """Set path to cache where temporary data of perturb-lib is stored."""
    path_to_cache = Path(path_to_cache)
    path_to_cache.mkdir(exist_ok=True, parents=True)
    os.environ["PERTURB_LIB_CACHE_DIR"] = str(path_to_cache)


def get_path_to_cache() -> Path:
    """Get path to cache where temporary data of perturb-lib is stored."""
    return Path(os.environ["PERTURB_LIB_CACHE_DIR"])


if os.environ.get("PERTURB_LIB_CACHE_DIR") is None:
    if "site-packages" in str(Path(__file__)):
        set_path_to_cache(Path(appdirs.user_cache_dir("perturb_lib")))
    else:
        set_path_to_cache(Path(__file__).parent.parent.resolve() / ".plib_cache")


def set_seed(seed: int):
    """Set random seed used in perturb-lib."""
    os.environ["PERTURB_LIB_RANDOM_SEED"] = str(seed)


def set_all_seeds(seed: int):
    """Set random seed for different libraries used within perturb-lib."""
    logger.info(f"Fixing random seeds of numpy, pytorch, random, and perturb-lib to {seed}.")
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.set_random_seed(seed)


def get_seed() -> int:
    """Get random seed used in perturb-lib."""
    return int(os.environ["PERTURB_LIB_RANDOM_SEED"])


# by default, we fix seed only for perturb-lib
set_seed(13)
