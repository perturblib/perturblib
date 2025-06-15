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

import gzip
import importlib
import os
import random
import shutil
import sys
from functools import wraps
from pathlib import Path
from tarfile import TarFile
from typing import Callable, List, Set
from zipfile import ZipFile

import numpy as np
import requests
from tqdm import tqdm

from perturb_lib.environment import logger


def try_import(module_name: str):
    dependency_groups = {
        "xgboost": ["xgboost"],
        "catboost": ["catboost"],
        "dev": ["jsonargparse", "pre_commit", "black", "isort", "pytest", "seaborn"],
    }
    dependency_to_group = {dependency: group for group, deps in dependency_groups.items() for dependency in deps}
    try:
        importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        if module_name in dependency_to_group:
            raise ImportError(
                f"Cannot import `{module_name}`. Make sure you have installed `{module_name}` "
                f"using `pip install perturb_lib[{dependency_to_group[module_name]}]`"
            )
        else:
            raise ImportError(f"Cannot import `{module_name}`. Make sure `{module_name}` is installed.")


def select_random_subset(the_set: Set | List, size: int, seed: int | None = None):
    if size < 0:
        raise ValueError("size must be non-negative")
    if size > len(the_set):
        raise ValueError("size cannot be larger than the set")
    return random.Random(seed).sample(sorted(the_set), size)


def split_by_ratio(arr: np.ndarray, *ratios):
    arr = np.random.permutation(arr)
    ind = np.add.accumulate(np.array(ratios) * len(arr)).astype(int)
    return [x.tolist() for x in np.split(arr, ind)][: len(ratios)]


def copy_doc(from_func: Callable) -> Callable:
    def decorator(to_func: Callable) -> Callable:
        to_func.__doc__ = from_func.__doc__
        return to_func

    return decorator


def download_file(url: str, save_path: Path):
    """Download helper with progress bar"""
    if save_path.exists():
        logger.info(f"{save_path.name} found in cache.")
    else:
        logger.info(f"Downloading {save_path.name}...")
        response = requests.get(url, stream=True, verify=False)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
        temp_save_path = save_path.with_suffix(".tmp")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(temp_save_path, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            # shutil.move(temp_save_path, save_path)
            temp_save_path.rename(save_path)
        except Exception as e:
            logger.error(f"Error occurred while downloading {save_path.name}: {str(e)}")
            if temp_save_path.exists():
                temp_save_path.unlink()
        finally:
            progress_bar.close()


def download_extract_zip_file(url: str, save_path: Path, data_path: Path):
    if save_path.exists():
        logger.info("Found .zip file in cache...")
    else:
        download_file(url, Path(f"{str(save_path)}.zip"))
        with ZipFile(Path(f"{str(save_path)}.zip"), "r") as zip_file:
            zip_file.extractall(path=data_path)


def download_extract_tar_file(url: str, save_path: Path, data_path: Path):
    if save_path.exists():
        logger.info("File found in the cache...")
    else:
        download_file(url, Path(f"{str(save_path)}.tar"))
        logger.info("Extracting tar file...")
        with TarFile(Path(f"{str(save_path)}.tar"), "r") as tar_file:
            tar_file.extractall(path=data_path)
        logger.info("Done!")
        os.system(f"gzip -k -d -f {data_path}/*.gz")


def download_extract_gz_file(url: str, save_path: Path):
    if save_path.exists():
        logger.info("File found in the cache...")
    else:
        download_file(url, Path(f"{str(save_path)}.gz"))
        logger.info("Extracting gz file...")
        with gzip.open(f"{str(save_path)}.gz", "rb") as f_in:
            with open(save_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        logger.info("Done!")


def list_python_modules_from_directory(path: Path):
    return [
        x.stem
        for x in path.glob("**/*")
        if x.is_file() and str(x).endswith(".py") and not str(x).endswith("__init__.py")
    ]


def silence_output(func):
    """Function decorator that silences output"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        original_stderr = sys.stderr
        original_stdout = sys.stdout
        sys.stderr = open(os.devnull, "w")
        sys.stdout = open(os.devnull, "w")
        try:
            return func(*args, **kwargs)
        finally:
            sys.stderr.close()
            sys.stderr = original_stderr
            sys.stdout.close()
            sys.stdout = original_stdout

    return wrapper
