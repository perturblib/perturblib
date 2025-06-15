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

Perturb Gym Utilities
"""

import hashlib
import itertools
import pickle
from pathlib import Path
from typing import Any

from perturb_gym.configs.base import TrainingConfig


def list_yaml_modules_from_directory(path_to_dir: Path):
    """Find and list all yaml files in the given directory."""
    return [x.stem for x in path_to_dir.glob("**/*") if x.is_file() and str(x).endswith(".yaml")]


def dict_product(the_dict: dict[str, list]) -> list[dict]:
    """Deriving all possible key,value pairs in a list, given a dictionary."""
    if not the_dict:
        return [{}]
    keys, values = zip(*the_dict.items())
    values = [v if isinstance(v, list) else [v] for v in values]
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def get_user_confirmation(prompt_message):
    """Prompting user to confirm its actions."""
    while True:
        user_input = input(prompt_message).strip().lower()
        if user_input in ["yes", "y"]:
            return True
        elif user_input in ["no", "n"]:
            return False
        else:
            print("Please enter 'yes' or 'no'.")


def flatten_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Flatten a nested dictionary."""

    def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        items: list[tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    return _flatten(d)


def hash_training_config_excluding_seed(training_config: TrainingConfig) -> str:
    """Hash the training configuration excluding the seed.

    Args:
        training_config: The training configuration to hash.

    Returns: A hash of the training configuration excluding the seed.

    """
    flattened_config = flatten_dict(training_config)
    flattened_config.pop("environment_config.seed", None)

    # Convert to ordered list of tuples
    items = sorted(flattened_config.items())
    h = hashlib.blake2b(pickle.dumps(items), digest_size=8)  # 8 bytes ⇒ 16‑hex‑char string
    digest = h.hexdigest()
    return digest


def get_same_models_with_different_seed(path_to_model: Path) -> list[Path]:
    """Get all models with the same arguments, hyperaparameters, and training data, but different random seeds.

    Args:
        path_to_model: Path to a .pt model file.

    Returns: A list of model paths

    """
    if not path_to_model.is_file():
        raise ValueError(f"{path_to_model} is not a file")

    current_seed_dir = path_to_model.parent
    multiple_seeds_dir = current_seed_dir.parent

    seed_dirs = [seed_dir for seed_dir in multiple_seeds_dir.iterdir() if (seed_dir / "model.pt").is_file()]

    # return the paths to the models
    return [path / "model.pt" for path in seed_dirs]
