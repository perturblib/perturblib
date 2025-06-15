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

Interface module for all the model-related operations.
"""

import copy
from pathlib import Path
from typing import Dict, List, Type

import torch

from .base import ModelMixin

model_catalogue: Dict[str, Type[ModelMixin]] = {}


def list_models() -> List[str]:
    """Get IDs of registered models.

    Returns:
        A list of model identifiers that are registered in Perturb-lib.
    """
    return sorted([x for x in list(model_catalogue.keys()) if "_on_" not in x])


def _verify_that_model_exists(model_id: str):
    if model_id not in list_models():
        raise ValueError(f"Non-existing model {model_id}; please chose one from: ", list_models())


def describe_model(model_id: str) -> str | None:
    """Describe specified model.

    Args:
        model_id: Identifier of the context.

    Returns:
        The description of the model given as a string.

    Raises:
        ValueError: If given model identifier is not recognized.
    """
    _verify_that_model_exists(model_id)
    return model_catalogue[model_id].__doc__


def load_model(model_id: str, model_args: Dict | None = None) -> ModelMixin:
    """Load specified model.

    Args:
        model_id:  ID of the model.
        model_args: Parameter dictionary to instantiate the model.

    Returns:
        Instantiated model.
    """
    _verify_that_model_exists(model_id)
    if model_args is None:
        model_args = {}
    return model_catalogue[model_id](**copy.deepcopy(model_args))


def save_trained_model(model: ModelMixin, path_to_model: Path, model_args: Dict | None = None):
    """Load specified model.

    Args:
        model:  Model to save.
        path_to_model: Path where the model should be saved.
        model_args: Model parameters.
    """
    model.save(path_to_model, model_args if model_args is not None else {})


def load_trained_model(path_to_model: Path) -> ModelMixin:
    """Load model on the specified path.

    Args:
        path_to_model: Path where the model is located.

    Returns:
        Loaded model.
    """
    load_result = torch.load(path_to_model)
    if isinstance(load_result, ModelMixin):
        model = load_result
    else:
        model_id, model_args, model_state = load_result
        model = load_model(model_id, model_args)
        model.load_state(model_state)
    return model


def register_model(model_class: Type[ModelMixin]):
    """Register new model to the collection.

    Example::

        import perturb_lib as plib
        import numpy as np


        @plib.register_model
        class CoolModel(plib.ModelMixin):
            def fit(self, traindata: plib.PlibData, valdata: plib.PlibData | None = None):
                pass

            def predict(self, data_x: plib.PlibData):
                return np.zeros(len(data_x))

    Args:
        model_class: model class to register
    Raises:
        ValueError: If model with the same name exists already.
    """
    model_id = model_class.__name__
    if model_id in list_models() and (
        getattr(model_class, "__module__", "") != getattr(model_catalogue[model_id], "__module__", "")
        or model_class.__qualname__ != model_catalogue[model_id].__qualname__
    ):
        raise ValueError(f"Existing id {model_id} already registered for a different model!")
    model_catalogue[model_id] = model_class
    return model_class
