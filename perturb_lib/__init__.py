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

import perturb_lib.utils as utils
from perturb_lib.data import ControlSymbol
from perturb_lib.data.access import (
    Vocabulary,
    describe_context,
    list_contexts,
    load_anndata,
    load_plibdata,
    register_context,
    split_plibdata_2fold,
    split_plibdata_3fold,
)
from perturb_lib.data.collection import *
from perturb_lib.data.plibdata import InMemoryPlibData, OnDiskPlibData, PlibData
from perturb_lib.data.preprocesing import DEFAULT_PREPROCESSING_TYPE, PreprocessingType, preprocessors
from perturb_lib.embeddings.access import (
    describe_embedding,
    list_embeddings,
    load_embedding,
    register_embedding,
)
from perturb_lib.embeddings.collection import *
from perturb_lib.environment import get_path_to_cache, get_seed, logger, set_all_seeds, set_path_to_cache, set_seed
from perturb_lib.evaluators.access import describe_evaluator, list_evaluators, load_evaluator, register_evaluator
from perturb_lib.evaluators.base import PlibEvaluatorMixin
from perturb_lib.evaluators.collection import *
from perturb_lib.models.access import (
    describe_model,
    list_models,
    load_model,
    load_trained_model,
    register_model,
    save_trained_model,
)
from perturb_lib.models.base import ModelMixin, SklearnModel
from perturb_lib.models.collection import *

__all__ = [
    "utils",
    "describe_context",
    "list_contexts",
    "load_anndata",
    "load_plibdata",
    "logger",
    "register_context",
    "split_plibdata_2fold",
    "split_plibdata_3fold",
    "set_seed",
    "set_all_seeds",
    "get_seed",
    "set_path_to_cache",
    "get_path_to_cache",
    "InMemoryPlibData",
    "OnDiskPlibData",
    "InMemoryPlibData",
    "PlibData",
    "describe_embedding",
    "list_embeddings",
    "load_embedding",
    "register_embedding",
    "describe_evaluator",
    "list_evaluators",
    "load_evaluator",
    "register_evaluator",
    "PlibEvaluatorMixin",
    "describe_model",
    "list_models",
    "load_model",
    "load_trained_model",
    "register_model",
    "save_trained_model",
    "ModelMixin",
    "SklearnModel",
    "Vocabulary",
    "PreprocessingType",
    "preprocessors",
    "DEFAULT_PREPROCESSING_TYPE",
    "ControlSymbol",
]
