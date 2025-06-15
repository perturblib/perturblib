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

Interface module for all the evaluation-related operations.
"""

from typing import Dict, List, Type

from perturb_lib.evaluators.base import PlibEvaluatorMixin

evaluator_catalogue: Dict[str, Type[PlibEvaluatorMixin]] = {}


def list_evaluators() -> List[str]:
    """Get IDs of registered evaluators.

    Returns:
        A list of evaluator identifiers that are registered in Perturb-lib.
    """
    return sorted(list(evaluator_catalogue.keys()))


def _verify_that_evaluator_exists(evaluator_id: str):
    if evaluator_id not in list_evaluators():
        raise ValueError(f"Unavailable evaluator {evaluator_id}: chose one of {list_evaluators()}")


def describe_evaluator(evaluator_id: str) -> str | None:
    """Describe specified evaluator.

    Args:
        evaluator_id: Identifier of the evaluator.

    Returns:
        The description of the evaluator given as a string.

    Raises:
        ValueError: If given evaluator identifier is not recognized.
    """
    _verify_that_evaluator_exists(evaluator_id)
    return evaluator_catalogue[evaluator_id].__doc__


def load_evaluator(evaluator_id: str) -> PlibEvaluatorMixin:
    """Load specified evaluator.

    Args:
        evaluator_id: Identifier of the evaluator.

    Returns:
        Specified evaluator.
    """
    _verify_that_evaluator_exists(evaluator_id)
    return evaluator_catalogue[evaluator_id]()


def register_evaluator(evaluator_class: Type[PlibEvaluatorMixin]):
    """Register new evaluator to the collection.

    Example::

        import perturb_lib as plib
        import numpy as np


        @plib.register_evaluator
        class CoolEvaluator(plib.PlibEvaluatorMixin):
            def _evaluate_predictions(self, predictions, true_values):
                return np.zeros(len(true_values))

    Args:
        evaluator_class: evaluator class to register
    Raises:
        ValueError: If evaluator with the same name exists already.
    """
    evaluator_id = evaluator_class.__name__
    if evaluator_id in list_evaluators() and (
        getattr(evaluator_class, "__module__", "") != getattr(evaluator_catalogue[evaluator_id], "__module__", "")
        or evaluator_class.__qualname__ != evaluator_catalogue[evaluator_id].__qualname__
    ):
        raise ValueError(f"Existing id {evaluator_id} already registered for a different evaluator!")
    evaluator_catalogue[evaluator_id] = evaluator_class
    return evaluator_class
