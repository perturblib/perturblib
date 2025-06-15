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

Fundamental/base classes for the evaluation module.
"""

from abc import ABCMeta, abstractmethod

from numpy.typing import NDArray

from perturb_lib import PlibData


class PlibEvaluatorMixin(metaclass=ABCMeta):
    """Mixin for all the Perturb-lib evaluators."""

    @abstractmethod
    def evaluate(self, predictions: NDArray, true_values: PlibData) -> float:
        """Base evaluation function.

        Args:
            predictions: predictions of the model to be evaluated
            true_values: true values to evaluate predictions on

        Returns:
            evaluation score
        """
