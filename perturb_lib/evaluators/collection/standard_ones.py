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

Collection of standard evaluators of perturbation models.
"""

from numpy.typing import NDArray
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from perturb_lib import PlibData
from perturb_lib.evaluators.access import register_evaluator
from perturb_lib.evaluators.base import PlibEvaluatorMixin
from perturb_lib.utils import inherit_docstring


@register_evaluator
@inherit_docstring
class RMSE(PlibEvaluatorMixin):
    """Root-mean-square error (RMSE)."""

    def evaluate(self, predictions: NDArray, true_values: PlibData) -> float:  # noqa: D102
        return root_mean_squared_error(true_values[:]["value"], predictions)


@register_evaluator
@inherit_docstring
class MAE(PlibEvaluatorMixin):
    """Mean absolute error (MAE)."""

    def evaluate(self, predictions: NDArray, true_values: PlibData) -> float:  # noqa: D102\
        return mean_absolute_error(true_values[:]["value"], predictions)


@register_evaluator
@inherit_docstring
class R2(PlibEvaluatorMixin):
    """R2 score function.

    Represents the proportion of variance (of y) that has been explained by the independent
    variables in the model.
    """

    def evaluate(self, predictions: NDArray, true_values: PlibData) -> float:  # noqa: D102
        return r2_score(true_values[:]["value"], predictions)


@register_evaluator
@inherit_docstring
class Pearson(PlibEvaluatorMixin):
    """Pearson correlation coefficient.

    Measures the linear relationship predictions and ground truth. Strictly speaking,
    Pearson’s correlation assumes that outputs be normally distributed.
    """

    def evaluate(self, predictions: NDArray, true_values: PlibData) -> float:  # noqa: D102
        return pearsonr(true_values[:]["value"], predictions).statistic
