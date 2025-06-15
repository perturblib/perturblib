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

Loading and processing of dummy data.
"""

import numpy as np
from scanpy import AnnData

from perturb_lib.data import ControlSymbol
from perturb_lib.data.access import register_context


@register_context
class DummyData(AnnData):
    """Dummy context that contains random numbers."""

    def __init__(self):
        super().__init__(
            X=np.random.rand(15, 15).astype(np.float32),
            obs={"perturbation": [ControlSymbol] + [f"DummyPerturbation{i}" for i in range(14)]},
            var={"readout": [f"DummyReadout{i}" for i in range(15)]},
        )


@register_context
class DummyDataLongStrings(AnnData):
    """Dummy context that contains random numbers and descriptions as long strings."""

    def __init__(self):
        super().__init__(
            X=np.random.rand(29, 30).astype(np.float32),
            obs={
                "perturbation": [ControlSymbol]
                + [f"DummyPerturbation{i:020d}" for i in range(14)]
                + [f"DummyPerturbation{i}" for i in range(14)]
            },
            var={"readout": [f"DummyReadout{i:020d}" for i in range(15)] + [f"DummyReadout{i}" for i in range(15)]},
        )
