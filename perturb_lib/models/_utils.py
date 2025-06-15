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

import re

import polars as pl
import torch
from pytorch_lightning.profilers import SimpleProfiler
from tabulate import tabulate
from torch import nn


class OneHotEmbedding(nn.Module):
    def __init__(self, num_categories):
        super().__init__()
        self.num_categories = num_categories

    def forward(self, tensor: torch.Tensor):
        return nn.functional.one_hot(tensor, self.num_categories)


def numeric_series2tensor(series: pl.Series) -> torch.Tensor:
    """Conversion to torch tensor."""
    if not series.dtype.is_numeric():
        raise ValueError(f"Expected numeric series, got {series.dtype}")

    return torch.from_numpy(series.to_numpy(writable=True))


class LPMProfiler(SimpleProfiler):
    include = {"Strategy", "LightningModule", "_TrainingEpochLoop", "_EvaluationLoop"}

    @staticmethod
    def _format_table_with_tabulate(table_lines):
        """Automatically format and align a table given a list of '|' separated strings."""
        table_data = [line.split("|")[1:-1] for line in table_lines]  # Remove empty edges
        table_data = [[col.strip() for col in row] for row in table_data]  # Strip spaces
        return tabulate(table_data, headers="firstrow", tablefmt="grid")  # Auto-format

    def summary(self):
        output = super().summary()
        lines = output.split("\n")
        header_lines, data_lines = lines[3:4], lines[5:-2]
        data_lines_filtered = []
        for line in data_lines:
            if any(word in line for word in self.include):
                line = re.sub(r"[\w.\[\]]*train_dataloader_next[\w.\[\]]*", "Get batch [training]", line)
                line = re.sub(r"[\w.\[\]]*.val_next[\w.\[\]]*", "Get batch [validation]", line)
                line = re.sub(r"[\w.\[\]]*training_step[\w.\[\]]*", "Forward propagation [training]", line)
                line = re.sub(r"[\w.\[\]]*validation_step[\w.\[\]]*", "Forward propagation [validation]", line)
                line = re.sub(r"[\w.\[\]]*optimizer_step[\w.\[\]]*", "Gradient update [backprop]", line)
                line = re.sub(r"[\w.\[\]]*backward[\w.\[\]]*", "Gradient calculation [backprop]", line)
                line = re.sub(r"[\w.\[\]]*batch_to_device[\w.\[\]]*", "Batch transfer", line)
                data_lines_filtered.append(line)
        sorted_lines = sorted(data_lines_filtered, key=lambda line: float(line.split("|")[4].strip()), reverse=True)
        sorted_lines = sorted_lines[:10]  # limit to 10 rows
        tabulated_lines = self._format_table_with_tabulate(header_lines + sorted_lines)
        return tabulated_lines
