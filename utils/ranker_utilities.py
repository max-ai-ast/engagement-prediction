"""Shared neural ranker components."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class LinearPredictionHead(nn.Module):
    """Linear-layer prediction head for candidate-pair encodings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        dropout_rate: float,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be in [0, 1]")

        hidden_dims = tuple(int(hidden_dim) for hidden_dim in hidden_dims)
        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden_dims must contain only positive values")

        layers: list[nn.Module] = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(float(dropout_rate)),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

        for module in self.network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, encoded_pair: torch.Tensor) -> torch.Tensor:
        return self.network(encoded_pair).squeeze(-1)
