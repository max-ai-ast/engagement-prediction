"""Shared neural ranker components."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from shared.input_data_helpers import AUTHOR_PAD_IDX, AUTHOR_UNK_IDX


class BSTPostAuthorFeatureEncoder(nn.Module):
    """Fuse MiniLM post embeddings with author embeddings for candidate-aware rankers."""

    def __init__(
        self,
        post_embedding_dim: int,
        author_table_num_rows: int,
        author_embedding_dim: int,
        content_projection_dim: int,
        author_projection_dim: int,
        model_dim: int,
        author_unknown_dropout_rate: float,
    ):
        super().__init__()
        if post_embedding_dim <= 0:
            raise ValueError("post_embedding_dim must be positive")
        if author_table_num_rows < 2:
            raise ValueError("author_table_num_rows must be at least 2")
        if author_embedding_dim <= 0:
            raise ValueError("author_embedding_dim must be positive")
        if content_projection_dim <= 0:
            raise ValueError("content_projection_dim must be positive")
        if author_projection_dim <= 0:
            raise ValueError("author_projection_dim must be positive")
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if not 0.0 <= author_unknown_dropout_rate <= 1.0:
            raise ValueError("author_unknown_dropout_rate must be in [0, 1]")

        self.post_embedding_dim = int(post_embedding_dim)
        self.content_projection_dim = int(content_projection_dim)
        self.author_projection_dim = int(author_projection_dim)
        self.model_dim = int(model_dim)
        self.author_unknown_dropout_rate = float(author_unknown_dropout_rate)
        self.author_embedding = nn.Embedding(
            num_embeddings=int(author_table_num_rows),
            embedding_dim=int(author_embedding_dim),
            padding_idx=AUTHOR_PAD_IDX,
        )
        nn.init.xavier_uniform_(self.author_embedding.weight)
        with torch.no_grad():
            self.author_embedding.weight[AUTHOR_PAD_IDX].zero_()

        self.content_projection = nn.Linear(
            int(post_embedding_dim),
            self.content_projection_dim,
        )
        self.author_projection = nn.Linear(
            int(author_embedding_dim),
            self.author_projection_dim,
        )
        self.projection_activation = nn.GELU()
        self.content_projection_norm = nn.LayerNorm(self.content_projection_dim)
        self.author_projection_norm = nn.LayerNorm(self.author_projection_dim)
        self.fusion_layer = nn.Linear(
            self.content_projection_dim + self.author_projection_dim,
            int(model_dim),
        )
        for layer in (self.content_projection, self.author_projection, self.fusion_layer):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        post_embeddings: torch.Tensor,
        author_indices: torch.Tensor,
    ) -> torch.Tensor:
        if post_embeddings.size(-1) != self.post_embedding_dim:
            raise ValueError(
                f"post_embeddings last dimension ({post_embeddings.size(-1)}) must match post_embedding_dim ({self.post_embedding_dim})"
            )
        if post_embeddings.shape[:-1] != author_indices.shape:
            raise ValueError("author_indices shape must match post_embeddings leading dimensions")

        author_indices = author_indices.to(device=post_embeddings.device, dtype=torch.long)
        if self.training and self.author_unknown_dropout_rate > 0.0:
            eligible = author_indices > AUTHOR_UNK_IDX
            if torch.any(eligible):
                dropout_mask = torch.rand(author_indices.shape, device=author_indices.device) < self.author_unknown_dropout_rate
                author_indices = torch.where(
                    eligible & dropout_mask,
                    torch.full_like(author_indices, AUTHOR_UNK_IDX),
                    author_indices,
                )

        author_embeddings = self.author_embedding(author_indices)
        content_features = self.content_projection_norm(
            self.projection_activation(self.content_projection(post_embeddings))
        )
        author_features = self.author_projection_norm(
            self.projection_activation(self.author_projection(author_embeddings))
        )
        fused_inputs = torch.cat([content_features, author_features], dim=-1)
        return self.fusion_layer(fused_inputs)


class LinearPredictionHead(nn.Module):
    """Linear-layer prediction head for candidate-pair encodings."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
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
