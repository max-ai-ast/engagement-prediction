"""Author embedding feature fusion shared by matrix ranking models."""

from __future__ import annotations

import torch
import torch.nn as nn

from shared.input_data_helpers import AUTHOR_PAD_IDX, AUTHOR_UNK_IDX


class PostAuthorFeatureEncoder(nn.Module):
    """Fuse post content embeddings with per-author embeddings."""

    def __init__(
        self,
        post_embedding_dim: int,
        author_table_num_rows: int,
        author_embedding_dim: int,
        author_unknown_dropout_rate: float,
    ):
        super().__init__()
        if author_table_num_rows < 2:
            raise ValueError("author_table_num_rows must be at least 2")
        if author_embedding_dim <= 0:
            raise ValueError("author_embedding_dim must be positive")
        if not 0.0 <= author_unknown_dropout_rate <= 1.0:
            raise ValueError("author_unknown_dropout_rate must be in [0, 1]")

        self.author_pad_idx = AUTHOR_PAD_IDX
        self.author_unk_idx = AUTHOR_UNK_IDX
        self.author_unknown_dropout_rate = float(author_unknown_dropout_rate)
        self.author_embedding = nn.Embedding(
            num_embeddings=author_table_num_rows,
            embedding_dim=author_embedding_dim,
            padding_idx=self.author_pad_idx,
        )
        nn.init.xavier_uniform_(self.author_embedding.weight)
        with torch.no_grad():
            self.author_embedding.weight[self.author_pad_idx].zero_()

        self.fusion_layer = nn.Linear(
            post_embedding_dim + author_embedding_dim,
            post_embedding_dim,
        )
        nn.init.xavier_uniform_(self.fusion_layer.weight)
        if self.fusion_layer.bias is not None:
            nn.init.zeros_(self.fusion_layer.bias)

    def forward(
        self,
        post_embeddings: torch.Tensor,
        author_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.training and self.author_unknown_dropout_rate > 0.0:
            eligible = author_indices > self.author_unk_idx
            if torch.any(eligible):
                dropout_mask = torch.rand(
                    author_indices.shape,
                    device=author_indices.device,
                ) < self.author_unknown_dropout_rate
                author_indices = torch.where(
                    eligible & dropout_mask,
                    torch.full_like(author_indices, self.author_unk_idx),
                    author_indices,
                )

        author_embeddings = self.author_embedding(author_indices)
        fused_inputs = torch.cat([post_embeddings, author_embeddings], dim=-1)
        return self.fusion_layer(fused_inputs)
