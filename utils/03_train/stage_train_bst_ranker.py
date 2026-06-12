#!/usr/bin/env python3

"""Stage 3 model components for a BST heavy ranker."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from shared.input_data_helpers import AUTHOR_PAD_IDX, AUTHOR_UNK_IDX


DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS = (1.0, 3.0, 6.0, 12.0, 24.0, 72.0, 168.0, 720.0, 2160.0)


def _validate_time_delta_bucket_boundaries(boundaries_hours: Sequence[float]) -> tuple[float, ...]:
    boundaries = tuple(float(boundary) for boundary in boundaries_hours)
    if len(boundaries) == 0:
        raise ValueError("time delta bucket boundaries must not be empty")
    previous = 0.0
    for boundary in boundaries:
        if boundary <= 0.0:
            raise ValueError("time delta bucket boundaries must be positive")
        if boundary <= previous:
            raise ValueError("time delta bucket boundaries must be strictly increasing")
        previous = boundary
    return boundaries


def bucketize_time_deltas_hours(
    time_deltas_hours: torch.Tensor,
    boundaries_hours: Sequence[float] = DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS,
) -> torch.Tensor:
    """Map raw hour deltas to embedding-table bucket IDs."""
    boundaries = _validate_time_delta_bucket_boundaries(boundaries_hours)
    deltas = time_deltas_hours
    if not torch.is_floating_point(deltas):
        deltas = deltas.to(dtype=torch.float32)
    deltas = torch.clamp(deltas, min=0.0)
    boundary_tensor = torch.tensor(boundaries, device=deltas.device, dtype=deltas.dtype)
    positive_bucket_ids = torch.bucketize(deltas, boundary_tensor, right=False) + 1
    zero_bucket_ids = torch.zeros_like(positive_bucket_ids)
    return torch.where(deltas <= 0.0, zero_bucket_ids, positive_bucket_ids).to(dtype=torch.long)


class BSTPostAuthorFeatureEncoder(nn.Module):
    """Fuse MiniLM post embeddings with author embeddings for the BST ranker."""

    def __init__(
        self,
        post_embedding_dim: int,
        author_table_num_rows: int,
        author_embedding_dim: int,
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
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if not 0.0 <= author_unknown_dropout_rate <= 1.0:
            raise ValueError("author_unknown_dropout_rate must be in [0, 1]")

        self.post_embedding_dim = int(post_embedding_dim)
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

        self.fusion_layer = nn.Linear(
            int(post_embedding_dim) + int(author_embedding_dim),
            int(model_dim),
        )
        nn.init.xavier_uniform_(self.fusion_layer.weight)
        if self.fusion_layer.bias is not None:
            nn.init.zeros_(self.fusion_layer.bias)

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
        fused_inputs = torch.cat([post_embeddings, author_embeddings], dim=-1)
        return self.fusion_layer(fused_inputs)


class BSTRanker(nn.Module):
    """Behavior Sequence Transformer encoder for one user-history/candidate pair."""

    def __init__(
        self,
        post_embedding_dim: int,
        author_table_num_rows: int,
        author_embedding_dim: int,
        model_dim: int,
        time_embedding_dim: int,
        num_attention_heads: int,
        num_transformer_layers: int,
        transformer_ff_dim: int,
        dropout_rate: float,
        author_unknown_dropout_rate: float,
        norm_first: bool,
        time_delta_bucket_boundaries_hours: Sequence[float],
    ):
        super().__init__()
        if time_embedding_dim <= 0:
            raise ValueError("time_embedding_dim must be positive")
        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if num_transformer_layers <= 0:
            raise ValueError("num_transformer_layers must be positive")
        if transformer_ff_dim <= 0:
            raise ValueError("transformer_ff_dim must be positive")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be in [0, 1]")

        self.post_embedding_dim = int(post_embedding_dim)
        self.model_dim = int(model_dim)
        self.time_embedding_dim = int(time_embedding_dim)
        self.time_delta_bucket_boundaries_hours = _validate_time_delta_bucket_boundaries(
            time_delta_bucket_boundaries_hours
        )
        self.num_time_delta_buckets = len(self.time_delta_bucket_boundaries_hours) + 2
        self.transformer_input_dim = self.model_dim + self.time_embedding_dim
        if self.transformer_input_dim % int(num_attention_heads) != 0:
            raise ValueError("model_dim + time_embedding_dim must be divisible by num_attention_heads")

        self.post_feature_encoder = BSTPostAuthorFeatureEncoder(
            post_embedding_dim=post_embedding_dim,
            author_table_num_rows=author_table_num_rows,
            author_embedding_dim=author_embedding_dim,
            model_dim=model_dim,
            author_unknown_dropout_rate=author_unknown_dropout_rate,
        )
        self.time_delta_embedding = nn.Embedding(
            num_embeddings=self.num_time_delta_buckets,
            embedding_dim=self.time_embedding_dim,
        )
        nn.init.xavier_uniform_(self.time_delta_embedding.weight)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_input_dim,
            nhead=int(num_attention_heads),
            dim_feedforward=int(transformer_ff_dim),
            dropout=float(dropout_rate),
            activation="gelu",
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(num_transformer_layers),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
        candidate_post_embeddings: torch.Tensor,
        history_author_indices: torch.Tensor,
        candidate_post_author_idx: torch.Tensor,
    ) -> torch.Tensor:
        if history_embeddings.dim() != 3:
            raise ValueError("history_embeddings must have shape [B, H, D]")
        if candidate_post_embeddings.dim() != 2:
            raise ValueError("candidate_post_embeddings must have shape [B, D]")
        batch_size, max_history_len, embed_dim = history_embeddings.shape
        if embed_dim != self.post_embedding_dim:
            raise ValueError(
                f"history_embeddings last dimension ({embed_dim}) must match post_embedding_dim ({self.post_embedding_dim})"
            )
        if candidate_post_embeddings.shape != (batch_size, self.post_embedding_dim):
            raise ValueError("candidate_post_embeddings must have shape [B, post_embedding_dim]")
        if history_mask.shape != (batch_size, max_history_len):
            raise ValueError("history_mask must have shape [B, H]")
        if history_time_deltas_hours.shape != (batch_size, max_history_len):
            raise ValueError("history_time_deltas_hours must have shape [B, H]")
        if history_author_indices.shape != (batch_size, max_history_len):
            raise ValueError("history_author_indices must have shape [B, H]")
        if candidate_post_author_idx.shape != (batch_size,):
            raise ValueError("candidate_post_author_idx must have shape [B]")

        device = history_embeddings.device
        history_mask = history_mask.to(device=device, dtype=torch.bool)
        history_time_deltas_hours = history_time_deltas_hours.to(device=device)
        candidate_post_embeddings = candidate_post_embeddings.to(device=device)
        history_author_indices = history_author_indices.to(device=device, dtype=torch.long)
        candidate_post_author_idx = candidate_post_author_idx.to(device=device, dtype=torch.long)

        history_post_vectors = self.post_feature_encoder(history_embeddings, history_author_indices)
        candidate_post_vector = self.post_feature_encoder(
            candidate_post_embeddings,
            candidate_post_author_idx,
        ).unsqueeze(1)
        post_sequence = torch.cat([history_post_vectors, candidate_post_vector], dim=1)

        candidate_time_delta = torch.zeros((batch_size, 1), device=device, dtype=history_time_deltas_hours.dtype)
        sequence_time_deltas = torch.cat([history_time_deltas_hours, candidate_time_delta], dim=1)
        time_bucket_ids = bucketize_time_deltas_hours(
            sequence_time_deltas,
            self.time_delta_bucket_boundaries_hours,
        )
        time_embeddings = self.time_delta_embedding(time_bucket_ids)
        transformer_input = torch.cat([post_sequence, time_embeddings], dim=-1)

        candidate_is_not_padding = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)
        src_key_padding_mask = torch.cat([~history_mask, candidate_is_not_padding], dim=1)
        encoded_sequence = self.transformer_encoder(
            transformer_input,
            src_key_padding_mask=src_key_padding_mask,
        )
        return encoded_sequence[:, -1, :]
