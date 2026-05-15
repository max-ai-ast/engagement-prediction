#!/usr/bin/env python3

"""
Stage 4 (Two-Tower): Train two-tower engagement prediction models with flexible user encoders.

This stage trains Two-Tower models, a popular architecture for large-scale recommendation
systems that separates user and item (post) representations into independent "towers",
enabling efficient candidate retrieval and ranking.

═══════════════════════════════════════════════════════════════════════════════
TWO-TOWER ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

Core concept: Project users and posts into a SHARED EMBEDDING SPACE where
engagement is predicted by computing similarity (dot product) between representations.

    User Tower:  history_sequence -> UserEncoder -> user_vector [shared_dim]
                (or "summarized": user_vector is provided by the dataset)
    Post Tower:  post_embedding -> (optional) PostTower -> post_vector [shared_dim]
                (or identity when use_post_encoder=False)
    Prediction:  sigmoid(user_vector · post_vector) -> engagement_probability

Benefits:
    ✓ Decoupled representations: User and post towers can be independently cached
    ✓ Efficient retrieval: Pre-compute all post_vectors, then find top-K by
      dot product similarity (can use approximate nearest neighbor search)
    ✓ Scalable: Avoids expensive cross-feature interactions until final dot product

═══════════════════════════════════════════════════════════════════════════════
USER ENCODER OPTIONS
═══════════════════════════════════════════════════════════════════════════════

This stage supports THREE user-history encoders, selected via `--user-encoder`:

0. **"summarized"** - Hand-crafted summarization (no trainable user encoder)
   ───────────────────────────────────────────────────────────────────────────
   Uses `SummarizedEngagementDataset` to pre-compute a fixed-size user summary
   vector (e.g. mean / EMA / linear recency). In this mode the "user tower" is
   effectively the identity function at training time because the dataset has
   already produced the user vector.

1. **"full_transformer"** - TransformerDualPoolingEncoder (Full Transformer Self-Attention)
   ───────────────────────────────────────────────────────────────────────────
   Uses transformer encoder with multi-head self-attention to capture complex
   inter-post relationships in user history. Best modeling capacity but highest
   computational cost.
   
   Architecture: Input projection -> Positional encoding -> Transformer encoder
                 layers -> Dual pooling (attention + mean) -> Output projection


2. **"cross_attention"** - CrossAttentionPoolingEncoder (Single-Query Cross-Attention)
   ───────────────────────────────────────────────────────────────────────────
   Skips expensive self-attention layers, using only a single learned-query
   cross-attention for aggregation. Significantly faster with fewer parameters.
   
   Architecture: Input projection -> Positional encoding -> Cross-attention
                 pooling (single query) + Mean pooling -> Output projection

═══════════════════════════════════════════════════════════════════════════════
TRAINING DETAILS
═══════════════════════════════════════════════════════════════════════════════

Loss:       Binary cross-entropy on sigmoid(dot_product)
Sampling:   Balanced positive/negative pairs (1:1 ratio)
Optimizer:  AdamW with weight decay for regularization
Scheduling: ReduceLROnPlateau based on validation AUC
Regularization: Gradient clipping, dropout, early stopping

The model is trained to maximize dot product for engaged pairs and minimize for
non-engaged pairs, learning a metric space where similar preferences cluster.

═══════════════════════════════════════════════════════════════════════════════

Inputs (from prior pipeline stages):
    - embeddings_*.npy memmap from 01_get_data
    - target_posts_*.parquet from 02_target_posts
    - history_posts_*.parquet from 03_user_history

Outputs under <run_dir>/04_train/<timestamp>/:
    - checkpoints/two_tower_best.pth (best-by-validation checkpoint during training)
    - checkpoints/two_tower_<timestamp>.pth (final model checkpoint)
    - plots/training_history_*.png (loss and AUC curves)
    - plots/{train,val,holdout}_performance_*.png (precision-recall, ROC curves)
    - logs/ (training logs)
    - training_config.json (hyperparameters and configuration)
    - stage_info.txt (pipeline metadata)
    - predictions/train.parquet (per-row predictions on the training set)
    - predictions/val.parquet (per-row predictions on the validation set)
    - predictions/holdout_unseen_users.parquet (predictions for user-split holdout)
    - predictions/holdout_seen_users.parquet (predictions for temporal holdout, if configured)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from contextlib import nullcontext
from tqdm import tqdm

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from torch.utils.data import DataLoader, Dataset

from utils.pipeline.core import Context
from utils.helpers import (
    get_stage_logger,
    log_operation_start,
    log_prior_stage_inputs,
    get_device,
    plot_model_performance,
    plot_training_history,
    clear_cuda_memory,
    set_random_seeds,
)
from utils.dataloaders import (
    AUTHOR_PAD_IDX,
    AUTHOR_UNK_IDX,
    build_author_table_lookup,
    load_training_data,
    SequenceEngagementDataset,
    SummarizedEngagementDataset,
    SummarizedUserTower,
    TransformerDualPoolingEncoder,
    CrossAttentionPoolingEncoder,
    create_data_loaders,
    get_summarizer,
)

STAGE_LOG_NAME = "STAGE_04_TRAIN_TWO_TOWER"


# =============================================================================
# Post Tower
# =============================================================================

class PostTower(nn.Module):
    """Post tower: Projects post embeddings to shared embedding space.
    
    This is the "item tower" of the two-tower architecture. It maps raw post
    embeddings into the same latent space as user representations, enabling
    dot-product similarity scoring.
    
    Architecture:
        Input: Post embedding [input_dim]
        Hidden: Linear -> LayerNorm -> GELU -> Dropout
        Output: Linear -> Shared space representation [output_dim]
    
    Design choices:
        - LayerNorm (not BatchNorm): Post embeddings don't have batch-level
          statistics during inference (we process posts independently)
        - GELU activation: Smooth gradients for better optimization
        - Single hidden layer: Posts are already semantic embeddings from a
          pre-trained model, so minimal transformation is often sufficient
    
    Args:
        input_dim: Dimensionality of input post embeddings (e.g., 384 for all-MiniLM)
        hidden_dim: Internal hidden layer size
        output_dim: Shared space dimensionality
        dropout_rate: Dropout probability for regularization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, post_embeddings: torch.Tensor) -> torch.Tensor:
        """Project post embeddings to shared space.
        
        Args:
            post_embeddings: Batch of post embeddings [batch, input_dim]
        
        Returns:
            Shared space representations [batch, output_dim]
        """
        return self.network(post_embeddings)


class L2NormalizedUserTower(nn.Module):
    """Wrap a user tower and optionally emit unit-length embeddings."""

    def __init__(self, tower: nn.Module, enabled: bool, eps: float = 1e-12):
        super().__init__()
        self.tower = tower
        self.enabled = bool(enabled)
        self.eps = float(eps)

    def forward(self, history_embeddings: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.tower(history_embeddings, history_mask)
        if not self.enabled:
            return embeddings
        return F.normalize(embeddings, p=2.0, dim=-1, eps=self.eps)


class L2NormalizedPostTower(nn.Module):
    """Wrap a post tower and optionally emit unit-length embeddings."""

    def __init__(self, tower: nn.Module, enabled: bool, eps: float = 1e-12):
        super().__init__()
        self.tower = tower
        self.enabled = bool(enabled)
        self.eps = float(eps)

    def forward(self, post_embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = self.tower(post_embeddings)
        if not self.enabled:
            return embeddings
        return F.normalize(embeddings, p=2.0, dim=-1, eps=self.eps)


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


class AuthorAwareUserTower(nn.Module):
    """User tower that fuses history post embeddings with author embeddings."""

    def __init__(
        self,
        post_author_feature_encoder: PostAuthorFeatureEncoder,
        user_tower: nn.Module,
    ):
        super().__init__()
        self.post_author_feature_encoder = post_author_feature_encoder
        self.user_tower = user_tower

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_author_indices: torch.Tensor,
    ) -> torch.Tensor:
        fused_history_embeddings = self.post_author_feature_encoder(
            history_embeddings,
            history_author_indices,
        )
        fused_history_embeddings = fused_history_embeddings.masked_fill(
            ~history_mask.unsqueeze(-1),
            0.0,
        )
        return self.user_tower(fused_history_embeddings, history_mask)


class AuthorAwarePostTower(nn.Module):
    """Post tower that fuses target post embeddings with author embeddings."""

    def __init__(
        self,
        post_author_feature_encoder: PostAuthorFeatureEncoder,
        post_tower: nn.Module,
    ):
        super().__init__()
        self.post_author_feature_encoder = post_author_feature_encoder
        self.post_tower = post_tower

    def forward(
        self,
        post_embeddings: torch.Tensor,
        target_author_indices: torch.Tensor,
    ) -> torch.Tensor:
        fused_post_embeddings = self.post_author_feature_encoder(
            post_embeddings,
            target_author_indices,
        )
        return self.post_tower(fused_post_embeddings)


def build_author_serving_mapping(
    author_idx_mapping_df: pl.DataFrame,
    author_idx_to_table_row: np.ndarray,
) -> pl.DataFrame:
    """Build the supported author DID -> embedding-table row artifact for serving."""
    required_cols = {"author_did", "author_idx", "author_train_count"}
    missing_cols = required_cols.difference(author_idx_mapping_df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"author_idx_mapping_df is missing required columns: {missing}")

    mapping_df = (
        author_idx_mapping_df
        .select(["author_did", "author_idx", "author_train_count"])
        .unique()
        .sort("author_idx")
    )
    author_table_rows: List[int] = []
    for raw_author_idx in mapping_df["author_idx"].to_list():
        if raw_author_idx is None:
            author_table_rows.append(AUTHOR_UNK_IDX)
            continue
        raw_author_idx_int = int(raw_author_idx)
        if 0 <= raw_author_idx_int < len(author_idx_to_table_row):
            author_table_rows.append(int(author_idx_to_table_row[raw_author_idx_int]))
        else:
            author_table_rows.append(AUTHOR_UNK_IDX)

    return (
        mapping_df
        .with_columns(
            pl.Series("author_table_row", author_table_rows, dtype=pl.UInt32),
        )
        .filter(pl.col("author_table_row") > AUTHOR_UNK_IDX)
        .select(["author_did", "author_idx", "author_train_count", "author_table_row"])
    )


# =============================================================================
# Two-Tower Engagement Model
# =============================================================================
class TwoTowerModel(nn.Module):
    """Two-tower engagement prediction model with pluggable user encoders.
    
    Implements the two-tower architecture where user and post representations
    are independently computed and combined via dot product similarity. This
    architecture is particularly well-suited for large-scale retrieval and
    ranking systems.
    
    Architecture:
        User Tower:
            - "full_transformer": TransformerDualPoolingEncoder(history_sequence, mask) -> user_vector [shared_dim]
            - "cross_attention": CrossAttentionPoolingEncoder(history_sequence, mask) -> user_vector [shared_dim]
            - "summarized": user_vector is provided by SummarizedEngagementDataset
              (the model treats the dataset-provided user summary as the user embedding)
        
        Post Tower:
            - use_post_encoder=True:  PostTower(post_embedding) -> post_vector [shared_dim]
            - use_post_encoder=False: post_vector is the raw post embedding (identity)
        
        Scoring: similarity(user_vector, post_vector) / temperature -> raw logit
                 sigmoid(raw_logit) -> engagement_probability
    
    Key characteristics:
        - Shared embedding space: Both towers output the same dimensionality for dot product
        - Independent computation: Towers never exchange information (until final dot product)
        - Modular encoders: User tower can be "summarized", "full_transformer", or "cross_attention"
    
    Deployment pattern:
        1. Pre-compute post_vectors for all candidate posts
        2. At inference, encode user history once -> user_vector
        3. Find top-K posts by dot product (can use ANN for scale)
        4. Return ranked candidates
    
    Args:
        post_embedding_dim: Dimensionality of input post embeddings
        shared_dim: Output dimension for both towers
        user_hidden_dim: User tower internal hidden size
        post_hidden_dim: Post tower internal hidden size
        num_attention_heads: Attention heads for TransformerDualPoolingEncoder
        num_attention_layers: Transformer layers for TransformerDualPoolingEncoder
        max_history_len: Maximum history sequence length
        dropout_rate: Dropout probability
        user_encoder_type: User tower architecture - "summarized", "full_transformer", or "cross_attention"
        use_post_encoder: If True, learn a post projection (PostTower). If False, use raw post embeddings.
    """

    def __init__(
        self,
        post_embedding_dim: int,
        shared_dim: int,
        user_hidden_dim: int,
        post_hidden_dim: int,
        num_attention_heads: int,
        num_attention_layers: int,
        max_history_len: int,
        dropout_rate: float,
        l2_normalize_embeddings: bool,
        similarity_temperature: float,
        user_encoder_type: str,
        use_post_encoder: bool,
        use_author_embedding_table: bool = False,
        author_table_num_rows: Optional[int] = None,
        author_embedding_dim: Optional[int] = None,
        author_unknown_dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.post_embedding_dim = post_embedding_dim
        if similarity_temperature <= 0.0:
            raise ValueError("similarity_temperature must be > 0")
        self.similarity_temperature = float(similarity_temperature)
        self.user_encoder_type = user_encoder_type
        self.use_post_encoder = use_post_encoder
        self.l2_normalize_embeddings = bool(l2_normalize_embeddings)
        self.use_author_embedding_table = bool(use_author_embedding_table)
        post_author_feature_encoder: Optional[PostAuthorFeatureEncoder] = None

        if self.use_author_embedding_table:
            if user_encoder_type == "summarized":
                raise ValueError("use_author_embedding_table is not supported with user_encoder_type='summarized'")
            if author_table_num_rows is None or author_table_num_rows < 2:
                raise ValueError("author_table_num_rows must be provided and >= 2 when use_author_embedding_table is True")
            if author_embedding_dim is None or author_embedding_dim <= 0:
                raise ValueError("author_embedding_dim must be provided and positive when use_author_embedding_table is True")
            post_author_feature_encoder = PostAuthorFeatureEncoder(
                post_embedding_dim=post_embedding_dim,
                author_table_num_rows=author_table_num_rows,
                author_embedding_dim=author_embedding_dim,
                author_unknown_dropout_rate=author_unknown_dropout_rate,
            )

        # Instantiate user tower based on encoder type
        if user_encoder_type == "cross_attention":
            raw_user_tower = CrossAttentionPoolingEncoder(
                input_dim=post_embedding_dim,
                hidden_dim=user_hidden_dim,
                output_dim=shared_dim,
                max_seq_len=max_history_len,
                dropout_rate=dropout_rate,
            )
        elif user_encoder_type == "full_transformer":
            raw_user_tower = TransformerDualPoolingEncoder(
                input_dim=post_embedding_dim,
                hidden_dim=user_hidden_dim,
                output_dim=shared_dim,
                num_attention_heads=num_attention_heads,
                num_attention_layers=num_attention_layers,
                max_seq_len=max_history_len,
                dropout_rate=dropout_rate,
            )
        elif user_encoder_type == "summarized":
            # In "summarized" mode, the dataset provides the user vector directly
            # (e.g., mean/EMA/linear-recency summary). The model treats the
            # history input as an already-encoded user embedding (placed at
            # position 0 in a padded sequence for a consistent forward() signature).
            raw_user_tower = SummarizedUserTower(embed_dim=post_embedding_dim)

            # If we still learn a post projection (use_post_encoder=True), its output
            # dimension must match the dataset-provided user embedding dimension.
            if use_post_encoder and (shared_dim != post_embedding_dim):
                raise ValueError(f"--shared-dim ({shared_dim}) and post embedding dim ({post_embedding_dim}) do not match! They must match for two tower with user summarization.")
        else:
            raise ValueError(
                f"Unknown user_encoder_type '{user_encoder_type}'. "
                "Choose 'summarized', 'full_transformer' or 'cross_attention'."
            )

        # If we don't project posts, then `post_embeddings` stays in the raw embedding
        # space and must match the user embedding dim for dot product scoring.
        #
        # In summarized mode, the user embedding is also in the raw embedding space.
        if (not use_post_encoder) and (user_encoder_type != "summarized") and (shared_dim != post_embedding_dim):
            raise ValueError(
                f"use_post_encoder=False requires --shared-dim ({shared_dim}) to equal post embedding dim ({post_embedding_dim}) "
                f"(or set use_post_encoder=True to project posts to shared_dim)."
            )

        if use_post_encoder:
            # Post tower is the same regardless of user encoder type
            raw_post_tower = PostTower(
                input_dim=post_embedding_dim,
                hidden_dim=post_hidden_dim,
                output_dim=shared_dim,
                dropout_rate=dropout_rate,
            )
        else:
            raw_post_tower = nn.Identity()

        base_user_tower = L2NormalizedUserTower(raw_user_tower, enabled=self.l2_normalize_embeddings)
        base_post_tower = L2NormalizedPostTower(raw_post_tower, enabled=self.l2_normalize_embeddings)

        if self.use_author_embedding_table:
            if post_author_feature_encoder is None:
                raise RuntimeError("post_author_feature_encoder must be initialized when author embeddings are enabled")
            self.user_tower = AuthorAwareUserTower(post_author_feature_encoder, base_user_tower)
            self.post_tower = AuthorAwarePostTower(post_author_feature_encoder, base_post_tower)
        else:
            self.user_tower = base_user_tower
            self.post_tower = base_post_tower

    @property
    def post_author_feature_encoder(self) -> Optional[PostAuthorFeatureEncoder]:
        """Return the shared author fusion module when author-aware towers are enabled."""
        if isinstance(self.user_tower, AuthorAwareUserTower):
            return self.user_tower.post_author_feature_encoder
        return None

    def encode_user(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_author_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode user engagement history into shared space representation.
        
        Args:
            history_embeddings: Padded history sequences [batch, seq_len, input_dim]
            history_mask: Boolean mask [batch, seq_len], True = valid position
        
        Returns:
            User vectors in shared space [batch, shared_dim].
            When `l2_normalize_embeddings=True`, these are unit-length.
        """
        if self.use_author_embedding_table:
            if history_author_indices is None:
                raise RuntimeError("history_author_indices are required when use_author_embedding_table is True")
            return self.user_tower(history_embeddings, history_mask, history_author_indices)
        return self.user_tower(history_embeddings, history_mask)

    def encode_post(
        self,
        post_embeddings: torch.Tensor,
        target_author_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode post embeddings into shared space representation.
        
        Args:
            post_embeddings: Raw post embeddings [batch, input_dim]
            target_author_indices: Author embedding table rows [batch], required
                when author embeddings are enabled.
        
        Returns:
            Post vectors for dot product scoring.
                - use_post_encoder=True: [batch, shared_dim]
                - otherwise: [batch, post_embedding_dim] (identity)
            When `l2_normalize_embeddings=True`, outputs are L2-normalized.
        """
        if self.use_author_embedding_table:
            if target_author_indices is None:
                raise RuntimeError("target_author_indices are required when use_author_embedding_table is True")
            return self.post_tower(post_embeddings, target_author_indices)
        return self.post_tower(post_embeddings)

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        post_embeddings: torch.Tensor,
        history_author_indices: Optional[torch.Tensor] = None,
        target_author_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute engagement scores via dot product in shared space.
        
        This is the core two-tower computation: encode user and post independently,
        then measure similarity via dot product. Higher dot product = higher
        predicted engagement probability.
        
        Args:
            history_embeddings: User history input.
                - all modes: padded history sequences [batch, seq_len, input_dim]
                  In "summarized" mode, the user summary is expected to be placed
                  at position 0 (and optionally padded to seq_len > 1).
            history_mask: History validity mask [batch, seq_len] (optional in summarized mode)
            post_embeddings: Target post embeddings [batch, input_dim]
            history_author_indices: Author table rows aligned with history items,
                required when author embeddings are enabled.
            target_author_indices: Author table rows for target posts, required
                when author embeddings are enabled.
        
        Returns:
            Raw engagement scores [batch] (logits before sigmoid)
        """
        user_emb = self.encode_user(history_embeddings, history_mask, history_author_indices)
        post_emb = self.encode_post(post_embeddings, target_author_indices)
        
        similarity_score = (user_emb * post_emb).sum(dim=-1)
        return similarity_score / self.similarity_temperature

    def compute_loss_and_preds(
        self,
        batch: Dict[str, Any],
        device: str,
        embed_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss and predictions for a batch.
        
        This method provides a unified interface for training, validation, and
        inference loops. It computes raw similarity scores (optionally cosine
        similarities when tower-level L2 normalization is enabled) and
        calculates binary cross-entropy loss directly from the logits for
        numerical stability.
        
        Args:
            batch: Batch dictionary. Expected keys depend on `user_encoder_type`:
                - "summarized": {"features", "label"} where features is
                  [B, 2*embed_dim] concatenated [user_summary || post_embedding]
                - otherwise: {"history_embeddings", "history_mask", "target_post_embedding", "label"}
            device: Device string (e.g. "cpu" or "cuda")
            embed_dim: Post embedding dimensionality D (used only to split "features" in summarized mode)
        
        Returns:
            Tuple of (loss, scores):
                - loss: Scalar BCE-with-logits loss tensor
                - scores: Raw similarity scores [batch] (before sigmoid)
        
        Note:
            Returns raw scores (not probabilities) for flexibility in evaluation.
            Apply sigmoid(scores) to get probabilities.
        """
        history_author_indices = None
        target_author_indices = None
        # unpack inputs
        if self.user_encoder_type == "summarized":
            features = batch["features"].to(device, non_blocking=True) # [B, embed_dim*2]
            user_summary = features[:, :embed_dim] # [B, embed_dim]
            history_embeddings = user_summary.unsqueeze(1)  # [B, 1, embed_dim] (summary token at position 0)
            post_embeddings = features[:, embed_dim:] # [B, embed_dim]
            # Cold-start handling: empty histories have an all-zero summary sentinel.
            # Use the mask to indicate whether the summary came from at least one
            # history item so the summarized user tower can inject a learnable
            # cold-start embedding.
            has_history = user_summary.abs().sum(dim=1) > 0 # [B]
            history_mask = has_history.unsqueeze(1).to(device=device, dtype=torch.bool) # [B, 1]
            assert history_embeddings.shape[-1] == post_embeddings.shape[-1]
        else:
            history_embeddings = batch["history_embeddings"].to(device, non_blocking=True) # [B, seq_len, embed_dim]
            history_mask = batch["history_mask"].to(device, non_blocking=True) # [B, seq_len]
            post_embeddings = batch["target_post_embedding"].to(device, non_blocking=True) # [B, embed_dim]
            if self.use_author_embedding_table:
                history_author_indices = batch["history_author_indices"].to(device, dtype=torch.long, non_blocking=True)
                target_author_indices = batch["target_post_author_idx"].to(device, dtype=torch.long, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        scores = self.forward(
            history_embeddings,
            history_mask,
            post_embeddings,
            history_author_indices,
            target_author_indices,
        )
        loss = F.binary_cross_entropy_with_logits(scores, labels.float())
        return loss, scores


def calc_ndcg_at_k(
    probs_df: pl.DataFrame,
    metrics_top_ks: list[int],
) -> dict[str, float]:
    """
    Calculate the normalized discounted cumulative gain for each user in a dataframe,
    and then take an average across users.
    """
    
    ranked_df = (
        probs_df
        .sort(["target_did", "prob"], descending=[False, True])
        .with_columns(
            pl.col("prob").cum_count().over("target_did").alias("pred_rank")
        )
        .with_columns(
            (pl.col("label") / (pl.col("pred_rank") + 1).log(2)).alias("dcg_gain")
        )
    )

    ideal_ranked_df = (
        probs_df
        .sort(["target_did", "label"], descending=[False, True])
        .with_columns(
            pl.col("label").cum_count().over("target_did").alias("ideal_rank")
        )
        .with_columns(
            (pl.col("label") / (pl.col("ideal_rank") + 1).log(2)).alias("idcg_gain")
        )
    )

    per_user_metrics = ranked_df.select("target_did").unique()

    for k in metrics_top_ks:
        dcg_at_k = (
            ranked_df
            .filter(pl.col("pred_rank") <= k)
            .group_by("target_did")
            .agg(pl.col("dcg_gain").sum().alias(f"dcg@{k}"))
        )

        idcg_at_k = (
            ideal_ranked_df
            .filter(pl.col("ideal_rank") <= k)
            .group_by("target_did")
            .agg(pl.col("idcg_gain").sum().alias(f"idcg@{k}"))
        )

        per_user_metrics = (
            per_user_metrics
            .join(dcg_at_k, on="target_did", how="left")
            .join(idcg_at_k, on="target_did", how="left")
            .with_columns(
                pl.col(f"dcg@{k}").fill_null(0.0),
                pl.col(f"idcg@{k}").fill_null(0.0),
            )
            .with_columns(
                pl.when(pl.col(f"idcg@{k}") > 0)
                .then(pl.col(f"dcg@{k}") / pl.col(f"idcg@{k}"))
                .otherwise(0.0)
                .alias(f"ndcg@{k}")
            )
        )

    ndcg_dict = {
        f"dcg@{k}": float(per_user_metrics.select(pl.col(f"dcg@{k}").mean()).item())
        for k in metrics_top_ks
    }
    ndcg_dict.update({
        f"ndcg@{k}": float(per_user_metrics.select(pl.col(f"ndcg@{k}").mean()).item())
        for k in metrics_top_ks
    })
    return ndcg_dict


def _run_one_epoch(
    train: bool,
    split_name: str,
    model: TwoTowerModel,
    device: str,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    disable_progress: bool,
    embed_dim: int,
    gradient_clip_max_norm: float,
    metrics_top_ks: list[int],
):
    if train:
        model.train()
    else:
        model.eval()

    loss_sum = torch.zeros((), device=device)
    batches = 0
    scores_chunks: List[torch.Tensor] = []
    labels_chunks: List[torch.Tensor] = []
    users_all: List[str] = []

    with nullcontext() if train else torch.inference_mode():
        for batch in tqdm(dataloader, desc=split_name, leave=False, disable=disable_progress):
            labels = batch["label"]

            if train:
                optimizer.zero_grad()

            loss, scores = model.compute_loss_and_preds(batch, device, embed_dim)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_max_norm)
                optimizer.step()

            loss_sum += loss.detach()
            batches += 1
            scores_chunks.append(scores.detach())
            labels_chunks.append(labels.detach())
            users_all.extend(batch["user_id"])

    loss = (loss_sum / max(batches, 1)).item()

    probs = (
        torch.sigmoid(torch.cat(scores_chunks)).float().cpu().numpy()
        if scores_chunks
        else np.array([])
    )
    labels_np = (
        torch.cat(labels_chunks).float().cpu().numpy()
        if labels_chunks
        else np.array([])
    )

    auc = roc_auc_score(labels_np, probs) if np.unique(labels_np).size > 1 else 0.5

    # ndcg@k
    probs_df = pl.DataFrame({
        "target_did": users_all,
        "prob": probs,
        "label": labels_np,
    })
    ndcg_dict = calc_ndcg_at_k(probs_df, metrics_top_ks)

    return loss, probs, labels_np, auc, ndcg_dict


# =============================================================================
# Training Loop
# =============================================================================

def train_two_tower_model(
    model: TwoTowerModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_unseen_loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    early_stopping_min_delta: float,
    checkpoints_dir: Optional[Path],
    disable_progress: bool,
    lr_scheduler_factor: float,
    lr_scheduler_patience: int,
    gradient_clip_max_norm: float,
    embed_dim: int,
    metrics_top_ks: list[int],
    experiment_tracker: Optional[Any] = None,
) -> Dict[str, Any]:

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_scheduler_factor, patience=lr_scheduler_patience
    )

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    best_val_auc = 0.0
    best_reset_val_auc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    for epoch in tqdm(range(epochs), desc="Training epochs", disable=disable_progress):
        train_loss, train_probs, train_labels_np, train_auc, train_ndcg_dict = _run_one_epoch(
            train=True,
            split_name="Train",
            model=model,
            device=device,
            dataloader=train_loader,
            optimizer=optimizer,
            disable_progress=disable_progress,
            embed_dim=embed_dim,
            gradient_clip_max_norm=gradient_clip_max_norm,
            metrics_top_ks=metrics_top_ks,
        )

        val_loss, _, _, val_auc, val_ndcg_dict = _run_one_epoch(
            train=False,
            split_name="Validation",
            model=model,
            device=device,
            dataloader=val_loader,
            optimizer=optimizer,
            disable_progress=disable_progress,
            embed_dim=embed_dim,
            gradient_clip_max_norm=gradient_clip_max_norm,
            metrics_top_ks=metrics_top_ks,
        )

        val_unseen_loss, val_unseen_probs, val_unseen_labels_np, val_unseen_auc, val_unseen_ndcg_dict = _run_one_epoch(
            train=False,
            split_name="Validation Unseen Users",
            model=model,
            device=device,
            dataloader=val_unseen_loader,
            optimizer=optimizer,
            disable_progress=disable_progress,
            embed_dim=embed_dim,
            gradient_clip_max_norm=gradient_clip_max_norm,
            metrics_top_ks=metrics_top_ks,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(float(train_auc))
        history["val_auc"].append(float(val_auc))

        if experiment_tracker is not None:
            iteration = epoch + 1
            experiment_tracker.log_scalar(
                title="Training Loss History",
                series="Train Loss",
                value=float(train_loss),
                iteration=iteration,
            )
            experiment_tracker.log_scalar(
                title="Training Loss History",
                series="Validation Loss",
                value=float(val_loss),
                iteration=iteration,
            )
            experiment_tracker.log_scalar(
                title="Training Loss History",
                series="Validation Unseen Users Loss",
                value=float(val_unseen_loss),
                iteration=iteration,
            )
            experiment_tracker.log_scalar(
                title="Training AUC History",
                series="Train AUC",
                value=float(train_auc),
                iteration=iteration,
            )
            experiment_tracker.log_scalar(
                title="Training AUC History",
                series="Validation AUC",
                value=float(val_auc),
                iteration=iteration,
            )
            experiment_tracker.log_scalar(
                title="Training AUC History",
                series="Validation Unseen Users AUC",
                value=float(val_unseen_auc),
                iteration=iteration,
            )
            for k in metrics_top_ks:
                experiment_tracker.log_scalar(
                    title=f"NDCG@{k}",
                    series=f"Train NDCG@{k}",
                    value=float(train_ndcg_dict[f"ndcg@{k}"]),
                    iteration=iteration,
                )
                experiment_tracker.log_scalar(
                    title=f"NDCG@{k}",
                    series=f"Validation NDCG@{k}",
                    value=float(val_ndcg_dict[f"ndcg@{k}"]),
                    iteration=iteration,
                )
                experiment_tracker.log_scalar(
                    title=f"NDCG@{k}",
                    series=f"Validation Unseen Users NDCG@{k}",
                    value=float(val_unseen_ndcg_dict[f"ndcg@{k}"]),
                    iteration=iteration,
                )

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if checkpoints_dir is not None:
                torch.save(
                    {"epoch": epoch, "model_state_dict": best_state_dict, "val_loss": val_loss, "val_auc": val_auc, "history": history},
                    checkpoints_dir / "two_tower_best.pth",
                )

        significant_improvement = (
            val_auc > best_reset_val_auc
            and (val_auc - best_reset_val_auc) >= early_stopping_min_delta
        )
        if significant_improvement:
            best_reset_val_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_auc": best_val_auc,
    }


# =============================================================================
# Evaluation
# =============================================================================

def _evaluate_two_tower_model(
    model: TwoTowerModel,
    data_loader: DataLoader,
    device: str,
    embed_dim: int,
) -> Dict[str, Any]:
    """Evaluate two-tower model and return metrics + predictions."""
    model = model.to(device)
    model.eval()

    labels_chunks: List[torch.Tensor] = []
    probs_chunks: List[torch.Tensor] = []
    all_user_ids: List[str] = []
    all_post_ids: List[str] = []

    with torch.inference_mode():
        for batch in data_loader:
            labels = batch["label"]

            _, scores = model.compute_loss_and_preds(batch, device, embed_dim)
            probs = torch.sigmoid(scores)

            probs_chunks.append(probs.detach())
            labels_chunks.append(labels.detach())
            all_user_ids.extend(batch["user_id"])
            all_post_ids.extend(batch["post_id"])

    probs_all = torch.cat(probs_chunks).float().cpu() if probs_chunks else torch.empty(0)
    labels_all = torch.cat(labels_chunks).float().cpu() if labels_chunks else torch.empty(0)

    y_true = labels_all.numpy()
    y_pred = probs_all.numpy()

    metrics: Dict[str, Any] = {
        "total_samples": len(y_true),
        "positive_samples": int(y_true.sum()),
        "negative_samples": int(len(y_true) - y_true.sum()),
    }

    if np.unique(y_true).size > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred))
        metrics["average_precision"] = float(average_precision_score(y_true, y_pred))

    metrics["accuracy_at_0.5"] = float(accuracy_score(y_true, (y_pred > 0.5).astype(int)))

    return {
        "metrics": metrics,
        "predictions": {
            "user_id": all_user_ids,
            "post_id": all_post_ids,
            "y_true": y_true,
            "y_pred": y_pred,
        },
    }


# =============================================================================
# Plotting
# =============================================================================

# =============================================================================
# Pipeline entry point
# =============================================================================

def run(context: Context, args) -> Dict[str, Any]:
    """Pipeline entry point for two-tower training."""
    device = get_device(args.device)
    timestamp = context.run_timestamp

    # --- output dirs ---
    run_tag = args.run_tag or ""
    out_dir = context.new_stage_dir("04_train", tag=run_tag)
    checkpoints_dir = out_dir / "checkpoints"
    plots_dir = out_dir / "plots"
    logs_dir = out_dir / "logs"
    for d in (checkpoints_dir, plots_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger = get_stage_logger(STAGE_LOG_NAME, log_file=out_dir / "stage.log")
    log_operation_start("Stage 4 Two-Tower training", STAGE_LOG_NAME, logger)
    t0 = time.time()

    # --- seeds & cuda ---
    clear_cuda_memory()
    random_seed = int(args.random_seed)
    set_random_seeds(random_seed)

    # --- load data from prior stages ---
    log_operation_start("Load training data from prior stages", STAGE_LOG_NAME, logger)
    embeddings_mmap, target_posts_df, history_df, author_idx_mapping_df, embed_dim = load_training_data(
        context, logger=logger,
    )
    log_prior_stage_inputs(context, logger)

    # --- hyperparams (extract all args once, use locals everywhere below) ---
    max_history_len = int(args.max_history_len)
    shared_dim = int(args.shared_dim)
    user_hidden_dim = int(args.user_hidden_dim)
    post_hidden_dim = int(args.post_hidden_dim)
    num_attention_heads = int(args.num_attention_heads)
    num_attention_layers = int(args.num_attention_layers)
    dropout_rate = float(args.dropout_rate_two_tower)
    l2_normalize_embeddings = bool(args.l2_normalize_embeddings)
    similarity_temperature = float(args.similarity_temperature)
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.weight_decay_two_tower)
    epochs = int(args.epochs)
    patience = int(args.patience)
    early_stopping_min_delta = float(args.early_stopping_min_delta)
    disable_progress = bool(args.disable_progress)
    user_encoder_type = args.user_encoder
    use_post_encoder = args.use_post_encoder
    generate_plots = not bool(args.no_plots)
    save_model = not bool(args.no_save_model)
    lr_scheduler_factor = float(args.lr_scheduler_factor)
    lr_scheduler_patience = int(args.lr_scheduler_patience)
    gradient_clip_max_norm = float(args.gradient_clip_max_norm)
    eval_holdout_type = str(args.eval_holdout_type)
    use_author_embedding_table = bool(args.use_author_embedding_table)
    author_embedding_dim = int(args.author_embedding_dim)
    min_author_support = int(args.min_author_support)
    author_unknown_dropout_rate = float(args.author_unknown_dropout_rate)
    metrics_top_ks = list(args.metrics_top_ks)

    if use_author_embedding_table and user_encoder_type == "summarized":
        raise ValueError("use_author_embedding_table is not supported with user_encoder_type='summarized'")
    if use_author_embedding_table and author_idx_mapping_df is None:
        raise FileNotFoundError(
            "author_idx artifact was not found in 02_target_posts output, but --use-author-embedding-table was enabled."
        )
    author_idx_to_table_row = None
    author_table_num_rows = 0
    author_serving_mapping_df = None
    if use_author_embedding_table:
        if author_idx_mapping_df is None:
            raise FileNotFoundError("author_idx_mapping_df is required when use_author_embedding_table is True")
        author_idx_to_table_row, author_table_num_rows = build_author_table_lookup(
            author_idx_mapping_df=author_idx_mapping_df,
            min_author_support=min_author_support,
        )
        author_serving_mapping_df = build_author_serving_mapping(
            author_idx_mapping_df=author_idx_mapping_df,
            author_idx_to_table_row=author_idx_to_table_row,
        )
        logger.info(
            "Author embedding table enabled: "
            f"min_author_support={min_author_support}, "
            f"author_embedding_dim={author_embedding_dim}, "
            f"author_table_num_rows={author_table_num_rows}, "
            f"serving_mapping_rows={len(author_serving_mapping_df)}"
        )

    # Worker settings
    num_workers = int(args.num_dataloader_workers)
    pin_memory = bool(args.dataloader_pin_memory)
    persistent_workers = bool(args.dataloader_persistent_workers)
    prefetch_factor = int(args.dataloader_prefetch_factor)

    # --- datasets ---
    log_operation_start("Create datasets", STAGE_LOG_NAME, logger)
    if user_encoder_type == "summarized":
        # get summarizer
        summarizer_name = args.user_summarization
        ema_alpha = float(args.ema_alpha)
        summarizer = get_summarizer(summarizer_name, ema_alpha=ema_alpha)
        
        train_dataset = SummarizedEngagementDataset(
            embeddings_mmap, target_posts_df, history_df, split="train", 
            summarizer=summarizer, embed_dim=embed_dim, logger=logger,
        )
        val_dataset = SummarizedEngagementDataset(
            embeddings_mmap, target_posts_df, history_df, split="val", 
            summarizer=summarizer, embed_dim=embed_dim, logger=logger,
        )
        val_unseen_dataset = SummarizedEngagementDataset(
            embeddings_mmap, target_posts_df, history_df, split="val_unseen_users", 
            summarizer=summarizer, embed_dim=embed_dim, logger=logger,
        )
    else:
        train_dataset = SequenceEngagementDataset(
            embeddings_mmap, target_posts_df, history_df, split="train",
            max_history_len=max_history_len, embed_dim=embed_dim,
            use_author_embedding_table=use_author_embedding_table,
            author_idx_to_table_row=author_idx_to_table_row,
            logger=logger,
        )
        val_dataset = SequenceEngagementDataset(
            embeddings_mmap, target_posts_df, history_df, split="val",
            max_history_len=max_history_len, embed_dim=embed_dim,
            use_author_embedding_table=use_author_embedding_table,
            author_idx_to_table_row=author_idx_to_table_row,
            logger=logger,
        )
        val_unseen_dataset = SequenceEngagementDataset(
            embeddings_mmap, target_posts_df, history_df, split="val_unseen_users",
            max_history_len=max_history_len, embed_dim=embed_dim,
            use_author_embedding_table=use_author_embedding_table,
            author_idx_to_table_row=author_idx_to_table_row,
            logger=logger,
        )

    # Create data loaders using centralized helper
    train_loader, val_loader, val_unseen_loader, _ = create_data_loaders(
        train_dataset, val_dataset, val_unseen_dataset, batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    logger.info(f"Post embedding dim: {embed_dim}")
    logger.info(f"Train items: {len(train_dataset)}, Val items: {len(val_dataset)}")

    # --- create model ---
    log_operation_start(f"Create two-tower model (user_encoder={user_encoder_type})", STAGE_LOG_NAME, logger)
    model = TwoTowerModel(
        post_embedding_dim=embed_dim,
        shared_dim=shared_dim,
        user_hidden_dim=user_hidden_dim,
        post_hidden_dim=post_hidden_dim,
        num_attention_heads=num_attention_heads,
        num_attention_layers=num_attention_layers,
        max_history_len=max_history_len,
        dropout_rate=dropout_rate,
        l2_normalize_embeddings=l2_normalize_embeddings,
        similarity_temperature=similarity_temperature,
        user_encoder_type=user_encoder_type,
        use_post_encoder=use_post_encoder,
        use_author_embedding_table=use_author_embedding_table,
        author_table_num_rows=author_table_num_rows if use_author_embedding_table else None,
        author_embedding_dim=author_embedding_dim if use_author_embedding_table else None,
        author_unknown_dropout_rate=author_unknown_dropout_rate,
    )

    if (not use_post_encoder) and (user_encoder_type == "summarized"):
        logger.info(f"Creating a simple dot-product model, no need for training")
        trained_model: TwoTowerModel = model
        training_results = None
    else:
        # --- train ---
        log_operation_start(f"Train two-tower (epochs={epochs}, batch_size={batch_size})", STAGE_LOG_NAME, logger)
        training_results = train_two_tower_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            val_unseen_loader=val_unseen_loader,
            device=device,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=patience,
            early_stopping_min_delta=early_stopping_min_delta,
            checkpoints_dir=checkpoints_dir,
            disable_progress=disable_progress,
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            gradient_clip_max_norm=gradient_clip_max_norm,
            embed_dim=embed_dim,
            metrics_top_ks=metrics_top_ks,
            experiment_tracker=context.tracker,
        )
        trained_model: TwoTowerModel = training_results["model"]
        clear_cuda_memory()

        # --- plots & evaluation ---
        hist = training_results["history"]

        if generate_plots:
            try:
                best_epoch = int(np.argmax(hist.get("val_auc", []))) + 1 if hist.get("val_auc") and len(hist.get("val_auc")) > 0 else None
            except Exception as e:
                logger.warning(f"Could not determine best epoch from training history: {e}")
                best_epoch = None
            plot_training_history(hist, plots_dir / f"training_history_{timestamp}.png", best_epoch=best_epoch)

    # Collect train + val predictions for performance plots & metrics
    train_eval = _evaluate_two_tower_model(trained_model, train_loader, device, embed_dim)
    val_eval = _evaluate_two_tower_model(trained_model, val_loader, device, embed_dim)
    logger.info(f"Train metrics: {train_eval['metrics']}")
    logger.info(f"Validation metrics: {val_eval['metrics']}")

    # calculate the best auc in the case that we didn't train a model (simple dot product version)
    if training_results is not None:
        best_val_auc = training_results["best_val_auc"]
    else:
        best_train_auc = roc_auc_score(train_eval["predictions"]["y_true"], train_eval["predictions"]["y_pred"])
        best_val_auc = roc_auc_score(val_eval["predictions"]["y_true"], val_eval["predictions"]["y_pred"])
        context.tracker.log_scalar(title="Training AUC History", series="Train AUC", value=float(best_train_auc), iteration=0)
        context.tracker.log_scalar(title="Training AUC History", series="Validation AUC", value=float(best_val_auc), iteration=0)

    if generate_plots:
        try:
            plot_model_performance(
                train_eval["predictions"]["y_true"],
                train_eval["predictions"]["y_pred"],
                plots_dir / f"train_performance_{timestamp}.png",
                title_suffix="(Train)",
            )
        except Exception as plot_exc:
            logger.warning(f"Train performance plotting failed: {plot_exc}")
        try:
            plot_model_performance(
                val_eval["predictions"]["y_true"],
                val_eval["predictions"]["y_pred"],
                plots_dir / f"val_performance_{timestamp}.png",
                title_suffix="(Validation)",
            )
        except Exception as plot_exc:
            logger.warning(f"Validation performance plotting failed: {plot_exc}")

    # --- save model ---
    model_path = None
    author_table_mapping_path = (
        checkpoints_dir / "author_table_mapping.parquet"
        if use_author_embedding_table and save_model
        else None
    )
    config = {
        "model_type": "two_tower",
        "user_encoder_type": user_encoder_type,
        "use_post_encoder": use_post_encoder,
        "post_embedding_dim": embed_dim,
        "shared_dim": shared_dim,
        "user_hidden_dim": user_hidden_dim,
        "post_hidden_dim": post_hidden_dim,
        "num_attention_heads": num_attention_heads,
        "num_attention_layers": num_attention_layers,
        "max_history_len": max_history_len,
        "dropout_rate": dropout_rate,
        "l2_normalize_embeddings": l2_normalize_embeddings,
        "similarity_temperature": similarity_temperature,
        "use_author_embedding_table": use_author_embedding_table,
        "author_embedding_dim": author_embedding_dim if use_author_embedding_table else None,
        "min_author_support": min_author_support if use_author_embedding_table else None,
        "author_unknown_dropout_rate": author_unknown_dropout_rate if use_author_embedding_table else None,
        "author_table_num_rows": author_table_num_rows if use_author_embedding_table else None,
        "author_pad_idx": AUTHOR_PAD_IDX,
        "author_unk_idx": AUTHOR_UNK_IDX,
        "author_table_mapping_path": str(author_table_mapping_path) if author_table_mapping_path else None,
    }
    if save_model:
        model_path = checkpoints_dir / f"two_tower_{timestamp}.pth"
        torch.save(
            {
                "model_state_dict": trained_model.state_dict(),
                "config": config,
                "training_history": training_results["history"] if training_results is not None else None,
                "best_val_auc": best_val_auc,
                "best_val_loss": training_results["best_val_loss"] if training_results is not None else None,
            },
            model_path,
        )
        logger.info(f"Model saved to: {model_path}")

        if use_author_embedding_table:
            if author_serving_mapping_df is None or author_table_mapping_path is None:
                raise RuntimeError("author_serving_mapping_df is required when author embeddings are enabled")
            author_serving_mapping_df.write_parquet(author_table_mapping_path, compression="zstd")
            author_mapping_id = context.tracker.log_artifact(
                name="author_table_mapping",
                path=author_table_mapping_path,
            )
            logger.info(f"Author table mapping artifact id: {author_mapping_id}")

        # Save TorchScript file, which is the format needed for ClearML serving
        # Save the post and user towers separately
        trained_model = trained_model.cpu()
        torchscript_user_name = "engagement_user_tower"
        torchscript_user_path = checkpoints_dir / f"{torchscript_user_name}.pt"
        torch.jit.script(trained_model.user_tower).save(torchscript_user_path)
        user_model_id = context.tracker.log_artifact(name=f"{torchscript_user_name}", path=torchscript_user_path)
        logger.info(f"User tower model id: {user_model_id}")

        torchscript_post_name = "engagement_post_tower"
        torchscript_post_path = checkpoints_dir / f"{torchscript_post_name}.pt"
        torch.jit.script(trained_model.post_tower).save(torchscript_post_path)
        post_model_id = context.tracker.log_artifact(name=f"{torchscript_post_name}", path=torchscript_post_path)
        logger.info(f"Post tower model id: {post_model_id}")

    # --- save predictions ---
    predictions_dir = out_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({
        "did": train_eval["predictions"]["user_id"],
        "post_id": train_eval["predictions"]["post_id"],
        "y_true": train_eval["predictions"]["y_true"],
        "y_pred_proba": train_eval["predictions"]["y_pred"],
    }).write_parquet(predictions_dir / "train.parquet")

    pl.DataFrame({
        "did": val_eval["predictions"]["user_id"],
        "post_id": val_eval["predictions"]["post_id"],
        "y_true": val_eval["predictions"]["y_true"],
        "y_pred_proba": val_eval["predictions"]["y_pred"],
    }).write_parquet(predictions_dir / "val.parquet")

    # --- holdout evaluation ---
    holdout_metrics: Dict[str, Any] = {}
    for holdout_type in ["unseen_users", "seen_users"]:
        split_name = f"holdout_{holdout_type}"
        try:
            if user_encoder_type == "summarized":
                holdout_dataset = SummarizedEngagementDataset(
                    embeddings_mmap, target_posts_df, history_df, split=split_name,
                    summarizer=summarizer, embed_dim=embed_dim, logger=logger,
                )
            else:
                holdout_dataset = SequenceEngagementDataset(
                    embeddings_mmap, target_posts_df, history_df, split=split_name,
                    max_history_len=max_history_len, embed_dim=embed_dim,
                    use_author_embedding_table=use_author_embedding_table,
                    author_idx_to_table_row=author_idx_to_table_row,
                    logger=logger,
                )
            if len(holdout_dataset) == 0:
                logger.info(f"No rows for split '{split_name}', skipping.")
                continue
            log_operation_start(f"Holdout evaluation ({holdout_type})", STAGE_LOG_NAME, logger)
            _, _, _, holdout_loader = create_data_loaders(
                train_dataset, val_dataset, val_unseen_dataset, batch_size,  # train/val loaders unused here
                holdout_dataset=holdout_dataset,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
            holdout_eval = _evaluate_two_tower_model(trained_model, holdout_loader, device, embed_dim)
            split_metrics = holdout_eval["metrics"]
            logger.info(f"Holdout metrics ({holdout_type}): {split_metrics}")
            if holdout_type == eval_holdout_type:
                holdout_metrics = split_metrics

            pl.DataFrame({
                "did": holdout_eval["predictions"]["user_id"],
                "post_id": holdout_eval["predictions"]["post_id"],
                "y_true": holdout_eval["predictions"]["y_true"],
                "y_pred_proba": holdout_eval["predictions"]["y_pred"],
            }).write_parquet(predictions_dir / f"{split_name}.parquet")

            if generate_plots and holdout_type == eval_holdout_type:
                try:
                    plot_model_performance(
                        holdout_eval["predictions"]["y_true"],
                        holdout_eval["predictions"]["y_pred"],
                        plots_dir / f"holdout_performance_{timestamp}.png",
                        title_suffix="(Holdout)",
                    )
                except Exception as plot_exc:
                    logger.warning(f"Holdout performance plotting failed: {plot_exc}")
        except Exception as exc:
            logger.warning(f"Holdout evaluation ({holdout_type}) failed (non-fatal): {exc}")

    # --- training config ---
    training_config = {
        **config,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "random_seed": random_seed,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_metrics": train_eval["metrics"],
        "val_metrics": val_eval["metrics"],
        "holdout_metrics": holdout_metrics,
        "best_val_auc": best_val_auc,
    }
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    # --- stage info ---
    runtime = time.time() - t0
    info_lines = [
        f"stage: train_two_tower",
        f"timestamp: {timestamp}",
        f"runtime_seconds: {runtime:.2f}",
        f"settings: batch_size={batch_size}, lr={learning_rate}, epochs={epochs}, user_encoder={user_encoder_type}, l2_norm={l2_normalize_embeddings}, early_stopping_min_delta={early_stopping_min_delta}, tau={similarity_temperature}",
        f"train_samples: {len(train_dataset)}",
        f"val_samples: {len(val_dataset)}",
        f"best_val_auc: {best_val_auc:.4f}",
    ]
    if holdout_metrics.get("auc_roc"):
        info_lines.append(f"holdout_auc: {holdout_metrics['auc_roc']:.4f}")
    (out_dir / "stage_info.txt").write_text("\n".join(info_lines) + "\n")

    logger.info(f"Two-Tower training completed in {runtime:.2f}s")

    return {
        "output_dir": out_dir,
        "artifacts": {
            "model_path": str(model_path) if model_path else None,
            "training_config": str(out_dir / "training_config.json"),
            "author_table_mapping_path": str(author_table_mapping_path) if author_table_mapping_path else None,
        },
    }
