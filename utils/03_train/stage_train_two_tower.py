#!/usr/bin/env python3

"""
Stage 3 (Two-Tower): Train two-tower engagement prediction models with flexible user encoders.

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
    Scoring:     user_matrix · candidate_matrix.T -> ranking scores

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

Loss:       Row-wise multi-positive softmax over same-hour candidate posts
Sampling:   Bucketed user-hour positives plus same-hour random candidate posts
Optimizer:  AdamW with weight decay for regularization
Scheduling: ReduceLROnPlateau based on validation NDCG@metrics_top_ks[0]
Regularization: Gradient clipping, dropout, early stopping

The model is trained to rank each user's engaged posts above other same-hour
candidate posts, learning a metric space where similar preferences cluster.

═══════════════════════════════════════════════════════════════════════════════

Inputs (from prior pipeline stages):
    - embeddings_*.npy, likes_core_*.parquet, posts_core_*.parquet from 01_get_data
    - history_posts_*.parquet from 02_user_history

Outputs under <run_dir>/03_train/<timestamp>/:
    - checkpoints/two_tower_best.pth (best-by-validation checkpoint during training)
    - checkpoints/two_tower_<timestamp>.pth (final model checkpoint)
    - logs/ (training logs)
    - training_config.json (hyperparameters and configuration)
    - stage_info.txt (pipeline metadata)
    - predictions/ (currently disabled for bucketed training to avoid materializing
      all user-candidate pairs)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.pipeline.core import Context
from utils.helpers import (
    get_stage_logger,
    log_operation_start,
    log_prior_stage_inputs,
    get_device,
    clear_cuda_memory,
    set_random_seeds,
)
from utils.dataloaders import (
    AUTHOR_PAD_IDX,
    AUTHOR_UNK_IDX,
    BucketedEngagementDataset,
    create_bucketed_data_loaders,
    get_author_table_num_rows,
    load_bucketed_training_data,
    SummarizedUserTower,
    TransformerDualPoolingEncoder,
    CrossAttentionPoolingEncoder,
)
from utils.author_features import PostAuthorFeatureEncoder
from utils.matrix_ranking import (
    DEFAULT_MAX_CLASSIFICATION_METRIC_PAIRS,
    FINAL_CLASSIFICATION_METRICS,
    calc_baseline_rank_metrics_for_batch,
    empty_rank_metric_sums,
    evaluate_matrix_model,
    finalize_rank_metrics,
    log_final_classification_metrics,
    optional_float_metric,
    rank_metric_sums_for_batch,
    ranking_rows_for_batch,
    run_matrix_epoch,
    stage_info_metric_lines,
    write_ranking_rows,
)

STAGE_LOG_NAME = "STAGE_03_TRAIN_TWO_TOWER"


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
        
        Scoring: user_matrix @ candidate_matrix.T / temperature -> raw rank scores
    
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
            target_author_indices: author_idx values [batch], required
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
        """Compute user-by-candidate engagement scores via dot product.
        
        This is the core bucketed two-tower computation: encode users and
        candidate posts independently, then score every user against every
        candidate post in the batch. Higher dot product = higher predicted
        engagement affinity.
        
        Args:
            history_embeddings: User history input.
                - all modes: padded history sequences [batch, seq_len, input_dim]
                  In "summarized" mode, the user summary is expected to be placed
                  at position 0 (and optionally padded to seq_len > 1).
            history_mask: History validity mask [batch, seq_len] (optional in summarized mode)
            post_embeddings: Candidate post embeddings [num_candidates, input_dim]
            history_author_indices: author_idx values aligned with history items,
                required when author embeddings are enabled.
            target_author_indices: author_idx values for candidate posts, required
                when author embeddings are enabled.
        
        Returns:
            Raw engagement scores [num_users, num_candidates].
        """
        user_emb = self.encode_user(history_embeddings, history_mask, history_author_indices)
        post_emb = self.encode_post(post_embeddings, target_author_indices)
        
        similarity_score = torch.matmul(user_emb, post_emb.T)
        return similarity_score / self.similarity_temperature

    def compute_loss_and_preds(
        self,
        batch: Dict[str, Any],
        device: str,
        embed_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-user multi-positive contrastive loss and score matrix.
        
        This method provides a unified interface for training, validation, and
        inference loops. It scores every user against every candidate post in a
        bucketed batch, then applies a row-wise softmax loss so each user
        contributes equally regardless of how many positives they have.
        
        Args:
            batch: Bucketed batch with history tensors, candidate post tensors,
                and label_matrix shaped [num_users, num_candidates].
            device: Device string (e.g. "cpu" or "cuda")
            embed_dim: Post embedding dimensionality D (kept for caller compatibility)
        
        Returns:
            Tuple of (loss, scores):
                - loss: Scalar row-wise contrastive loss tensor
                - scores: Raw similarity scores [num_users, num_candidates]
        """
        history_author_indices = None
        target_author_indices = None
        history_embeddings = batch["history_embeddings"].to(device, non_blocking=True) # [U, seq_len, embed_dim]
        history_mask = batch["history_mask"].to(device, non_blocking=True) # [U, seq_len]
        post_embeddings = batch["candidate_post_embeddings"].to(device, non_blocking=True) # [P, embed_dim]
        label_matrix = batch["label_matrix"].to(device, dtype=torch.float32, non_blocking=True) # [U, P]
        if self.use_author_embedding_table:
            history_author_indices = batch["history_author_indices"].to(device, dtype=torch.long, non_blocking=True)
            target_author_indices = batch["candidate_post_author_idx"].to(device, dtype=torch.long, non_blocking=True)

        scores = self.forward(
            history_embeddings,
            history_mask,
            post_embeddings,
            history_author_indices,
            target_author_indices,
        )
        if scores.shape != label_matrix.shape:
            raise RuntimeError("Expected scores and label_matrix to have matching [num_users, num_candidates] shapes")
        positive_counts = label_matrix.sum(dim=1, keepdim=True)
        if torch.any(positive_counts <= 0):
            raise RuntimeError("Each user row in label_matrix must contain at least one positive candidate")

        targets = label_matrix / positive_counts
        loss_per_user = -(targets * F.log_softmax(scores, dim=1)).sum(dim=1)
        loss = loss_per_user.mean()
        return loss, scores


def _empty_rank_metric_sums(metrics_top_ks: list[int]) -> Dict[str, float]:
    return empty_rank_metric_sums(metrics_top_ks)


def _calc_baseline_rank_metrics_for_batch(
    unranked_labels: torch.Tensor,
    metrics_top_ks: list[int],
) -> Tuple[Dict[str, float], int]:
    return calc_baseline_rank_metrics_for_batch(unranked_labels, metrics_top_ks)


def _rank_metric_sums_for_batch(
    ranked_labels: torch.Tensor,
    metrics_top_ks: list[int],
) -> Tuple[Dict[str, float], int]:
    return rank_metric_sums_for_batch(ranked_labels, metrics_top_ks)


def _finalize_rank_metrics(metric_sums: Dict[str, float], user_count: int) -> Dict[str, float]:
    return finalize_rank_metrics(metric_sums, user_count)


def _ranking_rows_for_batch(
    batch: Dict[str, Any],
    scores: torch.Tensor,
    labels: torch.Tensor,
    metrics_top_ks: list[int],
) -> List[Dict[str, Any]]:
    return ranking_rows_for_batch(batch, scores, labels, metrics_top_ks)


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
    calc_baseline_metrics: bool,
):
    return run_matrix_epoch(
        train=train,
        split_name=split_name,
        model=model,
        device=device,
        dataloader=dataloader,
        optimizer=optimizer,
        disable_progress=disable_progress,
        embed_dim=embed_dim,
        gradient_clip_max_norm=gradient_clip_max_norm,
        metrics_top_ks=metrics_top_ks,
        calc_baseline_metrics=calc_baseline_metrics,
    )


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

    if not metrics_top_ks:
        raise ValueError("metrics_top_ks must contain at least one value")
    primary_metric_name = f"ndcg@{metrics_top_ks[0]}"
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        f"train_{primary_metric_name}": [],
        f"val_{primary_metric_name}": [],
    }
    best_val_metric = float("-inf")
    best_reset_val_metric = float("-inf")
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    for epoch in tqdm(range(epochs), desc="Training epochs", disable=disable_progress):
        calc_baseline_metrics: bool = epoch == 0
        train_loss, train_metrics_dict, train_baseline_metrics_dict = _run_one_epoch(
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
            calc_baseline_metrics=calc_baseline_metrics,
        )

        val_loss, val_metrics_dict, val_baseline_metrics_dict = _run_one_epoch(
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
            calc_baseline_metrics=calc_baseline_metrics,
        )

        val_unseen_loss, val_unseen_metrics_dict, val_unseen_baseline_metrics_dict = _run_one_epoch(
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
            calc_baseline_metrics=calc_baseline_metrics,
        )

        train_primary_metric = float(train_metrics_dict[primary_metric_name])
        val_primary_metric = float(val_metrics_dict[primary_metric_name])
        val_unseen_primary_metric = float(val_unseen_metrics_dict[primary_metric_name])

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history[f"train_{primary_metric_name}"].append(train_primary_metric)
        history[f"val_{primary_metric_name}"].append(val_primary_metric)

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
                title=f"Primary Ranking Metric ({primary_metric_name})",
                series=f"Train {primary_metric_name}",
                value=train_primary_metric,
                iteration=iteration,
            )
            experiment_tracker.log_scalar(
                title=f"Primary Ranking Metric ({primary_metric_name})",
                series=f"Validation {primary_metric_name}",
                value=val_primary_metric,
                iteration=iteration,
            )
            experiment_tracker.log_scalar(
                title=f"Primary Ranking Metric ({primary_metric_name})",
                series=f"Validation Unseen Users {primary_metric_name}",
                value=val_unseen_primary_metric,
                iteration=iteration,
            )
            for k in metrics_top_ks:
                experiment_tracker.log_scalar(
                    title=f"NDCG@{k}",
                    series=f"Train NDCG@{k}",
                    value=float(train_metrics_dict[f"ndcg@{k}"]),
                    iteration=iteration,
                )
                experiment_tracker.log_scalar(
                    title=f"NDCG@{k}",
                    series=f"Validation NDCG@{k}",
                    value=float(val_metrics_dict[f"ndcg@{k}"]),
                    iteration=iteration,
                )
                experiment_tracker.log_scalar(
                    title=f"NDCG@{k}",
                    series=f"Validation Unseen Users NDCG@{k}",
                    value=float(val_unseen_metrics_dict[f"ndcg@{k}"]),
                    iteration=iteration,
                )
                experiment_tracker.log_scalar(
                    title=f"Recall@{k}",
                    series=f"Train Recall@{k}",
                    value=float(train_metrics_dict[f"recall@{k}"]),
                    iteration=iteration,
                )
                experiment_tracker.log_scalar(
                    title=f"Recall@{k}",
                    series=f"Validation Recall@{k}",
                    value=float(val_metrics_dict[f"recall@{k}"]),
                    iteration=iteration,
                )
                experiment_tracker.log_scalar(
                    title=f"Recall@{k}",
                    series=f"Validation Unseen Users Recall@{k}",
                    value=float(val_unseen_metrics_dict[f"recall@{k}"]),
                    iteration=iteration,
                )
                if calc_baseline_metrics:
                    experiment_tracker.log_scalar(
                        title=f"Baseline NDCG@{k}",
                        series=f"Train Baseline NDCG@{k}",
                        value=float(train_baseline_metrics_dict[f"ndcg@{k}"]),
                        iteration=iteration,
                    )
                    experiment_tracker.log_scalar(
                        title=f"Baseline NDCG@{k}",
                        series=f"Validation Baseline NDCG@{k}",
                        value=float(val_baseline_metrics_dict[f"ndcg@{k}"]),
                        iteration=iteration,
                    )
                    experiment_tracker.log_scalar(
                        title=f"Baseline NDCG@{k}",
                        series=f"Validation Unseen Users Baseline NDCG@{k}",
                        value=float(val_unseen_baseline_metrics_dict[f"ndcg@{k}"]),
                        iteration=iteration,
                    )
                    experiment_tracker.log_scalar(
                        title=f"Baseline Recall@{k}",
                        series=f"Train Baseline Recall@{k}",
                        value=float(train_baseline_metrics_dict[f"recall@{k}"]),
                        iteration=iteration,
                    )
                    experiment_tracker.log_scalar(
                        title=f"Baseline Recall@{k}",
                        series=f"Validation Baseline Recall@{k}",
                        value=float(val_baseline_metrics_dict[f"recall@{k}"]),
                        iteration=iteration,
                    )
                    experiment_tracker.log_scalar(
                        title=f"Baseline Recall@{k}",
                        series=f"Validation Unseen Users Baseline Recall@{k}",
                        value=float(val_unseen_baseline_metrics_dict[f"recall@{k}"]),
                        iteration=iteration,
                    )

        scheduler.step(val_primary_metric)

        if val_primary_metric > best_val_metric:
            best_val_metric = val_primary_metric
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if checkpoints_dir is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_state_dict,
                        "val_loss": val_loss,
                        "primary_metric_name": primary_metric_name,
                        "val_primary_metric": val_primary_metric,
                        "history": history,
                    },
                    checkpoints_dir / "two_tower_best.pth",
                )

        significant_improvement = (
            val_primary_metric > best_reset_val_metric
            and (val_primary_metric - best_reset_val_metric) >= early_stopping_min_delta
        )
        if significant_improvement:
            best_reset_val_metric = val_primary_metric
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
        "best_val_metric": best_val_metric,
        "primary_metric_name": primary_metric_name,
    }


# =============================================================================
# Evaluation
# =============================================================================

def _evaluate_two_tower_model(
    model: TwoTowerModel,
    data_loader: DataLoader,
    device: str,
    embed_dim: int,
    metrics_top_ks: list[int],
    max_classification_metric_pairs: Optional[int] = DEFAULT_MAX_CLASSIFICATION_METRIC_PAIRS,
    collect_ranking_rows: bool = False,
    progress_desc: Optional[str] = None,
    disable_progress: bool = True,
) -> Dict[str, Any]:
    return evaluate_matrix_model(
        model=model,
        data_loader=data_loader,
        device=device,
        embed_dim=embed_dim,
        metrics_top_ks=metrics_top_ks,
        max_classification_metric_pairs=max_classification_metric_pairs,
        collect_ranking_rows=collect_ranking_rows,
        progress_desc=progress_desc,
        disable_progress=disable_progress,
    )


def _optional_float_metric(value: Any) -> Optional[float]:
    return optional_float_metric(value)


def _split_metric_label(split_name: str) -> str:
    return split_name.replace("_", " ").title()


def _clearml_metric_label(metric_name: str) -> str:
    return {
        "auc_roc": "AUC-ROC",
        "average_precision": "Average Precision",
    }.get(metric_name, metric_name.replace("_", " ").title())


def _log_final_classification_metrics(
    experiment_tracker: Optional[Any],
    split_metrics: Dict[str, Dict[str, Any]],
    iteration: int,
) -> None:
    log_final_classification_metrics(experiment_tracker, split_metrics, iteration)


def _stage_info_metric_lines(split_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
    return stage_info_metric_lines(split_metrics)


def _write_ranking_rows(
    rows: List[Dict[str, Any]],
    output_path: Path,
    split_name: str,
    num_total_likes_by_user: Dict[str, int],
) -> None:
    write_ranking_rows(rows, output_path, split_name, num_total_likes_by_user)


def _find_author_idx_artifact_path(context: Context) -> Optional[Path]:
    get_data_dir = context.get_active_stage_inputs().get("01_get_data")
    if get_data_dir is None:
        get_data_dir = context.get_artifact_dir("get_data")
    if get_data_dir is None:
        return None
    candidates = sorted(
        Path(get_data_dir).glob("author_idx_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


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
    out_dir = context.new_stage_dir("03_train", tag=run_tag)
    checkpoints_dir = out_dir / "checkpoints"
    plots_dir = out_dir / "plots"
    logs_dir = out_dir / "logs"
    eval_dir = out_dir / "eval"
    for d in (checkpoints_dir, plots_dir, logs_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger = get_stage_logger(STAGE_LOG_NAME, log_file=out_dir / "stage.log")
    log_operation_start("Stage 3 Two-Tower training", STAGE_LOG_NAME, logger)
    t0 = time.time()

    # --- seeds & cuda ---
    clear_cuda_memory()
    random_seed = int(args.random_seed)
    set_random_seeds(random_seed)

    # --- load data from prior stages ---
    log_operation_start("Load training data from prior stages", STAGE_LOG_NAME, logger)
    embeddings_mmap, likes_core_df, posts_core_df, history_df, author_idx_mapping_df, embed_dim = load_bucketed_training_data(
        context, logger=logger,
    )
    log_prior_stage_inputs(context, logger)
    num_total_likes_by_user = {
        str(row["did"]): int(row["num_total_likes"])
        for row in (
            likes_core_df
            .group_by("did")
            .agg(pl.col("subject_uri").n_unique().alias("num_total_likes"))
            .iter_rows(named=True)
        )
    }

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
    author_unknown_dropout_rate = float(args.author_unknown_dropout_rate)
    metrics_top_ks = list(args.metrics_top_ks)
    if not metrics_top_ks:
        raise ValueError("metrics_top_ks must contain at least one value")
    primary_metric_name = f"ndcg@{metrics_top_ks[0]}"

    if user_encoder_type == "summarized":
        raise ValueError("Bucketed training does not support user_encoder_type='summarized'")
    if use_author_embedding_table and user_encoder_type == "summarized":
        raise ValueError("use_author_embedding_table is not supported with user_encoder_type='summarized'")
    if use_author_embedding_table and author_idx_mapping_df is None:
        raise FileNotFoundError(
            "author_idx artifact was not found in 01_get_data output, but --use-author-embedding-table was enabled."
        )
    author_table_num_rows = 0
    if use_author_embedding_table:
        if author_idx_mapping_df is None:
            raise FileNotFoundError("author_idx_mapping_df is required when use_author_embedding_table is True")
        author_table_num_rows = get_author_table_num_rows(author_idx_mapping_df)
        logger.info(
            "Author embedding table enabled: "
            f"author_embedding_dim={author_embedding_dim}, "
            f"author_table_num_rows={author_table_num_rows}"
        )
        author_idx_artifact_path = _find_author_idx_artifact_path(context)
        if author_idx_artifact_path is None:
            logger.warning("Author embedding table enabled, but no author_idx parquet path was found to log")
        else:
            author_idx_artifact_id = context.tracker.log_file_artifact(
                name="author_idx_mapping",
                path=author_idx_artifact_path,
            )
            logger.info(f"Author index mapping artifact id: {author_idx_artifact_id}")

    # Worker settings
    num_workers = int(args.num_dataloader_workers)
    pin_memory = bool(args.dataloader_pin_memory)
    persistent_workers = bool(args.dataloader_persistent_workers)
    prefetch_factor = int(args.dataloader_prefetch_factor)

    # --- datasets ---
    log_operation_start("Create datasets", STAGE_LOG_NAME, logger)
    train_dataset = BucketedEngagementDataset(
        embeddings_mmap, likes_core_df, posts_core_df, history_df, split="train",
        max_history_len=max_history_len, embed_dim=embed_dim,
        use_author_embedding_table=use_author_embedding_table,
        logger=logger,
    )
    val_dataset = BucketedEngagementDataset(
        embeddings_mmap, likes_core_df, posts_core_df, history_df, split="val",
        max_history_len=max_history_len, embed_dim=embed_dim,
        use_author_embedding_table=use_author_embedding_table,
        logger=logger,
    )
    val_unseen_dataset = BucketedEngagementDataset(
        embeddings_mmap, likes_core_df, posts_core_df, history_df, split="val_unseen_users",
        max_history_len=max_history_len, embed_dim=embed_dim,
        use_author_embedding_table=use_author_embedding_table,
        logger=logger,
    )

    # Create data loaders using centralized helper
    train_loader, val_loader, val_unseen_loader, _ = create_bucketed_data_loaders(
        train_dataset, val_dataset, val_unseen_dataset, batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        seed=random_seed,
    )

    logger.info(f"Post embedding dim: {embed_dim}")
    logger.info(f"Train user-hour rows: {len(train_dataset)}, Val user-hour rows: {len(val_dataset)}")

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
        if generate_plots:
            logger.info("Skipping two-tower training plots while per-row prediction materialization is disabled")

    # Collect split metrics without materializing per-user-candidate predictions
    train_eval = _evaluate_two_tower_model(
        trained_model, train_loader, device, embed_dim, metrics_top_ks,
        progress_desc="Evaluate train",
        disable_progress=disable_progress,
    )
    val_eval = _evaluate_two_tower_model(
        trained_model, val_loader, device, embed_dim, metrics_top_ks,
        progress_desc="Evaluate validation",
        disable_progress=disable_progress,
    )
    val_unseen_eval = _evaluate_two_tower_model(
        trained_model, val_unseen_loader, device, embed_dim, metrics_top_ks,
        progress_desc="Evaluate validation unseen users",
        disable_progress=disable_progress,
    )
    logger.info(f"Train metrics: {train_eval['metrics']}")
    logger.info(f"Validation metrics: {val_eval['metrics']}")
    logger.info(f"Validation unseen users metrics: {val_unseen_eval['metrics']}")

    if training_results is not None:
        best_val_metric = training_results["best_val_metric"]
        primary_metric_name = training_results["primary_metric_name"]
    else:
        best_val_metric = float(val_eval["metrics"].get(primary_metric_name, 0.0))
        if context.tracker is not None:
            context.tracker.log_scalar(
                title=f"Primary Ranking Metric ({primary_metric_name})",
                series=f"Train {primary_metric_name}",
                value=float(train_eval["metrics"].get(primary_metric_name, 0.0)),
                iteration=0,
            )
            context.tracker.log_scalar(
                title=f"Primary Ranking Metric ({primary_metric_name})",
                series=f"Validation {primary_metric_name}",
                value=best_val_metric,
                iteration=0,
            )
            context.tracker.log_scalar(
                title=f"Primary Ranking Metric ({primary_metric_name})",
                series=f"Validation Unseen Users {primary_metric_name}",
                value=float(val_unseen_eval["metrics"].get(primary_metric_name, 0.0)),
                iteration=0,
            )

    # --- save model ---
    model_path = None
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
        "author_unknown_dropout_rate": author_unknown_dropout_rate if use_author_embedding_table else None,
        "author_table_num_rows": author_table_num_rows if use_author_embedding_table else None,
        "author_pad_idx": AUTHOR_PAD_IDX,
        "author_unk_idx": AUTHOR_UNK_IDX,
    }
    if save_model:
        model_path = checkpoints_dir / f"two_tower_{timestamp}.pth"
        torch.save(
            {
                "model_state_dict": trained_model.state_dict(),
                "config": config,
                "training_history": training_results["history"] if training_results is not None else None,
                "primary_metric_name": primary_metric_name,
                "best_val_metric": best_val_metric,
                "best_val_loss": training_results["best_val_loss"] if training_results is not None else None,
            },
            model_path,
        )
        logger.info(f"Model saved to: {model_path}")

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

    # Prediction parquet writing is intentionally disabled for now. The bucketed
    # path produces one row per user-candidate pair, which can be hundreds of
    # millions of rows per split with current sampling settings.

    # --- holdout evaluation ---
    holdout_metrics: Dict[str, Any] = {}
    all_holdout_metrics: Dict[str, Dict[str, Any]] = {}
    for holdout_type in ["unseen_users", "seen_users"]:
        split_name = f"holdout_{holdout_type}"
        try:
            holdout_dataset = BucketedEngagementDataset(
                embeddings_mmap, likes_core_df, posts_core_df, history_df, split=split_name,
                max_history_len=max_history_len, embed_dim=embed_dim,
                use_author_embedding_table=use_author_embedding_table,
                logger=logger,
            )
            if len(holdout_dataset) == 0:
                logger.info(f"No rows for split '{split_name}', skipping.")
                continue
            log_operation_start(f"Holdout evaluation ({holdout_type})", STAGE_LOG_NAME, logger)
            _, _, _, holdout_loader = create_bucketed_data_loaders(
                train_dataset, val_dataset, val_unseen_dataset, batch_size,  # train/val loaders unused here
                holdout_dataset=holdout_dataset,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                seed=random_seed,
            )
            holdout_eval = _evaluate_two_tower_model(
                trained_model,
                holdout_loader,
                device,
                embed_dim,
                metrics_top_ks,
                collect_ranking_rows=True,
                progress_desc=f"Evaluate holdout {holdout_type}",
                disable_progress=disable_progress,
            )
            split_metrics = holdout_eval["metrics"]
            logger.info(f"Holdout metrics ({holdout_type}): {split_metrics}")

            all_holdout_metrics[split_name] = split_metrics

            ranking_rows_path = eval_dir / f"{split_name}_ranking_rows.parquet"
            write_ranking_rows(
                holdout_eval["ranking_rows"],
                ranking_rows_path,
                split_name,
                num_total_likes_by_user,
            )
            logger.info(f"Saved holdout ranking rows ({holdout_type}): {ranking_rows_path}")

            if holdout_type == eval_holdout_type:
                holdout_metrics = split_metrics
        except Exception as exc:
            logger.warning(f"Holdout evaluation ({holdout_type}) failed (non-fatal): {exc}")

    final_split_metrics: Dict[str, Dict[str, Any]] = {
        "train": train_eval["metrics"],
        "val": val_eval["metrics"],
        "val_unseen_users": val_unseen_eval["metrics"],
        **all_holdout_metrics,
    }
    final_metric_iteration = (
        len(training_results["history"]["train_loss"])
        if training_results is not None
        else 0
    )
    log_final_classification_metrics(
        context.tracker,
        final_split_metrics,
        final_metric_iteration,
    )

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
        "val_unseen_samples": len(val_unseen_dataset),
        "train_metrics": train_eval["metrics"],
        "val_metrics": val_eval["metrics"],
        "val_unseen_metrics": val_unseen_eval["metrics"],
        "holdout_metrics": holdout_metrics,
        "all_holdout_metrics": all_holdout_metrics,
        "primary_metric_name": primary_metric_name,
        "best_val_metric": best_val_metric,
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
        f"primary_metric_name: {primary_metric_name}",
        f"best_val_metric: {best_val_metric:.4f}",
    ]
    info_lines.extend(stage_info_metric_lines(final_split_metrics))
    if holdout_metrics.get(primary_metric_name) is not None:
        info_lines.append(f"holdout_{primary_metric_name}: {holdout_metrics[primary_metric_name]:.4f}")
    (out_dir / "stage_info.txt").write_text("\n".join(info_lines) + "\n")

    logger.info(f"Two-Tower training completed in {runtime:.2f}s")

    return {
        "output_dir": out_dir,
        "artifacts": {
            "model_path": str(model_path) if model_path else None,
            "training_config": str(out_dir / "training_config.json"),
        },
    }
