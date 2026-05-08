#!/usr/bin/env python3

"""
Shared dataloaders and user-encoder building blocks for engagement prediction.

A **modular framework for representing user engagement history**
in different formats, enabling flexible model architectures while maintaining
code reuse and memory efficiency.

═══════════════════════════════════════════════════════════════════════════════
MODULAR USER-HISTORY REPRESENTATION
═══════════════════════════════════════════════════════════════════════════════

User engagement history (the sequence of posts a user has liked) can be
represented in two fundamentally different ways, each optimized for different
model architectures:

1. **Fixed-Size Summary Vectors** (SummarizedEngagementDataset)
   ─────────────────────────────────────────────────────────────────────────
   Reduces variable-length history to a single fixed-dimensional vector using
   HAND-CRAFTED, DETERMINISTIC summarization strategies (no learnable parameters):
   
   • MeanSummarizer          : Simple arithmetic average of all liked posts
   • EMASummarizer           : Exponential moving average (recent posts weighted higher)
   • LinearRecencySummarizer : Linear decay weighting (most recent = highest weight)
   
   Output format: Concatenated [user_summary || post_embedding] vector
   Memory:        Pre-computed and cached in RAM (user summaries + pos/neg post
                  embeddings). Roughly ~3 * N * D * 4 bytes for float32 tensors
                  (e.g., N=178K, D=384 → ~0.8 GB).
   
2. **Variable-Length Sequences** (SequenceEngagementDataset)
   ─────────────────────────────────────────────────────────────────────────
   Preserves full temporal structure as padded/masked embedding sequences,
   enabling LEARNED, TRAINABLE encoders (neural networks with parameters) to
   discover optimal history aggregation during training:
   
   • TransformerDualPoolingEncoder  : Full transformer self-attention
                                      Dual pooling: attention-weighted + mean
   • CrossAttentionPoolingEncoder   : Single learned-query cross-attention pooling
                                      Faster and fewer parameters
   
   Output format: Dict with keys {"history_embeddings", "history_mask",
                  "target_post_embedding", "label", "user_id", "post_id"}
   Memory:        Sequences loaded on-the-fly via memmap (~13 GB if pre-computed)

═══════════════════════════════════════════════════════════════════════════════
KEY TERMINOLOGY
═══════════════════════════════════════════════════════════════════════════════

**SUMMARIZERS** (Hand-Crafted, Static)
  - Deterministic aggregation functions with NO learnable parameters
  - Apply predefined rules (mean, EMA, weighted average)
  - Fast, interpretable, work out-of-the-box
  - Examples: MeanSummarizer, EMASummarizer, LinearRecencySummarizer

**ENCODERS** (Learned, Trainable)
  - Neural network modules with trainable parameters
  - Learn optimal aggregation patterns from data during training
  - More flexible but require sufficient training data
  - Examples: TransformerDualPoolingEncoder, CrossAttentionPoolingEncoder

Both transform user history → fixed-size vector, but:
  - Summarizers use predetermined statistical operations
  - Encoders learn optimal transformations via backpropagation

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE PATTERNS
═══════════════════════════════════════════════════════════════════════════════

The modular design supports multiple training approaches:

    MLP + Summarizer             : SummarizedEngagementDataset + MLPModel(user_encoder_type="summarized")
    MLP + Full Transformer       : SequenceEngagementDataset + MLPModel(user_encoder_type="full_transformer")
    MLP + Cross-Attention        : SequenceEngagementDataset + MLPModel(user_encoder_type="cross_attention")
    Two-Tower + Full Transformer : SequenceEngagementDataset + TwoTowerModel(user_encoder_type="full_transformer")
    Two-Tower + Cross-Attention  : SequenceEngagementDataset + TwoTowerModel(user_encoder_type="cross_attention")

This separation allows experimentation with different history representations
without modifying model code, and vice versa.

═══════════════════════════════════════════════════════════════════════════════
MAIN COMPONENTS
═══════════════════════════════════════════════════════════════════════════════

Datasets:
    SummarizedEngagementDataset  -- Fixed-size [user_summary ‖ post_emb] vectors
    SequenceEngagementDataset    -- Padded variable-length history sequences + mask

Hand-Crafted Summarizers (deterministic, no learnable parameters):
    UserSummarizer               -- Abstract base class
    MeanSummarizer               -- Arithmetic mean
    EMASummarizer                -- Exponential moving average with recency bias
    LinearRecencySummarizer      -- Linear recency weighting

Learned Encoders (trainable neural networks):
    TransformerDualPoolingEncoder   -- Full transformer self-attention + dual pooling
    CrossAttentionPoolingEncoder    -- Efficient single-query cross-attention pooling

Utilities:
    load_training_data()         -- Locates and loads upstream pipeline artifacts
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.pipeline.core import Context, select_prior_output
from utils.helpers import (
    get_stage_logger,
    load_parquet_from_prior,
    log_operation_start,
)
from shared.input_data_helpers import get_padded_embedding_history_and_mask


PAD_IDX = 0
UNK_IDX = 1


def _map_raw_idx_to_table_row(
    raw_idx: Any,
    idx_to_table_row: np.ndarray,
) -> int:
    """Translate a raw Stage 2 embedding index into the model table row."""
    if raw_idx is None:
        return UNK_IDX
    try:
        if raw_idx != raw_idx:  # NaN
            return UNK_IDX
    except TypeError:
        pass
    raw_idx_int = int(raw_idx)
    if 0 <= raw_idx_int < len(idx_to_table_row):
        return int(idx_to_table_row[raw_idx_int])
    return UNK_IDX


def build_target_user_table_lookup(
    target_user_idx_mapping_df: pl.DataFrame,
    min_target_user_support: int,
) -> Tuple[np.ndarray, int]:
    """Map Stage 2 target_user indices to final embedding-table rows.

    Returns a pair ``(target_user_idx_to_table_row, target_user_table_num_rows)``.

    These represent two different index spaces:

    - ``target_user_idx_to_table_row`` is a lookup array keyed by raw Stage 2
      ``target_user_idx`` values. It only needs entries for raw target_user ids that may
      appear in the history artifact. Missing or out-of-range lookups are
      handled downstream as ``UNK_IDX``.
    - ``target_user_table_num_rows`` is the size of the actual embedding table used
      at training time. That table always reserves row ``0`` for padding and
      row ``1`` for unknown target_users, so it must be at least ``2`` even when no
      supported target_users survive filtering.
    """
    if min_target_user_support < 1:
        raise ValueError("min_target_user_support must be >= 1")
    required_cols = {"target_user_idx", "target_user_train_count"}
    missing_cols = required_cols.difference(target_user_idx_mapping_df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"target_user_idx_mapping_df is missing required columns: {missing}")

    if len(target_user_idx_mapping_df) == 0:
        # Empty Stage 2 mapping means there are no known raw target_user_idx values
        # to translate. The lookup therefore only needs a dummy slot, while the
        # embedding table still needs PAD=0 and UNK=1 rows.
        target_user_idx_to_table_row = np.full(1, UNK_IDX, dtype=np.uint32)
        target_user_table_num_rows = 2
        return target_user_idx_to_table_row, target_user_table_num_rows

    max_target_user_idx = int(target_user_idx_mapping_df["target_user_idx"].max()) # type: ignore
    if max_target_user_idx < 1:
        raise ValueError("target_user_idx values must start at 1")

    target_user_idx_to_table_row = np.full(max_target_user_idx + 1, UNK_IDX, dtype=np.uint32)
    supported_df = target_user_idx_mapping_df.filter(
        pl.col("target_user_train_count") >= int(min_target_user_support)
    )
    if len(supported_df) == 0:
        # Keep the full raw-target_user lookup shape so in-range Stage 2 target_user_idx
        # values still resolve deterministically to UNK after support filtering.
        target_user_table_num_rows = 2
        return target_user_idx_to_table_row, target_user_table_num_rows

    supported_target_user_indices = supported_df["target_user_idx"].to_numpy().astype(np.int64)
    target_user_idx_to_table_row[supported_target_user_indices] = (supported_target_user_indices + 1).astype(np.uint32)
    # Table row layout:
    #   0 -> PAD
    #   1 -> UNK
    #   raw target_user_idx N -> row N + 1 when that target_user survives filtering
    target_user_table_num_rows = int(supported_target_user_indices.max()) + 2
    return target_user_idx_to_table_row, target_user_table_num_rows


# ---------------------------------------------------------------------------
# User Summarizer strategies
# ---------------------------------------------------------------------------

class SummarizedUserTower(nn.Module):
    """User "tower" for summarized mode.

    Serving/training convention: represent the summarized user vector as a
    (possibly padded) length-T sequence where the summary lives at position 0:

        history_embeddings[:, 0, :] == user_summary

    This keeps model `forward(history_embeddings, history_mask, ...)` signatures
    consistent across encoder types.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = int(embed_dim)
        # Learnable cold-start embedding used when there is no user history.
        # Kept in the same space as the summary vector so Two-Tower dot products
        # can still rank posts for cold-start users.
        self.empty_user_embedding = nn.Parameter(torch.randn(self.embed_dim) * 0.02)

    def forward(self, history_embeddings: torch.Tensor, history_mask: Optional[torch.Tensor], target_user_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        if history_embeddings.dim() != 3:
            # Keep error message TorchScript-friendly (no dynamic shape formatting).
            raise RuntimeError("Expected history_embeddings with shape [B, T, D].")
        if history_embeddings.size(-1) != self.embed_dim:
            raise RuntimeError("Expected history_embeddings last dimension to match embed_dim.")
        summary = history_embeddings[:, 0, :]
        if history_mask is None:
            return summary
        if history_mask.dim() != 2:
            raise RuntimeError("Expected history_mask with shape [B, T].")
        if (
            history_mask.size(0) != history_embeddings.size(0)
            or history_mask.size(1) != history_embeddings.size(1)
        ):
            raise RuntimeError("Expected history_mask shape [B, T] to match history_embeddings.")

        history_mask = history_mask.to(device=history_embeddings.device, dtype=torch.bool)
        has_history = history_mask.any(dim=1)  # [B]
        has_history_f = has_history.to(dtype=summary.dtype).unsqueeze(1)  # [B, 1]
        empty = self.empty_user_embedding.unsqueeze(0)  # [1, D]
        return summary * has_history_f + empty * (1.0 - has_history_f)


class UserSummarizer(ABC):
    """Base class for hand-crafted user-history summarization strategies.
    
    Summarizers are DETERMINISTIC, HAND-CRAFTED aggregation functions with NO
    learnable parameters. They collapse a variable-length sequence of post
    embeddings (user's engagement history) into a single fixed-size vector using
    predefined statistical operations (mean, EMA, weighted average, etc.).
    
    All concrete summarizers must:
    - Handle empty histories gracefully (return zero vector)
    - Preserve embedding dimensionality: input [seq_len, D] -> output [D]
    - Expect embeddings sorted most-recent-first (index 0 = most recent like)
    - Be deterministic (no randomness, no learnable parameters)
    
    Design rationale:
        Pluggable summarizers allow experimentation with different hand-crafted
        aggregation strategies without changing dataset or model code. Simple
        strategies (mean, EMA) are fast and interpretable baseline alternatives
        to more complex learned encoders.
    """

    @abstractmethod
    def summarize(self, embeddings: np.ndarray) -> np.ndarray:
        """Aggregate a variable-length history into a single fixed-size vector.
        
        Args:
            embeddings: User's engagement history as embeddings, shape [seq_len, D].
                       Sorted most-recent-first (index 0 = most recent liked post).
                       May be empty (seq_len == 0) if user has no history.

        Returns:
            Single summary vector of shape [D]. For empty input, returns a zero
            vector to represent "no engagement history available".
            
        Implementation note:
            Subclasses should use float32 for consistency with PyTorch training.
        """
        ...


class MeanSummarizer(UserSummarizer):
    """Simple arithmetic mean summarizer - all history posts weighted equally.
    
    Treats all engagement equally regardless of recency. This is the simplest
    baseline summarization strategy.
    
    Computation: mean(embeddings) along the sequence dimension
    Complexity:  O(seq_len * D)

    """

    def summarize(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute unweighted mean of all history embeddings.
        
        Args:
            embeddings: History sequence, shape [seq_len, D]
            
        Returns:
            Mean vector [D], or zero vector if empty
        """
        if len(embeddings) == 0:
            return np.zeros(embeddings.shape[1], dtype=np.float32)
        return embeddings.mean(axis=0).astype(np.float32)


class EMASummarizer(UserSummarizer):
    """Exponential moving average summarizer with recency bias.
    
    Applies exponentially decaying weights to history, with recent likes weighted
    more heavily than older ones.
    
    Weight formula:
        For position i (0 = most recent): w_i = alpha * (1 - alpha)^i
        Weights are normalized to sum to 1.
    
    Args:
        alpha: Smoothing factor in (0, 1]. Higher values increase recency bias.
               α=0.1 (default) gives ~50% weight to the 7 most recent likes.
               α=0.3 gives ~50% weight to the 2 most recent likes.
               α=1.0 uses only the most recent like.
    
    Computation: O(seq_len * D)
    
    Raises:
        ValueError: If alpha is not in range (0, 1]
    """

    def __init__(self, alpha: float = 0.1):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha

    def summarize(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute exponentially-weighted mean favoring recent history.
        
        Args:
            embeddings: History sequence, shape [seq_len, D], most-recent-first
            
        Returns:
            Weighted mean vector [D], or zero vector if empty
        """
        if len(embeddings) == 0:
            return np.zeros(embeddings.shape[1], dtype=np.float32)
        n = len(embeddings)
        # Compute raw weights: alpha * (1-alpha)^i for i in [0, n)
        raw_weights = self.alpha * ((1.0 - self.alpha) ** np.arange(n, dtype=np.float64))
        # Normalize to sum to 1
        weights = (raw_weights / raw_weights.sum()).astype(np.float32)
        return (weights[:, None] * embeddings).sum(axis=0).astype(np.float32)


class LinearRecencySummarizer(UserSummarizer):
    """Linear recency weighting: most recent gets highest weight, oldest gets lowest.
    
    Applies linearly decreasing weights based on position in the history sequence.
    Simpler and more intuitive than EMA, with predictable weight distribution.
    
    Weight formula:
        For n items, position i gets weight (n - i), then normalize
        Example (n=4): weights = [4, 3, 2, 1] / 10 = [0.4, 0.3, 0.2, 0.1]
    
    Computation: O(seq_len * D)
    """

    def summarize(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute linearly-weighted mean with recency bias.
        
        Args:
            embeddings: History sequence, shape [seq_len, D], most-recent-first
            
        Returns:
            Weighted mean vector [D], or zero vector if empty
        """
        if len(embeddings) == 0:
            return np.zeros(embeddings.shape[1], dtype=np.float32)
        n = len(embeddings)
        # Weights: [n, n-1, ..., 2, 1] normalized
        raw_weights = np.arange(n, 0, -1, dtype=np.float32)
        weights = raw_weights / raw_weights.sum()
        return (weights[:, None] * embeddings).sum(axis=0).astype(np.float32)


def get_summarizer(name: str, **kwargs: Any) -> UserSummarizer:
    """Factory function: instantiate a summarizer by name.
    
    Args:
        name: One of "mean", "ema", "linear_recency"
        **kwargs: Summarizer-specific parameters:
                  - For "ema": alpha (or ema_alpha) controls recency bias
    
    Returns:
        Configured UserSummarizer instance
        
    Raises:
        ValueError: If name is not recognized
        
    Example:
        >>> summarizer = get_summarizer("ema", alpha=0.2)
        >>> summary = summarizer.summarize(history_embeddings)
    """
    if name == "mean":
        return MeanSummarizer()
    if name == "ema":
        alpha = kwargs.get("alpha", kwargs.get("ema_alpha", 0.1))
        return EMASummarizer(alpha=float(alpha))
    if name == "linear_recency":
        return LinearRecencySummarizer()
    raise ValueError(f"Unknown summarizer: {name!r}. Choose from: mean, ema, linear_recency")


# ---------------------------------------------------------------------------
# Learned user-history encoders
# ---------------------------------------------------------------------------

class BaseAttentionEncoder(nn.Module, ABC):
    """Shared building blocks for learned user-history encoders that use attention-style pooling.

    This base class centralizes the parts that are common across multiple "learned"
    history encoders:
      - projecting raw per-post embeddings into a model hidden space
      - adding learnable positional embeddings that encode *recency*
      - pooling a variable-length sequence into a fixed-size vector via:
          (1) learned-query attention pooling (content-aware)
          (2) masked mean pooling over content-only projected embeddings
              (coverage / stability)
      - projecting pooled features into a final `output_dim` representation

    Subclasses are responsible for defining the "sequence modeling" portion after
    positional encoding (e.g., a TransformerEncoder stack, or no self-attention at
    all) before attention pooling.

    Mask conventions:
      - Public `forward()` expects `history_mask` where **True means valid**.
      - PyTorch transformer modules often expect an inverted key-padding mask where
        **True means ignore**; subclasses should invert as needed.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_seq_len: int,
        dropout_rate: float,
        use_target_user_embedding_table: bool,
        target_user_table_num_rows: Optional[int] = None,
        target_user_embedding_dim: Optional[int] = None,
        target_user_unknown_dropout_rate: float = 0.0,
    ):
        """Construct shared layers used by attention-based history encoders.

        Args:
            input_dim: Dimensionality of each item in the input sequence (e.g. a
                post/content embedding size).
            hidden_dim: Internal model dimension used for attention/pooling.
            output_dim: Dimensionality of the final user representation produced by
                the encoder.
            max_seq_len: Maximum supported history length. This controls the size of
                the learnable positional-embedding table.
            dropout_rate: Dropout probability applied in projection MLPs.

        Notes:
            - Positional embeddings are learnable (not sinusoidal) and are applied
              after the input projection and content-only mean pooling.
            - The positional scheme is *recency-flipped*: position 0 corresponds to
              the most recent item (see `_forward_up_to_pos_embed`).
            - The final projection expects concatenated pooled features of size
              `2 * hidden_dim` (attention pooled + content-only mean pooled).
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.use_target_user_embedding_table = bool(use_target_user_embedding_table)

        # Project raw embeddings to transformer hidden dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # Learnable positional embeddings (one per position up to max_seq_len)
        self.positional_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Learnable query vector for attention pooling
        self.attention_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Learnable cold-start token used when a user has no history (all-masked).
        # This prevents degenerate all-zero user embeddings for cold-start users,
        # which would otherwise yield identical Two-Tower scores across posts.
        self.empty_history_embedding = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        # mean pool vector and cross attention vector
        output_projection_input_dim = hidden_dim * 2

        # user embedding table. initialize values because otherwise torchscript saving won't work
        self.target_user_embedding_dim = 1
        self.target_user_unk_idx = UNK_IDX
        self.target_user_pad_idx = PAD_IDX
        self.target_user_unknown_dropout_rate = float(target_user_unknown_dropout_rate)
        self.target_user_embedding_table = nn.Embedding(2, 1, padding_idx=self.target_user_pad_idx)

        if self.use_target_user_embedding_table:
            if target_user_table_num_rows is None or target_user_table_num_rows < 2:
                raise ValueError("target_user_table_num_rows must be provided and >= 2 when use_target_user_embedding_table is True")
            if target_user_embedding_dim is None or target_user_embedding_dim <= 0:
                raise ValueError("target_user_embedding_dim must be provided and positive when use_target_user_embedding_table is True")
            
            self.target_user_embedding_dim = target_user_embedding_dim
            self.target_user_unknown_dropout_rate = target_user_unknown_dropout_rate            
            
            self.target_user_embedding_table = nn.Embedding(
                num_embeddings=target_user_table_num_rows,
                embedding_dim=target_user_embedding_dim,
                padding_idx=self.target_user_pad_idx,
            )
            
            output_projection_input_dim += self.target_user_embedding_dim

        # Project concatenated dual-pooled features to final output dimension
        self.output_projection = nn.Sequential(
            nn.Linear(output_projection_input_dim, hidden_dim),  # 2x because we concatenate two pooling outputs
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def _init_weights(self):
        """Initialize weights using Xavier uniform for linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
        
        if self.use_target_user_embedding_table:
            nn.init.xavier_uniform_(self.target_user_embedding_table.weight)
            with torch.no_grad():
                self.target_user_embedding_table.weight[self.target_user_pad_idx].zero_()

    def _forward_up_to_pos_embed(
        self,
        history_embeddings: torch.Tensor,  # [B, seq_len, input_dim]
        history_mask: Optional[torch.Tensor],  # [B, seq_len] True = valid
    ) -> Tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare projected history, content mean, and position-encoded sequence.

        This helper performs the common "front half" of every attention-based encoder:
          1) ensure we have a boolean validity mask on the right device
          2) project raw input embeddings into `hidden_dim`
          3) inject a learnable content-space token for empty histories
          4) compute masked mean pooling before positional embeddings are added
          5) add a learnable positional embedding for each sequence position

        The positional embedding indexing is intentionally **flipped** so that the
        earliest positions in the tensor (index 0, 1, 2, ...) represent *more recent*
        events. This matches the typical "most recent first" intuition and allows
        the model to learn a consistent recency prior independent of padding.
        Mean pooling is computed before positional embeddings are added so it remains
        a content-only coverage signal.

        Args:
            history_embeddings: Padded input sequence tensor `[B, seq_len, input_dim]`.
            history_mask: Optional boolean validity mask `[B, seq_len]` where True
                means the corresponding position is real (not padding). If omitted,
                all positions are treated as valid.

        Returns:
            A 5-tuple `(B, seq_len, history_mask, x, mean_pooled)` where:
              - `B` and `seq_len` are extracted from the input for convenience.
              - `history_mask` is a boolean tensor on the same device as inputs.
              - `x` is the projected + position-encoded representation
                `[B, seq_len, hidden_dim]`.
              - `mean_pooled` is the masked mean over projected content embeddings,
                before positional embeddings are added, with the cold-start token
                included for empty-history rows.
        """
        B, seq_len, _ = history_embeddings.shape
        if seq_len == 0:
            raise RuntimeError("history_embeddings must have a non-zero sequence length.")
        device = history_embeddings.device

        # Default to all-valid mask if not provided
        if history_mask is None:
            history_mask = torch.ones(B, seq_len, dtype=torch.bool, device=device)
        else:
            history_mask = history_mask.to(device=device, dtype=torch.bool)

        # Project embeddings to hidden dimension
        x = self.input_projection(history_embeddings)

        # Cold-start handling: if an example has no valid history items, inject a
        # learnable token at position 0 and mark it as valid so downstream
        # pooling/self-attention has something to attend to.
        has_any = history_mask.any(dim=1)  # [B]
        if not has_any.all().item():
            inject = ~has_any  # [B], True where history is empty
            inject_f = inject.to(dtype=x.dtype).unsqueeze(1)  # [B, 1]

            # Mark position 0 as valid for empty-history rows.
            history_mask = history_mask.clone()
            history_mask[:, 0] = history_mask[:, 0] | inject

            # Overwrite x at position 0 for empty-history rows with the learnable token.
            token0 = self.empty_history_embedding.unsqueeze(0)  # [1, hidden]
            x = x.clone()
            x0 = x[:, 0, :]  # [B, hidden]
            x[:, 0, :] = x0 * (1.0 - inject_f) + token0.expand(B, -1) * inject_f

        # ─── Mean pooling (masked) ───
        mean_pooled = self._forward_mean_pooled(x, history_mask)

        # Add positional information (flipped: position 0 = most recent)
        # `positions` indexes into the learnable table `self.positional_embedding`.
        # We clamp to `max_seq_len - 1` so that longer sequences reuse the "oldest"
        # available positional embedding rather than indexing out of range.
        positions = torch.arange(seq_len, device=history_embeddings.device)
        positions = (self.max_seq_len - 1) - positions.clamp(max=self.max_seq_len - 1)
        pos_emb = self.positional_embedding(positions)
        x = x + pos_emb.unsqueeze(0)  # Broadcast across batch

        return B, seq_len, history_mask, x, mean_pooled

    def _forward_attention_pooled(
        self, 
        B: int, 
        x: torch.Tensor, 
        attn_mask_inv: torch.Tensor, 
        seq_len: int
    ) -> torch.Tensor:
        """Pool a sequence into a single vector using learned-query attention.

        Conceptually, this performs a single "cross-attention" step where a learned
        query vector attends over the sequence:
          - query: a trainable vector shared across all examples
          - keys/values: the sequence representations `x`

        This yields a content-aware weighted average of the sequence. The mask is
        applied so padding positions do not receive attention mass.

        Args:
            B: Batch size.
            x: Sequence representations `[B, seq_len, hidden_dim]`.
            attn_mask_inv: Inverted key-padding mask `[B, seq_len]` where True means
                "ignore this position" (PyTorch transformer convention).
            seq_len: Sequence length (used for a safe fallback in the all-masked case).

        Returns:
            Attention-pooled representations `[B, hidden_dim]`.
        """
        # Expand the shared learned query to one per batch element.
        query = self.attention_query.expand(B, -1, -1)  # [B, 1, hidden]

        # Compute raw dot-product scores between the query and each sequence element.
        attn_scores = torch.bmm(query, x.transpose(1, 2))  # [B, 1, seq]

        # TorchScript / backend compatibility note:
        # Avoid `-inf` + `softmax` + `nan_to_num` patterns that can create
        # version-sensitive graphs. Instead, use a finite large negative for
        # masked positions and explicitly renormalize.
        neg_inf = -1.0e9
        scores = attn_scores.masked_fill(attn_mask_inv.unsqueeze(1), neg_inf)
        max_scores = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - max_scores)
        exp_scores = exp_scores.masked_fill(attn_mask_inv.unsqueeze(1), 0.0)
        denom = exp_scores.sum(dim=-1, keepdim=True).clamp(min=1.0)
        attn_weights = exp_scores / denom

        # Weighted sum of values (the same `x` sequence) -> one vector per example.
        attention_pooled = torch.bmm(attn_weights, x).squeeze(1)  # [B, hidden]
        return attention_pooled
    
    def _forward_mean_pooled(
        self, 
        x: torch.Tensor, 
        history_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute a masked mean over the sequence dimension.

        Mean pooling acts as a robust "coverage" baseline: every valid item
        contributes equally, and padding contributes zero. Callers that want a
        content-only mean should pass projected embeddings before positional
        embeddings are added.

        Args:
            x: Sequence representations `[B, seq_len, hidden_dim]`.
            history_mask: Validity mask `[B, seq_len]` where True means valid.

        Returns:
            Mean-pooled representations `[B, hidden_dim]`.
        """
        # Expand the mask to match `[B, seq_len, hidden_dim]` for elementwise multiply.
        mask_expanded = history_mask.unsqueeze(-1).float()
        sum_x = (x * mask_expanded).sum(dim=1)

        # `count` is the number of valid (non-padding) positions. Clamp to 1 to
        # avoid division-by-zero for empty histories.
        count = mask_expanded.sum(dim=1).clamp(min=1)
        mean_pooled = sum_x / count
        return mean_pooled
    
    def _forward_target_user_embedding(
        self,
        target_user_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.training and self.target_user_unknown_dropout_rate > 0.0:
            eligible = target_user_idx > self.target_user_unk_idx
            if torch.any(eligible):
                dropout_mask = torch.rand(
                    target_user_idx.shape,
                    device=target_user_idx.device,
                ) < self.target_user_unknown_dropout_rate
                target_user_idx = torch.where(
                    eligible & dropout_mask,
                    torch.full_like(target_user_idx, self.target_user_unk_idx),
                    target_user_idx,
                )
        return self.target_user_embedding_table(target_user_idx)  # [batch, target_user_embedding_dim]

    @abstractmethod
    def forward(
        self,
        history_embeddings: torch.Tensor,  # [B, seq_len, input_dim]
        history_mask: torch.Tensor,  # [B, seq_len] True = valid
        target_user_idx: Optional[torch.Tensor] = None, # [B]
    ) -> torch.Tensor:
        """Encode a padded history sequence into a fixed-size representation.

        Subclasses should:
          1) call `_forward_up_to_pos_embed()` to obtain position-encoded `x`
             and content-only `mean_pooled`
          2) optionally apply a sequence model (e.g. TransformerEncoder) using the
             appropriate key-padding mask convention
          3) pool the position-aware sequence, often using `_forward_attention_pooled()`
          4) project pooled features to `[B, output_dim]`

        Args:
            history_embeddings: Padded history `[B, seq_len, input_dim]`.
            history_mask: Optional validity mask `[B, seq_len]` where True = valid.

        Returns:
            Encoded user representations `[B, output_dim]`.
        """
        ...


class _TS_TransformerBlock(nn.Module):
    """TorchScript-friendly transformer encoder block.

    Implements multi-head self-attention + FFN using basic tensor ops
    (matmul/softmax) to avoid `aten::scaled_dot_product_attention`, which can be
    missing in some libtorch builds shipped with serving stacks (e.g. Triton).
    """

    def __init__(self, hidden_dim: int, num_attention_heads: int, dropout_rate: float):
        super().__init__()
        if hidden_dim % num_attention_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_attention_heads ({num_attention_heads})"
            )
        self.num_attention_heads = int(num_attention_heads)
        self.head_dim = int(hidden_dim // num_attention_heads)

        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout_rate)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.ff1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ff2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.ff_dropout = nn.Dropout(dropout_rate)
        self.resid_dropout1 = nn.Dropout(dropout_rate)
        self.resid_dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, attn_mask_inv: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], attn_mask_inv: [B, T] where True means "ignore" (key padding mask)
        B, T, D = x.shape
        H = self.num_attention_heads
        Hd = self.head_dim
        scale = 1.0 / (float(Hd) ** 0.5)
        neg_inf = -1.0e9

        qkv = self.qkv_proj(x)  # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, H, T, Hd]
        q = q.view(B, T, H, Hd).permute(0, 2, 1, 3)
        k = k.view(B, T, H, Hd).permute(0, 2, 1, 3)
        v = v.view(B, T, H, Hd).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, T, T]
        scores = scores.masked_fill(attn_mask_inv[:, None, None, :], neg_inf)

        # Stable masked softmax (no -inf / nan_to_num).
        max_scores = scores.max(dim=-1, keepdim=True).values
        exp_scores = torch.exp(scores - max_scores)
        exp_scores = exp_scores.masked_fill(attn_mask_inv[:, None, None, :], 0.0)
        denom = exp_scores.sum(dim=-1, keepdim=True).clamp(min=1.0)
        attn = exp_scores / denom
        attn = self.attn_dropout(attn)

        ctx = torch.matmul(attn, v)  # [B, H, T, Hd]
        ctx = ctx.permute(0, 2, 1, 3).contiguous().view(B, T, D)  # [B, T, D]
        attn_out = self.out_proj(ctx)

        x = x + self.resid_dropout1(attn_out)
        x = self.norm1(x)

        ff = self.ff2(self.ff_dropout(F.gelu(self.ff1(x))))
        x = x + self.resid_dropout2(ff)
        x = self.norm2(x)
        return x


class TransformerDualPoolingEncoder(BaseAttentionEncoder):
    """Learned transformer-based user history encoder with dual pooling.
    
    This is a TRAINABLE NEURAL NETWORK with learnable parameters that encodes
    variable-length engagement history into a fixed user representation. Unlike
    hand-crafted SUMMARIZERS (MeanSummarizer, EMASummarizer), this encoder LEARNS
    optimal aggregation patterns from data during training via backpropagation.
    
    Uses transformer self-attention + dual pooling strategy to discover which
    historical engagement patterns are most predictive, rather than relying on
    predetermined statistical rules.
    
    Architecture:
        1. Input projection: Raw embeddings -> hidden_dim
        2. Content-only mean pooling: Robust coverage baseline
        3. Positional encoding: Explicit recency signal (position 0 = most recent)
        4. Transformer encoder: Multi-head self-attention captures inter-post relationships
        5. Learned-query attention pooling over the transformer output
        6. Output projection: Combined pooled features -> output_dim

    Dual pooling:
        - Learned-query attention pooling is adaptive, content-aware, and
          recency-aware through positional embeddings and transformer context.
        - Mean pooling is computed before positional embeddings are added, so it
          remains a content-only coverage signal.
    
    Design rationale:
        - Self-attention allows the model to identify complementary/contradictory
          preferences within a user's history
        - Dual pooling combines adaptive focus (attention) with comprehensive,
          content-only coverage (mean), providing robustness
        - Flipped positional embeddings ensure position 0 = most recent, matching
          the recency-biased intuition of hand-crafted summarizers
    
    Complexity:
        - Parameters: ~2M (typical settings) - ALL TRAINABLE
        - Forward pass: O(num_layers * seq_len^2 * hidden_dim) due to self-attention
        - Memory: Scales quadratically with sequence length
    
    Used by:
        - TwoTowerModel (user_encoder_type="full_transformer")
        - MLPModel (user_encoder_type="full_transformer")
    
    Args:
        input_dim: Dimensionality of input post embeddings
        hidden_dim: Internal transformer hidden size
        output_dim: Final user representation size
        num_attention_heads: Number of attention heads per layer
        num_attention_layers: Depth of transformer stack
        max_seq_len: Maximum history length for positional embeddings
        dropout_rate: Dropout probability for regularization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_attention_heads: int,
        num_attention_layers: int,
        max_seq_len: int,
        dropout_rate: float,
        use_target_user_embedding_table: bool,
        target_user_table_num_rows: Optional[int] = None,
        target_user_embedding_dim: Optional[int] = None,
        target_user_unknown_dropout_rate: float = 0.0,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            output_dim,
            max_seq_len,
            dropout_rate,
            use_target_user_embedding_table,
            target_user_table_num_rows,
            target_user_embedding_dim,
            target_user_unknown_dropout_rate,
        )
        if hidden_dim % num_attention_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_attention_heads ({num_attention_heads})"
            )

        # Custom transformer stack (TorchScript / Triton-friendly).
        # Avoid `nn.TransformerEncoderLayer` / `nn.MultiheadAttention` because
        # some serving stacks ship libtorch builds that can't load graphs
        # containing `aten::scaled_dot_product_attention`.
        self.transformer_layers = nn.ModuleList(
            [
                _TS_TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_attention_heads=num_attention_heads,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_attention_layers)
            ]
        )

        self._init_weights()

    def forward(
        self,
        history_embeddings: torch.Tensor,  # [B, seq_len, input_dim]
        history_mask: torch.Tensor,  # [B, seq_len] True = valid
        target_user_idx: Optional[torch.Tensor] = None, # [B]
    ) -> torch.Tensor:
        """Encode user history into fixed-size representation.
        
        Args:
            history_embeddings: Padded history sequences [batch, seq_len, input_dim]
            history_mask: Boolean mask [batch, seq_len], True = valid position
            
        Returns:
            User representations [batch, output_dim]
        """
        B, seq_len, history_mask, x, mean_pooled = self._forward_up_to_pos_embed(history_embeddings, history_mask)

        # Pass through custom transformer layers.
        # `attn_mask_inv` uses PyTorch convention: True = ignore.
        attn_mask_inv = ~history_mask
        for layer in self.transformer_layers:
            x = layer(x, attn_mask_inv)

        # ─── Attention-weighted pooling ───
        # Use learned query to compute content-aware weights over sequence
        attention_pooled = self._forward_attention_pooled(B, x, attn_mask_inv, seq_len)

        # Concatenate both pooling strategies and project to output dimension
        combined = torch.cat([attention_pooled, mean_pooled], dim=-1)

        if self.use_target_user_embedding_table:
            if target_user_idx is None:
                raise RuntimeError("target_user_idx is required when use_target_user_embedding_table is True")
            target_user_embedding = self._forward_target_user_embedding(target_user_idx)
            combined = torch.cat([attention_pooled, mean_pooled, target_user_embedding], dim=-1)

        return self.output_projection(combined)


class CrossAttentionPoolingEncoder(BaseAttentionEncoder):
    """Learned efficient user history encoder using single-query cross-attention pooling.
    
    This is a TRAINABLE NEURAL NETWORK with learnable parameters, designed as a
    faster alternative to TransformerDualPoolingEncoder. Unlike hand-crafted
    SUMMARIZERS (MeanSummarizer, EMASummarizer) which use predetermined statistical
    operations, this encoder LEARNS optimal aggregation patterns from data during
    training via backpropagation.
    
    Designed for production ranking scenarios where latency and throughput matter.
    
    Key difference from TransformerDualPoolingEncoder:
        **NO SELF-ATTENTION** - removes the expensive O(seq_len²) transformer layers
        that capture inter-post relationships. Instead, relies on:
        - Input projection + positional encoding to embed history
        - Single learned-query cross-attention to aggregate
        - Content-only mean pooling for stability
    
    This trades off some modeling capacity (can't capture complex inter-post
    dependencies) for efficiency gains.
    
    Architecture:
        1. Input projection: Raw embeddings -> hidden_dim
        2. Content-only mean pooling: Robust baseline aggregation
        3. Positional encoding: Explicit recency signal
        4. Cross-attention pooling: Single learned query attends to position-aware history
        5. Output projection: Combined features -> output_dim
    
    Complexity:
        - Parameters: ~150K (typical) - significantly fewer than TransformerDualPoolingEncoder, ALL TRAINABLE
        - Forward pass: O(seq_len * hidden_dim) - linear in sequence length
        - Memory: Scales linearly with sequence length

    Used by:
        - TwoTowerModel (user_encoder_type="cross_attention")
        - MLPModel (user_encoder_type="cross_attention")
    
    Args:
        input_dim: Dimensionality of input post embeddings
        hidden_dim: Internal hidden size
        output_dim: Final user representation size
        max_seq_len: Maximum history length for positional embeddings
        dropout_rate: Dropout probability for regularization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_seq_len: int,
        dropout_rate: float,use_target_user_embedding_table: bool,
        target_user_table_num_rows: Optional[int] = None,
        target_user_embedding_dim: Optional[int] = None,
        target_user_unknown_dropout_rate: float = 0.0,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            output_dim,
            max_seq_len,
            dropout_rate,
            use_target_user_embedding_table,
            target_user_table_num_rows,
            target_user_embedding_dim,
            target_user_unknown_dropout_rate,
        )
        self._init_weights()

    def forward(
        self,
        history_embeddings: torch.Tensor,  # [B, seq_len, input_dim]
        history_mask: torch.Tensor,  # [B, seq_len] True = valid
        target_user_idx: Optional[torch.Tensor] = None, # [B] 
    ) -> torch.Tensor:
        """Encode user history into fixed-size representation.
        
        Args:
            history_embeddings: Padded history sequences [batch, seq_len, input_dim]
            history_mask: Boolean mask [batch, seq_len], True = valid position
            
        Returns:
            User representations [batch, output_dim]
        """
        B, seq_len, history_mask, x, mean_pooled = self._forward_up_to_pos_embed(history_embeddings, history_mask)

        # ─── KEY DIFFERENCE: No TransformerEncoder here ───
        # We skip the expensive self-attention layers that capture inter-post
        # relationships. This is the primary source of speedup and parameter reduction.

        # ─── Cross-attention pooling ───
        # Learned query attends directly to the projected, position-encoded history
        attn_mask_inv = ~history_mask  # PyTorch convention: True = ignore
        attention_pooled = self._forward_attention_pooled(B, x, attn_mask_inv, seq_len)

        # Concatenate both pooling strategies and project to output dimension
        combined = torch.cat([attention_pooled, mean_pooled], dim=-1)

        if self.use_target_user_embedding_table:
            if target_user_idx is None:
                raise RuntimeError("target_user_idx is required when use_target_user_embedding_table is True")
            target_user_embedding = self._forward_target_user_embedding(target_user_idx)
            combined = torch.cat([attention_pooled, mean_pooled, target_user_embedding], dim=-1)

        return self.output_projection(combined)


# ---------------------------------------------------------------------------
# Shared data-loading helper
# ---------------------------------------------------------------------------

def load_training_data(
    context: Context,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, pl.DataFrame, pl.DataFrame, Optional[pl.DataFrame], int]:
    """Locate and load the three upstream pipeline artifacts needed for training.
    
    This function abstracts away the complexity of finding prior stage outputs,
    supporting both same-session pipeline runs (via context.artifacts) and
    multi-session workflows (via filesystem scanning).
    
    Resolution order for each artifact:
        1. context.artifacts - Set by pipeline when stages run in same session
        2. Filesystem scan of the canonical artifact store (context.artifacts_dir)
    
    The three required artifacts:
        1. embeddings_*.npy   : Memmap array of post embeddings from Stage 1
        2. target_posts_*.parquet : Train/val/holdout split assignments from Stage 2
        3. history_posts_*.parquet: User engagement history from Stage 3
    
    Args:
        context: Pipeline context with artifact tracking and configuration
        logger: Optional logger for progress reporting
    
    Returns:
        Tuple of (embeddings_mmap, target_posts_df, history_df, target_user_idx_mapping_df,embed_dim):
            - embeddings_mmap: Read-only numpy memmap [n_posts, D]
            - target_posts_df: Polars DataFrame with split, like_emb_idx, neg_emb_idx
            - history_df: Polars DataFrame with target_did, like_uri, prior_emb_indices
            - target_user_idx_mapping_df: Optional Polars DataFrame with Stage 2
              target_user_idx / target_user_train_count metadata
            - embed_dim: Integer embedding dimensionality D
    
    Raises:
        FileNotFoundError: If any required artifact cannot be located
        
    Example:
        >>> embeddings, targets, history, target_user_idx_mapping, dim = load_training_data(
        ...     context=context,
        ...     logger=logger
        ... )
        >>> print(f"Loaded {len(targets):,} target posts with {dim}-d embeddings")
    """
    if logger is None:
        logger = get_stage_logger("DATALOADERS")

    # --- 1. Embeddings memmap from 01_get_data ---
    # This is a read-only memory-mapped array that allows accessing post
    # embeddings without loading the entire matrix into RAM
    log_operation_start("Locate embeddings memmap", "DATALOADERS", logger)
    get_data_dir = _resolve_prior(context, stage_key="get_data", folder="01_get_data")
    emb_candidates = sorted(get_data_dir.glob("embeddings_*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not emb_candidates:
        raise FileNotFoundError(f"No embeddings_*.npy found under {get_data_dir}")
    embeddings_path = emb_candidates[0]
    embeddings_mmap: np.ndarray = np.load(str(embeddings_path), mmap_mode="r")
    embed_dim = embeddings_mmap.shape[1]
    logger.info(f"Loaded embeddings memmap: shape={embeddings_mmap.shape}, path={embeddings_path}")

    # --- 2. Target posts from 02_target_posts ---
    # Contains the train/val/holdout split assignments and negative sampling results
    log_operation_start("Locate target_posts", "DATALOADERS", logger)
    target_posts_dir = _resolve_prior(context, stage_key="target_posts", folder="02_target_posts")
    target_posts_df = load_parquet_from_prior(target_posts_dir, "target_posts_").collect()
    logger.info(f"Loaded target_posts: {len(target_posts_df):,} rows")

    try:
        target_user_idx_mapping_df = load_parquet_from_prior(target_posts_dir, "target_user_idx_").collect()
        logger.info(f"Loaded target_user_idx: {len(target_user_idx_mapping_df):,} rows")
    except FileNotFoundError:
        target_user_idx_mapping_df = None
        logger.info("No target_user_idx artifact found in target_posts stage output")

    # --- 3. User history from 03_user_history (or legacy 02_featurize) ---
    # Contains the (most-recent-first-ordered) list of post indices each user engaged with
    log_operation_start("Locate user_history", "DATALOADERS", logger)
    history_dir = _resolve_prior(context, stage_key="user_history", folder="03_user_history")
    history_df = load_parquet_from_prior(history_dir, "history_posts_").collect()
    logger.info(f"Loaded user_history: {len(history_df):,} rows")

    return embeddings_mmap, target_posts_df, history_df, target_user_idx_mapping_df, embed_dim


def _resolve_prior(
    context: Context,
    *,
    stage_key: str,
    folder: str,
) -> Path:
    """Resolve a prior stage output directory, trying context artifacts first,
    then a filesystem scan of the canonical artifact store."""
    # Try context artifacts first (same-session run)
    art_dir = context.get_artifact_dir(stage_key)
    if art_dir is not None and Path(art_dir).exists():
        return context.record_prior_input(folder, art_dir)
    # Filesystem scan (also records lineage for the active stage)
    return context.resolve_prior_output(folder, prior_path=context.prior_outputs.get(folder))


# ---------------------------------------------------------------------------
# Internal: shared filter + join used by both datasets and evaluation
# ---------------------------------------------------------------------------

def filter_split_and_join_history(
    target_posts_df: pl.DataFrame,
    history_df: pl.DataFrame,
    split: str,
) -> pl.DataFrame:
    """Filter target posts to a split and left-join with user history.

    This is the canonical implementation of the two-step operation that both
    the training dataloaders and the evaluation stage need:
      1. Keep only rows for the requested *split* that have a valid negative sample.
      2. Left-join with history to attach ``prior_emb_indices`` per row.

    Returns:
        Polars DataFrame with all columns from the filtered target posts plus
        ``prior_emb_indices`` from the history.
    """
    filtered = target_posts_df.filter(
        (pl.col("split") == split) & pl.col("neg_emb_idx").is_not_null()
    )
    return filtered.join(
        history_df.select(["target_did", "like_uri", "prior_emb_indices"]),
        on=["target_did", "like_uri"],
        how="left",
        maintain_order="left",
    )


# ---------------------------------------------------------------------------
# Internal: prepare the row-aligned index arrays shared by both datasets
# ---------------------------------------------------------------------------

def _prepare_split_data(
    target_posts_df: pl.DataFrame,
    history_df: pl.DataFrame,
    split: str,
    use_target_user_embedding_table: bool = False,
    target_user_idx_to_table_row: Optional[np.ndarray] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[str], List[str], List[str], Optional[np.ndarray]]:
    """Filter data to a single split and return aligned numpy arrays.
    
    This internal helper performs the core data preparation logic shared by both
    SummarizedEngagementDataset and SequenceEngagementDataset. It:
    1. Filters target_posts to the requested split (train/val/holdout)
    2. Drops rows with missing negative samples
    3. Joins with user history to get engagement sequences
    4. Converts to numpy arrays for fast indexing
    
    The returned arrays are **row-aligned**: position i in each array corresponds
    to the same user-post interaction. This alignment is critical for dataset
    __getitem__ to efficiently construct training samples.
    
    Args:
        target_posts_df: Full target posts DataFrame with all splits
        history_df: User engagement history DataFrame
        split: Split name to filter to ("train", "val", "holdout_unseen_users", or "holdout_seen_users")
        logger: Optional logger for progress reporting
    
    Returns:
        Tuple of (like_emb_idx, neg_emb_idx, prior_emb_indices_list, 
                  target_dids, like_uris, neg_uris, target_user_indices_list):
        
        - like_emb_idx: Indices into embeddings memmap for positive posts [N]
        - neg_emb_idx: Indices into embeddings memmap for negative posts [N]
        - prior_emb_indices_list: List of N variable-length arrays, each containing
                                  embedding indices for that user's history
                                  (most-recent-first, uint32)
        - target_dids: User IDs as string array [N]
        - like_uris: Liked post URIs as string array [N]
        - neg_uris: Non-liked post URIs as string array [N]
        - target_user_indices_list: Optional list of N integers mapping to user embedding table rows (if used)
        
        Where N = number of target posts in the requested split (after filtering).
    
    Note:
        Each target post row produces TWO training samples (positive + negative),
        so dataset length will be 2*N. This function returns N-length arrays that
        the datasets will expand to 2*N samples in their __getitem__ methods.
    """
    if logger is None:
        logger = get_stage_logger("DATALOADERS")

    joined = filter_split_and_join_history(target_posts_df, history_df, split)
    logger.info(f"  Split '{split}': {len(joined):,} target rows (after dropping null neg_emb_idx)")

    # Extract embedding indices for positive and negative posts
    like_emb_idx = joined["like_emb_idx"].to_numpy().astype(np.int64)
    neg_emb_idx = joined["neg_emb_idx"].to_numpy().astype(np.int64)
    target_dids = joined["target_did"].to_list()
    like_uris = joined["like_uri"].to_list()
    neg_uris = joined["neg_uri"].to_list()
    target_user_idx = None

    # Convert Polars List[UInt32] column to Python list of numpy arrays
    # This allows each user to have a different history length (variable-length)
    # while still supporting fast numpy indexing into the embeddings memmap
    prior_col = joined["prior_emb_indices"]
    prior_emb_indices_list: List[np.ndarray] = []

    if use_target_user_embedding_table and target_user_idx_to_table_row is None:
        raise ValueError("target_user_idx_to_table_row must be provided when author embeddings are enabled")

    if use_target_user_embedding_table:
        if 'target_user_idx' not in joined.columns:
            raise ValueError(f"target_posts_df must contain target_user_idx column when target_user embeddings are enabled")
        assert target_user_idx_to_table_row is not None
        target_user_idx = np.array([
            _map_raw_idx_to_table_row(raw_idx, target_user_idx_to_table_row)
            for raw_idx in joined["target_user_idx"].to_list()
        ], dtype=np.uint32)

    for row_val in prior_col.to_list():
        if row_val is None or len(row_val) == 0:
            # Users with no history get an empty array (will become zero vector
            # in SummarizedEngagementDataset or all-masked sequence in SequenceEngagementDataset)
            prior_emb_indices_list.append(np.array([], dtype=np.uint32))
        else:
            prior_emb_indices_list.append(np.array(row_val, dtype=np.uint32))

    return like_emb_idx, neg_emb_idx, prior_emb_indices_list, target_dids, like_uris, neg_uris, target_user_idx


# ---------------------------------------------------------------------------
# SummarizedEngagementDataset
# ---------------------------------------------------------------------------

class SummarizedEngagementDataset(Dataset):
    """Fixed-size feature vector dataset: [user_summary || post_embedding].
    
    This dataset represents user engagement history as FIXED-SIZE summary vectors
    computed by pluggable UserSummarizer strategies. It's designed for models that
    require consistent input dimensionality.
    
    ═══════════════════════════════════════════════════════════════════════════
    DATASET STRUCTURE
    ═══════════════════════════════════════════════════════════════════════════
    
    Each target post generates TWO training samples:
        - Positive sample (index 2*k):   [user_summary || liked_post_embedding] -> label=1
        - Negative sample (index 2*k+1): [user_summary || random_post_embedding] -> label=0
    
    This paired structure ensures balanced training and allows the model to learn
    discriminative features between engaged and non-engaged content.
    
    Sample indexing:
        len(dataset) = 2 * num_target_posts
        dataset[0]   = positive sample for target post 0
        dataset[1]   = negative sample for target post 0
        dataset[2]   = positive sample for target post 1
        ...
    
    ═══════════════════════════════════════════════════════════════════════════
    MEMORY & PERFORMANCE
    ═══════════════════════════════════════════════════════════════════════════
    
    **Pre-computation strategy**: All user summaries and post embeddings are
    materialized into contiguous float32 tensors at init time. This makes
    __getitem__ a pure in-memory index lookup with ZERO memmap I/O during
    training.
    
    Memory scales as ~3 * N * D * 4 bytes for float32 (user summaries + pos post
    embs + neg post embs). For example, N=178K and D=384 is ~0.8 GB.
    
    ═══════════════════════════════════════════════════════════════════════════
    
    Args:
        embeddings_mmap: Read-only numpy memmap of post embeddings [n_posts, D]
        target_posts_df: DataFrame with split, like_emb_idx, neg_emb_idx columns
        history_df: DataFrame with target_did, like_uri, prior_emb_indices columns
        split: Split to load ("train", "val", "holdout_unseen_users", or "holdout_seen_users")
        summarizer: UserSummarizer instance for aggregating engagement history
        embed_dim: Embedding dimensionality D
        logger: Optional logger for progress reporting
    
    Attributes:
        embed_dim: Embedding dimensionality
        target_dids: User IDs for each target post [N]
        like_uris: Post URIs for each target post [N]
    
    Returns (from __getitem__):
        Dictionary with keys:
            - "features": Concatenated [user_summary || post_emb] tensor [2*D]
            - "label": Binary label (1.0 for positive, 0.0 for negative)
            - "user_id": User ID string
            - "post_id": Post URI string (or "neg_uri" for negatives)
    """

    def __init__(
        self,
        embeddings_mmap: np.ndarray,
        target_posts_df: pl.DataFrame,
        history_df: pl.DataFrame,
        split: str,
        summarizer: UserSummarizer,
        embed_dim: int,
        logger: Optional[logging.Logger] = None,
    ):
        self.embed_dim = embed_dim

        # Prepare aligned arrays for the requested split
        (
            like_emb_idx,
            neg_emb_idx,
            prior_emb_indices,
            self.target_dids,
            self.like_uris,
            self.neg_uris,
            _,
        ) = _prepare_split_data(target_posts_df, history_df, split, logger=logger)

        self._n_rows = len(like_emb_idx)

        # ── Pre-compute user summaries [N, D] ────────────────────────
        # Apply the summarizer to each user's engagement history. This happens
        # once at init time so __getitem__ can be a pure lookup.
        if logger:
            logger.info(f"  Pre-computing user summaries for '{split}' ({self._n_rows:,} rows)…")
        user_summaries = np.zeros((self._n_rows, embed_dim), dtype=np.float32)
        for i, hist_indices in enumerate(prior_emb_indices):
            if len(hist_indices) > 0:
                # Fetch history embeddings from memmap and summarize
                hist_embs = embeddings_mmap[hist_indices]  # [seq, D]
                user_summaries[i] = summarizer.summarize(hist_embs)
            # else: Users with no history stay as zero vectors
        self._user_summaries = torch.from_numpy(user_summaries)

        # ── Pre-fetch post embeddings [N, D] for pos and neg ─────────
        # Materialize positive and negative post embeddings into contiguous tensors
        # for fast __getitem__ lookups. This is cheap compared to history sequences.
        pos_embs = np.array(embeddings_mmap[like_emb_idx], dtype=np.float32)
        neg_embs = np.array(embeddings_mmap[neg_emb_idx], dtype=np.float32)
        self._pos_post_embs = torch.from_numpy(pos_embs)
        self._neg_post_embs = torch.from_numpy(neg_embs)

        if logger:
            mem_mb = (user_summaries.nbytes + pos_embs.nbytes + neg_embs.nbytes) / (1024 * 1024)
            logger.info(
                f"  SummarizedEngagementDataset('{split}'): "
                f"{self._n_rows:,} rows -> {len(self):,} samples "
                f"(summarizer={type(summarizer).__name__}, "
                f"pre-computed {mem_mb:.1f} MB)"
            )

    def __len__(self) -> int:
        """Return total number of samples (2 per target post: positive + negative)."""
        return self._n_rows * 2

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training sample by index.
        
        Args:
            idx: Sample index in [0, 2*N). Even indices are positive samples,
                 odd indices are negative samples.
        
        Returns:
            Dictionary with:
                - "features": [2*D] concatenated [user_summary || post_embedding]
                - "label": 1.0 for positive, 0.0 for negative
                - "user_id": User identifier string
                - "post_id": Post URI (or "neg_uri" for negatives)
        """
        # Map dataset index to target post row and sample type
        row_idx = idx // 2  # Which target post
        is_positive = (idx % 2) == 0  # Even = positive, odd = negative

        # User summary is shared between positive and negative samples
        user_vec = self._user_summaries[row_idx]  # [D]
        
        # Select post embedding based on sample type
        if is_positive:
            post_vec = self._pos_post_embs[row_idx]  # [D]
            label = 1.0
            post_id = self.like_uris[row_idx]
        else:
            post_vec = self._neg_post_embs[row_idx]  # [D]
            label = 0.0
            post_id = self.neg_uris[row_idx]

        # Concatenate user summary and post embedding
        features = torch.cat([user_vec, post_vec])  # [2D]
        return {
            "features": features,
            "label": torch.tensor(label, dtype=torch.float32),
            "user_id": self.target_dids[row_idx],
            "post_id": post_id,
        }


# ---------------------------------------------------------------------------
# SequenceEngagementDataset
# ---------------------------------------------------------------------------

class SequenceEngagementDataset(Dataset):
    """Variable-length sequence dataset: padded history + mask + target post embedding.
    
    This dataset represents user engagement history as VARIABLE-LENGTH SEQUENCES
    with padding and masking. It preserves temporal structure and allows learned
    encoders (attention mechanisms) to discover optimal aggregation strategies.
    
    ═══════════════════════════════════════════════════════════════════════════
    DATASET STRUCTURE
    ═══════════════════════════════════════════════════════════════════════════
    
    Each target post generates TWO training samples:
        - Positive sample (index 2*k):   (history_seq, mask, liked_post_emb) -> label=1
        - Negative sample (index 2*k+1): (history_seq, mask, random_post_emb) -> label=0
    
    History sequences are:
        - Padded to max_history_len
        - Masked (True = valid position, False = padding)
        - Most-recent-first (index 0 = most recent engagement)
    
    Sample indexing:
        len(dataset) = 2 * num_target_posts
        dataset[0]   = positive sample for target post 0
        dataset[1]   = negative sample for target post 0
        ...
    
    ═══════════════════════════════════════════════════════════════════════════
    MEMORY & PERFORMANCE
    ═══════════════════════════════════════════════════════════════════════════
    
    **Hybrid pre-computation strategy**:
        ✓ Target post embeddings: Pre-computed into tensors (~560 MB for 178K samples)
        ✓ History sequences: Constructed on-the-fly from memmap during __getitem__
    
    Rationale:
        A fully materialized [N, max_seq, D] tensor would consume ~13 GB for typical
        parameters (N=178K, max_seq=50, D=384). By loading sequences on-demand, we
        keep memory footprint manageable while multi-worker DataLoaders pipeline the
        I/O to keep GPUs fed.
    
    ═══════════════════════════════════════════════════════════════════════════
    
    Args:
        embeddings_mmap: Read-only numpy memmap of post embeddings [n_posts, D]
        target_posts_df: DataFrame with split, like_emb_idx, neg_emb_idx columns
        history_df: DataFrame with target_did, like_uri, prior_emb_indices columns
        split: Split to load ("train", "val", "holdout_unseen_users", or "holdout_seen_users")
        max_history_len: Maximum sequence length for padding (truncate if longer)
        embed_dim: Embedding dimensionality D
        logger: Optional logger for progress reporting
    
    Attributes:
        embeddings: Reference to memmap for on-the-fly loading
        max_history_len: Maximum sequence length
        embed_dim: Embedding dimensionality
        prior_emb_indices: List of variable-length index arrays per target post
        target_dids: User IDs [N]
        like_uris: Post URIs [N]
    
    Returns (from __getitem__):
        Dictionary with keys:
            - "history_embeddings": Padded sequence [max_seq_len, D]
            - "history_mask": Boolean mask [max_seq_len], True = valid position
            - "target_post_embedding": Target post embedding [D]
            - "target_user_idx": Optional target user row for user embeddings
            - "label": Binary label (1.0 for positive, 0.0 for negative)
            - "user_id": User ID string
            - "post_id": Post URI string (or "neg_uri" for negatives)
    """

    def __init__(
        self,
        embeddings_mmap: np.ndarray,
        target_posts_df: pl.DataFrame,
        history_df: pl.DataFrame,
        split: str,
        max_history_len: int,
        embed_dim: int,
        use_target_user_embedding_table: bool = False,
        target_user_idx_to_table_row: Optional[np.ndarray] = None,
        logger: Optional[logging.Logger] = None,
    ):
        # Store memmap reference for on-the-fly sequence loading
        self.embeddings = embeddings_mmap
        self.max_history_len = max_history_len
        self.embed_dim = embed_dim

        # Prepare aligned arrays for the requested split
        (
            like_emb_idx,
            neg_emb_idx,
            self.prior_emb_indices,
            self.target_dids,
            self.like_uris,
            self.neg_uris,
            self.target_user_indices,
        ) = _prepare_split_data(
            target_posts_df,
            history_df,
            split,
            use_target_user_embedding_table,
            target_user_idx_to_table_row,
            logger
        )

        self._n_rows = len(like_emb_idx)

        # ── Pre-fetch post embeddings [N, D] for pos and neg (cheap: ~560 MB) ──
        # Target post embeddings are small enough to pre-compute, avoiding repeated
        # memmap lookups for the same posts during training
        pos_embs = np.array(embeddings_mmap[like_emb_idx], dtype=np.float32)
        neg_embs = np.array(embeddings_mmap[neg_emb_idx], dtype=np.float32)
        self._pos_post_embs = torch.from_numpy(pos_embs)
        self._neg_post_embs = torch.from_numpy(neg_embs)

        if logger:
            mem_mb = (pos_embs.nbytes + neg_embs.nbytes) / (1024 * 1024)
            # Estimate what the full pre-compute would have cost
            full_gb = (self._n_rows * max_history_len * embed_dim * 4) / (1024 ** 3)
            logger.info(
                f"  SequenceEngagementDataset('{split}'): "
                f"{self._n_rows:,} rows -> {len(self):,} samples "
                f"(max_history_len={max_history_len}, "
                f"post embs pre-computed {mem_mb:.1f} MB, "
                f"history sequences on-the-fly via workers "
                f"[would be {full_gb:.1f} GB if materialized])"
            )

    def __len__(self) -> int:
        """Return total number of samples (2 per target post: positive + negative)."""
        return self._n_rows * 2

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single training sample by index.
        
        This method performs on-the-fly loading of history sequences from the
        memmap. When using multi-worker DataLoaders, this I/O happens in parallel
        worker processes, keeping the main training loop fed.
        
        Args:
            idx: Sample index in [0, 2*N). Even indices are positive samples,
                 odd indices are negative samples.
        
        Returns:
            Dictionary with:
                - "history_embeddings": [max_seq_len, D] padded/truncated history
                - "history_mask": [max_seq_len] boolean, True = valid position
                - "target_post_embedding": [D] target post embedding
                - "target_post_embedding": Target post embedding [D]
                - "target_user_idx": Optional target user row for user embeddings
                - "label": 1.0 for positive, 0.0 for negative
                - "user_id": User identifier string
                - "post_id": Post URI (or "neg_uri" for negatives)
        """
        # Map dataset index to target post row and sample type
        row_idx = idx // 2
        is_positive = (idx % 2) == 0

        # --- Load and pad/truncate history sequence from memmap ---
        # This is the key difference from SummarizedEngagementDataset: we load
        # the raw sequence here rather than using a pre-computed summary
        hist_indices = self.prior_emb_indices[row_idx]
        hist_embeddings = self.embeddings[hist_indices]
        padded, mask = get_padded_embedding_history_and_mask(hist_embeddings, self.max_history_len, self.embed_dim)

        # --- Select target post embedding (pre-computed) ---
        if is_positive:
            post_vec = self._pos_post_embs[row_idx]  # [D]
            label = 1.0
            post_id = self.like_uris[row_idx]
        else:
            post_vec = self._neg_post_embs[row_idx]  # [D]
            label = 0.0
            post_id = self.neg_uris[row_idx]

        output = {
            "history_embeddings": torch.from_numpy(padded),  # [max_seq, D]
            "history_mask": torch.from_numpy(mask),  # [max_seq]
            "target_post_embedding": post_vec,
            "label": torch.tensor(label, dtype=torch.float32),
            "user_id": self.target_dids[row_idx],
            "post_id": post_id,
        }
        if self.target_user_indices is not None:
            output["target_user_idx"] = torch.tensor(
                self.target_user_indices[row_idx], dtype=torch.long
            )
        return output
    


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    holdout_dataset: Optional[Dataset] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
):
    """Create PyTorch DataLoaders for training, validation, and optionally holdout sets.
    
    Configures efficient data loading with multi-worker parallelism and GPU pinning.
    
    DataLoader configuration rationale:
        - num_workers > 0: Parallel data loading prevents GPU starvation
        - pin_memory: Speeds up CPU->GPU transfer by using page-locked memory
        - persistent_workers: Avoids worker process respawn overhead between epochs
        - prefetch_factor: Workers pre-load batches to hide data loading latency
        - drop_last=True for training: Ensures consistent batch sizes for BatchNorm
        
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Number of samples per batch
        holdout_dataset: Optional holdout dataset for final evaluation
        num_workers: Number of parallel data loading workers (0 = main process only)
        pin_memory: Use pinned (page-locked) memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch per worker
    
    Returns:
        Tuple of (train_loader, val_loader, holdout_loader).
        holdout_loader is None if holdout_dataset is not provided.
    
    Note:
        With SummarizedEngagementDataset (pre-computed tensors), workers just do
        index lookups and collation, so even a few workers eliminate CPU bottlenecks.
        With SequenceEngagementDataset (on-the-fly memmap loading), workers pipeline
        the I/O to keep GPUs fed during training.
    """
    from torch.utils.data import DataLoader
    
    # Base worker configuration
    worker_kw: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    # Add worker-specific options only when using multiple workers
    if num_workers > 0:
        worker_kw.update(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    # Create DataLoaders with appropriate settings for each split
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Shuffle for stochastic training
        drop_last=True,  # Drop incomplete final batch for BatchNorm stability
        **worker_kw
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # No shuffle for validation (deterministic evaluation)
        **worker_kw
    )
    holdout_loader = DataLoader(
        holdout_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        **worker_kw
    ) if holdout_dataset else None
    return train_loader, val_loader, holdout_loader
