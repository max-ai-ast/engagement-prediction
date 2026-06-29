#!/usr/bin/env python3

"""
Shared dataloaders and user-encoder building blocks for engagement prediction.

The active training path is bucketed matrix training. Each batch contains
positive likes from one hour bucket plus candidate posts from that same bucket,
so the trainer can score the full user x post matrix in one forward pass.

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

MAIN COMPONENTS
═══════════════════════════════════════════════════════════════════════════════

Datasets:
    BucketedEngagementDataset    -- User histories + same-hour candidate posts

Hand-Crafted Summarizers (deterministic, no learnable parameters):
    UserSummarizer               -- Abstract base class
    MeanSummarizer               -- Arithmetic mean
    EMASummarizer                -- Exponential moving average with recency bias
    LinearRecencySummarizer      -- Linear recency weighting

Learned Encoders (trainable neural networks):
    TransformerDualPoolingEncoder   -- Full transformer self-attention + dual pooling
    CrossAttentionPoolingEncoder    -- Efficient single-query cross-attention pooling

Utilities:
    load_bucketed_training_data() -- Locates and loads upstream pipeline artifacts
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from utils.pipeline.core import Context
from utils.helpers import (
    get_stage_logger,
    load_parquet_from_prior,
    log_operation_start,
    validate_dataframe_schema,
)
from shared.input_data_helpers import (
    get_padded_embedding_history_and_mask,
    get_padded_author_indices,
    get_padded_history_time_deltas,
    AUTHOR_UNK_IDX,
)


def _author_idx_or_unk(author_idx: Any) -> int:
    """Return the Stage 1 author_idx as an embedding-table row, mapping nulls to UNK."""
    if author_idx is None:
        return AUTHOR_UNK_IDX
    try:
        if author_idx != author_idx:  # NaN
            return AUTHOR_UNK_IDX
    except TypeError:
        pass
    return int(author_idx)


def get_author_table_num_rows(
    author_idx_mapping_df: pl.DataFrame,
) -> int:
    """Return the number of rows needed for the Stage 1 author embedding table."""
    required_cols = {"author_idx"}
    missing_cols = required_cols.difference(author_idx_mapping_df.columns)
    if missing_cols:
        missing = ", ".join(sorted(missing_cols))
        raise ValueError(f"author_idx_mapping_df is missing required columns: {missing}")

    if len(author_idx_mapping_df) == 0:
        return 2

    max_author_idx_value = author_idx_mapping_df.select(pl.col("author_idx").max()).item()
    if max_author_idx_value is None:
        raise ValueError("author_idx column must contain at least one non-null value")
    max_author_idx = int(max_author_idx_value)
    if max_author_idx < AUTHOR_UNK_IDX:
        raise ValueError("author_idx values must reserve rows 0 and 1 for PAD/UNK")
    return max(max_author_idx + 1, 2)


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

    def forward(self, history_embeddings: torch.Tensor, history_mask: Optional[torch.Tensor]) -> torch.Tensor:
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

        # Project concatenated dual-pooled features to final output dimension
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 2x because we concatenate two pooling outputs
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

    @abstractmethod
    def forward(
        self,
        history_embeddings: torch.Tensor,  # [B, seq_len, input_dim]
        history_mask: torch.Tensor,  # [B, seq_len] True = valid
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
    ):
        super().__init__(input_dim, hidden_dim, output_dim, max_seq_len, dropout_rate)
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
        dropout_rate: float,
    ):
        super().__init__(input_dim, hidden_dim, output_dim, max_seq_len, dropout_rate)
        self._init_weights()

    def forward(
        self,
        history_embeddings: torch.Tensor,  # [B, seq_len, input_dim]
        history_mask: torch.Tensor,  # [B, seq_len] True = valid
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
        return self.output_projection(combined)


# ---------------------------------------------------------------------------
# Shared data-loading helper
# ---------------------------------------------------------------------------

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
# Bucketed two-tower data path
# ---------------------------------------------------------------------------

def load_bucketed_training_data(
    context: Context,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, pl.DataFrame, pl.DataFrame, pl.DataFrame, Optional[pl.DataFrame], int]:
    """Locate and load artifacts for bucketed two-tower training."""
    if logger is None:
        logger = get_stage_logger("DATALOADERS")

    log_operation_start("Locate embeddings memmap", "DATALOADERS", logger)
    get_data_dir = _resolve_prior(context, stage_key="get_data", folder="01_get_data")
    emb_candidates = sorted(get_data_dir.glob("embeddings_*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not emb_candidates:
        raise FileNotFoundError(f"No embeddings_*.npy found under {get_data_dir}")
    embeddings_path = emb_candidates[0]
    embeddings_mmap: np.ndarray = np.load(str(embeddings_path), mmap_mode="r")
    embed_dim = int(embeddings_mmap.shape[1])
    logger.info(f"Loaded embeddings memmap: shape={embeddings_mmap.shape}, path={embeddings_path}")

    log_operation_start("Locate likes_core", "DATALOADERS", logger)
    likes_core_df = load_parquet_from_prior(get_data_dir, "likes_core_").collect()
    logger.info(f"Loaded likes_core: {len(likes_core_df):,} rows")

    log_operation_start("Locate posts_core", "DATALOADERS", logger)
    posts_core_df = load_parquet_from_prior(get_data_dir, "posts_core_").collect()
    logger.info(f"Loaded posts_core: {len(posts_core_df):,} rows")

    try:
        author_idx_mapping_df = load_parquet_from_prior(get_data_dir, "author_idx_").collect()
        logger.info(f"Loaded author_idx: {len(author_idx_mapping_df):,} rows")
    except FileNotFoundError:
        author_idx_mapping_df = None
        logger.info("No author_idx artifact found in get_data output")

    log_operation_start("Locate user_history", "DATALOADERS", logger)
    history_dir = _resolve_prior(context, stage_key="user_history", folder="02_user_history")
    history_df = load_parquet_from_prior(history_dir, "history_posts_").collect()
    logger.info(f"Loaded user_history: {len(history_df):,} rows")

    return embeddings_mmap, likes_core_df, posts_core_df, history_df, author_idx_mapping_df, embed_dim


def _post_split_window_for_like_split(split: str) -> str:
    if split == "val_unseen_users":
        return "val"
    if split.startswith("holdout"):
        return "holdout"
    return split


def _list_to_int_array(value: Any) -> np.ndarray:
    if value is None or len(value) == 0:
        return np.array([], dtype=np.int64)
    return np.array(value, dtype=np.int64)


def _list_to_float_array(value: Any) -> np.ndarray:
    if value is None or len(value) == 0:
        return np.array([], dtype=np.float32)
    return np.array(value, dtype=np.float32)


def _author_idx_list_to_table_rows(
    author_indices: Any,
) -> np.ndarray:
    if author_indices is None:
        return np.array([], dtype=np.uint32)
    return np.array([
        _author_idx_or_unk(author_idx)
        for author_idx in author_indices
    ], dtype=np.uint32)


class BucketedEngagementDataset(Dataset):
    """User-hour positives grouped by hour bucket with same-hour candidate posts."""

    def __init__(
        self,
        embeddings_mmap: np.ndarray,
        likes_core_df: pl.DataFrame,
        posts_core_df: pl.DataFrame,
        history_df: pl.DataFrame,
        split: str,
        max_history_len: int,
        embed_dim: int,
        use_author_embedding_table: bool = False,
        bst_additional_batch_negatives: Optional[int] = None,
        seed: int = 0,
        logger: Optional[logging.Logger] = None,
    ):
        if max_history_len <= 0:
            raise ValueError("max_history_len must be positive")
        if bst_additional_batch_negatives is not None and bst_additional_batch_negatives <= 0:
            raise ValueError("bst_additional_batch_negatives must be positive when provided")
        self.embeddings = embeddings_mmap
        self.split = str(split)
        self.max_history_len = int(max_history_len)
        self.embed_dim = int(embed_dim)
        self.use_author_embedding_table = bool(use_author_embedding_table)
        self.bst_additional_batch_negatives = int(bst_additional_batch_negatives) if bst_additional_batch_negatives is not None else None
        self.seed = int(seed)

        likes_columns = ["did", "subject_uri", "split", "like_hour_bucket", "emb_idx"]
        posts_columns = ["at_uri", "in_random_sample", "negative_hour_bucket", "split_window", "emb_idx"]
        history_columns = ["did", "like_hour_bucket", "prior_emb_indices"]
        self.has_history_time_deltas = "prior_like_age_hours_at_bucket_start" in history_df.columns
        if self.has_history_time_deltas:
            history_columns.append("prior_like_age_hours_at_bucket_start")
        elif logger:
            logger.warning(
                "BucketedEngagementDataset history input is missing "
                "prior_like_age_hours_at_bucket_start; emitting zero history_time_deltas_hours"
            )
        if self.use_author_embedding_table:
            likes_columns.append("author_idx")
            posts_columns.append("author_idx")
            history_columns.append("prior_author_indices")

        validate_dataframe_schema(
            likes_core_df,
            dict.fromkeys(likes_columns, None),
        )
        validate_dataframe_schema(
            posts_core_df,
            dict.fromkeys(posts_columns, None),
        )
        validate_dataframe_schema(
            history_df,
            dict.fromkeys(history_columns, None),
        )

        like_ordered_df = (
            likes_core_df
            .filter(pl.col("split") == self.split)
            .with_row_index(name="_like_order")
        )
        agg_exprs = [
            pl.col("subject_uri").sort_by("_like_order").alias("liked_post_ids"),
            pl.col("emb_idx").sort_by("_like_order").alias("liked_post_emb_indices"),
            pl.col("_like_order").min().alias("_first_like_order"),
        ]
        if self.use_author_embedding_table:
            agg_exprs.append(pl.col("author_idx").sort_by("_like_order").alias("liked_post_author_indices"))
        
        user_hour_df = (
            like_ordered_df
            .group_by(["did", "like_hour_bucket"])
            .agg(agg_exprs)
            .sort("_first_like_order")
        )

        joined = (
            user_hour_df
            .join(
                history_df.select(history_columns),
                on=["did", "like_hour_bucket"],
                how="left",
                maintain_order="left",
            )
        )
        if logger:
            logger.info(f"  BucketedEngagementDataset('{self.split}'): {len(joined):,} user-hour rows")

        self.user_ids = joined["did"].to_list()
        self.like_hour_buckets = joined["like_hour_bucket"].to_list()
        self.liked_post_ids = joined["liked_post_ids"].to_list()
        self.liked_post_emb_indices = [
            _list_to_int_array(value)
            for value in joined["liked_post_emb_indices"].to_list()
        ]
        self.prior_emb_indices = [
            _list_to_int_array(value)
            for value in joined["prior_emb_indices"].to_list()
        ]
        if self.has_history_time_deltas:
            self.prior_like_age_hours_at_bucket_start = [
                _list_to_float_array(value)
                for value in joined["prior_like_age_hours_at_bucket_start"].to_list()
            ]
        else:
            self.prior_like_age_hours_at_bucket_start = [
                np.array([], dtype=np.float32)
                for _ in self.prior_emb_indices
            ]

        self.liked_post_author_indices: Optional[List[np.ndarray]] = None
        self.prior_author_indices: Optional[List[np.ndarray]] = None
        if self.use_author_embedding_table:
            self.liked_post_author_indices = [
                _author_idx_list_to_table_rows(value)
                for value in joined["liked_post_author_indices"].to_list()
            ]
            self.prior_author_indices = [
                _author_idx_list_to_table_rows(value)
                for value in joined["prior_author_indices"].to_list()
            ]

        self.row_indices_by_bucket: Dict[Any, List[int]] = {}
        for row_idx, bucket in enumerate(self.like_hour_buckets):
            self.row_indices_by_bucket.setdefault(bucket, []).append(row_idx)

        post_split_window = _post_split_window_for_like_split(self.split)
        sampled_posts_df = posts_core_df.filter(
            (pl.col("split_window") == post_split_window)
            & pl.col("in_random_sample")
            & pl.col("negative_hour_bucket").is_not_null()
        )
        self.sampled_posts_by_bucket: Dict[Any, List[Dict[str, Any]]] = {}
        for row in sampled_posts_df.iter_rows(named=True):
            author_idx = _author_idx_or_unk(row.get("author_idx")) if self.use_author_embedding_table else None
            self.sampled_posts_by_bucket.setdefault(row["negative_hour_bucket"], []).append({
                "post_id": row["at_uri"],
                "emb_idx": int(row["emb_idx"]),
                "author_idx": author_idx,
            })

    def __len__(self) -> int:
        return len(self.user_ids)

    def __getitem__(self, idx: Any) -> Dict[str, Any]:
        if isinstance(idx, tuple):
            row_idx, epoch = idx
        else:
            row_idx = idx
            epoch = 0
        row_idx = int(row_idx)
        return {
            "row_idx": row_idx,
            "bucket": self.like_hour_buckets[row_idx],
            "user_id": self.user_ids[row_idx],
            "epoch": int(epoch),
        }

    def _padded_history_for_row(self, row_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        hist_indices = self.prior_emb_indices[row_idx]
        hist_embeddings = self.embeddings[hist_indices]
        padded, mask = get_padded_embedding_history_and_mask(hist_embeddings, self.max_history_len, self.embed_dim)
        return torch.from_numpy(padded), torch.from_numpy(mask)

    def _padded_author_history_for_row(self, row_idx: int) -> torch.Tensor:
        if self.prior_author_indices is None:
            raise ValueError("prior_author_indices must be available when author embeddings are enabled")
        mapped_author_indices = self.prior_author_indices[row_idx]
        padded = get_padded_author_indices(mapped_author_indices, self.max_history_len)
        return torch.from_numpy(padded)

    def _padded_time_deltas_for_row(self, row_idx: int) -> torch.Tensor:
        deltas = self.prior_like_age_hours_at_bucket_start[row_idx]
        padded = get_padded_history_time_deltas(deltas, self.max_history_len)
        return torch.from_numpy(padded)

    def _sample_candidate_posts_for_batch(
        self,
        row_indices: List[int],
        bucket: Any,
        epoch: int,
        candidate_to_idx: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        sampled_posts = [
            post
            for post in self.sampled_posts_by_bucket.get(bucket, [])
            if post["post_id"] not in candidate_to_idx
        ]
        if self.bst_additional_batch_negatives is None:
            return sampled_posts
        if len(sampled_posts) <= self.bst_additional_batch_negatives:
            return sampled_posts

        sorted_row_indices = sorted(int(row_idx) for row_idx in row_indices)
        row_seed = sum((pos + 1) * (row_idx + 1) for pos, row_idx in enumerate(sorted_row_indices))
        rng = np.random.default_rng(self.seed + int(epoch) * max(len(self.user_ids), 1) + row_seed)
        selected_indices = sorted(rng.choice(len(sampled_posts), size=self.bst_additional_batch_negatives, replace=False).tolist())
        return [sampled_posts[idx] for idx in selected_indices]

    def collate_batch(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not items:
            raise ValueError("BucketedEngagementDataset.collate_batch received an empty batch")

        row_indices = [int(item["row_idx"]) for item in items]
        epochs = {int(item.get("epoch", 0)) for item in items}
        if len(epochs) != 1:
            raise ValueError("Bucketed batches must contain rows from exactly one sampling epoch")
        epoch = next(iter(epochs))
        bucket = self.like_hour_buckets[row_indices[0]]
        if any(self.like_hour_buckets[row_idx] != bucket for row_idx in row_indices):
            raise ValueError("Bucketed batches must contain rows from exactly one hour bucket")

        user_ids = [self.user_ids[row_idx] for row_idx in row_indices]
        user_to_batch_idx = {
            user_id: user_idx
            for user_idx, user_id in enumerate(user_ids)
        }

        history_tensors = []
        mask_tensors = []
        time_delta_tensors = []
        for row_idx in row_indices:
            history, mask = self._padded_history_for_row(row_idx)
            history_tensors.append(history)
            mask_tensors.append(mask)
            time_delta_tensors.append(self._padded_time_deltas_for_row(row_idx))

        candidate_post_ids: List[str] = []
        candidate_emb_indices: List[int] = []
        candidate_author_indices: List[int] = []
        candidate_to_idx: Dict[str, int] = {}

        def add_candidate(post_id: str, emb_idx: int, author_idx: Optional[int]) -> None:
            if post_id in candidate_to_idx:
                return
            candidate_to_idx[post_id] = len(candidate_post_ids)
            candidate_post_ids.append(post_id)
            candidate_emb_indices.append(int(emb_idx))
            if self.use_author_embedding_table:
                candidate_author_indices.append(
                    int(author_idx) if author_idx is not None else AUTHOR_UNK_IDX
                )

        for row_idx in row_indices:
            author_indices = (
                self.liked_post_author_indices[row_idx]
                if self.liked_post_author_indices is not None
                else [None] * len(self.liked_post_ids[row_idx])
            )
            for post_id, emb_idx, author_idx in zip(
                self.liked_post_ids[row_idx],
                self.liked_post_emb_indices[row_idx],
                author_indices,
            ):
                add_candidate(post_id, int(emb_idx), int(author_idx) if author_idx is not None else None)

        for post in self._sample_candidate_posts_for_batch(row_indices, bucket, epoch, candidate_to_idx):
            add_candidate(post["post_id"], int(post["emb_idx"]), post.get("author_idx"))

        candidate_post_embeddings = torch.from_numpy(
            np.array(self.embeddings[np.array(candidate_emb_indices, dtype=np.int64)], dtype=np.float32)
        )
        label_matrix = torch.zeros((len(user_ids), len(candidate_post_ids)), dtype=torch.float32)
        for row_idx in row_indices:
            user_idx = user_to_batch_idx[self.user_ids[row_idx]]
            for post_id in self.liked_post_ids[row_idx]:
                candidate_idx = candidate_to_idx.get(post_id)
                if candidate_idx is not None:
                    label_matrix[user_idx, candidate_idx] = 1.0

        output: Dict[str, Any] = {
            "history_embeddings": torch.stack(history_tensors, dim=0),
            "history_mask": torch.stack(mask_tensors, dim=0),
            "history_time_deltas_hours": torch.stack(time_delta_tensors, dim=0),
            "candidate_post_embeddings": candidate_post_embeddings,
            "label_matrix": label_matrix,
            "user_id": user_ids,
            "candidate_post_id": candidate_post_ids,
            "bucket": bucket,
        }
        if self.use_author_embedding_table:
            output["history_author_indices"] = torch.stack(
                [self._padded_author_history_for_row(row_idx) for row_idx in row_indices],
                dim=0,
            )
            output["candidate_post_author_idx"] = torch.tensor(candidate_author_indices, dtype=torch.long)
        return output


class BucketedBatchSampler(Sampler[List[int]]):
    """Yield user-hour-row batches where each batch belongs to one hour bucket."""

    def __init__(
        self,
        dataset: BucketedEngagementDataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int,
        resample_candidates_each_epoch: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.resample_candidates_each_epoch = bool(resample_candidates_each_epoch)
        self._epoch = 0

    def __iter__(self) -> Iterator[List[Any]]:
        epoch = self._epoch
        rng = np.random.default_rng(self.seed + epoch)
        self._epoch += 1
        batches: List[List[int]] = []
        buckets = list(self.dataset.row_indices_by_bucket.keys())
        if self.shuffle:
            rng.shuffle(buckets)
        for bucket in buckets:
            row_indices = list(self.dataset.row_indices_by_bucket[bucket])
            if self.shuffle:
                rng.shuffle(row_indices)
            for start in range(0, len(row_indices), self.batch_size):
                batch = row_indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or (batch and not self.drop_last):
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        for batch in batches:
            if self.resample_candidates_each_epoch:
                yield [(int(row_idx), int(epoch)) for row_idx in batch]
            else:
                yield batch

    def __len__(self) -> int:
        total = 0
        for row_indices in self.dataset.row_indices_by_bucket.values():
            n_rows = len(row_indices)
            full_batches = n_rows // self.batch_size
            total += full_batches
            if n_rows % self.batch_size and not self.drop_last:
                total += 1
        return total


def create_bucketed_data_loaders(
    train_dataset: BucketedEngagementDataset,
    val_dataset: BucketedEngagementDataset,
    val_unseen_dataset: BucketedEngagementDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
    prefetch_factor: int,
    seed: int,
    holdout_dataset: Optional[BucketedEngagementDataset] = None,
    train_resample_candidates_each_epoch: bool = False,
):
    """Create DataLoaders for bucketed two-tower batches."""
    from torch.utils.data import DataLoader

    worker_kw: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        worker_kw.update(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=BucketedBatchSampler(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            seed=seed,
            resample_candidates_each_epoch=train_resample_candidates_each_epoch,
        ),
        collate_fn=train_dataset.collate_batch,
        **worker_kw,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=BucketedBatchSampler(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            seed=seed,
        ),
        collate_fn=val_dataset.collate_batch,
        **worker_kw,
    )
    val_unseen_loader = DataLoader(
        val_unseen_dataset,
        batch_sampler=BucketedBatchSampler(
            val_unseen_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            seed=seed,
        ),
        collate_fn=val_unseen_dataset.collate_batch,
        **worker_kw,
    )
    holdout_loader = None
    if holdout_dataset is not None:
        holdout_loader = DataLoader(
            holdout_dataset,
            batch_sampler=BucketedBatchSampler(
                holdout_dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                seed=seed,
            ),
            collate_fn=holdout_dataset.collate_batch,
            **worker_kw,
        )
    return train_loader, val_loader, val_unseen_loader, holdout_loader
