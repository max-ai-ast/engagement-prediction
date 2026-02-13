#!/usr/bin/env python3

"""
Shared dataloaders and user-encoder building blocks for engagement prediction.

Provides two dataset classes named by their **output representation**, not by
which model consumes them.  This keeps user-representation and model-head
concerns orthogonal:

    SummarizedEngagementDataset  -- fixed-size [user_summary ‖ post_emb] vectors
    SequenceEngagementDataset    -- padded variable-length history sequences + mask

Both read on-the-fly from a numpy memmap (embeddings), a Polars target_posts
DataFrame, and a Polars user-history DataFrame produced by the earlier pipeline
stages.

Also provides:
- Pluggable UserSummarizer strategies (mean, EMA, linear recency)
- Learned user-history encoders (``UserHistoryEncoder``,
  ``LightweightAttentionEncoder``) shared by both MLP and Two-Tower models
- A shared ``load_training_data()`` helper that locates and opens the three
  upstream artifacts
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
    log_operation_start,
)


# ---------------------------------------------------------------------------
# User Summarizer strategies
# ---------------------------------------------------------------------------

class UserSummarizer(ABC):
    """Base class for turning a variable-length history of embeddings into a
    single fixed-size vector."""

    @abstractmethod
    def summarize(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Args:
            embeddings: shape ``[seq_len, D]``, sorted most-recent-first.
                        May be empty (``seq_len == 0``).

        Returns:
            A single vector of shape ``[D]``.
        """
        ...


class MeanSummarizer(UserSummarizer):
    """Simple arithmetic mean over all history embeddings."""

    def summarize(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) == 0:
            raise ValueError("MeanSummarizer received empty embeddings; caller should handle the empty case")
        return embeddings.mean(axis=0).astype(np.float32)


class EMASummarizer(UserSummarizer):
    """Exponential moving average weighted towards more recent likes.

    Embeddings are expected most-recent-first.  Weight for position *i* is
    ``alpha * (1 - alpha)^i``, normalised to sum to 1.

    Args:
        alpha: smoothing factor in (0, 1].  Higher = more weight on recent.
    """

    def __init__(self, alpha: float = 0.1):
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha

    def summarize(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) == 0:
            raise ValueError("EMASummarizer received empty embeddings; caller should handle the empty case")
        n = len(embeddings)
        raw_weights = self.alpha * ((1.0 - self.alpha) ** np.arange(n, dtype=np.float64))
        weights = (raw_weights / raw_weights.sum()).astype(np.float32)
        return (weights[:, None] * embeddings).sum(axis=0).astype(np.float32)


class LinearRecencySummarizer(UserSummarizer):
    """Linear recency weighting: most-recent gets weight *n*, next *n-1*, etc."""

    def summarize(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) == 0:
            raise ValueError("LinearRecencySummarizer received empty embeddings; caller should handle the empty case")
        n = len(embeddings)
        raw_weights = np.arange(n, 0, -1, dtype=np.float32)  # [n, n-1, ..., 1]
        weights = raw_weights / raw_weights.sum()
        return (weights[:, None] * embeddings).sum(axis=0).astype(np.float32)


def get_summarizer(name: str, **kwargs: Any) -> UserSummarizer:
    """Factory: ``"mean"`` | ``"ema"`` | ``"linear_recency"``."""
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

class UserHistoryEncoder(nn.Module):
    """Encodes a variable-length sequence of liked post embeddings into a fixed
    user representation via self-attention + dual pooling (attention + mean).

    Used as the user tower in both the Two-Tower model and AttentionMLP.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_attention_heads: int = 4,
        num_attention_layers: int = 2,
        max_seq_len: int = 50,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.positional_embedding = nn.Embedding(max_seq_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_attention_layers,
        )

        self.attention_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(
        self,
        history_embeddings: torch.Tensor,  # [B, seq_len, input_dim]
        history_mask: Optional[torch.Tensor] = None,  # [B, seq_len] True = valid
    ) -> torch.Tensor:
        B, seq_len, _ = history_embeddings.shape
        device = history_embeddings.device

        if history_mask is None:
            history_mask = torch.ones(B, seq_len, dtype=torch.bool, device=device)

        x = self.input_projection(history_embeddings)

        # Positional embeddings flipped for recency (position 0 = most recent)
        positions = torch.arange(seq_len, device=device)
        positions = (self.max_seq_len - 1) - positions.clamp(max=self.max_seq_len - 1)
        pos_emb = self.positional_embedding(positions)
        x = x + pos_emb.unsqueeze(0)

        # Transformer (PyTorch uses inverted mask: True = ignore)
        attn_mask = ~history_mask
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)

        # Attention-weighted pooling
        query = self.attention_query.expand(B, -1, -1)
        attn_scores = torch.bmm(query, x.transpose(1, 2))
        attn_scores = attn_scores.masked_fill(attn_mask.unsqueeze(1), float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=1.0 / max(seq_len, 1))
        attention_pooled = torch.bmm(attn_weights, x).squeeze(1)

        # Mean pooling (masked)
        mask_expanded = history_mask.unsqueeze(-1).float()
        sum_x = (x * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1)
        mean_pooled = sum_x / count

        combined = torch.cat([attention_pooled, mean_pooled], dim=-1)
        return self.output_projection(combined)


class LightweightAttentionEncoder(nn.Module):
    """Lightweight user-history encoder: input projection + positional encoding
    + single learned-query cross-attention pooling.

    Same ``forward(history_embeddings, history_mask) -> [B, output_dim]``
    interface as :class:`UserHistoryEncoder` but **without** the expensive
    ``TransformerEncoder`` self-attention layers.  This reduces complexity from
    O(num_layers * seq_len^2 * hidden_dim) to O(seq_len * hidden_dim) and
    parameter count from ~2 M to ~150 K (typical settings).

    Designed as the user tower for a *light-ranker* two-tower model that needs
    to score large candidate sets efficiently.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        max_seq_len: int = 50,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.positional_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Learned query for cross-attention pooling (single head)
        self.attention_query = nn.Parameter(torch.randn(1, 1, hidden_dim))

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(
        self,
        history_embeddings: torch.Tensor,  # [B, seq_len, input_dim]
        history_mask: Optional[torch.Tensor] = None,  # [B, seq_len] True = valid
    ) -> torch.Tensor:
        B, seq_len, _ = history_embeddings.shape
        device = history_embeddings.device

        if history_mask is None:
            history_mask = torch.ones(B, seq_len, dtype=torch.bool, device=device)

        x = self.input_projection(history_embeddings)

        # Positional embeddings flipped for recency (position 0 = most recent)
        positions = torch.arange(seq_len, device=device)
        positions = (self.max_seq_len - 1) - positions.clamp(max=self.max_seq_len - 1)
        pos_emb = self.positional_embedding(positions)
        x = x + pos_emb.unsqueeze(0)

        # --- NO TransformerEncoder here (the key cost saving) ---

        # Cross-attention pooling: learned query attends to projected history
        attn_mask_inv = ~history_mask  # True = ignore
        query = self.attention_query.expand(B, -1, -1)
        attn_scores = torch.bmm(query, x.transpose(1, 2))  # [B, 1, seq]
        attn_scores = attn_scores.masked_fill(attn_mask_inv.unsqueeze(1), float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=1.0 / max(seq_len, 1))
        attention_pooled = torch.bmm(attn_weights, x).squeeze(1)  # [B, hidden]

        # Mean pooling (masked)
        mask_expanded = history_mask.unsqueeze(-1).float()
        sum_x = (x * mask_expanded).sum(dim=1)
        count = mask_expanded.sum(dim=1).clamp(min=1)
        mean_pooled = sum_x / count

        combined = torch.cat([attention_pooled, mean_pooled], dim=-1)
        return self.output_projection(combined)


# ---------------------------------------------------------------------------
# Shared data-loading helper
# ---------------------------------------------------------------------------

def load_training_data(
    run_dir: Path,
    context: Context,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, pl.DataFrame, pl.DataFrame, int]:
    """Locate and load the three upstream artifacts needed for training.

    Resolution order for each artifact:
    1. ``context.artifacts`` (populated during a same-session pipeline run)
    2. ``select_prior_output()`` filesystem scan

    Returns:
        ``(embeddings_mmap, target_posts_df, history_df, embed_dim)``

        * ``embeddings_mmap`` -- read-only numpy memmap, shape ``[n_posts, D]``
        * ``target_posts_df`` -- collected Polars DataFrame from Stage 2
        * ``history_df``      -- collected Polars DataFrame from Stage 3
        * ``embed_dim``       -- int, the embedding dimensionality *D*
    """
    if logger is None:
        logger = get_stage_logger("DATALOADERS")
    run_dir = Path(run_dir).resolve()

    # --- 1. Embeddings memmap from 01_get_data ---
    log_operation_start("Locate embeddings memmap", "DATALOADERS", logger)
    get_data_dir = _resolve_prior(run_dir, context, stage_key="get_data", folder="01_get_data")
    emb_candidates = sorted(get_data_dir.glob("embeddings_*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not emb_candidates:
        raise FileNotFoundError(f"No embeddings_*.npy found under {get_data_dir}")
    embeddings_path = emb_candidates[0]
    embeddings_mmap: np.ndarray = np.load(str(embeddings_path), mmap_mode="r")
    embed_dim = embeddings_mmap.shape[1]
    logger.info(f"Loaded embeddings memmap: shape={embeddings_mmap.shape}, path={embeddings_path}")

    # --- 2. Target posts from 02_target_posts ---
    log_operation_start("Locate target_posts", "DATALOADERS", logger)
    target_posts_dir = _resolve_prior(run_dir, context, stage_key="target_posts", folder="02_target_posts")
    tp_candidates = sorted(target_posts_dir.glob("target_posts_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not tp_candidates:
        raise FileNotFoundError(f"No target_posts_*.parquet found under {target_posts_dir}")
    target_posts_df = pl.read_parquet(tp_candidates[0])
    logger.info(f"Loaded target_posts: {len(target_posts_df):,} rows from {tp_candidates[0].name}")

    # --- 3. User history from 03_user_history (or legacy 02_featurize) ---
    log_operation_start("Locate user_history", "DATALOADERS", logger)
    history_dir = _resolve_prior(
        run_dir, context,
        stage_key="user_history",
        folder="03_user_history",
        fallback_folder="02_featurize",
    )
    hist_candidates = sorted(history_dir.glob("history_posts_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not hist_candidates:
        raise FileNotFoundError(f"No history_posts_*.parquet found under {history_dir}")
    history_df = pl.read_parquet(hist_candidates[0])
    logger.info(f"Loaded user_history: {len(history_df):,} rows from {hist_candidates[0].name}")

    return embeddings_mmap, target_posts_df, history_df, embed_dim


def _resolve_prior(
    run_dir: Path,
    context: Context,
    *,
    stage_key: str,
    folder: str,
    fallback_folder: Optional[str] = None,
) -> Path:
    """Resolve a prior stage output directory, trying context artifacts first,
    then ``select_prior_output`` with an optional legacy fallback folder."""
    # Try context artifacts first (same-session run)
    art_dir = context.get_artifact_dir(stage_key)
    if art_dir is not None and Path(art_dir).exists():
        return Path(art_dir)
    # Filesystem scan
    result = select_prior_output(
        run_dir, folder,
        use_latest=context.use_latest,
        prior_path=context.prior_outputs.get(folder),
    )
    if result is not None:
        return result
    # Fallback folder (e.g. legacy '02_featurize' for user_history)
    if fallback_folder is not None:
        result = select_prior_output(
            run_dir, fallback_folder,
            use_latest=context.use_latest,
            prior_path=context.prior_outputs.get(fallback_folder),
        )
        if result is not None:
            return result
    raise FileNotFoundError(
        f"Could not find output for stage '{stage_key}' "
        f"(looked for '{folder}'"
        + (f" and '{fallback_folder}'" if fallback_folder else "")
        + f") under {run_dir}"
    )


# ---------------------------------------------------------------------------
# Internal: prepare the row-aligned index arrays shared by both datasets
# ---------------------------------------------------------------------------

def _prepare_split_data(
    target_posts_df: pl.DataFrame,
    history_df: pl.DataFrame,
    split: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    """Filter to a single split and return aligned numpy arrays.

    Returns:
        ``(like_emb_idx, neg_emb_idx, prior_emb_indices_list,
          target_dids, like_uris)``

    Where:
        * ``like_emb_idx``  -- int32 array [N]
        * ``neg_emb_idx``   -- int32 array [N]
        * ``prior_emb_indices_list`` -- Python list of N numpy arrays (each
          variable-length uint32, most-recent-first)
        * ``target_dids``   -- object array [N] of user-id strings
        * ``like_uris``     -- object array [N] of like-uri strings
    """
    if logger is None:
        logger = get_stage_logger("DATALOADERS")

    # Filter target_posts to requested split, drop rows with null neg_emb_idx
    tp = target_posts_df.filter(
        (pl.col("split") == split) & pl.col("neg_emb_idx").is_not_null()
    )
    logger.info(f"  Split '{split}': {len(tp):,} target rows (after dropping null neg_emb_idx)")

    # History is 1:1 aligned by row with target_posts.  Join on (target_did, like_uri)
    # to get the correct history for each target row.
    joined = tp.join(
        history_df.select(["target_did", "like_uri", "prior_emb_indices"]),
        on=["target_did", "like_uri"],
        how="left",
    )

    like_emb_idx = joined["like_emb_idx"].to_numpy().astype(np.int64)
    neg_emb_idx = joined["neg_emb_idx"].to_numpy().astype(np.int64)
    target_dids = joined["target_did"].to_list()
    like_uris = joined["like_uri"].to_list()

    # prior_emb_indices is a List[UInt32] column.  Convert each element to a
    # numpy array; null (no history match) becomes an empty array.
    prior_col = joined["prior_emb_indices"]
    prior_emb_indices_list: List[np.ndarray] = []
    for row_val in prior_col.to_list():
        if row_val is None or len(row_val) == 0:
            prior_emb_indices_list.append(np.array([], dtype=np.uint32))
        else:
            prior_emb_indices_list.append(np.array(row_val, dtype=np.uint32))

    return like_emb_idx, neg_emb_idx, prior_emb_indices_list, np.array(target_dids), np.array(like_uris)


# ---------------------------------------------------------------------------
# SummarizedEngagementDataset
# ---------------------------------------------------------------------------

class SummarizedEngagementDataset(Dataset):
    """Produces fixed-size feature vectors ``[user_summary || post_embedding]``.

    Each target-posts row yields **two** training samples (positive + negative).
    Index ``2*k`` is the positive sample for row *k*; index ``2*k+1`` is the
    negative.

    Currently consumed by the MLP; future-proof for any model that accepts a
    fixed-length feature vector.

    **Performance**: all user summaries and post embeddings are pre-computed
    into contiguous float32 tensors at init time, so ``__getitem__`` is a
    pure in-memory index lookup with zero memmap I/O.
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

        (
            like_emb_idx,
            neg_emb_idx,
            prior_emb_indices,
            self.target_dids,
            self.like_uris,
        ) = _prepare_split_data(target_posts_df, history_df, split, logger)

        self._n_rows = len(like_emb_idx)

        # ── Pre-compute user summaries [N, D] ────────────────────────
        if logger:
            logger.info(f"  Pre-computing user summaries for '{split}' ({self._n_rows:,} rows)…")
        user_summaries = np.zeros((self._n_rows, embed_dim), dtype=np.float32)
        for i, hist_indices in enumerate(prior_emb_indices):
            if len(hist_indices) > 0:
                hist_embs = embeddings_mmap[hist_indices]  # [seq, D]
                user_summaries[i] = summarizer.summarize(hist_embs)
            # else: stays zeros
        self._user_summaries = torch.from_numpy(user_summaries)

        # ── Pre-fetch post embeddings [N, D] for pos and neg ─────────
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
        return self._n_rows * 2

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row_idx = idx // 2
        is_positive = (idx % 2) == 0

        user_vec = self._user_summaries[row_idx]            # [D]
        if is_positive:
            post_vec = self._pos_post_embs[row_idx]          # [D]
            label = 1.0
            post_id = self.like_uris[row_idx]
        else:
            post_vec = self._neg_post_embs[row_idx]          # [D]
            label = 0.0
            post_id = f"neg_{row_idx}"

        features = torch.cat([user_vec, post_vec])           # [2D]
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
    """Produces padded history sequences + mask + target post embedding.

    Each target-posts row yields **two** training samples (positive + negative).

    Currently consumed by the Two-Tower model; future-proof for any model that
    operates on variable-length embedding sequences (e.g. MLP + attention head).

    **Performance**: post embeddings (small: ``2 * N * D * 4`` bytes) are
    pre-computed into contiguous tensors.  History sequences are constructed
    on-the-fly from the memmap to avoid the prohibitive memory cost of
    materializing a dense ``[N, max_seq, D]`` float32 tensor (which would be
    ~13 GB at N=178K, max_seq=50, D=384).  Multi-worker DataLoaders pipeline
    the per-sample memmap reads so the GPU stays fed.
    """

    def __init__(
        self,
        embeddings_mmap: np.ndarray,
        target_posts_df: pl.DataFrame,
        history_df: pl.DataFrame,
        split: str,
        max_history_len: int,
        embed_dim: int,
        logger: Optional[logging.Logger] = None,
    ):
        self.embeddings = embeddings_mmap
        self.max_history_len = max_history_len
        self.embed_dim = embed_dim

        (
            like_emb_idx,
            neg_emb_idx,
            self.prior_emb_indices,
            self.target_dids,
            self.like_uris,
        ) = _prepare_split_data(target_posts_df, history_df, split, logger)

        self._n_rows = len(like_emb_idx)

        # ── Pre-fetch post embeddings [N, D] for pos and neg (cheap: ~560 MB) ──
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
        return self._n_rows * 2

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row_idx = idx // 2
        is_positive = (idx % 2) == 0

        # --- history sequence: pad / truncate from memmap (done in worker) ---
        hist_indices = self.prior_emb_indices[row_idx]
        seq_len = min(len(hist_indices), self.max_history_len)

        padded = np.zeros((self.max_history_len, self.embed_dim), dtype=np.float32)
        mask = np.zeros(self.max_history_len, dtype=bool)

        if seq_len > 0:
            used_indices = hist_indices[: self.max_history_len]
            padded[:seq_len] = self.embeddings[used_indices]
            mask[:seq_len] = True

        # --- post embedding: pre-computed tensor lookup ---
        if is_positive:
            post_vec = self._pos_post_embs[row_idx]          # [D]
            label = 1.0
            post_id = self.like_uris[row_idx]
        else:
            post_vec = self._neg_post_embs[row_idx]          # [D]
            label = 0.0
            post_id = f"neg_{row_idx}"

        return {
            "history_embeddings": torch.from_numpy(padded),            # [max_seq, D]
            "history_mask": torch.from_numpy(mask),                    # [max_seq]
            "target_post_embedding": post_vec,
            "label": torch.tensor(label, dtype=torch.float32),
            "user_id": self.target_dids[row_idx],
            "post_id": post_id,
        }


def sequence_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate for ``SequenceEngagementDataset`` -- stacks pre-padded tensors."""
    return {
        "history_embeddings": torch.stack([b["history_embeddings"] for b in batch]),
        "history_mask": torch.stack([b["history_mask"] for b in batch]),
        "target_post_embedding": torch.stack([b["target_post_embedding"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "user_ids": [b["user_id"] for b in batch],
        "post_ids": [b["post_id"] for b in batch],
    }
