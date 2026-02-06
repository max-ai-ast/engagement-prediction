#!/usr/bin/env python3

"""
Shared dataloaders for engagement prediction training.

Provides two dataset classes named by their **output representation**, not by
which model consumes them.  This keeps user-representation and model-head
concerns orthogonal:

    SummarizedEngagementDataset  -- fixed-size [user_summary ‖ post_emb] vectors
    SequenceEngagementDataset    -- padded variable-length history sequences + mask

Both read on-the-fly from a numpy memmap (embeddings), a Polars target_posts
DataFrame, and a Polars user-history DataFrame produced by the earlier pipeline
stages.

Also provides pluggable UserSummarizer strategies (mean, EMA, linear recency)
and a shared ``load_training_data()`` helper that locates and opens the three
upstream artifacts.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from utils.pipeline.core import Context, select_prior_output
from utils.helpers import (
    get_stage_logger,
    log_operation_start,
    load_parquet_from_prior,
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
        self.embeddings = embeddings_mmap
        self.summarizer = summarizer
        self.embed_dim = embed_dim

        (
            self.like_emb_idx,
            self.neg_emb_idx,
            self.prior_emb_indices,
            self.target_dids,
            self.like_uris,
        ) = _prepare_split_data(target_posts_df, history_df, split, logger)

        self._n_rows = len(self.like_emb_idx)
        if logger:
            logger.info(
                f"  SummarizedEngagementDataset('{split}'): "
                f"{self._n_rows:,} rows -> {len(self):,} samples "
                f"(summarizer={type(summarizer).__name__})"
            )

    def __len__(self) -> int:
        return self._n_rows * 2

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row_idx = idx // 2
        is_positive = (idx % 2) == 0

        # --- user history summary ---
        hist_indices = self.prior_emb_indices[row_idx]
        if len(hist_indices) > 0:
            hist_embs = self.embeddings[hist_indices]  # [seq, D]
            user_vec = self.summarizer.summarize(hist_embs)
        else:
            user_vec = np.zeros(self.embed_dim, dtype=np.float32)

        # --- post embedding ---
        if is_positive:
            post_vec = np.array(self.embeddings[self.like_emb_idx[row_idx]], dtype=np.float32)
            label = 1.0
            post_id = self.like_uris[row_idx]
        else:
            post_vec = np.array(self.embeddings[self.neg_emb_idx[row_idx]], dtype=np.float32)
            label = 0.0
            post_id = f"neg_{row_idx}"

        features = np.concatenate([user_vec, post_vec])  # [2D]
        return {
            "features": torch.from_numpy(features),
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
            self.like_emb_idx,
            self.neg_emb_idx,
            self.prior_emb_indices,
            self.target_dids,
            self.like_uris,
        ) = _prepare_split_data(target_posts_df, history_df, split, logger)

        self._n_rows = len(self.like_emb_idx)
        if logger:
            logger.info(
                f"  SequenceEngagementDataset('{split}'): "
                f"{self._n_rows:,} rows -> {len(self):,} samples "
                f"(max_history_len={max_history_len})"
            )

    def __len__(self) -> int:
        return self._n_rows * 2

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row_idx = idx // 2
        is_positive = (idx % 2) == 0

        # --- user history sequence (pad / truncate) ---
        hist_indices = self.prior_emb_indices[row_idx]
        seq_len = min(len(hist_indices), self.max_history_len)

        padded = np.zeros((self.max_history_len, self.embed_dim), dtype=np.float32)
        mask = np.zeros(self.max_history_len, dtype=bool)

        if seq_len > 0:
            # hist_indices are already most-recent-first; take the first max_history_len
            used_indices = hist_indices[: self.max_history_len]
            padded[:seq_len] = self.embeddings[used_indices]
            mask[:seq_len] = True

        # --- post embedding ---
        if is_positive:
            post_vec = np.array(self.embeddings[self.like_emb_idx[row_idx]], dtype=np.float32)
            label = 1.0
            post_id = self.like_uris[row_idx]
        else:
            post_vec = np.array(self.embeddings[self.neg_emb_idx[row_idx]], dtype=np.float32)
            label = 0.0
            post_id = f"neg_{row_idx}"

        return {
            "history_embeddings": torch.from_numpy(padded),
            "history_mask": torch.from_numpy(mask),
            "target_post_embedding": torch.from_numpy(post_vec),
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
