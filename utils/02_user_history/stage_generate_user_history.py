#!/usr/bin/env python3

"""
Stage 2: Generate User-Hour History Directory

Creates a directory-style artifact that maps each (user, like-hour bucket) to a
list of prior liked post embedding indices and, when author metadata is
available, author indices. This enables efficient on-the-fly embedding retrieval
during training and stable author-history features.

Inputs:
- likes_core_*.parquet from 01_get_data: Contains
  {did, subject_uri, record_created_at, like_hour_bucket, emb_idx, author_idx}
- author_idx_*.parquet from 01_get_data, when available: Author index mapping with
  {author_did, author_train_count, author_idx}

Outputs under <run_dir>/02_user_history/<timestamp>/:
- history_posts_<timestamp>.parquet:
  {did, like_hour_bucket, prior_emb_indices, prior_like_age_hours_at_bucket_start}
  and, when author_idx is available, {prior_author_indices}
  where prior_emb_indices is a List[UInt32] of embedding indices sorted by recency (most recent first),
  prior_like_age_hours_at_bucket_start is a List[Float32] aligned element-wise with
  prior_emb_indices and measured from the target like_hour_bucket,
  prior_author_indices is a List[UInt32] aligned element-wise with
  prior_emb_indices, and user-hour rows where the user has no prior likes in
  the dataset get empty lists.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import logging
import polars as pl
import time

from utils.pipeline.core import Context
from utils.helpers import (
    get_stage_logger,
    log_operation_start,
    validate_dataframe_schema,
    load_parquet_from_prior,
    TIMESTAMP_COL_NAME,
)
from utils.memory_helpers import MemoryTracker


def _build_user_history_directory(
    likes_lf: pl.LazyFrame,
    max_prior_likes: Optional[int],
    logger: logging.Logger,
) -> pl.LazyFrame:
    """
    Build a directory mapping each (user, like-hour bucket) to prior liked embedding indices.

    Uses vectorized Polars operations for efficiency:
    1. Get distinct user-hour pairs from likes_core
    2. Join each pair to that user's likes
    3. Filter to likes that occurred before the hour bucket
    4. Group by (did, like_hour_bucket) and collect emb_idx values sorted by recency
    5. Left-join back to ensure every user-hour appears, including empty histories

    Args:
        likes_lf: LazyFrame with columns [did, like_hour_bucket, record_created_at, emb_idx]
        max_prior_likes: Optional cap on prior likes per target (None = no cap)
        logger: Logger instance

    Returns:
        LazyFrame with columns [did, like_hour_bucket, prior_emb_indices, raw_prior_count,
        prior_like_age_hours_at_bucket_start]
        where raw_prior_count is the uncapped number of prior likes (for distribution analysis).
    """
    logger.info("Building user history directory...")
    likes_schema = likes_lf.collect_schema()
    include_author_idx = "author_idx" in likes_schema

    user_bucket_pairs_lf = (
        likes_lf
        .select(['did', 'like_hour_bucket'])
        .unique()
    ) # [did, like_hour_bucket]

    # Join targets with likes on user identity
    # This creates one row per (target, like) pair for each user
    likes_cols = ['did', TIMESTAMP_COL_NAME, 'emb_idx']
    if include_author_idx:
        likes_cols.append('author_idx')
    pairs_with_prior_likes_lf = (
        user_bucket_pairs_lf
        .join(
            likes_lf.select(likes_cols),
            on="did",
            how="left"
        )
        .filter(pl.col(TIMESTAMP_COL_NAME) < pl.col("like_hour_bucket"))
        .with_columns(
            (
                (pl.col("like_hour_bucket") - pl.col(TIMESTAMP_COL_NAME))
                .dt.total_seconds()
                / 3600.0
            )
            .cast(pl.Float32)
            .alias("prior_like_age_hours_at_bucket_start")
        )
    ) # [did, like_hour_bucket, record_created_at, emb_idx, (author_idx)]

    def _get_agg_expr(col_name: str):
        # Build aggregation expression: sort by recency (descending) and optionally cap
        # The result is a list of emb_idx values, most recent first
        agg_expr = (
            pl.col(col_name)
            .sort_by(pl.col(TIMESTAMP_COL_NAME), descending=True)
        )
        if max_prior_likes is not None and max_prior_likes > 0:
            agg_expr = agg_expr.head(max_prior_likes)
        return agg_expr
    
    agg_exprs = [
        _get_agg_expr("emb_idx").alias("prior_emb_indices"),
        pl.len().alias("raw_prior_count"),
        _get_agg_expr("prior_like_age_hours_at_bucket_start").alias("prior_like_age_hours_at_bucket_start"),
    ]
    if include_author_idx:
        agg_exprs += [_get_agg_expr("author_idx").alias("prior_author_indices")]

    # Group by user and hour bucket, and collect prior emb_idx as list.
    # Also compute raw (uncapped) count for distribution analysis.
    history_lists_lf = (
        pairs_with_prior_likes_lf
        .group_by(["did", "like_hour_bucket"])
        .agg(agg_exprs)
    )
    pairs_with_history_list_lf = (
        user_bucket_pairs_lf
        .join(history_lists_lf, on=["did", "like_hour_bucket"], how="left")
        .with_columns(
            pl.when(pl.col("prior_emb_indices").is_null())
            .then(pl.lit([]).cast(pl.List(pl.UInt32)))
            .otherwise(pl.col("prior_emb_indices").cast(pl.List(pl.UInt32)))
            .alias("prior_emb_indices"),
            pl.col("raw_prior_count").fill_null(0),
            pl.when(pl.col("prior_like_age_hours_at_bucket_start").is_null())
            .then(pl.lit([]).cast(pl.List(pl.Float32)))
            .otherwise(pl.col("prior_like_age_hours_at_bucket_start").cast(pl.List(pl.Float32)))
            .alias("prior_like_age_hours_at_bucket_start"),
        )
    ) # [did, like_hour_bucket, prior_emb_indices, raw_prior_count, prior_like_age_hours_at_bucket_start]
    
    if include_author_idx:
        pairs_with_history_list_lf = (
            pairs_with_history_list_lf
            .with_columns(
                pl.when(pl.col("prior_author_indices").is_null())
                .then(pl.lit([]).cast(pl.List(pl.UInt32)))
                .otherwise(pl.col("prior_author_indices").cast(pl.List(pl.UInt32)))
                .alias("prior_author_indices"),
            )
        ) # [did, like_hour_bucket, prior_emb_indices, raw_prior_count, (prior_author_indices)]

    return pairs_with_history_list_lf


def _log_and_plot_history_distribution(
    directory_df: pl.DataFrame,
    max_prior_likes: Optional[int],
    out_dir: Path,
    logger: logging.Logger,
) -> None:
    """
    Log summary statistics and save a histogram of the per-user history length
    distribution (measured at each user's last target post) before and after
    the max_prior_likes cap.

    For each user we look at their chronologically last target post, which has
    the maximum available history.  The distribution of these counts across
    users reveals how many users are actually affected by the cap.
    """
    # For each user, find the raw prior count at their last target post
    last_target_per_user = directory_df.group_by("did").agg(
        pl.col("raw_prior_count").sort_by("like_hour_bucket").last().alias("history_len_before"),
    )

    before = last_target_per_user["history_len_before"]
    n_users = len(before)

    if n_users == 0:
        logger.warning("No users found for history distribution analysis")
        return

    if max_prior_likes is not None:
        last_target_per_user = last_target_per_user.with_columns(
            pl.col("history_len_before").clip(upper_bound=max_prior_likes).alias("history_len_after")
        )
        after = last_target_per_user["history_len_after"]
    else:
        after = before

    # --- Log summary statistics ---
    logger.info("=" * 60)
    logger.info("Per-user history distribution (at each user's last target post)")
    logger.info(f"  Number of unique users: {n_users:,}")

    for label, dist in [("Before capping", before), ("After capping", after)]:
        logger.info(f"  {label}:")
        logger.info(f"    mean={dist.mean():.1f}, median={dist.median():.1f}")
        logger.info(
            f"    p25={int(dist.quantile(0.25, 'nearest') or 0)}, "
            f"p75={int(dist.quantile(0.75, 'nearest') or 0)}, "
            f"p90={int(dist.quantile(0.90, 'nearest') or 0)}, "
            f"p95={int(dist.quantile(0.95, 'nearest') or 0)}, "
            f"p99={int(dist.quantile(0.99, 'nearest') or 0)}"
        )
        logger.info(f"    min={dist.min()}, max={dist.max()}")

    if max_prior_likes is not None:
        n_capped = int((before > max_prior_likes).sum())
        pct_capped = 100.0 * n_capped / n_users
        logger.info(
            f"  Users affected by cap ({max_prior_likes}): "
            f"{n_capped:,} ({pct_capped:.1f}%)"
        )
        total_before = int(before.sum())
        total_after = int(after.sum())
        dropped = total_before - total_after
        pct_dropped = 100.0 * dropped / max(total_before, 1)
        logger.info(
            f"  Total prior likes (last target): before={total_before:,}, "
            f"after={total_after:,}, dropped={dropped:,} ({pct_dropped:.1f}%)"
        )

    logger.info("=" * 60)

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available, skipping distribution plot")
        return

    before_np = before.to_numpy().astype(float)
    after_np = after.to_numpy().astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: before capping
    ax = axes[0]
    max_val = int(before_np.max()) if len(before_np) > 0 else 1
    n_bins = min(100, max(max_val + 1, 2))
    bins = np.linspace(-0.5, max_val + 0.5, n_bins)
    ax.hist(before_np, bins=bins, alpha=0.8, color="steelblue",
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Prior likes count")
    ax.set_ylabel("Number of users")
    ax.set_title("Before capping")
    if before_np.max() > 0:
        ax.set_yscale("log")
    if max_prior_likes is not None:
        ax.axvline(max_prior_likes, color="red", linestyle="--",
                    linewidth=1.5, label=f"cap = {max_prior_likes}")
        ax.legend()

    # Right panel: after capping
    ax = axes[1]
    max_val_after = int(after_np.max()) if len(after_np) > 0 else 1
    n_bins_after = min(100, max(max_val_after + 1, 2))
    bins_after = np.linspace(-0.5, max_val_after + 0.5, n_bins_after)
    ax.hist(after_np, bins=bins_after, alpha=0.8, color="darkorange",
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Prior likes count")
    ax.set_ylabel("Number of users")
    cap_label = (f" (max_prior_likes={max_prior_likes})"
                 if max_prior_likes is not None else " (no cap)")
    ax.set_title(f"After capping{cap_label}")
    if after_np.max() > 0:
        ax.set_yscale("log")

    fig.suptitle(
        "Distribution of history length per user (last target post)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    plot_path = out_dir / "history_distribution.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"✓ Saved history distribution plot to {plot_path.name}")


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    """
    Stage 2: Generate user history directory.

    Creates a parquet file mapping each target row to a list of prior liked
    post embedding indices for efficient on-the-fly lookup during training.
    """
    out_dir = context.new_stage_dir('02_user_history')

    # Initialize logger and memory tracker
    logger = get_stage_logger('STAGE_02_USER_HISTORY', log_file=out_dir / 'stage.log')
    t0 = time.time()
    mem_tracker = MemoryTracker(logger=logger)
    mem_tracker.checkpoint("stage_start")

    # === Locate prior stage outputs ===

    # 1. Likes from get_data stage
    prior_get_data = context.resolve_prior_output(
        '01_get_data',
        prior_path=context.prior_outputs.get('01_get_data'),
    )

    # === Get CLI args ===
    max_prior_likes: Optional[int] = args.max_prior_likes
    if max_prior_likes is not None and max_prior_likes <= 0:
        max_prior_likes = None  # Treat 0 or negative as "no cap"

    # === Load data ===
    log_operation_start('Load likes_core from prior stage', 'STAGE_02_USER_HISTORY', logger)
    likes_lf: pl.LazyFrame = load_parquet_from_prior(prior_get_data, "likes_core_")

    # Validate likes schema
    likes_schema = {
        "did": str,
        TIMESTAMP_COL_NAME: pl.Datetime,
        "subject_uri": str,
        "emb_idx": int,
    }
    validate_dataframe_schema(likes_lf, likes_schema)
    logger.info("✓ likes_core schema validated")

    mem_tracker.checkpoint("after_load_inputs", quiet=True)

    # Log input sizes (collect counts efficiently)
    n_likes = likes_lf.select(pl.len()).collect().item()
    logger.info(f"Input: {n_likes:,} likes")

    # === Build user history directory ===
    log_operation_start('Build user history directory', 'STAGE_02_USER_HISTORY', logger)

    directory_lf = _build_user_history_directory(
        likes_lf=likes_lf,
        max_prior_likes=max_prior_likes,
        logger=logger,
    )

    mem_tracker.checkpoint("after_build_history", quiet=True)

    # === Write output ===
    log_operation_start('Write user history directory', 'STAGE_02_USER_HISTORY', logger)

    # Collect using the streaming engine so that the intermediate fan-out join
    # is processed in batches rather than fully materialised in memory.
    # Falls back to the default engine automatically if the plan can't be streamed.
    directory_df = directory_lf.collect(engine="streaming")

    # Log and plot the per-user history distribution before/after capping
    _log_and_plot_history_distribution(directory_df, max_prior_likes, out_dir, logger)

    # Select only the required persisted output columns. Author history is
    # optional so user-history outputs without author columns can still feed
    # training when the author embedding feature is not being used.
    user_history_output_path = out_dir / f"history_posts_{out_dir.name}.parquet"
    directory_df.write_parquet(user_history_output_path, compression="zstd")

    n_output = len(directory_df)

    n_with_history = directory_df.filter(pl.col("prior_emb_indices").list.len() > 0).height
    n_empty_history = n_output - n_with_history

    # Stats on prior likes counts
    prior_counts = directory_df["prior_emb_indices"].list.len()
    mean_prior = prior_counts.mean()
    max_prior = prior_counts.max()
    min_prior = prior_counts.filter(prior_counts > 0).min() if n_with_history > 0 else 0

    mem_tracker.checkpoint("after_write_output", quiet=True)

    logger.info(f"✓ Wrote {n_output:,} directory entries to {user_history_output_path.name}")
    logger.info(f"  With history: {n_with_history:,} ({100*n_with_history/n_output:.1f}%)")
    logger.info(f"  Empty history: {n_empty_history:,} ({100*n_empty_history/n_output:.1f}%)")
    logger.info(f"  Prior likes per target: mean={mean_prior:.1f}, min={min_prior}, max={max_prior}")

    # Memory summary
    mem_tracker.summary()

    runtime = time.time() - t0

    # === Stage info ===
    info_lines = [
        f"stage: user_history",
        f"runtime_seconds: {runtime:.2f}",
        f"settings: max_prior_likes={max_prior_likes}",
        f"inputs: likes_core ({n_likes:,})",
        f"outputs: user_history_directory ({n_output:,} entries)",
        f"stats: with_history={n_with_history:,}, empty_history={n_empty_history:,}",
        f"stats: mean_prior={mean_prior:.1f}, max_prior={max_prior}",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    logger.info(f"Stage 2 (user_history) completed in {runtime:.2f}s")

    return {
        'output_dir': out_dir,
        'artifacts': {
            'user_history_directory_path': str(user_history_output_path),
        }
    }
