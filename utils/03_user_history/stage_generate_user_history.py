#!/usr/bin/env python3

"""
Stage 2: Generate User History Directory

Creates a directory-style artifact that maps each target row to a list of prior
liked post embedding indices, enabling efficient on-the-fly embedding retrieval
during training.

Inputs:
- likes_core_*.parquet from 01_get_data: Contains {did, subject_uri, record_created_at, emb_idx}
- target_posts_*.parquet from 02_target_posts: Wide format with
  {target_did, seen_at, like_uri, like_emb_idx, ..., neg_uri, neg_emb_idx, ..., split}

Outputs under <run_dir>/02_featurize/<timestamp>/:
- history_posts_<timestamp>.parquet: {target_did, like_uri, prior_emb_indices}
  where prior_emb_indices is a List[UInt32] of embedding indices sorted by recency (most recent first),
  indexed on (target_did, like_uri) so there is one history per (user, like-event) pair.
  Rows where the user has no prior likes in the dataset get an empty list.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List
import argparse
import logging
import polars as pl
import time

from utils.pipeline.core import new_stage_timestamp_dir, select_prior_output, Context
from utils.helpers import get_stage_logger, log_operation_start, validate_dataframe_schema, load_parquet_from_prior, TIMESTAMP_COL_NAME
from utils.memory_helpers import MemoryTracker


def _build_user_history_directory(
    targets_lf: pl.LazyFrame,
    likes_lf: pl.LazyFrame,
    max_prior_likes: Optional[int],
    logger: logging.Logger,
    history_buffer_hours: Optional[float] = None,
) -> pl.LazyFrame:
    """
    Build a directory mapping each target (user, like-event) to prior liked embedding indices.

    The target posts use a wide format where each row represents a (user, like-event)
    training pair.  The user history depends only on the user (target_did) and the
    event timestamp (seen_at), so a single history list is produced per pair.
    The output is indexed on (target_did, like_uri).

    Internally, the heavy join/filter/group_by operations use an integer target_idx
    to avoid carrying expensive string columns (like_uri) through the fan-out join.
    The string keys are mapped back only at the end.

    Uses vectorized Polars operations for efficiency:
    1. Assign integer target_idx to each target row
    2. Join with likes on user (target_did == did) carrying only target_idx
    3. Filter to likes that occurred before the target timestamp (seen_at),
       optionally with a buffer period subtracted from seen_at
    4. Group by target_idx and collect emb_idx values sorted by recency
    5. Left-join back to ensure every target appears (empty list for no history)
    6. Map target_idx back to (target_did, like_uri) for the final output

    Args:
        targets_lf: LazyFrame with at least columns [target_did, seen_at, like_uri, ...]
        likes_lf: LazyFrame with columns [did, subject_uri, record_created_at, emb_idx]
        max_prior_likes: Optional cap on prior likes per target (None = no cap)
        logger: Logger instance
        history_buffer_hours: Optional buffer in hours to subtract from seen_at when
            determining prior likes.  When set, a like must satisfy
            ``like_ts < seen_at - buffer`` instead of ``like_ts < seen_at``.
            Useful for simulating information delay or avoiding leakage near the
            boundary.  None or 0 means no buffer (original behaviour).

    Returns:
        LazyFrame with columns [target_did, like_uri, seen_at, prior_emb_indices, raw_prior_count]
        where raw_prior_count is the uncapped number of prior likes (for distribution analysis).
    """
    logger.info("Building user history directory...")

    # Assign an integer row index for memory-efficient keying during the
    # expensive fan-out join.  Carrying a UInt32 target_idx through hundreds
    # of millions of intermediate rows is far cheaper than carrying like_uri
    # strings (~80-100 bytes each).
    targets_indexed = targets_lf.with_row_index("target_idx")

    # For the heavy join, only carry target_idx + the columns needed for
    # joining (target_did) and filtering (seen_at).  like_uri is NOT included
    # here to save memory during the fan-out.
    join_keys = targets_indexed.select(["target_idx", "target_did", "seen_at"])

    # Rename likes columns to avoid collision after join
    # We need: did (join key), record_created_at (for filtering/sorting), emb_idx (the result)
    likes_renamed = likes_lf.select([
        pl.col("did"),
        pl.col(TIMESTAMP_COL_NAME).alias("like_ts"),
        pl.col("emb_idx").alias("like_emb_idx"),
    ])

    # Join targets with likes on user identity
    # This creates one row per (target, like) pair for each user
    joined = join_keys.join(
        likes_renamed,
        left_on="target_did",
        right_on="did",
        how="left",
    )

    # Filter to likes that occurred BEFORE the target timestamp (minus optional buffer).
    # This ensures we only include prior history, not future likes.
    if history_buffer_hours is not None and history_buffer_hours > 0:
        buffer_us = int(history_buffer_hours * 3_600_000_000)  # hours → microseconds
        cutoff = pl.col("seen_at") - pl.duration(microseconds=buffer_us)
        logger.info(f"  Applying history buffer of {history_buffer_hours}h (like_ts < seen_at - {history_buffer_hours}h)")
    else:
        cutoff = pl.col("seen_at")

    prior_likes = joined.filter(
        pl.col("like_ts") < cutoff
    )

    # Build aggregation expression: sort by recency (descending) and optionally cap
    # The result is a list of emb_idx values, most recent first
    agg_expr = (
        pl.col("like_emb_idx")
        .sort_by(pl.col("like_ts"), descending=True)
    )

    if max_prior_likes is not None and max_prior_likes > 0:
        agg_expr = agg_expr.head(max_prior_likes)
        logger.info(f"  Capping prior likes to {max_prior_likes} per target")
    else:
        logger.info("  No cap on prior likes (using all available history)")

    # Group by integer target_idx (cheap) and collect prior emb_idx as list.
    # Also compute raw (uncapped) count for distribution analysis.
    directory_lf = prior_likes.group_by("target_idx").agg(
        agg_expr.alias("prior_emb_indices"),
        pl.len().alias("raw_prior_count"),
    )

    # Left-join back to ensure every target row appears, including those with
    # no prior likes (e.g. a user's first like in the dataset).
    # Also map back from target_idx to the meaningful (target_did, like_uri) keys.
    #
    # NOTE: The left side preserves the original row order from the target_posts
    # file.  This means the output rows are 1:1 aligned with target_posts, even
    # though (target_did, like_uri) is the official join key.  Downstream
    # consumers should join on the key columns, but positional alignment can be
    # relied upon as an optimization if needed.
    all_target_keys = targets_indexed.select(["target_idx", "target_did", "like_uri", "seen_at"])
    directory_lf = all_target_keys.join(
        directory_lf,
        on="target_idx",
        how="left",
    ).with_columns(
        pl.when(pl.col("prior_emb_indices").is_null())
        .then(pl.lit([]).cast(pl.List(pl.UInt32)))
        .otherwise(pl.col("prior_emb_indices").cast(pl.List(pl.UInt32)))
        .alias("prior_emb_indices"),
        pl.col("raw_prior_count").fill_null(0),
    ).select(["target_did", "like_uri", "seen_at", "prior_emb_indices", "raw_prior_count"])

    return directory_lf


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
    last_target_per_user = directory_df.group_by("target_did").agg(
        pl.col("raw_prior_count").sort_by("seen_at").last().alias("history_len_before"),
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
    Stage 3: Generate user history directory.

    Creates a parquet file mapping each target row to a list of prior liked
    post embedding indices for efficient on-the-fly lookup during training.
    """
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '03_user_history')

    # Initialize logger and memory tracker
    logger = get_stage_logger('STAGE_03_USER_HISTORY', log_file=out_dir / 'stage.log')
    t0 = time.time()
    mem_tracker = MemoryTracker(logger=logger)
    mem_tracker.checkpoint("stage_start")

    # === Locate prior stage outputs ===

    # 1. Likes from get_data stage
    prior_get_data = select_prior_output(
        run_dir, '01_get_data',
        use_latest=context.use_latest,
        prior_path=context.prior_outputs.get('01_get_data'),
    )
    if prior_get_data is None:
        raise FileNotFoundError("Could not find 01_get_data output")

    # 2. Target posts from 02_target_posts stage
    prior_target_posts_dir = select_prior_output(
        run_dir, '02_target_posts',
        use_latest=context.use_latest,
        prior_path=context.prior_outputs.get('02_target_posts'),
    )
    if prior_target_posts_dir is None:
        raise FileNotFoundError(
            "Could not find 02_target_posts output. "
            "Run the target_posts stage first or provide --prior-output-target-posts."
        )
    # Find the parquet file inside the target posts output directory
    target_posts_candidates = sorted(
        prior_target_posts_dir.glob("target_posts_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not target_posts_candidates:
        raise FileNotFoundError(
            f"No target_posts_*.parquet found under {prior_target_posts_dir}"
        )
    target_posts_path = target_posts_candidates[0]

    logger.info(f"Using likes from: {prior_get_data}")
    logger.info(f"Using target posts from: {target_posts_path}")

    # === Get CLI args ===
    max_prior_likes: Optional[int] = getattr(args, 'max_prior_likes', None)
    if max_prior_likes is not None and max_prior_likes <= 0:
        max_prior_likes = None  # Treat 0 or negative as "no cap"

    history_buffer_hours: Optional[float] = getattr(args, 'history_buffer_hours', None)
    if history_buffer_hours is not None and history_buffer_hours <= 0:
        history_buffer_hours = None  # Treat 0 or negative as "no buffer"

    # === Load data ===
    log_operation_start('Load likes_core from prior stage', 'STAGE_03_USER_HISTORY', logger)
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

    log_operation_start('Load target_posts', 'STAGE_03_USER_HISTORY', logger)
    targets_lf: pl.LazyFrame = pl.scan_parquet(target_posts_path)

    # Validate target posts schema (wide format)
    targets_schema = {
        "target_did": str,
        "seen_at": pl.Datetime,
        "like_uri": str,
    }
    validate_dataframe_schema(targets_lf, targets_schema)
    logger.info("✓ target_posts schema validated")

    mem_tracker.checkpoint("after_load_inputs", quiet=True)

    # Log input sizes (collect counts efficiently)
    n_likes = likes_lf.select(pl.len()).collect().item()
    n_targets = targets_lf.select(pl.len()).collect().item()
    logger.info(f"Input sizes: {n_likes:,} likes, {n_targets:,} targets")

    # === Build user history directory ===
    log_operation_start('Build user history directory', 'STAGE_03_USER_HISTORY', logger)

    directory_lf = _build_user_history_directory(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=max_prior_likes,
        logger=logger,
        history_buffer_hours=history_buffer_hours,
    )

    mem_tracker.checkpoint("after_build_history", quiet=True)

    # === Write output ===
    log_operation_start('Write user history directory', 'STAGE_03_USER_HISTORY', logger)
    output_path = out_dir / f"history_posts_{out_dir.name}.parquet"

    # Collect using the streaming engine so that the intermediate fan-out join
    # is processed in batches rather than fully materialised in memory.
    # Falls back to the default engine automatically if the plan can't be streamed.
    directory_df = directory_lf.collect(engine="streaming")

    # Log and plot the per-user history distribution before/after capping
    _log_and_plot_history_distribution(directory_df, max_prior_likes, out_dir, logger)

    # Select only the required output columns (drop analysis columns)
    directory_df = directory_df.select(["target_did", "like_uri", "prior_emb_indices"])

    directory_df.write_parquet(output_path, compression="zstd")

    n_output = len(directory_df)

    # Sanity check: history_posts must have exactly the same number of rows as target_posts
    if n_output != n_targets:
        logger.error(
            f"Row count mismatch! history_posts has {n_output:,} rows but "
            f"target_posts has {n_targets:,} rows. These should be 1:1."
        )
        raise ValueError(
            f"history_posts row count ({n_output:,}) != target_posts row count ({n_targets:,})"
        )
    logger.info(f"✓ Row count check passed: {n_output:,} history rows == {n_targets:,} target rows")

    n_with_history = directory_df.filter(pl.col("prior_emb_indices").list.len() > 0).height
    n_empty_history = n_output - n_with_history

    # Stats on prior likes counts
    prior_counts = directory_df["prior_emb_indices"].list.len()
    mean_prior = prior_counts.mean()
    max_prior = prior_counts.max()
    min_prior = prior_counts.filter(prior_counts > 0).min() if n_with_history > 0 else 0

    mem_tracker.checkpoint("after_write_output", quiet=True)

    logger.info(f"✓ Wrote {n_output:,} directory entries to {output_path.name}")
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
        f"settings: max_prior_likes={max_prior_likes}, history_buffer_hours={history_buffer_hours}",
        f"inputs: likes_core ({n_likes:,}), target_posts ({n_targets:,})",
        f"outputs: user_history_directory ({n_output:,} entries)",
        f"stats: with_history={n_with_history:,}, empty_history={n_empty_history:,}",
        f"stats: mean_prior={mean_prior:.1f}, max_prior={max_prior}",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    logger.info(f"Stage 3 (user_history) completed in {runtime:.2f}s")

    return {
        'output_dir': out_dir,
        'artifacts': {
            'user_history_directory_path': str(output_path),
        }
    }
