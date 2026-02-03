#!/usr/bin/env python3

"""
Stage 1: Get and filter data using streaming Polars + hash-based sampling.

This stage produces two core datasets:
- likes_core.parquet: filtered likes (user sampling, per-user caps, min-likes)
- posts_core.parquet: liked posts + hash-based random sample, with expanded embeddings

================================================================================
FILTERING SEQUENCE
================================================================================

PHASE 0: Inputs and safety check
-------------------------------------------------------------------------
  - List GCS parquet files in the requested time ranges.
  - Optional memory safety check (check_data_load_safe) before loading.

PHASE 1: Likes Processing (_load_likes_core_polars)
-------------------------------------------------------------------------
Pass 1 - Count likes per user (streaming):
  - Scan all likes parquet files (no batching) and apply --likes-start/--likes-end.
  - Count total likes and per-user like counts.
  - Pre-filter users with fewer than --min-likes-per-user.
  - If --max-liking-users is set, hash-sample eligible users
    using --cap-random-seed (deterministic).

Pass 2 - Filter likes to sampled users (streaming):
  - Semi-join likes to the sampled user list.

Per-user random cap (--max-likes-per-user):
  - For each user, hash-rank subject_uri with the seed and keep top-K.
  - This is random but deterministic and avoids recency bias.

Finalization:
  - Keep did, subject_uri, record_created_at (convert to UTC datetime if needed).
  - Collect a final in-memory likes_core_df for downstream steps and stats.

PHASE 2: Posts Processing (_load_posts_core_polars)
-------------------------------------------------------------------------
Extract liked post URIs:
  - Get unique subject_uri values from likes_core_df.

Process posts with early embedding expansion (single scan):
  - Scan all posts parquet files (no batching) and apply --posts-start/--posts-end.
  - Count total posts to compute a hash threshold for --negative-posts-sample.
  - Hash-sample posts by at_uri (seeded) and left-join liked URIs.
  - Keep posts that are in the random sample OR are liked.
  - Expand embeddings into columns and drop the raw embedding blob.
  - Write posts_core parquet (with embeddings), then read back a slim
    posts_core_df (without embeddings) for downstream use.

PHASE 3: Post-join min-likes verification (_run_greenearth_pipeline)
-------------------------------------------------------------------------
  - Filter likes to posts that exist in posts_core (by subject_uri -> at_uri).
  - Re-apply --min-likes-per-user after the join; drop users who fall below.
  - Update stats to reflect join losses and final counts.

================================================================================
DESIGN RATIONALE
================================================================================

Streaming Polars scans:
  Keeps memory bounded without materializing full tables.

Pre-filtering before sampling:
  Ensures the user cap is spent on users who already meet min-likes.

Hash-based sampling/capping:
  Deterministic and avoids recency bias in per-user caps.

Early embedding expansion:
  Converts large list blobs into per-dimension columns, reducing peak memory.

Statistically independent random sample:
  The random sample is drawn from ALL posts, independent of like status.
  Some liked posts will appear in the random sample; this is expected.

================================================================================
OUTPUTS
================================================================================

Under <run_dir>/01_get_data/<timestamp>/:
  - likes_core_*.parquet: did, subject_uri, record_created_at
  - posts_core_*.parquet: post columns + post_emb_0..N + is_liked + in_random_sample
  - summary.json: full filtering statistics and parameters
  - stage.log: detailed execution log with memory checkpoints
  - stage_info.txt: human-readable run summary

Using the outputs:
  - Random sample (for population stats): filter posts_core where in_random_sample=True
  - Liked posts: join likes_core.subject_uri with posts_core.at_uri. is_liked is also set
  - Negative examples for training: sample from random sample excluding user's likes
"""

from __future__ import annotations

import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
import polars as pl
import logging
from datetime import datetime, timezone
import re
from google.cloud import storage
import numpy as np

from utils.pipeline.core import (
    new_stage_timestamp_dir, 
    Context,
)
from utils.helpers import (
    get_stage_logger,
    log_operation_start,
    parse_one_ts,
    validate_dataframe_schema,
    apply_time_filter,
    get_embed_dim,
    expand_embeddings_polars,
)
from utils.memory_helpers import (
    check_data_load_safe,
    MemoryTracker,
)

# ----------------------------------------
# Data IO helpers (Green Earth Ingex + GCS)
# ----------------------------------------
# For parsing GCS Ingex filenames
TIMESTAMP_SUFFIX_GCS = "_(\\d{8})_(\\d{6})\\.parquet$"


def _parse_ts_from_name_ingex_gcs(
    blob_name: str, 
    blob_prefix: str
) -> Optional[datetime]:
    """Parse timestamp from GCS blob name based on Ingex naming convention."""
    pattern = re.compile(blob_prefix + TIMESTAMP_SUFFIX_GCS)
    m = pattern.match(blob_name)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def _list_files_with_timestamps_ingex_gcs(
    gcs_bucket: str, 
    blob_prefix: str, 
    start: Optional[datetime], 
    end: Optional[datetime],
) -> Tuple[list[str], list[datetime]]:
    """
    List GCS blob URIs and their timestamps within specified time range.
    
    Returns:
        Tuple of (uris, timestamps) where both lists are aligned by index.
    """
    client = storage.Client()
    blobs = client.list_blobs(gcs_bucket)
    uris = []
    timestamps = []
    for b in blobs:
        ts = _parse_ts_from_name_ingex_gcs(blob_name=b.name, blob_prefix=blob_prefix)
        if ts is None:
            continue
        if start is not None and ts < start:
            continue
        if end is not None and ts >= end:
            continue
        uris.append(f"gs://{gcs_bucket}/{b.name}")
        timestamps.append(ts)
    return uris, timestamps


def _plot_data_density_histogram(
    likes_timestamps: list[datetime],
    posts_timestamps: list[datetime],
    likes_start: Optional[datetime],
    likes_end: Optional[datetime],
    posts_start: Optional[datetime],
    posts_end: Optional[datetime],
    out_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Create a histogram showing data file density (files/day) for likes and posts.
    
    Helps users visualize:
    - Data collection coverage within their time window
    - Gaps in data availability
    - Relative density of likes vs posts data
    
    Args:
        likes_timestamps: Parsed timestamps from likes parquet files
        posts_timestamps: Parsed timestamps from posts parquet files
        likes_start, likes_end: Requested time range for likes
        posts_start, posts_end: Requested time range for posts
        out_dir: Directory to save the histogram
        logger: Logger instance
        
    Returns:
        Dict with density statistics for both likes and posts
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from collections import Counter
        
        stats = {}
        
        # Helper to compute daily counts
        def compute_daily_counts(timestamps: list[datetime], name: str) -> Tuple[list, list, Dict]:
            if not timestamps:
                return [], [], {'total_files': 0, 'days_with_data': 0, 'mean_files_per_day': 0}
            
            # Group by date
            date_counts = Counter(ts.date() for ts in timestamps)
            dates = sorted(date_counts.keys())
            counts = [date_counts[d] for d in dates]
            
            # Compute stats
            total_files = sum(counts)
            days_with_data = len(dates)
            mean_per_day = total_files / days_with_data if days_with_data > 0 else 0
            
            # Check for gaps (missing days)
            if len(dates) >= 2:
                expected_days = (dates[-1] - dates[0]).days + 1
                missing_days = expected_days - days_with_data
            else:
                expected_days = days_with_data
                missing_days = 0
            
            return dates, counts, {
                'total_files': total_files,
                'days_with_data': days_with_data,
                'expected_days': expected_days,
                'missing_days': missing_days,
                'mean_files_per_day': round(mean_per_day, 1),
                'min_files_per_day': min(counts) if counts else 0,
                'max_files_per_day': max(counts) if counts else 0,
            }
        
        likes_dates, likes_counts, likes_stats = compute_daily_counts(likes_timestamps, 'likes')
        posts_dates, posts_counts, posts_stats = compute_daily_counts(posts_timestamps, 'posts')
        
        stats['likes'] = likes_stats
        stats['posts'] = posts_stats
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
        
        # Plot likes density
        if likes_dates:
            ax1.bar(likes_dates, likes_counts, color='steelblue', alpha=0.7, edgecolor='white', linewidth=0.5)
            ax1.axhline(y=likes_stats['mean_files_per_day'], color='red', linestyle='--', 
                       linewidth=1.5, label=f"Mean: {likes_stats['mean_files_per_day']:.1f}")
            ax1.set_ylabel('Files per Day')
            ax1.set_title(f"Likes Data Density ({likes_stats['total_files']:,} files across {likes_stats['days_with_data']} days)")
            ax1.legend(loc='upper right')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add gap warning if applicable
            if likes_stats['missing_days'] > 0:
                ax1.text(0.02, 0.98, f"Warning: {likes_stats['missing_days']} missing day(s)", 
                        transform=ax1.transAxes, fontsize=9, verticalalignment='top',
                        color='orange', fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No likes files found in range', transform=ax1.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            ax1.set_title('Likes Data Density (no data)')
        
        # Plot posts density
        if posts_dates:
            ax2.bar(posts_dates, posts_counts, color='forestgreen', alpha=0.7, edgecolor='white', linewidth=0.5)
            ax2.axhline(y=posts_stats['mean_files_per_day'], color='red', linestyle='--',
                       linewidth=1.5, label=f"Mean: {posts_stats['mean_files_per_day']:.1f}")
            ax2.set_ylabel('Files per Day')
            ax2.set_xlabel('Date')
            ax2.set_title(f"Posts Data Density ({posts_stats['total_files']:,} files across {posts_stats['days_with_data']} days)")
            ax2.legend(loc='upper right')
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add gap warning if applicable
            if posts_stats['missing_days'] > 0:
                ax2.text(0.02, 0.98, f"Warning: {posts_stats['missing_days']} missing day(s)",
                        transform=ax2.transAxes, fontsize=9, verticalalignment='top',
                        color='orange', fontweight='bold')
        else:
            ax2.text(0.5, 0.5, 'No posts files found in range', transform=ax2.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            ax2.set_title('Posts Data Density (no data)')
            ax2.set_xlabel('Date')
        
        plt.tight_layout()
        
        # Save to output directory
        plot_path = out_dir / 'data_density_histogram.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Data density histogram saved to {plot_path}")
        logger.info(f"  Likes: {likes_stats['total_files']:,} files, "
                   f"{likes_stats['days_with_data']} days with data, "
                   f"{likes_stats['missing_days']} gaps")
        logger.info(f"  Posts: {posts_stats['total_files']:,} files, "
                   f"{posts_stats['days_with_data']} days with data, "
                   f"{posts_stats['missing_days']} gaps")
        
        return stats
        
    except Exception as e:
        logger.warning(f"Failed to create data density histogram: {e}")
        return {}


def _get_sampled_users_with_min_likes(
    likes_lf: pl.LazyFrame,
    min_likes_per_user: int,
    max_liking_users: Optional[int],
    random_seed: int
) -> Tuple[pl.DataFrame, int, int, int]:
    # get total user and like count
    likes_summary_df = likes_lf.select(
        pl.col('did').n_unique().alias('user_count'),
        pl.len().alias('like_count')
    ).collect(engine="streaming")
    n_users_initial = likes_summary_df["user_count"][0]
    n_likes_initial = likes_summary_df["like_count"][0]

    # ===== Count likes per user =====
    user_counts_lf = (
        likes_lf.group_by('did')
        .agg(pl.len().alias('like_count'))
    )
    # ===== Pre-filter users by min_likes_per_user before sampling =====
    if min_likes_per_user > 0:
        user_counts_lf = user_counts_lf.filter(
            pl.col('like_count') >= min_likes_per_user
        )
    # get count of eligible users
    n_users_eligible = (
        user_counts_lf
        .select(pl.len().alias('n'))
        .collect(engine="streaming")
        .item()
    )
    # ===== Sample users if cap is set =====
    if max_liking_users is not None and n_users_eligible > max_liking_users:
        threshold_hash = _compute_random_sample_threshold(n_users_eligible, max_liking_users)
        user_counts_lf = (
            user_counts_lf.with_columns(
                pl.col("did").hash(seed=random_seed).alias("_hash_key"),
            ).filter(
                pl.col("_hash_key") <= threshold_hash
            )
        )
    return user_counts_lf.select("did").collect(engine="streaming"), n_users_initial, n_likes_initial, n_users_eligible


def _apply_per_user_random_cap(
    likes_lf: pl.LazyFrame,
    max_likes_per_user: int,
    random_seed: int
) -> pl.LazyFrame:
    if max_likes_per_user <= 0:
        return likes_lf
    # Add deterministic pseudo-random order per user, then keep top-K
    return (
        likes_lf
        .with_columns(
            pl.col('subject_uri').hash(seed=random_seed).alias('_rand_key')
        ).with_columns(
            pl.col('_rand_key').rank('ordinal').over('did').alias('_rand_order')
        ).filter(
            pl.col('_rand_order') <= max_likes_per_user
        ).drop(
            ['_rand_key', '_rand_order']
        )
    )


def _load_likes_core_polars(
    start_str: Optional[str],
    end_str: Optional[str],
    paths: List[str],
    *,
    max_liking_users: Optional[int],
    max_likes_per_user: int,
    min_likes_per_user: int,
    random_seed: int,
    logger: logging.Logger,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """
    Load and filter likes data using a streaming Polars pipeline.
    
    High-level flow:
    1. Streamed pass: count likes per user
    2. Pre-filter users who don't meet min_likes_per_user
    3. Sample users from eligible pool (if cap is set)
    4. Streamed pass: keep only likes from sampled users
    5. Apply per-user random caps (NOT recency-based)
    6. Verify min-likes threshold (handles edge cases from per-user caps)
    
    Returns:
        Tuple of (likes_lf: pl.DataFrame, stats: Dict with filtering statistics)
    """
    if not paths:
        raise ValueError(f"No likes parquet files found for time range {start_str} to {end_str}")
    
    logger.info(f"Found {len(paths)} likes parquet files")
    
    raw_lf = pl.scan_parquet(paths)
    base_lf = apply_time_filter(raw_lf, start_str, end_str)

    # ===== PASS 1: Filter users =====
    logger.info("Pass 1: Counting likes per user (streaming)...")

    # Filter users by min likes and then sample down to max liking users
    # n_users_eligible is the number of users that had the minimum number of likes, before we randomly sample
    sampled_users_df, n_users_initial, n_likes_initial, n_users_eligible = _get_sampled_users_with_min_likes(
        base_lf, 
        min_likes_per_user,
        max_liking_users,
        random_seed
    )
    logger.info(f"Pass 1 complete: {n_likes_initial:,} likes from {n_users_initial:,} users")
    
    stats: Dict[str, Any] = {
        'n_likes_initial': n_likes_initial,
        'n_users_initial': n_users_initial,
    }

    # record stats and log stuff
    n_users_filtered = n_users_initial - n_users_eligible
    n_users_sampled = sampled_users_df.height
    logger.info(f"Pre-filtering: {n_users_eligible:,} users meet min-likes threshold ({min_likes_per_user}), "
            f"excluded {n_users_filtered:,} users with too few likes")
    stats['n_users_eligible_for_sampling'] = n_users_eligible
    stats['n_users_excluded_min_likes'] = n_users_filtered
    stats['n_users_sampled'] = n_users_sampled
    logger.info(
        f"Sampled {n_users_sampled:,} liking users "
        f"({100*n_users_sampled/n_users_eligible:.1f}% of eligible)"
    )
    
    # ===== PASS 2: Filter likes to sampled users (lazy) =====
    logger.info("Pass 2: Filtering likes to sampled users")
    likes_lf = base_lf.join(sampled_users_df.lazy(), on='did', how='semi')
    
    # Compute counts per user before per-user cap
    counts_pre_cap_df = (
        likes_lf.group_by('did')
        .agg(pl.len().alias('like_count'))
        .collect(engine='streaming')
    )
    
    n_after_user_sample = int(counts_pre_cap_df['like_count'].sum()) if counts_pre_cap_df.height > 0 else 0
    pct_retained = 100.0 * n_after_user_sample / n_likes_initial if n_likes_initial > 0 else 0
    logger.info(f"Pass 2 complete: {n_after_user_sample:,} likes ({pct_retained:.1f}% retained)")
    stats['n_likes_after_user_sample'] = n_after_user_sample
    
    # ===== Capture like count distribution BEFORE cap (for analysis/plotting) =====
    # This shows how many likes each sampled user has before we apply the per-user cap
    if counts_pre_cap_df.height > 0:
        likes_per_user_before_cap = counts_pre_cap_df['like_count'].to_list()
        stats['likes_per_user_distribution'] = likes_per_user_before_cap
        stats['likes_per_user_mean'] = float(np.mean(likes_per_user_before_cap))
        stats['likes_per_user_median'] = float(np.median(likes_per_user_before_cap))
        stats['likes_per_user_max'] = int(max(likes_per_user_before_cap))
        stats['likes_per_user_p90'] = float(np.percentile(likes_per_user_before_cap, 90))
        stats['likes_per_user_p99'] = float(np.percentile(likes_per_user_before_cap, 99))
        logger.info(f"Likes per sampled user: mean={stats['likes_per_user_mean']:.1f}, "
             f"median={stats['likes_per_user_median']:.0f}, max={stats['likes_per_user_max']}, "
             f"p90={stats['likes_per_user_p90']:.0f}, p99={stats['likes_per_user_p99']:.0f}")
    
    # ===== Apply per-user random cap (NOT recency-based) =====
    if max_likes_per_user > 0 and n_after_user_sample > 0:
        likes_lf = _apply_per_user_random_cap(likes_lf, max_likes_per_user, random_seed)
    
    likes_df = likes_lf.select(['did', 'subject_uri', 'record_created_at']).unique().collect(engine="streaming")

    # Compute post-cap counts
    n_after_cap = likes_df.height
    pct_retained = 100.0 * n_after_cap / n_after_user_sample if n_after_user_sample > 0 else 0
    logger.info(f"After per-user cap ({max_likes_per_user}): {n_after_cap:,} likes ({pct_retained:.1f}% retained)")
    stats['n_likes_after_per_user_cap'] = n_after_cap

    # Convert record_created_at to datetime if it exists and is not already datetime
    schema = likes_df.schema
    if 'record_created_at' in schema and schema['record_created_at'] != pl.Datetime:
        likes_df = likes_df.with_columns(
            pl.col('record_created_at').str.to_datetime(time_zone="UTC").alias('record_created_at')
        )

    stats['n_likes_final'] = n_after_cap
    
    return likes_df, stats


def _load_posts_core_polars(
    start_str: Optional[str],
    end_str: Optional[str],
    liked_post_uris_df: pl.DataFrame,
    paths: List[str],
    *,
    negative_posts_sample: int,
    embedding_model: str,
    random_seed: int,
    logger: logging.Logger,
    out_dir: Path,
    # the below inputs are for testing: they allow this function to be called without expanding real embeddings
    expand_embeddings_fn: Optional[Callable[[pl.LazyFrame, str, int], pl.LazyFrame]] = expand_embeddings_polars,
    get_embed_dim_fn: Callable[[pl.LazyFrame, str], int] = get_embed_dim,
    embed_dim_override: Optional[int] = None,
    skip_embedding_expansion: bool = False,
) -> Tuple[pl.DataFrame, Dict[str, Any], int, Path]:
    """
    Load posts data with a single Polars scan (no batching) and expand embeddings early.

    Processing flow:
    1. Scan all parquet files and apply the time filter.
    2. Count total posts to set a hash-sampling threshold.
    3. Hash-sample posts (random_seed) and left-join liked_post_uris.
    4. Keep posts that are either in the random sample or liked.
    5. Expand embeddings into columns and drop the raw embeddings blob.
    6. Write posts_core parquet (with embeddings), then read back a slim version
       (without embeddings) for downstream use.

    Statistical independence:
    The random sample is drawn from ALL posts, not filtered by like status.
    Posts that are both liked AND randomly sampled appear once with in_random_sample=True.
    This keeps the random sample usable for unbiased population statistics.

    Output columns:
    - in_random_sample: True if post was selected by hash-sampling,
                        False if included only because it was liked
    - is_liked: True if post is in likes core dataset, False otherwise

    Returns:
        Tuple of (posts_df: pl.DataFrame, stats: Dict, embedding_dim: int)
    """
    if not paths:
        raise ValueError(f"No posts parquet files found for time range {start_str} to {end_str}")
    
    logger.info(f"Found {len(paths)} posts parquet files")

    posts_lf = pl.scan_parquet(paths)
    posts_lf = apply_time_filter(posts_lf, start_str, end_str)

    # get the total number of posts and calc threshold
    n_posts_total = posts_lf.select(pl.len()).collect().item()
    logger.info(f"n_posts_total: {n_posts_total:,}")
    threshold_hash = _compute_random_sample_threshold(n_posts_total, negative_posts_sample)

    cols_no_emb = ["at_uri", "record_created_at", "did", "record_text"]
    cols_with_emb = cols_no_emb + ["embeddings"]
    
    # get posts: sampled via hash, or in liked_post_uris:
    negs_and_likes_lf = _build_posts_candidate_lf(
        posts_lf=posts_lf,
        liked_post_uris_df=liked_post_uris_df,
        threshold_hash=threshold_hash,
        random_seed=random_seed,
        cols_with_emb=cols_with_emb,
    )

    # get embedding dim
    if embed_dim_override is not None:
        embed_dim = embed_dim_override
    else:
        embed_dim = get_embed_dim_fn(posts_lf, embedding_model)
    logger.info(f"Detected embedding dimension: {embed_dim}")

    # expand embeddings into columns
    if expand_embeddings_fn is None:
        skip_embedding_expansion = True
    posts_core_lf = _expand_posts_embeddings(
        negs_and_likes_lf,
        embedding_model,
        embed_dim,
        expand_embeddings_fn,
        skip_embedding_expansion,
    )
    
    # Validate posts_core_lf schema
    posts_schema_with_embs = {
        'at_uri': str,
        'in_random_sample': bool,
        'did': str,
        'record_created_at': str,
        'record_text': str,
        'is_liked': bool,
    }
    if not skip_embedding_expansion:
        for i in range(embed_dim):
            posts_schema_with_embs[f'post_emb_{i}'] = float
    validate_dataframe_schema(posts_core_lf, posts_schema_with_embs, allow_extra_columns=False)

    # write out
    logger.info(f"✓ posts_core schema validated (embed_dim={embed_dim})")

    # Save outputs as parquet
    log_operation_start('Save likes core dataset as parquet', 'STAGE_01_GET_DATA', logger)
    ts_name = out_dir.name
    posts_core_path = out_dir / f"posts_core_{ts_name}.parquet"
    
    # low row_group_size because embeddings are very large. keeps memory low
    posts_core_lf.sink_parquet(
        posts_core_path,
        compression="zstd",
        engine="streaming",
        row_group_size=128
    )

    # read back from the parquet file, withOUT embeddings
    posts_core_df = (
        pl
        .scan_parquet(posts_core_path)
        .select(cols_no_emb + ['is_liked', 'in_random_sample'])
        .collect(engine="streaming")
    )

    # Validate posts_core_lf schema
    posts_schema_no_embs = {
        'at_uri': str,
        'in_random_sample': bool,
        'did': str,
        'record_created_at': str,
        'record_text': str,
        'is_liked': bool,
    }
    validate_dataframe_schema(posts_core_df, posts_schema_no_embs, allow_extra_columns=False)

    # calculate metrics
    n_posts_core = posts_core_df.height
    n_liked_only = posts_core_df.filter(pl.col("is_liked") & ~pl.col("in_random_sample")).height
    n_liked_in_random = posts_core_df.filter(pl.col("is_liked") & pl.col("in_random_sample")).height
    n_random_sample = posts_core_df.filter(pl.col("in_random_sample")).height

    logger.info(f"Saved posts_core: {posts_core_path} ({n_posts_core:,} rows)")
    logger.info(f"All posts in raw data: {n_posts_total:,}")
    logger.info(f"Liked only: {n_liked_only:,}")
    logger.info(f"Liked in random sample: {n_liked_in_random:,}")
    logger.info(f"Random sample total: {n_random_sample:,}")

    # Total liked posts = those only in liked set + those also in random sample
    n_total_liked_posts = n_liked_only + n_liked_in_random
    liked_post_match_rate = 100.0 * n_total_liked_posts / liked_post_uris_df.height
    logger.info(f"Loaded {n_total_liked_posts:,} liked posts ({liked_post_match_rate:.1f}% match rate)")
    
    stats = {
        'n_posts_total': n_posts_total,
        'n_liked_posts': n_total_liked_posts,
        'n_liked_only': n_liked_only,  # Liked posts not in random sample
        'n_liked_in_random_sample': n_liked_in_random,  # Liked posts that are also in random sample
        'liked_post_match_rate': liked_post_match_rate,
        'n_random_sample': n_random_sample,
    }
    
    logger.info(f"posts_core: {n_posts_core:,} rows ({n_liked_only:,} liked-only + {n_random_sample:,} random sample)")
    logger.info(f"Embeddings already expanded during loading (dim={embed_dim})")
    
    stats['n_posts_core'] = n_posts_core
    stats['embedding_dim'] = embed_dim
    
    return posts_core_df, stats, embed_dim, posts_core_path


def _compute_random_sample_threshold(n_rows_total: int, n_sample: int) -> int:
    if n_rows_total <= 0:
        return 0
    max_hash = 2**64 - 1
    if n_sample >= n_rows_total:
        return max_hash
    if n_sample <= 0:
        return 0
    # Use integer math to avoid float rounding issues at large ranges.
    return (n_sample * max_hash) // n_rows_total


def _build_posts_candidate_lf(
    posts_lf: pl.LazyFrame,
    liked_post_uris_df: pl.DataFrame,
    threshold_hash: int,
    random_seed: int,
    cols_with_emb: List[str],
) -> pl.LazyFrame:
    """Samples random posts (liked or not) and also includes posts from liked_post_uris_df"""
    return (
        posts_lf
        .select(cols_with_emb)
        .with_columns(
            pl.col("at_uri").hash(seed=random_seed).alias("_hash_key"),
        )
        .join(
            liked_post_uris_df.with_columns(pl.lit(True).alias("_is_liked")).lazy(),
            left_on="at_uri",
            right_on="subject_uri",
            how="left",
        )
        .with_columns(
            (pl.col("_hash_key") <= threshold_hash).alias("in_random_sample"),
            pl.col("_is_liked").fill_null(False).alias("is_liked"),
        )
        .filter(pl.col("in_random_sample") | pl.col("is_liked"))
        .drop(["_is_liked", "_hash_key"])
    )


def _expand_posts_embeddings(
    posts_lf: pl.LazyFrame,
    embedding_model: str,
    embed_dim: int,
    expand_embeddings_fn: Optional[Callable[[pl.LazyFrame, str, int], pl.LazyFrame]],
    skip_embedding_expansion: bool,
) -> pl.LazyFrame:
    if skip_embedding_expansion:
        return posts_lf.drop("embeddings")
    if expand_embeddings_fn is None:
        return posts_lf
    return expand_embeddings_fn(posts_lf, embedding_model, embed_dim)


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '01_get_data')

    # Initialize logger
    logger = get_stage_logger('01_GET_DATA', log_file=out_dir / 'stage.log')

    t0 = time.time()

    # Get input parameters
    gcs_bucket = args.gcs_bucket
    posts_start = args.posts_start
    posts_end = args.posts_end
    likes_start = args.likes_start
    likes_end = args.likes_end
    max_liking_users = args.max_liking_users
    if max_liking_users is not None:
        max_liking_users = int(max_liking_users)
    max_likes_per_user = int(args.max_likes_per_user)
    min_likes_per_user = int(args.min_likes_per_user)
    negative_posts_sample = int(args.negative_posts_sample)
    cap_random_seed = int(args.cap_random_seed)
    embedding_model = args.embedding_model
    memory_check = str(args.memory_check)  # "full", "ignore", or "skip"
    max_memory_gb = args.max_memory_gb
    if max_memory_gb is not None:
        max_memory_gb = float(max_memory_gb)
    max_memory_pct = float(args.max_memory_pct)

    # Use Polars-based filtering pipeline for GreenEarth Ingex data
    likes_core_df, posts_core_df, posts_core_path, embed_dim, all_stats = _run_greenearth_pipeline(
        logger=logger, 
        out_dir=out_dir,
        gcs_bucket=gcs_bucket,
        likes_start=likes_start,
        likes_end=likes_end,
        posts_start=posts_start,
        posts_end=posts_end,
        memory_check=memory_check,
        max_memory_gb=max_memory_gb,
        max_memory_pct=max_memory_pct,
        max_liking_users=max_liking_users,
        max_likes_per_user=max_likes_per_user,
        min_likes_per_user=min_likes_per_user,
        negative_posts_sample=negative_posts_sample,
        cap_random_seed=cap_random_seed,
        embedding_model=embedding_model
    )

    # Validate likes_core schema
    log_operation_start('Validate likes core output schema', '01_GET_DATA', logger)
    likes_schema = {
        'did': str,
        'subject_uri': str,
        'record_created_at': 'datetime',
    }
    validate_dataframe_schema(likes_core_df, likes_schema, allow_extra_columns=False)
    logger.info("✓ likes_core schema validated")

    n_likes = len(likes_core_df)
    n_posts = len(posts_core_df)

    # Save likes as parquet (posts are saved in load_posts_core_polars() in helpers.py)
    log_operation_start('Save likes core dataset as parquet', '01_GET_DATA', logger)
    ts_name = out_dir.name
    likes_core_path = out_dir / f"likes_core_{ts_name}.parquet"
    likes_core_df.write_parquet(likes_core_path)
    logger.info(f"Saved likes_core: {likes_core_path} ({n_likes:,} rows)")

    # Summary
    log_operation_start('Write summary files', '01_GET_DATA', logger)
    
    summary = {
        'gcs_bucket': gcs_bucket,
        'posts_start': posts_start,
        'posts_end': posts_end,
        'likes_start': likes_start,
        'likes_end': likes_end,
        'parameters': {
            'max_liking_users': max_liking_users,
            'max_likes_per_user': max_likes_per_user,
            'min_likes_per_user': min_likes_per_user,
            'negative_posts_sample': negative_posts_sample,
            'cap_random_seed': cap_random_seed,
            'embedding_model': embedding_model,
        },
        'outputs': {
            'likes_core_rows': n_likes,
            'posts_core_rows': n_posts,
            'embedding_dim': embed_dim,
        },
        'filtering_stats': all_stats,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Primary outputs
    context.tracker.log_single_value(name="Output - Likes (final)", value=n_likes)
    context.tracker.log_single_value(name="Output - Posts (final)", value=n_posts)
    context.tracker.log_single_value(name="Output - Embedding Dim", value=embed_dim)

    # Log to experiment tracker - comprehensive metrics for sweep analysis
    # Metric names use readable format: "Category - Metric Name" for better CSV exports
    _attrition_stats_to_experiment_tracker(all_stats, context, max_likes_per_user, logger)

    runtime = time.time() - t0
    
    # Stage info
    info_lines = [
        f"stage: get_data",
        f"runtime_seconds: {runtime:.2f}",
        f"settings: max_liking_users={max_liking_users}, max_likes_per_user={max_likes_per_user}, "
        f"min_likes_per_user={min_likes_per_user}, negative_posts_sample={negative_posts_sample}",
        f"inputs: GCS bucket={gcs_bucket}",
        f"N_likes_core: {n_likes}",
        f"N_posts_core: {n_posts}",
        f"embedding_dim: {embed_dim}",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')
    
    logger.info(f"Stage 1 completed in {runtime:.2f}s")

    return {
        'output_dir': out_dir,
        'artifacts': {
            'likes_core_path': str(likes_core_path),
            'posts_core_path': str(posts_core_path),
        },
    }


def _run_greenearth_pipeline(
    logger: logging.Logger,
    out_dir: Path,
    gcs_bucket: str,
    likes_start: str,
    likes_end: str,
    posts_start: str,
    posts_end: str,
    memory_check: str,
    max_memory_gb: Optional[float],
    max_memory_pct: float,
    max_liking_users: Optional[int],
    max_likes_per_user: int,
    min_likes_per_user: int,
    negative_posts_sample: int,
    cap_random_seed: int,
    embedding_model: str
) -> Tuple[pl.DataFrame, pl.DataFrame, Path, int, Dict[str, Any]]:
    """
    Run the new Polars-based filtering pipeline for GreenEarth Ingex data.
    
    Returns:
        Tuple of (likes_core_df, posts_core_df, posts_core_path, embed_dim, stats_dict)
    """
    all_stats = {}
    
    # Initialize memory tracker for actual memory monitoring
    mem_tracker = MemoryTracker(logger=logger)
    mem_tracker.checkpoint("pipeline_start")
    
    # Pre-flight memory safety check
    log_operation_start('Pre-flight memory safety check', '01_GET_DATA', logger)
    
    # Get file paths for memory estimation
    likes_start_dt = parse_one_ts(likes_start)
    likes_end_dt = parse_one_ts(likes_end)
    posts_start_dt = parse_one_ts(posts_start)
    posts_end_dt = parse_one_ts(posts_end)
    
    # List files with timestamps for density analysis
    likes_paths, likes_timestamps = _list_files_with_timestamps_ingex_gcs(
        gcs_bucket=gcs_bucket,
        blob_prefix='bsky_likes',
        start=likes_start_dt,
        end=likes_end_dt,
    )
    posts_paths, posts_timestamps = _list_files_with_timestamps_ingex_gcs(
        gcs_bucket=gcs_bucket,
        blob_prefix='bsky_posts',
        start=posts_start_dt,
        end=posts_end_dt,
    )
    
    # Generate data density histogram for observability
    log_operation_start('Generate data density histogram', '01_GET_DATA', logger)
    density_stats = _plot_data_density_histogram(
        likes_timestamps=likes_timestamps,
        posts_timestamps=posts_timestamps,
        likes_start=likes_start_dt,
        likes_end=likes_end_dt,
        posts_start=posts_start_dt,
        posts_end=posts_end_dt,
        out_dir=out_dir,
        logger=logger,
    )
    all_stats['data_density'] = density_stats
    
    memory_estimate = None
    if memory_check == "skip":
        logger.info("Memory estimation skipped (--memory-check skip)")
    elif memory_check in ("full", "ignore"):
        # Smart memory check that accounts for filtering parameters
        memory_estimate = check_data_load_safe(
            likes_paths=likes_paths,
            posts_paths=posts_paths,
            embedding_dim=384,  # Standard MiniLM dimension
            max_memory_gb=max_memory_gb,
            max_memory_pct=max_memory_pct,
            max_liking_users=max_liking_users,
            max_likes_per_user=max_likes_per_user,
            min_likes_per_user=min_likes_per_user,
            negative_posts_sample=negative_posts_sample,
            skip_safety_check=(memory_check == "ignore"),
            logger=logger,
        )
        all_stats['memory_estimate'] = memory_estimate
        if memory_check == "full":
            logger.info("Memory check passed, proceeding with data load")
        else:
            logger.info("Memory estimation complete (ignore mode), proceeding regardless")
        
    mem_tracker.checkpoint("after_memory_check", quiet=True)
    
    # Step 1: Load and filter likes
    log_operation_start('Load and filter likes data', '01_GET_DATA', logger)
    likes_core_df, likes_stats = _load_likes_core_polars(
        start_str=likes_start,
        end_str=likes_end,
        paths=likes_paths,
        max_liking_users=max_liking_users,
        max_likes_per_user=max_likes_per_user,
        min_likes_per_user=min_likes_per_user,
        random_seed=cap_random_seed,
        logger=logger,
    )
    all_stats['likes'] = likes_stats
    n_users_final = likes_core_df['did'].n_unique()
    all_stats['likes']['n_users_final'] = n_users_final
    mem_tracker.checkpoint("after_likes_load", quiet=True)
    
    # Step 2: Extract liked post URIs
    log_operation_start('Extract liked post URIs', '01_GET_DATA', logger)
    liked_post_uris_df: pl.DataFrame = likes_core_df.select(pl.col('subject_uri').unique())
    logger.info(f"Extracted {len(liked_post_uris_df):,} unique liked post URIs")
    mem_tracker.checkpoint("after_uri_extraction", quiet=True)
    
    # Step 3: Load posts with early embedding expansion
    # This loads posts, filters to liked + negative sample, and expands embeddings
    # Returned dataframe does *NOT* include embeddings. Those are written to parquet though.
    log_operation_start('Load posts with early embedding expansion', '01_GET_DATA', logger)
    posts_core_df, posts_stats, embed_dim, posts_core_path = _load_posts_core_polars(
        start_str=posts_start,
        end_str=posts_end,
        liked_post_uris_df=liked_post_uris_df,
        paths=posts_paths,
        negative_posts_sample=negative_posts_sample,
        embedding_model=embedding_model,
        random_seed=cap_random_seed,
        logger=logger,
        out_dir=out_dir,
    )
    all_stats['posts'] = posts_stats
    all_stats['embedding_dim'] = embed_dim
    mem_tracker.checkpoint("after_posts_load_and_expansion", quiet=True)

    # Step 4: Re-verify min-likes after like-post join
    # Since like-post joining isn't perfect (some posts may be missing, deleted, or outside time range),
    # we need to re-check that users still meet min-likes threshold based on likes that have matching posts.
    log_operation_start('Re-verify min-likes after like-post join', '01_GET_DATA', logger)
    likes_core_df, join_stats = _filter_likes_after_post_join(
        likes_core_df=likes_core_df,
        posts_core_df=posts_core_df,
        min_likes_per_user=min_likes_per_user,
        random_seed=cap_random_seed,
        logger=logger,
        n_users_before_join_verify=n_users_final,
    )
    
    # Update stats with join verification results
    if 'likes' in all_stats:
        all_stats['likes'].update(join_stats)
    
    mem_tracker.checkpoint("after_join_verification", quiet=True)
    
    # Memory summary: compare actual vs estimated
    memory_summary = mem_tracker.summary()
    all_stats['memory_actual'] = memory_summary
    
    # Log comparison
    if memory_estimate and 'estimated_peak_gb' in memory_estimate:
        actual_peak = memory_summary.get('peak_process_gb', 0)
        estimated_peak = memory_estimate.get('estimated_peak_gb', 0)
        if estimated_peak > 0:
            accuracy_pct = 100.0 * actual_peak / estimated_peak
            logger.info(f"Memory estimation accuracy: actual peak {actual_peak:.3f} GB vs estimated {estimated_peak:.2f} GB ({accuracy_pct:.1f}%)")
    
    # Log comprehensive attrition report

    max_liking_users_for_report = max_liking_users if max_liking_users is not None else 0
    _log_data_attrition_report(all_stats, memory_estimate, min_likes_per_user, max_likes_per_user, max_liking_users_for_report, logger)
    
    return likes_core_df, posts_core_df, posts_core_path, embed_dim, all_stats


def _filter_likes_after_post_join(
    likes_core_df: pl.DataFrame,
    posts_core_df: pl.DataFrame,
    min_likes_per_user: int,
    random_seed: int,
    logger: logging.Logger,
    n_users_before_join_verify: int,
) -> Tuple[pl.DataFrame, Dict[str, int]]:
    """Once we've filtered posts, now filter likes to only those that have matching posts in the dataset"""
    # Get set of post URIs that actually exist in posts_core
    existing_post_uris = set(posts_core_df['at_uri'].unique().to_list())
    logger.info(f"Found {len(existing_post_uris):,} unique post URIs in posts_core")
    
    # Filter likes to only those with matching posts
    n_likes_before_join_filter = len(likes_core_df)
    likes_core_df = likes_core_df.filter(
        pl.col('subject_uri').is_in(existing_post_uris)
    )
    n_likes_after_join_filter = len(likes_core_df)
    n_likes_removed_by_join = n_likes_before_join_filter - n_likes_after_join_filter
    logger.info(f"After join filter: {n_likes_before_join_filter:,} -> {n_likes_after_join_filter:,} likes ({n_likes_removed_by_join:,} removed)")
    
    # Re-verify min-likes per user
    n_users_removed_by_join_verify = 0
    if min_likes_per_user > 0:
        # Filter users by min likes
        # (no need to use the max_liking_users functionality)
        sampled_users_df, n_users_before_join_verify, _, n_users_after_join_verify = _get_sampled_users_with_min_likes(
            likes_lf=likes_core_df.lazy(),
            min_likes_per_user=min_likes_per_user,
            max_liking_users=None,
            random_seed=random_seed
        )

        n_users_after_join_verify = sampled_users_df.height
        n_users_removed_by_join_verify = n_users_before_join_verify - n_users_after_join_verify

        if n_users_removed_by_join_verify > 0:
            logger.info(
                f"Min-likes verification after join: {n_users_before_join_verify:,} -> {n_users_after_join_verify:,} users "
                f"({n_users_removed_by_join_verify:,} removed due to insufficient likes after join)"
            )
            # Filter likes to only users who still meet threshold
            valid_user_dids = set(sampled_users_df['did'].to_list())
            likes_core_df = likes_core_df.filter(
                pl.col('did').is_in(valid_user_dids)
            )
            n_likes_after_final_filter = len(likes_core_df)
            logger.info(f"Final likes after user filter: {n_likes_after_final_filter:,} likes")
        else:
            logger.info(f"All {n_users_before_join_verify:,} users still meet min-likes threshold after join")

    stats = {
        'n_likes_removed_by_join': n_likes_removed_by_join,
        'n_users_removed_by_join_verify': n_users_removed_by_join_verify,
        'n_likes_final_after_join': len(likes_core_df),
        'n_users_final_after_join': likes_core_df['did'].n_unique() if len(likes_core_df) > 0 else 0,
    }
    return likes_core_df, stats


def _log_data_attrition_report(
    all_stats: Dict[str, Any],
    memory_estimate: Optional[Dict[str, Any]],
    min_likes_per_user: int,
    max_likes_per_user: int,
    max_liking_users: int,
    logger,
) -> None:
    """
    Log a comprehensive data attrition report showing case counts and memory
    usage at each filtering stage.
    
    This consolidates the scattered statistics into a single, easy-to-read
    summary at the end of the data-getting process.
    """
    likes_stats = all_stats.get('likes', {})
    posts_stats = all_stats.get('posts', {})
    memory_actual = all_stats.get('memory_actual', {})
    
    # Helper for safe percentage calculation
    def pct(part, whole):
        return 100.0 * part / whole if whole > 0 else 0.0
    
    # Helper for formatting numbers with commas
    def fmt(n):
        if n is None:
            return '-'
        return f"{n:,}"
    
    # Build memory checkpoint lookup
    mem_checkpoints = {}
    for cp in memory_actual.get('checkpoints', []):
        mem_checkpoints[cp['name']] = cp
    
    # Extract likes pipeline stats
    n_likes_initial = likes_stats.get('n_likes_initial', 0)
    n_users_initial = likes_stats.get('n_users_initial', 0)
    n_users_eligible = likes_stats.get('n_users_eligible_for_sampling', 0)
    n_users_excluded_min = likes_stats.get('n_users_excluded_min_likes', 0)
    n_users_sampled = likes_stats.get('n_users_sampled', 0)
    n_likes_after_user_sample = likes_stats.get('n_likes_after_user_sample', 0)
    n_likes_after_cap = likes_stats.get('n_likes_after_per_user_cap', 0)
    n_likes_final = likes_stats.get('n_likes_final', 0)
    n_users_final = likes_stats.get('n_users_final', 0)
    n_likes_removed_join = likes_stats.get('n_likes_removed_by_join', 0)
    n_users_removed_join = likes_stats.get('n_users_removed_by_join_verify', 0)
    n_users_final_join = likes_stats.get('n_users_final_after_join', 0)
    n_likes_final_join = likes_stats.get('n_likes_final_after_join', 0)
    
    # Extract posts pipeline stats
    n_posts_total = posts_stats.get('n_posts_total', 0)
    n_liked_posts = posts_stats.get('n_liked_posts', 0)
    n_liked_only = posts_stats.get('n_liked_only', 0)
    n_liked_in_random = posts_stats.get('n_liked_in_random_sample', 0)
    n_random_sample = posts_stats.get('n_random_sample', 0)
    n_posts_core = posts_stats.get('n_posts_core', 0)
    match_rate = posts_stats.get('liked_post_match_rate', 0)
    
    # Calculate derived stats
    n_likes_after_join = n_likes_final - n_likes_removed_join if n_likes_removed_join else n_likes_final
    n_users_before_join_verify = n_users_final
    
    # Memory stats
    peak_actual = memory_actual.get('peak_process_gb', 0)
    mem_after_likes = mem_checkpoints.get('after_likes_load', {}).get('process_gb', 0)
    mem_after_posts = mem_checkpoints.get('after_posts_load_and_expansion', {}).get('process_gb', 0)
    
    # Build the report
    sep = "=" * 80
    sep2 = "-" * 80
    
    logger.info("")
    logger.info(sep)
    logger.info("DATA ATTRITION REPORT")
    logger.info(sep)
    
    # === LIKES PIPELINE ===
    logger.info("")
    logger.info("LIKES PIPELINE")
    logger.info(sep2)
    logger.info(f"{'Stage':<45} {'Users':>12} {'Likes':>15} {'Mem(GB)':>10}")
    logger.info(sep2)
    
    # 1. Initial scan
    logger.info(f"{'1. Initial scan (time-filtered)':<45} {fmt(n_users_initial):>12} {fmt(n_likes_initial):>15} {'':>10}")
    
    # 2. Min-likes pre-filter
    if n_users_excluded_min > 0:
        excluded_pct = pct(n_users_excluded_min, n_users_initial)
        logger.info(f"{'2. Min-likes pre-filter (>=' + str(min_likes_per_user) + ')':<45} {fmt(n_users_eligible):>12} {'(n/a)':>15} {'':>10}")
        logger.info(f"{'   - Excluded users':<45} {'-' + fmt(n_users_excluded_min):>12} {f'({excluded_pct:.1f}%)':>15} {'':>10}")
    else:
        logger.info(f"{'2. Min-likes pre-filter (>=' + str(min_likes_per_user) + ')':<45} {fmt(n_users_eligible):>12} {'(n/a)':>15} {'':>10}")
    
    # 3. User sampling
    if max_liking_users > 0 and n_users_eligible > 0:
        sample_pct = pct(n_users_sampled, n_users_eligible)
        logger.info(f"{'3. User sampling (' + fmt(max_liking_users) + ')':<45} {fmt(n_users_sampled):>12} {'(n/a)':>15} {'':>10}")
        logger.info(f"{'   - Retained':<45} {f'{sample_pct:.1f}%':>12} {'':>15} {'':>10}")
    else:
        logger.info(f"{'3. User sampling (no cap)':<45} {fmt(n_users_sampled):>12} {'(n/a)':>15} {'':>10}")
    
    # 4. Collect sampled user likes
    if n_likes_initial > 0:
        likes_sample_pct = pct(n_likes_after_user_sample, n_likes_initial)
        logger.info(f"{'4. Collect sampled user likes':<45} {'(n/a)':>12} {fmt(n_likes_after_user_sample):>15} {f'{mem_after_likes:.2f}':>10}")
        logger.info(f"{'   - Retained from initial':<45} {'':>12} {f'({likes_sample_pct:.1f}%)':>15} {'':>10}")
    
    # 5. Per-user cap
    if n_likes_after_user_sample > 0:
        cap_pct = pct(n_likes_after_cap, n_likes_after_user_sample)
        logger.info(f"{'5. Per-user cap (' + str(max_likes_per_user) + ')':<45} {'(n/a)':>12} {fmt(n_likes_after_cap):>15} {'':>10}")
        logger.info(f"{'   - Retained':<45} {'':>12} {f'({cap_pct:.1f}%)':>15} {'':>10}")
    
    # 6. Min-likes verification
    n_likes_removed_verify = n_likes_after_cap - n_likes_final if n_likes_after_cap > n_likes_final else 0
    n_users_removed_verify = n_users_sampled - n_users_final if n_users_sampled > n_users_final else 0
    logger.info(f"{'6. Min-likes verification':<45} {fmt(n_users_final):>12} {fmt(n_likes_final):>15} {'':>10}")
    if n_users_removed_verify > 0 or n_likes_removed_verify > 0:
        logger.info(f"{'   - Removed (edge cases)':<45} {'-' + fmt(n_users_removed_verify):>12} {'-' + fmt(n_likes_removed_verify):>15} {'':>10}")
    
    logger.info(sep2)
    
    # === POSTS PIPELINE ===
    logger.info("")
    logger.info("POSTS PIPELINE")
    logger.info(sep2)
    logger.info(f"{'Stage':<45} {'Posts':>15} {'Mem(GB)':>10}")
    logger.info(sep2)
    
    logger.info(f"{'1. Time-filtered scan':<45} {fmt(n_posts_total):>15} {'':>10}")
    logger.info(f"{'2. Liked posts extracted':<45} {fmt(n_liked_posts):>15} {f'{mem_after_posts:.2f}':>10}")
    logger.info(f"{'   - Match rate vs liked URIs':<45} {f'{match_rate:.1f}%':>15} {'':>10}")
    logger.info(f"{'3. Random sample (reservoir)':<45} {fmt(n_random_sample):>15} {'':>10}")
    if n_liked_in_random > 0:
        logger.info(f"{'   - Overlap with liked posts':<45} {fmt(n_liked_in_random):>15} {'':>10}")
    logger.info(f"{'4. Combined output':<45} {fmt(n_posts_core):>15} {'':>10}")
    logger.info(f"   ({fmt(n_liked_only)} liked-only + {fmt(n_random_sample)} random)")
    
    logger.info(sep2)
    
    # === POST-JOIN VERIFICATION ===
    logger.info("")
    logger.info("POST-JOIN VERIFICATION")
    logger.info(sep2)
    logger.info(f"{'Stage':<45} {'Users':>12} {'Likes':>15}")
    logger.info(sep2)
    
    logger.info(f"{'Before join filter':<45} {fmt(n_users_final):>12} {fmt(n_likes_final):>15}")
    
    if n_likes_removed_join > 0:
        join_filter_likes_pct = pct(n_likes_removed_join, n_likes_final)
        n_users_after_join_filter = n_users_final - n_users_removed_join if n_users_removed_join else n_users_final
        n_likes_after_join_filter = n_likes_final - n_likes_removed_join
        logger.info(f"{'After join filter':<45} {fmt(n_users_before_join_verify):>12} {fmt(n_likes_after_join_filter):>15}")
        logger.info(f"{'  - Likes removed (no matching post)':<45} {'':>12} {'-' + fmt(n_likes_removed_join) + f' ({join_filter_likes_pct:.1f}%)':>15}")
    
    if n_users_removed_join > 0:
        join_users_pct = pct(n_users_removed_join, n_users_before_join_verify)
        logger.info(f"{'After min-likes re-verify':<45} {fmt(n_users_final_join):>12} {fmt(n_likes_final_join):>15}")
        logger.info(f"{'  - Users removed (<' + str(min_likes_per_user) + ' joinable)':<45} {'-' + fmt(n_users_removed_join) + f' ({join_users_pct:.1f}%)':>12} {'':>15}")
    
    logger.info(sep2)
    
    # === FINAL OUTPUT ===
    logger.info("")
    logger.info("FINAL OUTPUT")
    logger.info(sep2)
    logger.info(f"likes_core.parquet:  {fmt(n_users_final_join)} users, {fmt(n_likes_final_join)} likes")
    logger.info(f"posts_core.parquet:  {fmt(n_posts_core)} posts ({fmt(n_liked_only)} liked + {fmt(n_random_sample)} random)")
    logger.info(sep2)
    
    # === OVERALL ATTRITION SUMMARY ===
    logger.info("")
    logger.info("OVERALL ATTRITION SUMMARY")
    logger.info(sep2)
    
    if n_users_initial > 0 and n_users_final_join > 0:
        users_retained_pct = pct(n_users_final_join, n_users_initial)
        logger.info(f"Users: {fmt(n_users_initial)} -> {fmt(n_users_final_join)} ({users_retained_pct:.2f}% retained)")
    
    if n_likes_initial > 0 and n_likes_final_join > 0:
        likes_retained_pct = pct(n_likes_final_join, n_likes_initial)
        logger.info(f"Likes: {fmt(n_likes_initial)} -> {fmt(n_likes_final_join)} ({likes_retained_pct:.2f}% retained)")
    
    if n_posts_total > 0 and n_posts_core > 0:
        posts_retained_pct = pct(n_posts_core, n_posts_total)
        logger.info(f"Posts: {fmt(n_posts_total)} -> {fmt(n_posts_core)} ({posts_retained_pct:.2f}% retained)")
    
    logger.info(sep2)
    
    # === MEMORY SUMMARY ===
    logger.info("")
    logger.info("MEMORY SUMMARY")
    logger.info(sep2)
    logger.info(f"{'Phase':<40} {'Memory (GB)':>12} {'Elapsed (s)':>12}")
    logger.info(sep2)
    
    for cp in memory_actual.get('checkpoints', []):
        name = cp.get('name', 'unknown')
        mem_gb = cp.get('process_gb', 0)
        elapsed = cp.get('elapsed_sec', 0)
        # Make checkpoint names more readable
        display_name = name.replace('_', ' ').title()
        logger.info(f"{display_name:<40} {mem_gb:>12.2f} {elapsed:>12.1f}")
    
    logger.info(sep2)
    
    # Peak memory comparison
    if memory_estimate and memory_estimate.get('estimated_peak_gb', 0) > 0:
        peak_estimated = memory_estimate.get('estimated_peak_gb', 0) if memory_estimate else 0
        accuracy = pct(peak_actual, peak_estimated)
        logger.info(f"Peak process memory: {peak_actual:.2f} GB (estimated: {peak_estimated:.2f} GB, accuracy: {accuracy:.1f}%)")
    else:
        logger.info(f"Peak process memory: {peak_actual:.2f} GB")
    
    logger.info(sep)
    logger.info("")


def _log_likes_distribution_plot(
    tracker,
    likes_per_user: list,
    cap_value: int,
    logger,
) -> None:
    """
    Log a histogram of likes per user to the experiment tracker.
    Shows the distribution with a vertical line at the per-user cap.
    
    This helps visualize how many users are affected by the cap.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram
        likes_arr = np.array(likes_per_user)
        
        # Determine bin edges - use log scale if distribution is very skewed
        max_val = likes_arr.max()
        if max_val > 10 * np.median(likes_arr):
            # Log-spaced bins for skewed distribution
            bins = np.logspace(0, np.log10(max_val + 1), 50)
            ax.set_xscale('log')
        else:
            bins = 50
        
        # Plot histogram
        counts, bin_edges, patches = ax.hist(
            likes_arr, 
            bins=bins, 
            alpha=0.7, 
            color='steelblue',
            edgecolor='white',
            linewidth=0.5,
        )
        
        # Add vertical line at cap value
        ax.axvline(
            x=cap_value, 
            color='red', 
            linestyle='--', 
            linewidth=2,
            label=f'Per-user cap = {cap_value}'
        )
        
        # Calculate stats
        n_above_cap = np.sum(likes_arr > cap_value)
        pct_above_cap = 100.0 * n_above_cap / len(likes_arr)
        
        # Add annotations
        ax.set_xlabel('Likes per User (before cap)')
        ax.set_ylabel('Number of Users')
        ax.set_title(f'Likes Distribution for Sampled Users\n'
                     f'({n_above_cap:,} users ({pct_above_cap:.1f}%) exceed cap)')
        ax.legend()
        
        # Add text box with stats
        stats_text = (f'N users: {len(likes_arr):,}\n'
                     f'Mean: {np.mean(likes_arr):.1f}\n'
                     f'Median: {np.median(likes_arr):.0f}\n'
                     f'P90: {np.percentile(likes_arr, 90):.0f}\n'
                     f'Max: {max_val:,}')
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Log to ClearML
        tracker.log_plot(
            title="Likes Distribution",
            series="Sampled Users",
            figure=fig,
        )
        
        plt.close(fig)
        
        if logger:
            logger.info(f"Logged likes distribution plot to experiment tracker "
                       f"({pct_above_cap:.1f}% of users exceed cap)")
    
    except Exception as e:
        if logger:
            logger.warning(f"Failed to create likes distribution plot: {e}")


def _attrition_stats_to_experiment_tracker(
    all_stats: Dict[str, Any], 
    context: Context, 
    max_likes_per_user: int, 
    logger: logging.Logger
):
    # Likes pipeline attrition metrics
    if 'likes' in all_stats:
        likes_stats = all_stats['likes']
        
        # Initial counts
        context.tracker.log_single_value(
            name="Likes - 1 Initial Users", 
            value=likes_stats.get('n_users_initial', 0)
        )
        context.tracker.log_single_value(
            name="Likes - 1 Initial Likes", 
            value=likes_stats.get('n_likes_initial', 0)
        )
        
        # After each filtering stage
        context.tracker.log_single_value(
            name="Likes - 2 Eligible Users (min-likes)",
            value=likes_stats.get('n_users_eligible_for_sampling', 0)
        )
        context.tracker.log_single_value(
            name="Likes - 3 Sampled Users",
            value=likes_stats.get('n_users_sampled', 0)
        )
        context.tracker.log_single_value(
            name="Likes - 4 Likes After User Sample",
            value=likes_stats.get('n_likes_after_user_sample', 0)
        )
        context.tracker.log_single_value(
            name="Likes - 5 Likes After Per-User Cap",
            value=likes_stats.get('n_likes_after_per_user_cap', 0)
        )
        
        # Final counts (before and after join)
        context.tracker.log_single_value(
            name="Likes - 6 Final Users (pre-join)", 
            value=likes_stats.get('n_users_final', 0)
        )
        context.tracker.log_single_value(
            name="Likes - 6 Final Likes (pre-join)",
            value=likes_stats.get('n_likes_final', 0)
        )
        context.tracker.log_single_value(
            name="Likes - 7 Final Users (post-join)",
            value=likes_stats.get('n_users_final_after_join', 0)
        )
        context.tracker.log_single_value(
            name="Likes - 7 Final Likes (post-join)",
            value=likes_stats.get('n_likes_final_after_join', 0)
        )
        
        # Attrition percentages (for easy comparison)
        n_users_initial = likes_stats.get('n_users_initial', 0)
        n_users_final = likes_stats.get('n_users_final_after_join', 0)
        n_likes_initial = likes_stats.get('n_likes_initial', 0)
        n_likes_final = likes_stats.get('n_likes_final_after_join', 0)
        
        if n_users_initial > 0:
            context.tracker.log_single_value(
                name="Retention - Users %",
                value=100.0 * n_users_final / n_users_initial
            )
        if n_likes_initial > 0:
            context.tracker.log_single_value(
                name="Retention - Likes %",
                value=100.0 * n_likes_final / n_likes_initial
            )
        
        # Like count distribution stats (for understanding cap impact)
        if 'likes_per_user_mean' in likes_stats:
            context.tracker.log_single_value(
                name="Distribution - Likes/User Mean",
                value=likes_stats.get('likes_per_user_mean', 0)
            )
            context.tracker.log_single_value(
                name="Distribution - Likes/User Median",
                value=likes_stats.get('likes_per_user_median', 0)
            )
            context.tracker.log_single_value(
                name="Distribution - Likes/User Max",
                value=likes_stats.get('likes_per_user_max', 0)
            )
            context.tracker.log_single_value(
                name="Distribution - Likes/User P90",
                value=likes_stats.get('likes_per_user_p90', 0)
            )
            context.tracker.log_single_value(
                name="Distribution - Likes/User P99",
                value=likes_stats.get('likes_per_user_p99', 0)
            )
        
        # Plot histogram of likes per user (with cap line)
        if 'likes_per_user_distribution' in likes_stats:
            _log_likes_distribution_plot(
                context.tracker, 
                likes_stats['likes_per_user_distribution'],
                max_likes_per_user,
                logger
            )
    
    # Posts pipeline metrics
    if 'posts' in all_stats:
        posts_stats = all_stats['posts']
        context.tracker.log_single_value(
            name="Posts - 1 Total (time-filtered)",
            value=posts_stats.get('n_posts_total', 0)
        )
        context.tracker.log_single_value(
            name="Posts - 2 Liked Posts Found",
            value=posts_stats.get('n_liked_posts', 0)
        )
        context.tracker.log_single_value(
            name="Posts - 3 Random Sample",
            value=posts_stats.get('n_random_sample', 0)
        )
        context.tracker.log_single_value(
            name="Posts - Match Rate %",
            value=posts_stats.get('liked_post_match_rate', 0)
        )
    
    # Memory metrics (critical for sweep analysis)
    if 'memory_actual' in all_stats:
        mem_stats = all_stats['memory_actual']
        context.tracker.log_single_value(
            name="Memory - Peak GB",
            value=mem_stats.get('peak_process_gb', 0)
        )
        context.tracker.log_single_value(
            name="Memory - Start GB",
            value=mem_stats.get('start_process_gb', 0)
        )
        context.tracker.log_single_value(
            name="Memory - End GB",
            value=mem_stats.get('end_process_gb', 0)
        )
        context.tracker.log_single_value(
            name="Memory - Growth GB",
            value=mem_stats.get('growth_gb', 0)
        )
    
    # Memory estimate vs actual (for estimator accuracy tracking)
    if 'memory_estimate' in all_stats:
        est_stats = all_stats['memory_estimate']
        context.tracker.log_single_value(
            name="Memory - Estimated Peak GB",
            value=est_stats.get('estimated_peak_gb', 0)
        )
        
        # Estimation accuracy
        actual_peak = all_stats.get('memory_actual', {}).get('peak_process_gb', 0)
        estimated_peak = est_stats.get('estimated_peak_gb', 0)
        if estimated_peak > 0:
            context.tracker.log_single_value(
                name="Memory - Estimate Accuracy %",
                value=100.0 * actual_peak / estimated_peak
            )
