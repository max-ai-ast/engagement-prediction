#!/usr/bin/env python3

"""
Stage 1: Get and filter data using memory-efficient incremental processing.

This stage implements a two-pass filtering pipeline to produce core datasets:
- likes_core.parquet: filtered likes (user sampling, per-user caps, min-likes)
- posts_core.parquet: liked posts + random negative sample, with expanded embeddings

================================================================================
FILTERING SEQUENCE
================================================================================

PHASE 1: Likes Processing (load_likes_core_polars in helpers.py)
-------------------------------------------------------------------------
Pass 1 - Count likes per user:
  - Scan all likes parquet files in batches (20 files at a time)
  - Build a Dict[user_did, like_count] to track likes per user
  - Apply time-range filter (--likes-start / --likes-end)

Pre-filter for minimum likes:
  - Exclude users with fewer than --min-likes-per-user from the eligible pool
  - This happens BEFORE sampling, ensuring we don't waste sample slots on users
    who would later be filtered out

Sample users (if --max-liking-users is set):
  - Randomly sample from the eligible user pool (not all users)
  - Uses --cap-random-seed for reproducibility

Pass 2 - Collect likes for sampled users:
  - Scan files again, keeping only likes from sampled users
  - Accumulates likes incrementally to bound memory usage

Per-user random cap (--max-likes-per-user):
  - For each user, randomly select up to the cap limit
  - IMPORTANT: Random selection, NOT recency-based, to avoid the model learning
    spurious time patterns (e.g., "recent likes predict engagement")

Final min-likes verification:
  - Re-check that users still meet --min-likes-per-user after the per-user cap
  - Handles edge cases where capping reduced a user below the threshold

PHASE 2: Posts Processing (load_posts_core_polars in helpers.py)
-------------------------------------------------------------------------
Extract liked post URIs:
  - Get unique subject_uri values from the filtered likes

Process posts in batches with early embedding expansion:
  - Scan posts parquet files in batches (20 files at a time)
  - Apply time-range filter (--posts-start / --posts-end)
  - For each batch:
    a) Reservoir sampling for random sample (from ALL posts, independent of like status)
    b) Extract liked posts NOT already in the random sample
    c) Expand embeddings immediately (float list -> individual columns)
    d) Drop the raw embedding blob to free memory
    e) Accumulate only the slim expanded data

Reservoir sampling for random sample (--negative-posts-sample):
  - Maintain a reservoir of posts sampled from ALL posts (not filtered by like status)
  - This ensures the random sample is STATISTICALLY INDEPENDENT of like status
  - Can be used for unbiased population statistics and as negative examples
  - Early embedding expansion reduces memory footprint
  - Some posts in the random sample may also be liked (this is expected and correct)

Combine outputs:
  - Concatenate liked-only posts + random sample
  - Add 'in_random_sample' flag:
    * True = post was collected via reservoir sampling (proper random sample)
    * False = post was collected only because it was liked
  - To identify liked posts: join with likes_core on subject_uri = at_uri

================================================================================
DESIGN RATIONALE
================================================================================

Two-pass processing:
  Memory usage is bounded regardless of total data size. Pass 1 only stores
  user DIDs and counts; Pass 2 only stores likes for sampled users.

Pre-filtering before sampling:
  Without this, sampling 100k users might yield only 67k after min-likes filter.
  Pre-filtering ensures the full sample budget is used on valid users.

Random capping (not recency-based):
  If we kept only the N most recent likes per user, the model could learn that
  "recency = engagement" rather than content-based patterns. Random selection
  prevents this temporal leakage.

Early embedding expansion:
  Raw embeddings are ~150KB/post (serialized float lists). Expanded columns are
  ~2KB/post. Expanding per-batch reduces peak memory by ~98%.

Statistically independent random sample:
  The random sample is drawn from ALL posts, not filtered by like status. This
  is critical for:
  1. Unbiased population statistics (e.g., computing base rates)
  2. Proper negative sampling (liked posts appearing in random sample is correct
     behavior - it reflects the true probability of a random post being liked)
  3. Downstream analyses that require a representative sample
  
  Note: Some posts in the random sample will also be liked by our users. This is
  expected and correct - excluding liked posts would bias the sample toward less
  engaging content.

Reservoir sampling algorithm:
  Ensures the random sample is representative of the full time range, not
  biased toward early or late files in the scan order.

================================================================================
OUTPUTS
================================================================================

Under <run_dir>/01_get_data/<timestamp>/:
  - likes_core.parquet: did, subject_uri, record_created_at
  - posts_core.parquet: post columns + post_emb_0..N + in_random_sample flag
  - summary.json: full filtering statistics and parameters
  - stage.log: detailed execution log with memory checkpoints

Using the outputs:
  - Random sample (for population stats): filter posts_core where in_random_sample=True
  - Liked posts: join likes_core.subject_uri with posts_core.at_uri
  - Negative examples for training: sample from random sample excluding user's likes
"""

from __future__ import annotations

import json
import argparse
import time
from pathlib import Path
from typing import Dict, Any

from utils.pipeline.core import new_stage_timestamp_dir, Context
from utils.helpers import (
    get_stage_logger,
    log_operation_start,
    load_likes_core_polars,
    load_posts_core_polars,
    # Memory safety checks and tracking
    check_data_load_safe,
    MemoryTracker,
    log_memory_checkpoint,
    list_files_in_range_ingex_gcs,
    parse_one_ts,
    # Legacy imports for DigitalOcean fallback
    load_most_recent_raw_data_digital_ocean,
)


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '01_get_data')

    # Initialize logger
    logger = get_stage_logger('STAGE_01_GET_DATA', log_file=out_dir / 'stage.log')

    # Parameters (defaults are set in cli.py)
    data_source = args.data_source

    t0 = time.time()

    if data_source == 'greenearth':
        # Use new Polars-based filtering pipeline
        likes_core_df, posts_core_df, embed_dim, all_stats = _run_greenearth_pipeline(
            args, logger, context
        )
    elif data_source == 'digitalocean':
        # Legacy path: DigitalOcean Spaces (kept for backwards compatibility)
        likes_core_df, posts_core_df, embed_dim, all_stats = _run_digitalocean_legacy(
            args, logger, context
        )
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    # Save outputs as parquet
    log_operation_start('Save core datasets as parquet', 'STAGE_01_GET_DATA', logger)
    ts_name = out_dir.name
    
    likes_core_path = out_dir / f"likes_core_{ts_name}.parquet"
    posts_core_path = out_dir / f"posts_core_{ts_name}.parquet"
    
    likes_core_df.write_parquet(likes_core_path)
    posts_core_df.write_parquet(posts_core_path)
    
    logger.info(f"Saved likes_core: {likes_core_path} ({len(likes_core_df):,} rows)")
    logger.info(f"Saved posts_core: {posts_core_path} ({len(posts_core_df):,} rows)")

    # Summary
    log_operation_start('Write summary files', 'STAGE_01_GET_DATA', logger)
    
    # Get parameters for summary
    gcs_bucket = getattr(args, 'gcs_bucket', None)
    posts_start = getattr(args, 'posts_start', None)
    posts_end = getattr(args, 'posts_end', None)
    likes_start = getattr(args, 'likes_start', None)
    likes_end = getattr(args, 'likes_end', None)
    max_liking_users = int(getattr(args, 'max_liking_users', 0))
    max_likes_per_user = int(getattr(args, 'max_likes_per_user', 100))
    min_likes_per_user = int(getattr(args, 'min_likes_per_user', 2))
    negative_posts_sample = int(getattr(args, 'negative_posts_sample', 100000))
    cap_random_seed = int(getattr(args, 'cap_random_seed', 42))
    embedding_model = getattr(args, 'embedding_model', 'all_MiniLM_L6_v2')
    
    summary = {
        'data_source': data_source,
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
            'likes_core_rows': len(likes_core_df),
            'posts_core_rows': len(posts_core_df),
            'embedding_dim': embed_dim,
        },
        'filtering_stats': all_stats,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Log to experiment tracker
    n_likes = len(likes_core_df)
    n_posts = len(posts_core_df)
    context.tracker.log_single_value(name="get_data/n_likes_core", value=n_likes)
    context.tracker.log_single_value(name="get_data/n_posts_core", value=n_posts)
    context.tracker.log_single_value(name="get_data/embedding_dim", value=embed_dim)
    
    if 'likes' in all_stats:
        context.tracker.log_single_value(
            name="get_data/n_users_initial", 
            value=all_stats['likes'].get('n_users_initial', 0)
        )
        context.tracker.log_single_value(
            name="get_data/n_users_final", 
            value=all_stats['likes'].get('n_users_final', 0)
        )

    runtime = time.time() - t0
    
    # Stage info
    info_lines = [
        f"stage: get_data",
        f"runtime_seconds: {runtime:.2f}",
        f"data_source: {data_source}",
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
    args: argparse.Namespace,
    logger,
    context: Context,
):
    """
    Run the new Polars-based filtering pipeline for GreenEarth Ingex data.
    
    Returns:
        Tuple of (likes_core_df, posts_core_df, embed_dim, stats_dict)
    """
    import polars as pl
    
    gcs_bucket = args.gcs_bucket
    posts_start = args.posts_start
    posts_end = args.posts_end
    likes_start = args.likes_start
    likes_end = args.likes_end
    
    max_liking_users = int(getattr(args, 'max_liking_users', 0))
    max_likes_per_user = int(getattr(args, 'max_likes_per_user', 100))
    min_likes_per_user = int(getattr(args, 'min_likes_per_user', 2))
    negative_posts_sample = int(getattr(args, 'negative_posts_sample', 100000))
    cap_random_seed = int(getattr(args, 'cap_random_seed', 42))
    embedding_model = getattr(args, 'embedding_model', 'all_MiniLM_L6_v2')
    max_memory_gb = float(getattr(args, 'max_memory_gb', 0))
    max_memory_pct = float(getattr(args, 'max_memory_pct', 0.75))
    skip_memory_check = bool(getattr(args, 'skip_memory_check', False))
    
    all_stats = {}
    
    # Initialize memory tracker for actual memory monitoring
    mem_tracker = MemoryTracker(logger=logger)
    mem_tracker.checkpoint("pipeline_start")
    
    # Pre-flight memory safety check
    log_operation_start('Pre-flight memory safety check', 'STAGE_01_GET_DATA', logger)
    
    # Get file paths for memory estimation
    likes_start_dt = parse_one_ts(likes_start)
    likes_end_dt = parse_one_ts(likes_end)
    posts_start_dt = parse_one_ts(posts_start)
    posts_end_dt = parse_one_ts(posts_end)
    
    likes_paths = list_files_in_range_ingex_gcs(
        gcs_bucket=gcs_bucket,
        blob_prefix='bsky_likes',
        start=likes_start_dt,
        end=likes_end_dt,
    )
    posts_paths = list_files_in_range_ingex_gcs(
        gcs_bucket=gcs_bucket,
        blob_prefix='bsky_posts',
        start=posts_start_dt,
        end=posts_end_dt,
    )
    
    # Smart memory check that accounts for filtering parameters
    if skip_memory_check:
        logger.warning("=" * 60)
        logger.warning("SKIPPING MEMORY CHECK (--skip-memory-check flag set)")
        logger.warning("Monitor memory usage carefully - OOM may occur!")
        logger.warning("=" * 60)
        memory_estimate = {'skipped': True, 'reason': 'skip_memory_check flag set'}
    else:
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
            logger=logger,
        )
        logger.info("Memory check passed, proceeding with data load")
    all_stats['memory_estimate'] = memory_estimate
    
    mem_tracker.checkpoint("after_memory_check")
    
    # Step 1: Load and filter likes
    log_operation_start('Load and filter likes data', 'STAGE_01_GET_DATA', logger)
    likes_core_df, likes_stats = load_likes_core_polars(
        gcs_bucket=gcs_bucket,
        start_str=likes_start,
        end_str=likes_end,
        max_liking_users=max_liking_users,
        max_likes_per_user=max_likes_per_user,
        min_likes_per_user=min_likes_per_user,
        random_seed=cap_random_seed,
        logger=logger,
    )
    all_stats['likes'] = likes_stats
    
    mem_tracker.checkpoint("after_likes_load")
    
    # Step 2: Extract liked post URIs
    log_operation_start('Extract liked post URIs', 'STAGE_01_GET_DATA', logger)
    liked_post_uris = set(likes_core_df['subject_uri'].unique().to_list())
    logger.info(f"Extracted {len(liked_post_uris):,} unique liked post URIs")
    
    mem_tracker.checkpoint("after_uri_extraction")
    
    # Step 3: Load posts with early embedding expansion
    # This loads posts, filters to liked + negative sample, and expands embeddings per-batch
    # Early expansion dramatically reduces memory by dropping raw embedding blobs immediately
    log_operation_start('Load posts with early embedding expansion', 'STAGE_01_GET_DATA', logger)
    posts_core_df, posts_stats, embed_dim = load_posts_core_polars(
        gcs_bucket=gcs_bucket,
        start_str=posts_start,
        end_str=posts_end,
        liked_post_uris=liked_post_uris,
        negative_posts_sample=negative_posts_sample,
        embedding_model=embedding_model,
        random_seed=cap_random_seed,
        logger=logger,
    )
    all_stats['posts'] = posts_stats
    all_stats['embedding_dim'] = embed_dim
    
    mem_tracker.checkpoint("after_posts_load_and_expansion")
    
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
    
    return likes_core_df, posts_core_df, embed_dim, all_stats


def _run_digitalocean_legacy(
    args: argparse.Namespace,
    logger,
    context: Context,
):
    """
    Legacy path for DigitalOcean Spaces data.
    Converts to the new output format for compatibility.
    
    Returns:
        Tuple of (likes_core_df, posts_core_df, embed_dim, stats_dict)
    """
    import polars as pl
    import pandas as pd
    import numpy as np
    
    max_files = int(args.max_files_per_table)
    max_liking_users = int(getattr(args, 'max_liking_users', 0))
    cap_random_seed = int(getattr(args, 'cap_random_seed', 42))
    
    log_operation_start('Load data from DigitalOcean Spaces (legacy)', 'STAGE_01_GET_DATA', logger)
    posts_pdf, likes_pdf, metadata_df = load_most_recent_raw_data_digital_ocean(max_files)
    
    all_stats = {
        'legacy_mode': True,
        'max_files_per_table': max_files,
    }
    
    n_users_before = 0
    n_users_after = 0
    
    # User-level downsampling
    if 'did' in likes_pdf.columns:
        unique_users = likes_pdf['did'].unique()
        n_users_before = len(unique_users)
        
        if max_liking_users > 0 and n_users_before > max_liking_users:
            log_operation_start(f'Downsample liking users: {n_users_before} -> {max_liking_users}', 'STAGE_01_GET_DATA', logger)
            rng = np.random.RandomState(cap_random_seed)
            sampled_users = set(rng.choice(unique_users, size=max_liking_users, replace=False))
            likes_pdf = likes_pdf[likes_pdf['did'].isin(sampled_users)]
            n_users_after = max_liking_users
        else:
            n_users_after = n_users_before
    
    all_stats['likes'] = {
        'n_users_initial': n_users_before,
        'n_users_final': n_users_after,
        'n_likes_final': len(likes_pdf),
    }
    all_stats['posts'] = {
        'n_posts_total': len(posts_pdf),
        'n_posts_core': len(posts_pdf),
    }
    
    # Convert to polars
    likes_core_df = pl.from_pandas(likes_pdf)
    posts_core_df = pl.from_pandas(posts_pdf)
    
    # Add in_random_sample flag (all false for legacy mode - no random sampling done)
    posts_core_df = posts_core_df.with_columns(pl.lit(False).alias('in_random_sample'))
    
    # No embeddings in DO data by default
    embed_dim = 0
    
    return likes_core_df, posts_core_df, embed_dim, all_stats
