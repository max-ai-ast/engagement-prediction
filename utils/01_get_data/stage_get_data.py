#!/usr/bin/env python3

"""
Stage 1: Get and filter data using Polars-based lazy evaluation.

This stage implements efficient filtering to produce core datasets:
- likes_core.parquet: filtered likes (user sampling, per-user caps, min-likes)
- posts_core.parquet: liked posts + random sample for negatives, with expanded embeddings

Filtering Sequence:
1. Scan likes parquets for liking user DIDs
2. Sample liking users if --max-liking-users cap is set
3. Filter likes to sampled users
4. Random cap likes per user (NOT recency-based, to avoid model learning spurious time patterns)
5. Filter users with fewer than --min-likes-per-user
6. Extract liked post URIs
7. Load liked posts from posts parquets
8. Sample random posts for negative cases
9. Expand pre-computed embeddings into columns

Outputs under <run_dir>/01_get_data/<timestamp>/:
- likes_core.parquet: did, subject_uri, record_created_at
- posts_core.parquet: all post columns + post_emb_* + is_liked flag
- summary.json: filtering statistics and parameters
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
    expand_embeddings_polars,
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
    all_stats['memory_estimate'] = memory_estimate
    logger.info("Memory check passed, proceeding with data load")
    
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
    
    # Step 3: Load posts (liked + negative sample)
    log_operation_start('Load and sample posts data', 'STAGE_01_GET_DATA', logger)
    posts_df, posts_stats = load_posts_core_polars(
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
    
    mem_tracker.checkpoint("after_posts_load")
    
    # Step 4: Expand embeddings
    log_operation_start('Expand embeddings', 'STAGE_01_GET_DATA', logger)
    posts_core_df, embed_dim = expand_embeddings_polars(
        posts_df,
        embedding_model=embedding_model,
        logger=logger,
    )
    all_stats['embedding_dim'] = embed_dim
    
    mem_tracker.checkpoint("after_embedding_expansion")
    
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
    
    # Add is_liked flag (all true for legacy mode - no negative sampling)
    posts_core_df = posts_core_df.with_columns(pl.lit(True).alias('is_liked'))
    
    # No embeddings in DO data by default
    embed_dim = 0
    
    return likes_core_df, posts_core_df, embed_dim, all_stats
