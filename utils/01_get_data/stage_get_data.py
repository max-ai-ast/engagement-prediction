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

POST-JOIN MIN-LIKES VERIFICATION:
  - After Phase 2, re-check min-likes threshold based on likes that have matching posts
  - Since like-post joining isn't perfect (some posts may be missing, deleted, or outside
    time range), users who met the threshold in Phase 1 may drop below it after the join
  - Filter out likes for posts that don't exist in posts_core
  - Remove users who no longer meet --min-likes-per-user after the join filter
  - This ensures all users in the final dataset have at least min-likes with valid posts

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
from typing import Dict, Any, Optional

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
    # Data validation
    validate_dataframe_schema,
)


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '01_get_data')

    # Initialize logger
    logger = get_stage_logger('STAGE_01_GET_DATA', log_file=out_dir / 'stage.log')

    t0 = time.time()

    # Use Polars-based filtering pipeline for GreenEarth Ingex data
    likes_core_df, posts_core_df, embed_dim, all_stats = _run_greenearth_pipeline(
        args, logger, context
    )

    # Validate output schemas before saving
    log_operation_start('Validate output schemas', 'STAGE_01_GET_DATA', logger)
    
    # Validate likes_core schema
    likes_schema = {
        'did': str,
        'subject_uri': str,
        'record_created_at': 'datetime',
    }
    validate_dataframe_schema(likes_core_df, likes_schema, allow_extra_columns=False)
    logger.info("✓ likes_core schema validated")
    
    # Validate posts_core schema (dynamic embedding columns + all extra columns)
    posts_schema = {
        'at_uri': str,
        'in_random_sample': bool,
        # Extra columns from source data
        'did': str,
        'record_created_at': 'datetime',
        'record_text': str,
        'is_liked': bool,
    }
    # Add embedding columns dynamically
    for i in range(embed_dim):
        posts_schema[f'post_emb_{i}'] = float
    validate_dataframe_schema(posts_core_df, posts_schema, allow_extra_columns=False)
    
    logger.info(f"✓ posts_core schema validated (embed_dim={embed_dim})")

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
            'likes_core_rows': len(likes_core_df),
            'posts_core_rows': len(posts_core_df),
            'embedding_dim': embed_dim,
        },
        'filtering_stats': all_stats,
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Log to experiment tracker - comprehensive metrics for sweep analysis
    # Metric names use readable format: "Category - Metric Name" for better CSV exports
    n_likes = len(likes_core_df)
    n_posts = len(posts_core_df)
    
    # Primary outputs
    context.tracker.log_single_value(name="Output - Likes (final)", value=n_likes)
    context.tracker.log_single_value(name="Output - Posts (final)", value=n_posts)
    context.tracker.log_single_value(name="Output - Embedding Dim", value=embed_dim)
    
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
    
    max_liking_users = args.max_liking_users
    if max_liking_users is not None:
        max_liking_users = int(max_liking_users)
    max_likes_per_user = int(args.max_likes_per_user)
    min_likes_per_user = int(args.min_likes_per_user)
    negative_posts_sample = int(args.negative_posts_sample)
    cap_random_seed = int(args.cap_random_seed)
    embedding_model = args.embedding_model
    max_memory_gb = args.max_memory_gb
    if max_memory_gb is not None:
        max_memory_gb = float(max_memory_gb)
    max_memory_pct = float(args.max_memory_pct)
    skip_memory_check = bool(args.skip_memory_check)
    
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
    
    memory_estimate = None
    if not skip_memory_check:
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
            skip_safety_check=skip_memory_check,
            logger=logger,
        )
        all_stats['memory_estimate'] = memory_estimate
        logger.info("Memory check passed, proceeding with data load")
    else:
        # Log additional warning if skip_memory_check is set
        logger.warning("=" * 60)
        logger.warning("SKIPPING MEMORY SAFETY CHECK (--skip-memory-check flag set)")
        logger.warning("Proceeding anyway - monitor memory usage carefully, OOM may occur!")
        logger.warning("=" * 60)
        
    
    mem_tracker.checkpoint("after_memory_check")
    
    # Step 1: Load and filter likes
    log_operation_start('Load and filter likes data', 'STAGE_01_GET_DATA', logger)
    likes_core_df, likes_stats = load_likes_core_polars(
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
    
    mem_tracker.checkpoint("after_likes_load")
    
    # Step 2: Extract liked post URIs
    log_operation_start('Extract liked post URIs', 'STAGE_01_GET_DATA', logger)
    # liked_post_uris = set(likes_core_df['subject_uri'].unique().to_list())
    liked_post_uris_df: pl.DataFrame = likes_core_df.select(pl.col('subject_uri').unique())
    # logger.info(f"Extracted {len(liked_post_uris_df):,} unique liked post URIs")
    
    mem_tracker.checkpoint("after_uri_extraction")
    
    # Step 3: Load posts with early embedding expansion
    # This loads posts, filters to liked + negative sample, and expands embeddings per-batch
    # Early expansion dramatically reduces memory by dropping raw embedding blobs immediately
    log_operation_start('Load posts with early embedding expansion', 'STAGE_01_GET_DATA', logger)
    posts_core_df, posts_stats, embed_dim = load_posts_core_polars(
        start_str=posts_start,
        end_str=posts_end,
        liked_post_uris_df=liked_post_uris_df,
        paths=posts_paths,
        negative_posts_sample=negative_posts_sample,
        embedding_model=embedding_model,
        random_seed=cap_random_seed,
        logger=logger,
    )
    all_stats['posts'] = posts_stats
    all_stats['embedding_dim'] = embed_dim
    
    mem_tracker.checkpoint("after_posts_load_and_expansion")
    
    # Step 4: Re-verify min-likes after like-post join
    # Since like-post joining isn't perfect (some posts may be missing, deleted, or outside time range),
    # we need to re-check that users still meet min-likes threshold based on likes that have matching posts.
    log_operation_start('Re-verify min-likes after like-post join', 'STAGE_01_GET_DATA', logger)
    
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
        user_like_counts = (
            likes_core_df.group_by('did')
            .agg(pl.count().alias('like_count'))
        )
        
        n_users_before_join_verify = len(user_like_counts)
        users_meeting_threshold = user_like_counts.filter(
            pl.col('like_count') >= min_likes_per_user
        )
        n_users_after_join_verify = len(users_meeting_threshold)
        n_users_removed_by_join_verify = n_users_before_join_verify - n_users_after_join_verify
        
        if n_users_removed_by_join_verify > 0:
            logger.warning(
                f"Min-likes verification after join: {n_users_before_join_verify:,} -> {n_users_after_join_verify:,} users "
                f"({n_users_removed_by_join_verify:,} removed due to insufficient likes after join)"
            )
            
            # Filter likes to only users who still meet threshold
            valid_user_dids = set(users_meeting_threshold['did'].to_list())
            likes_core_df = likes_core_df.filter(
                pl.col('did').is_in(valid_user_dids)
            )
            n_likes_after_final_filter = len(likes_core_df)
            logger.info(f"Final likes after user filter: {n_likes_after_final_filter:,} likes")
        else:
            logger.info(f"All {n_users_before_join_verify:,} users still meet min-likes threshold after join")
    
    # Update stats with join verification results
    if 'likes' in all_stats:
        all_stats['likes']['n_likes_removed_by_join'] = n_likes_removed_by_join
        all_stats['likes']['n_users_removed_by_join_verify'] = n_users_removed_by_join_verify
        all_stats['likes']['n_users_final_after_join'] = likes_core_df['did'].n_unique() if len(likes_core_df) > 0 else 0
        all_stats['likes']['n_likes_final_after_join'] = len(likes_core_df)
    
    mem_tracker.checkpoint("after_join_verification")
    
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
    _log_data_attrition_report(all_stats, memory_estimate, args, logger)
    
    return likes_core_df, posts_core_df, embed_dim, all_stats


def _log_data_attrition_report(
    all_stats: Dict[str, Any],
    memory_estimate: Optional[Dict[str, Any]],
    args: argparse.Namespace,
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
    
    # Get parameters for display
    min_likes = int(args.min_likes_per_user)
    max_likes = int(args.max_likes_per_user)
    max_users = int(args.max_liking_users) if args.max_liking_users is not None else 0
    neg_sample = int(args.negative_posts_sample)
    
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
        logger.info(f"{'2. Min-likes pre-filter (>=' + str(min_likes) + ')':<45} {fmt(n_users_eligible):>12} {'(n/a)':>15} {'':>10}")
        logger.info(f"{'   - Excluded users':<45} {'-' + fmt(n_users_excluded_min):>12} {f'({excluded_pct:.1f}%)':>15} {'':>10}")
    else:
        logger.info(f"{'2. Min-likes pre-filter (>=' + str(min_likes) + ')':<45} {fmt(n_users_eligible):>12} {'(n/a)':>15} {'':>10}")
    
    # 3. User sampling
    if max_users > 0 and n_users_eligible > 0:
        sample_pct = pct(n_users_sampled, n_users_eligible)
        logger.info(f"{'3. User sampling (' + fmt(max_users) + ')':<45} {fmt(n_users_sampled):>12} {'(n/a)':>15} {'':>10}")
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
        logger.info(f"{'5. Per-user cap (' + str(max_likes) + ')':<45} {'(n/a)':>12} {fmt(n_likes_after_cap):>15} {'':>10}")
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
        logger.info(f"{'  - Users removed (<' + str(min_likes) + ' joinable)':<45} {'-' + fmt(n_users_removed_join) + f' ({join_users_pct:.1f}%)':>12} {'':>15}")
    
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


