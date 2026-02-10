#!/usr/bin/env python3

"""
Stage 2: Build the target-posts dataset by pairing likes with sampled negatives and
assigning temporal splits for train/val/holdout.

Inputs:
- posts_core_* (from 01_get_data)
- likes_core_* (from 01_get_data)

Outputs under <run_dir>/target_posts/<timestamp>/:
- target_posts_<timestamp>.parquet
- stage_info.txt
- stage.log
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import polars as pl
import time
from datetime import datetime
import logging

from utils.pipeline.core import new_stage_timestamp_dir, select_prior_output, Context
from utils.helpers import (
    get_stage_logger, 
    log_operation_start, 
    validate_dataframe_schema, 
    load_parquet_from_prior, 
    parse_one_ts,
    parse_one_ts_strict
)

STAGE_NAME_FOR_LOGGING = '02_TARGET_POSTS'
RAW_TS_COL_NAME = 'record_created_at'


def _get_liked_target_posts(
    likes_lf: pl.LazyFrame,
    posts_lf: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Get all of the liked target posts (positive examples). 
    For now this is just all of the liked posts in the dataset.
    We could filter out a user's first like in the entire dataset but this
    probably makes more sense in the downstream user history creation.
    """
    return (
        likes_lf
        .select([
            pl.col('did').alias('target_did'),
            pl.col(RAW_TS_COL_NAME).alias('seen_at'),
            pl.col('subject_uri').alias('like_uri'),
            pl.col('emb_idx').alias('like_emb_idx')
        ])
        .join(
            posts_lf.select([
                pl.col('at_uri'),
                pl.col(RAW_TS_COL_NAME).alias('like_posted_at'),
                pl.col('did').alias('like_author_did')
            ]),
            left_on='like_uri',
            right_on='at_uri',
            how='inner'
        )
    ) # 'target_did', 'seen_at', 'like_uri', 'like_emb_idx', 'like_posted_at', 'like_author_did'


def _get_negative_target_posts(
    args: argparse.Namespace,
    posts_lf: pl.LazyFrame,
    liked_target_posts_lf: pl.LazyFrame,
    logger: logging.Logger,
    context: Context
) -> pl.LazyFrame:
    """
    Samples one negative (unliked) post per like by bucketing posts in time and
    selecting a deterministic index within each bucket based on the like metadata.
    Ensures the selected negative was not liked by the target user.
    """
    random_seed = args.random_seed
    bucket = args.neg_sample_bucket
    if bucket is None:
        raise ValueError("No bucket size specified for negative samples!")
    
    posts_lf = (
        posts_lf
        .select([
            pl.col('at_uri').alias('neg_uri'),
            pl.col(RAW_TS_COL_NAME).alias('neg_posted_at'),
            pl.col('emb_idx').alias('neg_emb_idx'),
            pl.col('did').alias('neg_author_did'),
        ])
        .with_columns(
            pl.col('neg_posted_at').dt.truncate(bucket).alias('bucket')
        )
        # stable ordering so idx assignment is deterministic
        .sort(['bucket', 'neg_posted_at', 'neg_uri'])
        .with_columns(
            # polars cum_count is 1-based; shift to 0-based to match neg_idx
            (pl.col('neg_uri').cum_count().over('bucket') - 1).cast(pl.Int64).alias('idx_in_bucket'),
        )
    ) # 'neg_uri', 'neg_posted_at', 'neg_emb_idx', 'neg_author_did', 'bucket', 'idx_in_bucket'

    bucket_sizes_lf = (
        posts_lf
        .group_by('bucket')
        .len()
        .rename({'len': 'bucket_size'})
    ) # 'bucket', 'bucket_size'

    likes_with_bucket_lf = (
        liked_target_posts_lf
        .with_columns([
            # using posted_at time of like so that it can match up with negatives
            # (for which we only currently have the posted_at time)
            pl.col('like_posted_at').dt.truncate(bucket).alias('bucket')
        ])
    )

    liked_idx_by_user_bucket_lf = (
        liked_target_posts_lf
        .join(
            posts_lf.select(['neg_uri', 'bucket', 'idx_in_bucket']),
            left_on='like_uri',
            right_on='neg_uri',
            how='inner'
        )
        .group_by(['target_did', 'bucket'])
        .agg(
            pl.col('idx_in_bucket').unique().sort().alias('liked_idx_list')
        )
        .with_columns(
            pl.col('liked_idx_list').list.len().alias('liked_count')
        )
    ) # 'target_did', 'bucket', 'liked_idx_list', 'liked_count'

    user_bucket_candidates_lf = (
        likes_with_bucket_lf
        .join(bucket_sizes_lf, on="bucket", how="inner")
        .select(['target_did', 'bucket', 'bucket_size'])
        .unique()
        .join(liked_idx_by_user_bucket_lf, on=['target_did', 'bucket'], how='left')
        .with_columns(
            pl.col('liked_idx_list')
            .fill_null(pl.lit([], dtype=pl.List(pl.Int64)))
            .alias('liked_idx_list')
        )
        .with_columns(
            pl.int_ranges(0, pl.col('bucket_size')).alias('all_idx'),
        )
        .with_columns(
            pl.col('all_idx')
            .list
            .set_difference(pl.col('liked_idx_list'))
            .list
            .sort()
            .alias('unliked_idx_list')
        )
        .with_columns(
            pl.col('unliked_idx_list').list.len().alias('unliked_count')
        )
        .select(['target_did', 'bucket', 'unliked_idx_list', 'unliked_count'])
    )

    likes_lf = (
        likes_with_bucket_lf
        # join to get bucket_size for the like's bucket
        .join(bucket_sizes_lf, on="bucket", how="inner")
        .with_columns(
            # deterministic "random" seed per (user, liked_post)
            pl.struct(
                [pl.col('target_did'), pl.col('like_uri')]
            ).hash(seed=random_seed).cast(pl.UInt64).alias('seed'),
        )
        .join(user_bucket_candidates_lf, on=['target_did', 'bucket'], how='left')
        .with_columns(
            # seed and neg_rank together assign a random *unliked* post in the bucket to each like
            # (multiple likes can be assigned to the same negative post, which is what we want)
            pl.when(pl.col('unliked_count') > 0)
            .then((pl.col('seed') % pl.col('unliked_count').cast(pl.UInt64)).cast(pl.Int64))
            .otherwise(None)
            .alias('neg_rank'),
        )
        .with_columns(
            pl.when(pl.col('neg_rank').is_not_null())
            .then(pl.col('unliked_idx_list').list.get(pl.col('neg_rank')))
            .otherwise(None)
            .alias('neg_idx'),
        )
    ) # 'target_did', 'seen_at', 'like_uri', 'like_emb_idx', 'like_posted_at', 'like_author_did' 'bucket', 'bucket_size', 'seed', 'neg_idx'

    # (Shouldn't happen but) keep track of any potential lost likes by joining to bucket sizes.
    n_likes_orig = liked_target_posts_lf.select(pl.len()).collect(engine='streaming').item()
    n_likes_after_bucket_join = likes_lf.select(pl.len()).collect(engine='streaming').item()
    n_likes_lost_from_bucket_join = n_likes_orig - n_likes_after_bucket_join
    logger.info(f"Started with {n_likes_orig:,} likes; have {n_likes_after_bucket_join:,} after joining to post buckets. (Lost {n_likes_lost_from_bucket_join:,} likes).")
    context.tracker.log_single_value('Target Posts - Dropped Likes from Bucket Join', n_likes_lost_from_bucket_join)

    # If a user liked every post in a bucket, there is no valid negative for that like.
    n_likes_without_neg = (
        likes_lf
        .filter(pl.col('neg_idx').is_null())
        .select(pl.len())
        .collect(engine='streaming')
        .item()
    )
    if n_likes_without_neg > 0:
        logger.info(f"Dropping {n_likes_without_neg:,} likes with no unliked negatives in the bucket.")
        context.tracker.log_single_value('Target Posts - Dropped Likes with No Unliked Negatives', n_likes_without_neg)
        likes_lf = likes_lf.filter(pl.col('neg_idx').is_not_null())

    posts_keyed_lf = posts_lf.with_columns(
        pl.concat_str(['bucket', 'idx_in_bucket']).alias('key')
    ).select(['key', 'neg_uri', 'neg_posted_at', 'neg_emb_idx', 'neg_author_did'])

    final_lf = (
        likes_lf
        .with_columns(pl.concat_str([pl.col('bucket'), pl.col(f"neg_idx")]).alias('key'))
        .join(posts_keyed_lf, on='key', how='left')
        .select([
            'target_did',
            'seen_at',
            'like_uri',
            'like_emb_idx',
            'like_author_did',
            'neg_uri',
            'neg_emb_idx',
            'neg_author_did',
        ])
    ) 
    return final_lf
    

def _log_final_metrics(
    all_target_posts_lf: pl.LazyFrame,
    liked_target_posts_lf: pl.LazyFrame,
    logger: logging.Logger,
    context: Context
):
    # how many negative posts were actually liked by the user??
    final_counts = (
        all_target_posts_lf
        .select(['neg_uri', 'target_did'])
        .join(
            liked_target_posts_lf.select(['target_did', 'like_uri']), 
            left_on=['target_did', 'neg_uri'], 
            right_on=['target_did', 'like_uri'], 
            how='left',
            coalesce=False
        )
        .select(
            pl.len().alias("total_rows"),
            pl.col("like_uri").is_not_null().sum().alias("matched_rows"),
        )
        .collect(engine="streaming")
    )
    n_total_target_pairs, n_negs_liked_by_target_user = final_counts.row(0)
    logger.info(f"Total target pairs: {n_total_target_pairs:,}.")
    logger.info(f"(False) Negatives that were actually liked by the user: {n_negs_liked_by_target_user:,}.")
    context.tracker.log_single_value('Target Posts - Total Target Pairs', n_total_target_pairs)
    context.tracker.log_single_value('Target Posts - False Negatives Actually Liked By User', n_negs_liked_by_target_user)

    # how many negatives were selected more than once? how many negatives were selected more than once for the same user?
    counts_per_user_and_neg_lf = all_target_posts_lf.group_by(['target_did', 'neg_uri']).len()
    counts_per_neg_lf = counts_per_user_and_neg_lf.group_by(['neg_uri']).agg(pl.col("len").sum().alias("len"))
    n_same_neg_for_user = counts_per_user_and_neg_lf.filter(pl.col('len') > 1).select(pl.len()).collect(engine='streaming').item()
    n_same_neg_overall = counts_per_neg_lf.filter(pl.col('len') > 1).select(pl.len()).collect(engine='streaming').item()
    logger.info(f"Number of negatives that were assigned to the same target user more than once: {n_same_neg_for_user:,} ({n_same_neg_for_user/n_total_target_pairs*100:.1f}%)")
    logger.info(f"Number of negatives that were assigned more than once, total: {n_same_neg_overall:,} ({n_same_neg_overall/n_total_target_pairs*100:.1f}%")
    context.tracker.log_single_value('Target Posts - Negatives Assigned Multiple Times to Same User', n_same_neg_for_user)
    context.tracker.log_single_value('Target Posts - Negatives Assigned Multiple Times', n_same_neg_overall)


def _get_target_posts(
    args: argparse.Namespace,
    posts_lf: pl.LazyFrame,
    likes_lf: pl.LazyFrame,
    logger: logging.Logger,
    context: Context
) -> pl.LazyFrame:
    liked_target_posts_lf = _get_liked_target_posts(likes_lf, posts_lf)
    all_target_posts_lf = _get_negative_target_posts(args, posts_lf, liked_target_posts_lf, logger, context)
    _log_final_metrics(all_target_posts_lf, liked_target_posts_lf, logger, context)
    return all_target_posts_lf


def _get_train_start(
    args: argparse.Namespace
) -> str:
    """
    Resolves the training window start date, falling back to posts_start and likes_start if
    not explicitly specified.
    """
    if args.train_start is not None:
        return args.train_start
    if args.posts_start is not None:
        return args.posts_start
    if args.likes_start is not None:
        return args.likes_start
    raise ValueError("Could not infer train window start from input arguments!")


def _apply_temporal_splits(
    args: argparse.Namespace,
    lf: pl.LazyFrame,
) -> pl.LazyFrame:
    """
    Appends a column, "split", that specifies "train", "val", or "holdout".
    If holdout_start is not specified in args, only outputs "train" or "val".
    """
    # note we use 
    final_ts_col_name = 'seen_at'

    train_start_str: str = _get_train_start(args)
    train_start: datetime = parse_one_ts_strict(train_start_str)
    if args.val_start is None:
        raise ValueError("Validation window start not supplied in input arguments!")
    val_start: datetime = parse_one_ts_strict(args.val_start)
    if val_start <= train_start:
        raise ValueError("Train start date is greater than or equal to val start date!")
    holdout_start: Optional[datetime] = parse_one_ts(args.holdout_start)

    if holdout_start is not None:
        if holdout_start <= val_start:
            raise ValueError("Validation start date is greater than or equal to holdout start date!")
        return lf.with_columns(
            pl.when((pl.col(final_ts_col_name) >= pl.lit(train_start)) & (pl.col(final_ts_col_name) < pl.lit(val_start)))
              .then(pl.lit("train"))
              .when((pl.col(final_ts_col_name) >= pl.lit(val_start)) & (pl.col(final_ts_col_name) < pl.lit(holdout_start)))
              .then(pl.lit("val"))
              .when(pl.col(final_ts_col_name) >= pl.lit(holdout_start))
              .then(pl.lit("holdout"))
              .otherwise(None)
              .alias("split")
        )
    else:
        return lf.with_columns(
            pl.when((pl.col(final_ts_col_name) >= pl.lit(train_start)) & (pl.col(final_ts_col_name) < pl.lit(val_start)))
              .then(pl.lit("train"))
              .when(pl.col(final_ts_col_name) >= pl.lit(val_start))
              .then(pl.lit("val"))
              .otherwise(None)
              .alias("split")
        )


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '02_target_posts')

    # Initialize logger
    logger: logging.Logger = get_stage_logger(STAGE_NAME_FOR_LOGGING, log_file=out_dir / 'stage.log')

    # get args
    t0 = time.time()

    prior_stage_path = select_prior_output(run_dir, '01_get_data', use_latest=context.use_latest, prior_path=context.prior_outputs.get('01_get_data'))
    if prior_stage_path is None:
        raise FileNotFoundError(f"Could not find outputs in prior 01_get_data directory!")
    
    posts_core_lf: pl.LazyFrame = load_parquet_from_prior(prior_stage_path, "posts_core_")
    validate_dataframe_schema(posts_core_lf, {
        'did': str, 
        'at_uri': str, 
        'record_created_at': datetime, 
        'emb_idx': int, 
        'record_text': str, 
        'is_liked': bool, 
        'in_random_sample': bool
    })
    likes_core_lf: pl.LazyFrame = load_parquet_from_prior(prior_stage_path, "likes_core_")
    validate_dataframe_schema(likes_core_lf, {
        'did': str, 
        'subject_uri': str, 
        'record_created_at': datetime, 
        'emb_idx': int
    })

    log_operation_start('Generate target posts dataset from likes and posts', STAGE_NAME_FOR_LOGGING, logger)
    # The core logic. (1) Generate target posts and (2) split into train/val/holdout
    target_posts_lf: pl.LazyFrame = _get_target_posts(args, posts_core_lf, likes_core_lf, logger, context)
    target_posts_lf = _apply_temporal_splits(args, target_posts_lf)
    validate_dataframe_schema(target_posts_lf, {
        'target_did': str,
        'seen_at': datetime,
        'like_uri': str,
        'like_emb_idx': int,
        'like_author_did': str,
        'neg_uri': str,
        'neg_emb_idx': int,
        'neg_author_did': str,
    })

    # Write out result
    target_posts_output_path = out_dir / f"target_posts_{out_dir.name}.parquet"
    target_posts_lf.sink_parquet(target_posts_output_path)

    # Stage info
    info_lines = [
        f"stage: target_posts",
        f"runtime_seconds: {time.time()-t0:.2f}",
        f"inputs: posts_core, likes_core",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    return {
        'output_dir': out_dir,
        'artifacts': {
            'user_summary_path': str(target_posts_output_path),
        }
    }
