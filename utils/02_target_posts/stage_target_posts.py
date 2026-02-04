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
TS_COL_NAME = 'record_created_at'


def _get_liked_target_posts(
    likes_lf: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Get all of the liked target posts (positive examples). 
    For now this is just all of the liked posts in the dataset.
    We could filter out a user's first like in the entire dataset but this
    probably makes more sense in the downstream user history creation.
    """
    return (
        likes_lf
        .select(['did', TS_COL_NAME, 'subject_uri', 'emb_idx'])
        .with_columns(pl.lit(True).alias("was_liked"))
    ) # 'did', 'record_created_at', 'subject_uri', 'was_liked'


def _get_negative_target_posts(
    args: argparse.Namespace,
    posts_lf: pl.LazyFrame,
    liked_target_posts_lf: pl.LazyFrame
) -> pl.LazyFrame:
    """
    Samples one negative (unliked) post per like by bucketing posts in time and
    selecting a deterministic index within each bucket based on the like metadata.
    """
    random_seed = args.random_seed
    bucket = args.neg_sample_bucket
    if bucket is None:
        raise ValueError("No bucket size specified for negative samples!")
    
    post_ts_col_name = 'post_'+TS_COL_NAME
    posts_lf = (
        posts_lf
        .select([
            pl.col('at_uri'),
            pl.col(TS_COL_NAME).alias(post_ts_col_name),
            pl.col('emb_idx'),
        ])
        .with_columns(
            pl.col(post_ts_col_name).dt.truncate(bucket).alias('bucket')
        )
        # stable ordering so idx assignment is deterministic
        .sort(['bucket', post_ts_col_name, 'at_uri'])
        .with_columns(
            # polars cum_count is 1-based; shift to 0-based to match neg_idx
            (pl.col('at_uri').cum_count().over('bucket') - 1).cast(pl.Int64).alias('idx_in_bucket'),
        )
    ) # 'at_uri', 'post_record_created_at', 'bucket', 'idx_in_bucket'

    bucket_sizes_lf = (
        posts_lf
        .group_by('bucket')
        .len()
        .rename({'len': 'bucket_size'})
    ) # 'bucket', 'bucket_size'

    like_ts_col_name = 'like_'+TS_COL_NAME
    likes_lf = (
        liked_target_posts_lf
        .select([
            pl.col('did'),
            pl.col('subject_uri'),
            pl.col(TS_COL_NAME).alias(like_ts_col_name)
        ])
        .with_columns([
            pl.col(like_ts_col_name).dt.truncate(bucket).alias('bucket')
        ])
        # join to get bucket_size for the like's bucket
        .join(bucket_sizes_lf, on="bucket", how="inner")
        .with_columns(
            # deterministic "random" seed per (user, liked_post, time)
            pl.struct(
                [pl.col('did'), pl.col('subject_uri'), pl.col(like_ts_col_name)]
            ).hash(seed=random_seed).cast(pl.UInt64).alias('seed'),
        )
        # seed and neg_idx together effectively assign a random post in the bucket to each like
        # (multiple likes can be assigned to the same negative post, which is what we want - a true random sample)
        .with_columns(
            (pl.col('seed') % pl.col('bucket_size').cast(pl.UInt64)).cast(pl.Int64).alias('neg_idx'),
        )
    ) # 'did', 'subject_uri', 'like_record_created_at', 'bucket', 'bucket_size', 'seed', 'neg_idx'

    posts_keyed_lf = posts_lf.with_columns(
        pl.concat_str(['bucket', 'idx_in_bucket']).alias('key')
    ).select(['key', 'at_uri', post_ts_col_name, 'emb_idx'])
    # 'key', 'at_uri', 'post_record_created_at'

    return (
        likes_lf
        .with_columns(pl.concat_str([pl.col('bucket'), pl.col(f"neg_idx")]).alias('key'))
        .join(posts_keyed_lf, on='key', how='left')
        .select([
            pl.col('did'),
            pl.col(post_ts_col_name).alias(TS_COL_NAME),
            pl.col('at_uri').alias('subject_uri'),
            pl.col('emb_idx'),
            pl.lit(False).alias('was_liked')
        ]) # 'did', 'record_created_at', 'subject_uri', 'was_liked'
    ) 
    

def _get_target_posts(
    args: argparse.Namespace,
    posts_lf: pl.LazyFrame,
    likes_lf: pl.LazyFrame
) -> pl.LazyFrame:
    liked_target_posts_lf = _get_liked_target_posts(likes_lf)
    negative_target_posts_lf = _get_negative_target_posts(args, posts_lf, liked_target_posts_lf)
    return pl.concat([liked_target_posts_lf, negative_target_posts_lf], how="vertical")


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
            pl.when((pl.col(TS_COL_NAME) >= pl.lit(train_start)) & (pl.col(TS_COL_NAME) < pl.lit(val_start)))
              .then(pl.lit("train"))
              .when((pl.col(TS_COL_NAME) >= pl.lit(val_start)) & (pl.col(TS_COL_NAME) < pl.lit(holdout_start)))
              .then(pl.lit("val"))
              .when(pl.col(TS_COL_NAME) >= pl.lit(holdout_start))
              .then(pl.lit("holdout"))
              .otherwise(None)
              .alias("split")
        )
    else:
        return lf.with_columns(
            pl.when((pl.col(TS_COL_NAME) >= pl.lit(train_start)) & (pl.col(TS_COL_NAME) < pl.lit(val_start)))
              .then(pl.lit("train"))
              .when(pl.col(TS_COL_NAME) >= pl.lit(val_start))
              .then(pl.lit("val"))
              .otherwise(None)
              .alias("split")
        )


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '02_target_posts')

    # Initialize logger
    logger = get_stage_logger(STAGE_NAME_FOR_LOGGING, log_file=out_dir / 'stage.log')

    # get args
    t0 = time.time()

    prior_stage_path = select_prior_output(run_dir, '01_get_data', use_latest=context.use_latest, prior_path=context.prior_outputs.get('01_get_data'))
    if prior_stage_path is None:
        raise FileNotFoundError(f"Could not find outputs in prior 01_get_data directory!")
    
    log_operation_start('Load raw posts data from prior stage', STAGE_NAME_FOR_LOGGING, logger)
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
    
    log_operation_start('Load raw likes data from prior stage', STAGE_NAME_FOR_LOGGING, logger)
    likes_core_lf: pl.LazyFrame = load_parquet_from_prior(prior_stage_path, "likes_core_")
    validate_dataframe_schema(likes_core_lf, {
        'did': str, 
        'subject_uri': str, 
        'record_created_at': datetime, 
        'emb_idx': int
    })

    log_operation_start('Generate target posts dataset from likes and posts', STAGE_NAME_FOR_LOGGING, logger)

    # The core logic. (1) Generate target posts and (2) split into train/val/holdout
    target_posts_lf: pl.LazyFrame = _get_target_posts(args, posts_core_lf, likes_core_lf)
    target_posts_lf = _apply_temporal_splits(args, target_posts_lf)
    validate_dataframe_schema(target_posts_lf, {
        'did': str, 
        'subject_uri': str, 
        'record_created_at': datetime, 
        'emb_idx': int, 
        'was_liked': bool, 
        'split': str
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
