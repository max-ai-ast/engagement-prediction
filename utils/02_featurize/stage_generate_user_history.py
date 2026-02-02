#!/usr/bin/env python3

"""
Stage 2: 

Inputs:


Outputs under <run_dir>/featurize/<timestamp>/:

"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import polars as pl
import time

from utils.pipeline.core import new_stage_timestamp_dir, select_prior_output, Context
from utils.helpers import get_stage_logger, log_operation_start, validate_dataframe_schema, load_parquet_from_prior, TIMESTAMP_COL_NAME


# TODO: Add an "end_window_lookback"?
# e.g. if bucket is at t, only include likes up to bucket t-k (instead of up to t or t-1)
def _generate_user_history_from_likes(
    likes_core_lf: pl.LazyFrame, 
    bucket_duration: str,
    num_buckets_lookback: int,
    max_likes_per_bucket: Optional[int],
    random_seed: Optional[int],
) -> pl.LazyFrame:
    
    # set random seed
    if random_seed is not None:
        pl.set_random_seed(random_seed)

    # repeat likes num_buckets_lookback times
    if bucket_duration == 'daily':
        user_history_lf = likes_core_lf.with_columns(
            (pl.col(TIMESTAMP_COL_NAME).dt.truncate("1d") + pl.duration(days=1)).alias("_ceil")
        ).with_columns(
            pl.int_ranges(0, num_buckets_lookback).alias("bucket_offset")
        ).explode("bucket_offset").with_columns(
            (pl.col("_ceil") + pl.duration(days=pl.col("bucket_offset"))).alias("timestamp_bucket")
        )
    elif bucket_duration == 'hourly':
        user_history_lf = likes_core_lf.with_columns(
            (pl.col(TIMESTAMP_COL_NAME).dt.truncate("1h") + pl.duration(hours=1)).alias("_ceil")
        ).with_columns(
            pl.int_ranges(0, num_buckets_lookback).alias("bucket_offset")
        ).explode("bucket_offset").with_columns(
            (pl.col("_ceil") + pl.duration(hours=pl.col("bucket_offset"))).alias("timestamp_bucket")
        )
    else:
        raise ValueError(f"Unsupported bucket_duration: {bucket_duration}")
    
    # get unique likes per bucket, and count them
    user_history_lf = user_history_lf.drop(
        "bucket_offset", "_ceil", TIMESTAMP_COL_NAME
    ).group_by(["did", "timestamp_bucket"]).agg(
        pl.col("subject_uri").unique().alias("subject_uri")
    ).with_columns(
        pl.col("subject_uri").list.len().alias("num_likes_in_bucket")
    )

    # random sample likes per bucket if max_likes_per_bucket is set
    if max_likes_per_bucket is not None and max_likes_per_bucket > 0:
        user_history_lf = user_history_lf.with_columns(
            pl.col("subject_uri")
            .list.sample(
                pl.min_horizontal([
                    pl.col("subject_uri").list.len(),
                    pl.lit(max_likes_per_bucket),
                ]),
                with_replacement=False,
                shuffle=False,
            )
            .alias("subject_uri")
        )
    return user_history_lf.explode("subject_uri").drop("num_likes_in_bucket")


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '02_featurize')

    # Initialize logger
    logger = get_stage_logger('STAGE_02_FEATURIZE', log_file=out_dir / 'stage.log')

    # Try to use prior get_data output when available
    prior_path = select_prior_output(run_dir, '01_get_data', use_latest=context.use_latest, prior_path=context.prior_outputs.get('01_get_data'))
    if prior_path is None:
        raise FileNotFoundError(f"Could not find raw data in 01_get_data")

    # get args
    bucket_duration = str(args.bucket_duration)
    num_buckets_lookback = int(args.num_buckets_lookback)
    max_likes_per_bucket = int(args.max_likes_per_bucket)
    random_seed = args.random_seed

    log_operation_start('Load raw data from prior stage', 'STAGE_02_FEATURIZE', logger)
    t0 = time.time()
    likes_core_lf: pl.LazyFrame = load_parquet_from_prior(prior_path, "likes_core_")
    validate_dataframe_schema(likes_core_lf, {"did": str, TIMESTAMP_COL_NAME: pl.Datetime, "subject_uri": str})

    log_operation_start('Aggregate likes into user history store', 'STAGE_02_FEATURIZE', logger)
    user_history_lf: pl.LazyFrame = _generate_user_history_from_likes(likes_core_lf, bucket_duration, num_buckets_lookback, max_likes_per_bucket, random_seed)
    validate_dataframe_schema(user_history_lf, {"did": str, "subject_uri": str, "timestamp_bucket": pl.Datetime})

    # Write out result
    user_history_output_path = out_dir / f"user_history_{out_dir.name}.parquet"
    user_history_lf.sink_parquet(user_history_output_path)

    # Stage info
    info_lines = [
        f"stage: featurize",
        f"runtime_seconds: {time.time()-t0:.2f}",
        f"settings: bucket_duration={bucket_duration}, num_buckets_lookback={num_buckets_lookback}, max_likes_per_bucket={max_likes_per_bucket}",
        f"inputs: likes_core",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    return {
        'output_dir': out_dir,
        'artifacts': {
            'user_history_path': str(user_history_output_path),
        }
    }
