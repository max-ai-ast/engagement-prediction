#!/usr/bin/env python3

"""
Stage 3: 

Inputs:


Outputs under <run_dir>/featurize/<timestamp>/:

"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import argparse
import polars as pl
import time

from utils.pipeline.core import new_stage_timestamp_dir, select_prior_output, Context
from utils.helpers import get_stage_logger, log_operation_start, validate_dataframe_schema, load_parquet_from_prior, get_embed_cols_from_lf


def _calc_time_weighted_exp_mov_avg(
    lf: pl.LazyFrame,
    value_cols: list[str],
    group_cols: list[str],
    tau_hours: int,
) -> pl.LazyFrame:
    lf = (
        lf
        .sort(group_cols + ["inserted_at"])
        .with_columns(
            pl.col("inserted_at").shift().over(group_cols).alias("prev_ts")
        )
        .with_columns(
            (pl.col("inserted_at") - pl.col("prev_ts"))
            .dt.total_hours(fractional=True)
            .alias("time_diff")
        )
        .with_columns(
            (-pl.col("time_diff") / pl.lit(tau_hours)).exp().alias("one_minus_alpha_i")
        )
        .with_columns(
            (pl.lit(1.0) - pl.col("one_minus_alpha_i")).fill_null(1.0).alias("alpha_i")
        )
        .with_columns(
            pl.col("one_minus_alpha_i")
            .shift(-1)
            .cum_prod(reverse=True)
            .over(group_cols)
            .fill_null(1.0)
            .alias("cumprod_one_minus_alpha")
        )
        .with_columns(
            (pl.col("alpha_i") * pl.col("cumprod_one_minus_alpha")).alias("weight")
        )
    )
    return lf.group_by(group_cols).agg([
        (pl.col("weight") * pl.col(c)).sum().alias(f"weighted_{c}")
        for c in value_cols
    ])


def _generate_user_summary_from_history(
    posts_core_lf: pl.LazyFrame,
    user_history_lf: pl.LazyFrame, 
    group_cols: List[str],
    tau_hours: int,
) -> pl.LazyFrame:
    embedding_cols = get_embed_cols_from_lf(posts_core_lf)
    # join user_history to posts_core to get embeddings and timestamps
    posts_core_lf_cols = ["subject_uri", "inserted_at"] + embedding_cols

    user_history_lf = user_history_lf.join(
        posts_core_lf.select(posts_core_lf_cols),
        on="subject_uri",
        how="inner",
    )
    # agg over did, bucket in some fashion to generate summary
    return _calc_time_weighted_exp_mov_avg(
        user_history_lf,
        value_cols=embedding_cols,
        group_cols=group_cols,
        tau_hours=tau_hours,
    )


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '02_featurize')

    # Initialize logger
    logger = get_stage_logger('STAGE_02_FEATURIZE', log_file=out_dir / 'stage.log')

    # get args
    tau_hours = args.tau_hours
    t0 = time.time()

    log_operation_start('Load raw data from prior stage', 'STAGE_02_FEATURIZE', logger)
    posts_core_path = select_prior_output(run_dir, '01_get_data', use_latest=context.use_latest, prior_path=context.prior_outputs.get('01_get_data'))
    if posts_core_path is None:
        raise FileNotFoundError(f"Could not find posts_core_*.parquet in 01_get_data")
    posts_core_lf: pl.LazyFrame = load_parquet_from_prior(posts_core_path, "posts_core_")
    validate_dataframe_schema(posts_core_lf, {})
    
    user_history_path = select_prior_output(run_dir, '02_featurize', use_latest=context.use_latest, prior_path=context.prior_outputs.get('02_featurize'))
    if user_history_path is None:
        raise FileNotFoundError(f"Could not find user_history_*.parquet in 02_featurize")
    user_history_lf: pl.LazyFrame = load_parquet_from_prior(user_history_path, "user_history_")
    validate_dataframe_schema(user_history_lf, {"did": str, "subject_uri": str, "inserted_at_bucket": pl.Datetime})

    log_operation_start('Aggregate likes into user history store', 'STAGE_02_FEATURIZE', logger)
    user_summary_lf: pl.LazyFrame = _generate_user_summary_from_history(posts_core_lf, user_history_lf, ['did'], tau_hours)
    validate_dataframe_schema(user_summary_lf, {})

    # Write out result
    user_summary_output_path = out_dir / f"user_summary_{out_dir.name}.parquet"
    user_summary_lf.sink_parquet(user_summary_output_path)

    # Stage info
    info_lines = [
        f"stage: user_summary",
        f"runtime_seconds: {time.time()-t0:.2f}",
        f"settings: ",
        f"inputs: posts_core, user_history",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    return {
        'output_dir': out_dir,
        'artifacts': {
            'user_summary_path': str(user_summary_output_path),
        }
    }
