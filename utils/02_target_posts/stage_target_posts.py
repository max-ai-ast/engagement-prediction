#!/usr/bin/env python3

"""
Stage 2: Build the target-posts dataset by pairing likes with sampled negatives and
assigning train/val/holdout splits.

Two distinct holdout sets are produced:

* **holdout_unseen_users** – a deterministic fraction of users (controlled by
  ``holdout_user_fraction`` and ``holdout_user_seed``) whose rows are *never*
  seen during training or validation.  Bounded above by ``holdout_end``.
* **holdout_seen_users** – rows for the *remaining* (train/val) users that fall
  after ``holdout_start`` (i.e. after the validation window).  Bounded above by
  ``holdout_end``.

The remaining rows for non-holdout users are split temporally into train/val
using ``train_start`` / ``val_start`` / ``holdout_start``.

Inputs:
- posts_core_* (from 01_get_data)
- likes_core_* (from 01_get_data)

Outputs under <run_dir>/target_posts/<timestamp>/:
- target_posts_<timestamp>.parquet
- author_idx_<timestamp>.parquet
- stage_info.txt
- stage.log
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import argparse
import hashlib
import polars as pl
import time
from datetime import datetime
import logging

from utils.pipeline.core import Context
from utils.helpers import (
    get_stage_logger, 
    log_operation_start, 
    log_prior_stage_inputs,
    validate_dataframe_schema, 
    load_parquet_from_prior, 
    parse_one_ts,
    parse_one_ts_strict
)

STAGE_NAME_FOR_LOGGING = '02_TARGET_POSTS'
RAW_TS_COL_NAME = 'record_created_at'


def _resolve_negative_bucket_index(row: Dict[str, Any]) -> Optional[int]:
    """
    Convert an unliked rank within a bucket into the actual post index while only
    storing the sorted liked indices for that (user, bucket) pair.
    """
    neg_rank = row.get('neg_rank')
    bucket_size = row.get('bucket_size')
    liked_idx_list = row.get('liked_idx_list') or []

    if neg_rank is None or bucket_size is None:
        return None

    actual_idx = int(neg_rank)
    for liked_idx in liked_idx_list:
        liked_idx = int(liked_idx)
        if liked_idx > actual_idx:
            break
        actual_idx += 1

    return actual_idx if actual_idx < int(bucket_size) else None


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
            .alias('liked_idx_list'),
            pl.col('liked_count')
            .fill_null(0)
            .cast(pl.Int64)
            .alias('liked_count'),
            (pl.col('bucket_size') - pl.col('liked_count').fill_null(0).cast(pl.Int64)).alias('unliked_count'),
        )
        .select(['target_did', 'bucket', 'bucket_size', 'liked_idx_list', 'unliked_count'])
    )

    likes_lf = (
        likes_with_bucket_lf
        .join(user_bucket_candidates_lf, on=['target_did', 'bucket'], how='left')
        .with_columns(
            # deterministic "random" seed per (user, liked_post)
            pl.struct(
                [pl.col('target_did'), pl.col('like_uri')]
            ).hash(seed=random_seed).cast(pl.UInt64).alias('seed'),
            # seed and neg_rank together assign a random *unliked* post in the bucket to each like
            # (multiple likes can be assigned to the same negative post, which is what we want)
        )
        .with_columns(
            pl.when(pl.col('unliked_count') > 0)
            .then((pl.col('seed') % pl.col('unliked_count').cast(pl.UInt64)).cast(pl.Int64))
            .otherwise(None)
            .alias('neg_rank'),
        )
        .with_columns(
            pl.struct(['liked_idx_list', 'neg_rank', 'bucket_size'])
            .map_elements(_resolve_negative_bucket_index, return_dtype=pl.Int64)
            .alias('neg_idx')
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

    final_lf = (
        likes_lf
        .join(
            posts_lf.select(['bucket', 'idx_in_bucket', 'neg_uri', 'neg_emb_idx', 'neg_author_did']),
            left_on=['bucket', 'neg_idx'],
            right_on=['bucket', 'idx_in_bucket'],
            how='left'
        )
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


def _user_is_holdout(did: str, seed: int, fraction: float) -> bool:
    """Deterministic hash-based holdout assignment.

    Stable across dataset rebuilds: adding or removing a user does not
    change the assignment of other users.
    """
    digest = hashlib.sha256(f"{seed}:{did}".encode()).hexdigest()
    return (int(digest, 16) % 10_000) / 10_000 < fraction


def _apply_splits(
    args: argparse.Namespace,
    target_posts_lf: pl.LazyFrame,
    logger: logging.Logger,
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """Assign ``split`` column with two distinct holdout sets.

    **holdout_unseen_users** – rows for users selected by a deterministic
    hash of ``(holdout_user_seed, target_did)``.  Bounded above by
    ``holdout_end`` when set.

    **holdout_seen_users** – rows for non-holdout (train/val) users whose
    ``seen_at >= holdout_start``.  Bounded above by ``holdout_end`` when
    set.  Only produced when ``holdout_start`` is supplied.

    Non-holdout users are otherwise split temporally:
    ``train`` (``train_start <= seen_at < val_start``) and ``val``
    (``val_start <= seen_at < holdout_start``, or unbounded when
    ``holdout_start`` is ``None``).

    Holdout assignment is computed from the unique ``target_did`` values in a
    lazy side table, then joined back so the full dataset can still be streamed
    to disk via ``sink_parquet`` without materializing all rows in memory.
    """
    ts_col = "seen_at"

    train_start_str: str = _get_train_start(args)
    train_start: datetime = parse_one_ts_strict(train_start_str)
    if args.val_start is None:
        raise ValueError("Validation window start not supplied in input arguments!")
    val_start: datetime = parse_one_ts_strict(args.val_start)
    if val_start <= train_start:
        raise ValueError("Train start date is greater than or equal to val start date!")

    holdout_fraction: float = float(args.holdout_user_fraction)
    holdout_seed: int = int(args.holdout_user_seed)
    holdout_start: Optional[datetime] = parse_one_ts(args.holdout_start)
    holdout_end: Optional[datetime] = parse_one_ts(args.holdout_end)

    if holdout_start is not None and holdout_start <= val_start:
        raise ValueError(
            "holdout_start must be after val_start "
            f"(got holdout_start={holdout_start}, val_start={val_start})"
        )

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(
            f"holdout_user_fraction must be in (0, 1), got {holdout_fraction}"
        )

    holdout_users_lf = (
        target_posts_lf
        .select("target_did")
        .unique()
        .with_columns(
            pl.col("target_did")
            .map_elements(
                lambda did: _user_is_holdout(did, holdout_seed, holdout_fraction),
                return_dtype=pl.Boolean,
            )
            .alias("is_holdout_user")
        )
    )

    user_split_counts = (
        holdout_users_lf
        .group_by("is_holdout_user")
        .len()
        .collect(engine="streaming")
    )
    counts_by_holdout = {
        row["is_holdout_user"]: row["len"]
        for row in user_split_counts.iter_rows(named=True)
    }
    n_holdout = int(counts_by_holdout.get(True, 0))
    n_trainval = int(counts_by_holdout.get(False, 0))
    logger.info(
        f"User split: {n_holdout} holdout (unseen) users, "
        f"{n_trainval} train/val users "
        f"(fraction={holdout_fraction}, seed={holdout_seed})"
    )

    target_posts_lf = target_posts_lf.join(holdout_users_lf, on="target_did", how="left")
    is_holdout_user = pl.col("is_holdout_user").fill_null(False)
    before_end = (pl.col(ts_col) < pl.lit(holdout_end)) if holdout_end is not None else pl.lit(True)

    # --- unseen-users holdout ---
    unseen_expr = pl.when(is_holdout_user & before_end).then(pl.lit("holdout_unseen_users"))

    # --- seen-users holdout (only when holdout_start is set) ---
    if holdout_start is not None:
        after_holdout_start = pl.col(ts_col) >= pl.lit(holdout_start)
        seen_holdout_expr = (
            unseen_expr
            .when(~is_holdout_user & after_holdout_start & before_end)
            .then(pl.lit("holdout_seen_users"))
        )
    else:
        seen_holdout_expr = unseen_expr

    # --- val window upper bound ---
    if holdout_start is not None:
        val_upper = pl.col(ts_col) < pl.lit(holdout_start)
    else:
        val_upper = pl.lit(True)

    split_expr = (
        seen_holdout_expr
        .when(
            ~is_holdout_user
            & (pl.col(ts_col) >= pl.lit(train_start))
            & (pl.col(ts_col) < pl.lit(val_start))
        )
        .then(pl.lit("train"))
        .when(
            ~is_holdout_user
            & (pl.col(ts_col) >= pl.lit(val_start))
            & val_upper
        )
        .then(pl.lit("val"))
        .otherwise(None)
        .alias("split")
    )

    target_posts_lf = target_posts_lf.with_columns(split_expr).drop("is_holdout_user")

    # ============================================================
    # Create author index map
    # ============================================================

    target_posts_train_lf = target_posts_lf.filter(pl.col("split") == "train")
    author_idx_lf = (
        pl
        .concat([
            target_posts_train_lf
            .group_by("like_author_did")
            .agg(pl.count().alias("author_train_count"))
            .rename({"like_author_did": "author_did"}),
            target_posts_train_lf
            .group_by("neg_author_did")
            .agg(pl.count().alias("author_train_count"))
            .rename({"neg_author_did": "author_did"}),
        ])
        .group_by(['author_did'])
        .agg(pl.sum("author_train_count"))
        .with_row_index(name="author_idx")
    )

    # join back to target posts 
    target_posts_lf = (
        target_posts_lf
        .join(
            author_idx_lf,
            left_on="like_author_did",
            right_on="author_did",
            how="left",
        )
        .rename({"author_idx": "like_author_idx"})
        .drop("author_train_count")
        .join(
            author_idx_lf,
            left_on="neg_author_did",
            right_on="author_did",
            how="left",
        )
        .rename({"author_idx": "neg_author_idx"})
        .drop("author_train_count")
    )

    return target_posts_lf, author_idx_lf


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = context.new_stage_dir('02_target_posts')

    # Initialize logger
    logger: logging.Logger = get_stage_logger(STAGE_NAME_FOR_LOGGING, log_file=out_dir / 'stage.log')

    # get args
    t0 = time.time()

    prior_stage_path = context.resolve_prior_output('01_get_data', prior_path=context.prior_outputs.get('01_get_data'))
    log_prior_stage_inputs(context, logger)
    
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
    target_posts_lf: pl.LazyFrame = _get_target_posts(args, posts_core_lf, likes_core_lf, logger, context)
    target_posts_lf, author_idx_lf = _apply_splits(args, target_posts_lf, logger)
    validate_dataframe_schema(target_posts_lf, {
        'target_did': str,
        'seen_at': datetime,
        'like_uri': str,
        'like_emb_idx': int,
        'like_author_did': str,
        'like_author_idx': int,
        'neg_uri': str,
        'neg_emb_idx': int,
        'neg_author_did': str,
        'neg_author_idx': int,
        'split': str,
    })

    # Write out result (streaming to avoid full materialization in memory)
    target_posts_output_path = out_dir / f"target_posts_{out_dir.name}.parquet"
    target_posts_lf.sink_parquet(target_posts_output_path)

    author_idx_output_path = out_dir / f"author_idx_{out_dir.name}.parquet"
    author_idx_lf.sink_parquet(author_idx_output_path)

    # Log split counts from the written file
    split_counts = (
        pl.scan_parquet(target_posts_output_path)
        .group_by("split")
        .len()
        .sort("split")
        .collect()
    )
    for row in split_counts.iter_rows(named=True):
        logger.info(f"  split={row['split']!s:>10s}: {row['len']:>8,} rows")

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
            'target_posts_path': str(target_posts_output_path),
            'author_idx_path': str(author_idx_output_path),
        }
    }
