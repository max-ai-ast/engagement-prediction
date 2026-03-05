#!/usr/bin/env python3

"""
Stage 5: Evaluate a trained model using modular evaluation framework.

This stage orchestrates the evaluation pipeline by:
1. Loading training data (target_posts, user history) via shared dataloaders
2. Locating holdout predictions from Stage 4 (04_train)
3. Computing user metadata (number of embedding likes per user)
4. Creating an EvalContext and running all discovered evaluation modules

Evaluation modules are auto-discovered from utils/05_evaluate/evals/ and each
produces its own set of artifacts (plots, CSVs, JSON summaries).

Inputs (from prior pipeline stages):
- target_posts_*.parquet from 02_target_posts  (includes train/val/holdout split column)
- history_posts_*.parquet from 03_user_history
- predictions/holdout_<type>.parquet from 04_train  (e.g. holdout_unseen_users.parquet)

Outputs under <train_dir>/evals/<timestamp>/
- eval_summary.json: Combined results from all modules
- stage_info.txt: Stage metadata
- <module_name>/: Subdirectory for each evaluation module's artifacts
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import polars as pl

from utils.pipeline.core import select_prior_output, Context
from utils.helpers import get_stage_logger, log_operation_start
from utils.dataloaders import load_training_data

# ---------------------------------------------------------------------------
# Import evaluation framework
# ---------------------------------------------------------------------------
# We use importlib with sys.modules registration to avoid issues with the
# numeric directory name (05_evaluate is not a valid Python identifier).
import sys
import importlib.util


def _import_evals_module():
    """Import the evals package from the numeric directory.

    We set ``submodule_search_locations`` so that the sub-modules
    (cold_start_curves, performance_inequality, ...) can use relative imports
    like ``from . import EvalContext``.
    """
    evals_dir = Path(__file__).parent / 'evals'
    evals_init = evals_dir / '__init__.py'
    spec = importlib.util.spec_from_file_location(
        'utils._evals',
        evals_init,
        submodule_search_locations=[str(evals_dir)],
    )
    evals = importlib.util.module_from_spec(spec)
    sys.modules['utils._evals'] = evals  # Register before exec so dataclasses resolve
    spec.loader.exec_module(evals)
    return evals


_evals = _import_evals_module()
EvalContext = _evals.EvalContext
discover_modules = _evals.discover_modules
run_all_modules = _evals.run_all_modules

STAGE_LOG_NAME = 'STAGE_05_EVALUATE'


# ---------------------------------------------------------------------------
# Asset resolution
# ---------------------------------------------------------------------------

def resolve_train_output(
    run_dir: Path,
    context: Context,
) -> Path:
    """Locate the training stage output directory.

    Tries same-session artifacts (``train_mlp`` / ``train_two_tower``) first,
    then falls back to filesystem scanning under ``04_train/``.
    """
    for stage_key in ("train_mlp", "train_two_tower"):
        art_dir = context.get_artifact_dir(stage_key)
        if art_dir is not None and Path(art_dir).exists():
            return Path(art_dir)

    train_dir = select_prior_output(
        run_dir, '04_train',
        use_latest=context.use_latest,
        prior_path=context.prior_outputs.get('04_train'),
    )
    if train_dir is None:
        raise FileNotFoundError(
            "Could not find training output (04_train). "
            "Please ensure Stage 4 (train) has completed."
        )
    return train_dir


# ---------------------------------------------------------------------------
# Holdout predictions
# ---------------------------------------------------------------------------

def load_holdout_predictions(
    predictions_dir: Optional[Path],
    holdout_type: str,
    logger=None,
) -> pl.DataFrame:
    """Load pre-computed holdout predictions from the training stage.

    Both MLP and Two-Tower training stages save predictions to
    ``predictions/holdout_<holdout_type>.parquet`` with columns
    [did, post_id, y_true, y_pred_proba].

    Returns:
        Polars DataFrame with columns [did, post_id, y_true, y_pred_proba].
    """
    if predictions_dir is not None:
        pred_parquet = predictions_dir / f'holdout_{holdout_type}.parquet'
        if pred_parquet.exists():
            if logger:
                logger.info(f"Loading pre-computed predictions from {pred_parquet}")
            return pl.read_parquet(pred_parquet)

    raise FileNotFoundError(
        f"No holdout predictions found (expected predictions/holdout_{holdout_type}.parquet "
        "in the 04_train output directory). Please ensure Stage 4 (train) "
        "completed successfully with holdout evaluation enabled."
    )


# ---------------------------------------------------------------------------
# Holdout enrichment with history lengths
# ---------------------------------------------------------------------------

def _build_holdout_with_history(
    target_posts: pl.DataFrame,
    history: pl.DataFrame,
    holdout_split: str,
) -> pl.DataFrame:
    """Join holdout target rows with pre-computed history lengths.

    Args:
        target_posts: Full target posts DataFrame with split column.
        history: DataFrame with columns [target_did, like_uri, num_embedding_likes].
        holdout_split: Split name to filter to (e.g. ``"holdout_unseen_users"``).

    Returns:
        Holdout target rows enriched with ``num_embedding_likes``.
    """
    holdout = target_posts.filter(
        (pl.col("split") == holdout_split)
        & pl.col("neg_emb_idx").is_not_null()
    )
    return holdout.join(
        history,
        on=["target_did", "like_uri"],
        how="left",
    ).with_columns(
        pl.col("num_embedding_likes").fill_null(0)
    )


def enrich_predictions_with_history_len(
    predictions: pl.DataFrame,
    holdout_with_hist: pl.DataFrame,
    logger,
) -> pl.DataFrame:
    """Add per-row ``num_embedding_likes`` to predictions via key-based join.

    Builds a ``(did, post_id)`` lookup from the holdout target rows -- one entry
    for the positive (``like_uri``) and one for the negative (``neg_uri``) of
    each target row -- then left-joins predictions against it.

    When the same ``(did, post_id)`` pair appears in multiple target rows (e.g.
    the same ``neg_uri`` sampled for two different likes of the same user), the
    maximum ``num_embedding_likes`` value is kept.  This is conservative: the
    values will be nearly identical (same user, close in time) and this avoids
    row duplication from a many-to-many join.
    """
    pos_lookup = holdout_with_hist.select(
        pl.col("target_did").alias("did"),
        pl.col("like_uri").alias("post_id"),
        "num_embedding_likes",
    )
    neg_lookup = holdout_with_hist.select(
        pl.col("target_did").alias("did"),
        pl.col("neg_uri").alias("post_id"),
        "num_embedding_likes",
    )
    lookup = (
        pl.concat([pos_lookup, neg_lookup])
        .group_by("did", "post_id")
        .agg(pl.col("num_embedding_likes").max())
    )

    enriched = predictions.join(
        lookup, on=["did", "post_id"], how="left",
    ).with_columns(
        pl.col("num_embedding_likes").fill_null(0)
    )

    n_total = enriched.height
    n_with_hist = enriched.filter(pl.col("num_embedding_likes") > 0).height
    logger.info(
        f"  Enrichment: {n_with_hist}/{n_total} predictions matched "
        f"with history length > 0"
    )
    return enriched


# ---------------------------------------------------------------------------
# User metadata
# ---------------------------------------------------------------------------

def compute_user_metadata(
    predictions: pl.DataFrame,
    target_posts: pl.DataFrame,
    holdout_with_hist: pl.DataFrame,
) -> pl.DataFrame:
    """Compute per-user metadata including number of embedding likes.

    The number of embedding likes is derived from the user history: for each
    holdout (user, like_event) pair, it is the count of prior embedding indices.
    When a user has multiple holdout rows, we take the maximum history length.

    Returns:
        Polars DataFrame with columns [did, num_embedding_likes, num_total_likes].
    """
    holdout_dids = predictions.select(
        pl.col("did").cast(pl.Utf8).unique()
    )

    user_hist = (
        holdout_with_hist
        .group_by("target_did")
        .agg(pl.col("num_embedding_likes").max())
        .rename({"target_did": "did"})
    )

    total_likes = (
        target_posts
        .filter(pl.col("target_did").cast(pl.Utf8).is_in(holdout_dids["did"]))
        .group_by("target_did")
        .agg(pl.col("like_uri").n_unique().alias("num_total_likes"))
        .rename({"target_did": "did"})
    )

    metadata = (
        holdout_dids
        .join(user_hist, on="did", how="left")
        .join(total_likes, on="did", how="left")
        .with_columns(
            pl.col("num_embedding_likes").fill_null(0).cast(pl.Int64),
            pl.col("num_total_likes").fill_null(0).cast(pl.Int64),
        )
    )

    return metadata.select("did", "num_embedding_likes", "num_total_likes")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run(context: Context, args) -> Dict[str, Any]:
    """Main entry point for Stage 5: Evaluation.

    Loads holdout predictions from the training stage and runs all evaluation
    modules.
    """
    run_dir = Path(context.run_dir).resolve()
    t0 = time.time()

    # --- hyperparams ---
    eval_batch_size = int(args.eval_batch_size)
    eval_holdout_type = str(args.eval_holdout_type)
    holdout_split = f"holdout_{eval_holdout_type}"
    skip_modules = args.skip_modules
    if skip_modules and isinstance(skip_modules, str):
        skip_modules = [m.strip() for m in skip_modules.split(',')]

    # Resolve training output first so we can nest eval outputs inside it
    train_dir = resolve_train_output(run_dir, context)
    predictions_dir = train_dir / 'predictions'

    # Create output directory inside the training directory
    evals_base = train_dir / 'evals'
    evals_base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = evals_base / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger
    logger = get_stage_logger(STAGE_LOG_NAME, log_file=out_dir / 'stage.log')
    log_operation_start('Stage 5: Evaluation', STAGE_LOG_NAME, logger)
    logger.info(f"Training output dir: {train_dir}")
    logger.info(f"Holdout type for evaluation: {eval_holdout_type} (split={holdout_split})")

    # Step 1: Load training data from prior stages.
    # load_training_data returns pandas; we convert to polars immediately.
    log_operation_start('Load training data from prior stages', STAGE_LOG_NAME, logger)
    _, target_posts_pd, history_pd, embed_dim = load_training_data(
        run_dir, context, logger=logger,
    )

    target_posts = pl.from_pandas(target_posts_pd)
    history = (
        pl.from_pandas(history_pd)
        .select(
            "target_did",
            "like_uri",
            pl.col("prior_emb_indices").list.len().fill_null(0).alias("num_embedding_likes"),
        )
    )
    del target_posts_pd, history_pd

    holdout_target_rows = target_posts.filter(pl.col("split") == holdout_split)
    holdout_users = (
        holdout_target_rows
        .select(pl.col("target_did").cast(pl.Utf8).unique())
        ["target_did"].to_list()
    )

    if not holdout_users:
        raise RuntimeError(
            f"No holdout users found in target_posts (no rows with split='{holdout_split}'). "
            f"Did the target_posts stage produce a {eval_holdout_type} holdout split? "
            f"Check --holdout-user-fraction and --holdout-start in your pipeline configuration."
        )
    logger.info(f"Holdout users ({eval_holdout_type}): {len(holdout_users)}")
    logger.info(f"Holdout target rows ({eval_holdout_type}): {holdout_target_rows.height}")

    # Step 2: Load holdout predictions (polars)
    log_operation_start('Load holdout predictions', STAGE_LOG_NAME, logger)
    predictions = load_holdout_predictions(
        predictions_dir=predictions_dir if predictions_dir.exists() else None,
        holdout_type=eval_holdout_type,
        logger=logger,
    )

    if predictions.height == 0:
        raise RuntimeError("No holdout predictions available")

    logger.info(
        f"Loaded {predictions.height} predictions "
        f"for {predictions['did'].n_unique()} users"
    )

    # Step 2b: Join holdout targets with history (shared by enrichment + metadata)
    log_operation_start('Join holdout targets with history', STAGE_LOG_NAME, logger)
    holdout_with_hist = _build_holdout_with_history(target_posts, history, holdout_split)

    # Step 2c: Enrich predictions with per-row history length (key-based join)
    log_operation_start('Enrich predictions with per-row history length', STAGE_LOG_NAME, logger)
    predictions = enrich_predictions_with_history_len(predictions, holdout_with_hist, logger)
    logger.info("Enriched predictions with num_embedding_likes (per-post history length)")

    # Step 3: Compute user metadata
    log_operation_start('Compute user metadata', STAGE_LOG_NAME, logger)
    user_metadata = compute_user_metadata(predictions, target_posts, holdout_with_hist)
    logger.info(f"Computed metadata for {user_metadata.height} users")

    # Step 4: Create EvalContext.
    # Eval modules consume pandas DataFrames, so convert at the boundary.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    eval_config: Dict[str, Any] = {
        'batch_size': eval_batch_size,
        'embed_dim': embed_dim,
    }

    ctx = EvalContext(
        predictions_df=predictions.to_pandas(),
        user_metadata_df=user_metadata.to_pandas(),
        output_dir=out_dir,
        timestamp=timestamp,
        config=eval_config,
    )

    # Step 5: Discover and run evaluation modules
    log_operation_start('Discover and run evaluation modules', STAGE_LOG_NAME, logger)
    logger.info("Running evaluation modules...")
    module_results = run_all_modules(ctx, skip_modules=skip_modules)

    # Step 6: Save combined summary
    log_operation_start('Save evaluation summary', STAGE_LOG_NAME, logger)

    eval_summary = {
        'timestamp': timestamp,
        'runtime_seconds': time.time() - t0,
        'num_holdout_users': ctx.num_holdout_users,
        'num_predictions': ctx.num_predictions,
        'train_dir': str(train_dir),
        'embed_dim': embed_dim,
        'modules': module_results,
    }

    summary_path = out_dir / 'eval_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(eval_summary, f, indent=2, default=str)

    info_lines = [
        "stage: evaluate",
        f"runtime_seconds: {time.time() - t0:.2f}",
        f"timestamp: {timestamp}",
        f"num_holdout_users: {ctx.num_holdout_users}",
        f"num_predictions: {ctx.num_predictions}",
        f"modules_run: {', '.join(module_results.keys())}",
        f"inputs: target_posts, user_history, holdout predictions",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    logger.info(f"Evaluation complete. Output: {out_dir}")

    return {
        'output_dir': out_dir,
        'artifacts': eval_summary,
    }
