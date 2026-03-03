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
- holdout_eval/predictions.parquet from 04_train

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

import pandas as pd
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
from pathlib import Path as _Path


def _import_evals_module():
    """Import the evals package from the numeric directory.

    We set ``submodule_search_locations`` so that the sub-modules
    (cold_start_curves, performance_inequality, …) can use relative imports
    like ``from . import EvalContext``.
    """
    evals_dir = _Path(__file__).parent / 'evals'
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
    """
    Locate the training stage output directory.

    Tries same-session artifacts (``train_mlp`` / ``train_two_tower``) first,
    then falls back to filesystem scanning under ``04_train/``.

    Returns:
        Path to the training stage timestamp directory.

    Raises:
        FileNotFoundError: If no training output can be found.
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
    holdout_eval_dir: Optional[Path],
    logger=None,
) -> pd.DataFrame:
    """
    Load pre-computed holdout predictions from the training stage.

    Both MLP and Two-Tower training stages save predictions to
    ``holdout_eval/predictions.parquet`` with columns
    [did, post_id, y_true, y_pred_proba].

    Returns:
        DataFrame with columns [did, post_id, y_true, y_pred_proba]

    Raises:
        FileNotFoundError: If no predictions file can be found.
    """
    if holdout_eval_dir is not None:
        pred_parquet = holdout_eval_dir / 'predictions.parquet'
        pred_csv = holdout_eval_dir / 'predictions.csv'

        if pred_parquet.exists():
            if logger:
                logger.info(f"Loading pre-computed predictions from {pred_parquet}")
            return pd.read_parquet(pred_parquet)
        elif pred_csv.exists():
            if logger:
                logger.info(f"Loading pre-computed predictions from {pred_csv}")
            return pd.read_csv(pred_csv)

    raise FileNotFoundError(
        "No holdout predictions found (expected holdout_eval/predictions.parquet "
        "in the 04_train output directory). Please ensure Stage 4 (train) "
        "completed successfully with holdout evaluation enabled."
    )


# ---------------------------------------------------------------------------
# Per-prediction history length enrichment
# ---------------------------------------------------------------------------

def enrich_predictions_with_history_len(
    predictions_df: pd.DataFrame,
    target_posts_df: pl.DataFrame,
    history_df: pl.DataFrame,
    holdout_split: str = "holdout_unseen_users",
) -> pd.DataFrame:
    """
    Add a per-row ``num_embedding_likes`` column to ``predictions_df``.

    Uses **positional assignment**: predictions are emitted in strict
    alternating order ``[pos_0, neg_0, pos_1, neg_1, …]`` by the training
    stage's ``_collect_predictions``, so ``predictions_df[i]`` always
    belongs to holdout target row ``i // 2``.  Both the positive and
    negative sample from the same target row therefore receive the same
    history length.

    A join-based approach cannot be used because ``post_id`` is
    ``like_uri`` for positives but ``neg_uri`` for negatives, and
    collisions between the two sets make a key-based merge ambiguous.

    Raises:
        ValueError: If the number of predictions is not exactly twice the
            number of holdout target rows (which would indicate the
            positional assumption is violated).

    Returns:
        Copy of ``predictions_df`` with ``num_embedding_likes`` column added.
    """
    holdout_targets = target_posts_df.filter(pl.col("split") == holdout_split)

    joined = holdout_targets.join(
        history_df.select(["target_did", "like_uri", "prior_emb_indices"]),
        on=["target_did", "like_uri"],
        how="left",
    ).with_columns(
        pl.col("prior_emb_indices").list.len().alias("num_embedding_likes")
    )

    hist_per_row = joined["num_embedding_likes"].fill_null(0).to_list()
    n_target_rows = len(hist_per_row)
    n_predictions = len(predictions_df)

    if n_predictions != 2 * n_target_rows:
        raise ValueError(
            f"predictions_df has {n_predictions} rows but holdout targets "
            f"have {n_target_rows} rows (expected 2x). "
            f"Cannot assign history by position."
        )

    enriched = predictions_df.copy()
    enriched["num_embedding_likes"] = [
        hist_per_row[i // 2] for i in range(n_predictions)
    ]
    return enriched


# ---------------------------------------------------------------------------
# User metadata
# ---------------------------------------------------------------------------

def compute_user_metadata(
    predictions_df: pd.DataFrame,
    target_posts_df: pl.DataFrame,
    history_df: pl.DataFrame,
    holdout_split: str = "holdout_unseen_users",
) -> pd.DataFrame:
    """
    Compute per-user metadata including number of embedding likes.

    The number of embedding likes is derived from the user history: for each
    holdout (user, like_event) pair, it is the count of prior embedding indices.
    When a user has multiple holdout rows, we take the maximum history length.

    Returns:
        DataFrame with columns [did, num_embedding_likes, num_total_likes]
    """
    # Get holdout users from predictions
    holdout_user_ids = set(predictions_df['did'].astype(str).unique())

    # Get holdout target rows
    holdout_targets = target_posts_df.filter(pl.col("split") == holdout_split)

    # Join with history to get prior_emb_indices
    joined = holdout_targets.join(
        history_df.select(["target_did", "like_uri", "prior_emb_indices"]),
        on=["target_did", "like_uri"],
        how="left",
    )

    # Compute per-user history length (max across their holdout rows)
    user_history_lens = (
        joined
        .with_columns(
            pl.col("prior_emb_indices").list.len().alias("history_len")
        )
        .group_by("target_did")
        .agg([
            pl.col("history_len").max().alias("num_embedding_likes"),
            pl.col("like_uri").n_unique().alias("num_holdout_targets"),
        ])
        .rename({"target_did": "did"})
    )

    # Also count total likes per user from all splits
    total_likes = (
        target_posts_df
        .filter(pl.col("target_did").is_in(list(holdout_user_ids)))
        .group_by("target_did")
        .agg(pl.col("like_uri").n_unique().alias("num_total_likes"))
        .rename({"target_did": "did"})
    )

    # Merge
    metadata_pl = user_history_lens.join(total_likes, on="did", how="left")
    metadata_df = metadata_pl.to_pandas()
    metadata_df['did'] = metadata_df['did'].astype(str)
    metadata_df['num_embedding_likes'] = metadata_df['num_embedding_likes'].fillna(0).astype(int)
    metadata_df['num_total_likes'] = metadata_df['num_total_likes'].fillna(0).astype(int)

    # Add any users from predictions that are missing from metadata
    pred_users = set(predictions_df['did'].astype(str).unique())
    existing_users = set(metadata_df['did'].astype(str).unique())
    missing_users = pred_users - existing_users

    if missing_users:
        missing_df = pd.DataFrame({
            'did': list(missing_users),
            'num_embedding_likes': 0,
            'num_total_likes': 0,
        })
        metadata_df = pd.concat([metadata_df, missing_df], ignore_index=True)

    return metadata_df[['did', 'num_embedding_likes', 'num_total_likes']]


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run(context: Context, args) -> Dict[str, Any]:
    """
    Main entry point for Stage 5: Evaluation.

    Loads holdout predictions from the training stage and runs all evaluation
    modules.
    """
    run_dir = Path(context.run_dir).resolve()
    t0 = time.time()

    # --- hyperparams ---
    eval_batch_size = int(args.eval_batch_size)
    eval_holdout_type = str(args.eval_holdout_type)
    holdout_split = f"holdout_{eval_holdout_type}"
    skip_modules = getattr(args, 'skip_modules', None)
    if skip_modules and isinstance(skip_modules, str):
        skip_modules = [m.strip() for m in skip_modules.split(',')]
    cold_start_bin_edges = getattr(args, 'cold_start_bin_edges', None)

    # Resolve training output first so we can nest eval outputs inside it
    train_dir = resolve_train_output(run_dir, context)
    holdout_eval_dir = train_dir / 'holdout_eval'

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

    # Step 1: Load training data from prior stages (target_posts + history for metadata)
    log_operation_start('Load training data from prior stages', STAGE_LOG_NAME, logger)
    _, target_posts_df, history_df, embed_dim = load_training_data(
        run_dir, context, logger=logger,
    )

    holdout_target_rows = target_posts_df.filter(pl.col("split") == holdout_split)
    holdout_users = holdout_target_rows["target_did"].unique().to_list()
    holdout_users = [str(u) for u in holdout_users]

    if not holdout_users:
        raise RuntimeError(
            f"No holdout users found in target_posts (no rows with split='{holdout_split}'). "
            f"Did the target_posts stage produce a {eval_holdout_type} holdout split? "
            f"Check --holdout-user-fraction and --holdout-start in your pipeline configuration."
        )
    logger.info(f"Holdout users ({eval_holdout_type}): {len(holdout_users)}")
    logger.info(f"Holdout target rows ({eval_holdout_type}): {len(holdout_target_rows)}")

    # Step 2: Load holdout predictions
    log_operation_start('Load holdout predictions', STAGE_LOG_NAME, logger)
    predictions_df = load_holdout_predictions(
        holdout_eval_dir=holdout_eval_dir if holdout_eval_dir.exists() else None,
        logger=logger,
    )

    if len(predictions_df) == 0:
        raise RuntimeError("No holdout predictions available")

    logger.info(f"Loaded {len(predictions_df)} predictions for {predictions_df['did'].nunique()} users")

    # Step 2b: Enrich predictions with per-row history length
    log_operation_start('Enrich predictions with per-row history length', STAGE_LOG_NAME, logger)
    predictions_df = enrich_predictions_with_history_len(
        predictions_df=predictions_df,
        target_posts_df=target_posts_df,
        history_df=history_df,
        holdout_split=holdout_split,
    )
    logger.info(f"Enriched predictions with num_embedding_likes (post-level history length)")

    # Step 3: Compute user metadata
    log_operation_start('Compute user metadata', STAGE_LOG_NAME, logger)
    user_metadata_df = compute_user_metadata(
        predictions_df=predictions_df,
        target_posts_df=target_posts_df,
        history_df=history_df,
        holdout_split=holdout_split,
    )
    logger.info(f"Computed metadata for {len(user_metadata_df)} users")

    # Step 4: Create EvalContext
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    eval_config: Dict[str, Any] = {
        'batch_size': eval_batch_size,
        'embed_dim': embed_dim,
    }
    if cold_start_bin_edges is not None:
        eval_config['cold_start_bin_edges'] = cold_start_bin_edges

    ctx = EvalContext(
        predictions_df=predictions_df,
        user_metadata_df=user_metadata_df,
        bundle={},
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
