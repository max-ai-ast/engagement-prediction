#!/usr/bin/env python3

"""
Stage 5: Evaluate a trained model using modular evaluation framework.

This stage orchestrates the evaluation pipeline by:
1. Loading holdout ranking rows from Stage 4 (04_train)
2. Computing user metadata from those rows
3. Creating an EvalContext and running all discovered evaluation modules

Evaluation modules are auto-discovered from utils/05_evaluate/evals/ and each
produces its own set of artifacts (plots, CSVs, JSON summaries).

Inputs (from prior pipeline stages):
- eval/holdout_<type>_ranking_rows.parquet from 04_train

Outputs under artifacts/05_evaluate/<stage_run_id>/
- eval_summary.json: Combined results from all modules
- stage_info.txt: Stage metadata
- <module_name>/: Subdirectory for each evaluation module's artifacts
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from utils.pipeline.core import Context, generate_run_timestamp
from utils.helpers import get_stage_logger, log_operation_start

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
    (cold_start_curves, performance_inequality, …) can use relative imports
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
            return context.record_prior_input("04_train", art_dir)

    return context.resolve_prior_output("04_train", prior_path=context.prior_outputs.get("04_train"))


# ---------------------------------------------------------------------------
# Holdout ranking rows
# ---------------------------------------------------------------------------

def load_holdout_ranking_rows(
    eval_dir: Optional[Path],
    holdout_type: str,
    logger=None,
) -> Optional[pd.DataFrame]:
    if eval_dir is None:
        return None
    ranking_path = eval_dir / f'holdout_{holdout_type}_ranking_rows.parquet'
    if not ranking_path.exists():
        return None
    if logger:
        logger.info(f"Loading ranking rows from {ranking_path}")
    return pd.read_parquet(ranking_path)


def compute_user_metadata_from_ranking_rows(ranking_rows_df: pd.DataFrame) -> pd.DataFrame:
    metadata_df = (
        ranking_rows_df
        .groupby('did', as_index=False)
        .agg(
            num_embedding_likes=('num_embedding_likes', 'max'),
            num_total_likes=('num_total_likes', 'max'),
        )
    )
    metadata_df['did'] = metadata_df['did'].astype(str)
    metadata_df['num_embedding_likes'] = metadata_df['num_embedding_likes'].fillna(0).astype(int)
    metadata_df['num_total_likes'] = metadata_df['num_total_likes'].fillna(0).astype(int)
    return metadata_df[['did', 'num_embedding_likes', 'num_total_likes']]


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run(context: Context, args) -> Dict[str, Any]:
    """
    Main entry point for Stage 5: Evaluation.

    Loads holdout ranking rows from the training stage and runs all evaluation
    modules.
    """
    t0 = time.time()

    # --- hyperparams ---
    eval_batch_size = int(args.eval_batch_size)
    eval_holdout_type = str(args.eval_holdout_type)
    skip_modules = args.skip_modules
    if skip_modules and isinstance(skip_modules, str):
        skip_modules = [m.strip() for m in skip_modules.split(',')]

    # Resolve training output (inputs)
    train_dir = resolve_train_output(context)
    train_eval_dir = train_dir / 'eval'

    # Canonical stage output
    out_dir = context.new_stage_dir("05_evaluate", tag=eval_holdout_type)

    # Initialize logger
    logger = get_stage_logger(STAGE_LOG_NAME, log_file=out_dir / 'stage.log')
    log_operation_start('Stage 5: Evaluation', STAGE_LOG_NAME, logger)
    logger.info(f"Training output dir: {train_dir}")
    logger.info(f"Holdout type for evaluation: {eval_holdout_type}")

    log_operation_start('Load evaluation artifact', STAGE_LOG_NAME, logger)
    ranking_rows_df = load_holdout_ranking_rows(
        eval_dir=train_eval_dir if train_eval_dir.exists() else None,
        holdout_type=eval_holdout_type,
        logger=logger,
    )
    if ranking_rows_df is None:
        raise FileNotFoundError(
            f"No holdout ranking rows found. Expected {train_eval_dir / f'holdout_{eval_holdout_type}_ranking_rows.parquet'}. "
            "Please rerun Stage 4 training so it writes matrix ranking-row artifacts."
        )
    predictions_df = pd.DataFrame(columns=['did', 'post_id', 'y_true', 'y_pred_proba'])
    user_metadata_df = compute_user_metadata_from_ranking_rows(ranking_rows_df)
    embed_dim = None
    logger.info(f"Loaded {len(ranking_rows_df)} ranking rows for {ranking_rows_df['did'].nunique()} users")

    # Step 4: Create EvalContext
    timestamp = generate_run_timestamp()

    eval_config: Dict[str, Any] = {
        'batch_size': eval_batch_size,
        'embed_dim': embed_dim,
        'eval_mode': 'ranking_rows',
    }

    ctx = EvalContext(
        predictions_df=predictions_df,
        user_metadata_df=user_metadata_df,
        output_dir=out_dir,
        timestamp=timestamp,
        config=eval_config,
        ranking_rows_df=ranking_rows_df,
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
        'num_ranking_rows': ctx.num_ranking_rows,
        'train_dir': str(train_dir),
        'embed_dim': embed_dim,
        'eval_mode': eval_config['eval_mode'],
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
        f"num_ranking_rows: {ctx.num_ranking_rows}",
        f"modules_run: {', '.join(module_results.keys())}",
        "inputs: ranking rows",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    logger.info(f"Evaluation complete. Output: {out_dir}")

    return {
        'output_dir': out_dir,
        'artifacts': eval_summary,
    }
