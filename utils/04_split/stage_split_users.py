#!/usr/bin/env python3

"""
Stage 4: Split users into train/val/holdout.

Inputs:
- embedding_bundle_*.pkl from Stage 2
- Optionally retained_users.json and/or user_topic_mixtures.parquet from Stage 3

Outputs under <run_dir>/split/<timestamp>/:
- user_splits.json (train_users, val_users, holdout_users)
- summary.json (counts)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from utils.pipeline.core import new_stage_timestamp_dir, select_prior_output
from utils.helpers import get_stage_logger, log_operation_start
import time


def run(context, args) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '04_split')

    # Initialize logger
    logger = get_stage_logger('STAGE_04_SPLIT', log_file=out_dir / 'stage.log')

    # Locate embedding bundle
    log_operation_start('Locate embedding bundle from prior stage', 'STAGE_04_SPLIT', logger)
    prior_featurize = select_prior_output(run_dir, '02_featurize', use_latest=context.use_latest, prior_path=context.prior_outputs.get('02_featurize'))
    if prior_featurize is None:
        raise FileNotFoundError("Featurize output not found.")
    bundle_candidates = sorted(prior_featurize.glob('embedding_bundle_*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not bundle_candidates:
        raise FileNotFoundError(f"No embedding_bundle_*.pkl found under {prior_featurize}")
    bundle_path = bundle_candidates[0]

    # Load bundle
    log_operation_start('Load embedding bundle', 'STAGE_04_SPLIT', logger)
    import pickle
    with open(bundle_path, 'rb') as f:
        bundle = pickle.load(f)
    posts_emb_df: pd.DataFrame = bundle['posts_emb_df']
    likes_df: pd.DataFrame = bundle['likes_df']
    join_like: str = str(bundle['join_like'])
    join_post: str = str(bundle['join_post'])

    # Eligible users
    available_posts = set(posts_emb_df[join_post].astype(str).unique())
    likes_df_local = likes_df.copy()
    if join_like not in likes_df_local.columns:
        raise KeyError(f"likes_df missing join_like column: {join_like}")
    likes_df_local[join_like] = likes_df_local[join_like].astype(str)
    likes_joinable = likes_df_local[likes_df_local[join_like].isin(available_posts)]
    min_likes_per_user = int(getattr(args, 'min_likes_per_user', 4))
    
    log_operation_start(f'Compute eligible users (min_likes_per_user={min_likes_per_user})', 'STAGE_04_SPLIT', logger)
    counts = likes_joinable.groupby('did', observed=True)[join_like].nunique().astype(int)
    eligible_users = counts[counts >= min_likes_per_user].index.astype(str).tolist()

    # Apply retained_users if present (from relevel stage)
    log_operation_start('Apply retained users filter (if available)', 'STAGE_04_SPLIT', logger)
    prior_relevel = select_prior_output(run_dir, '03_relevel', use_latest=context.use_latest, prior_path=context.prior_outputs.get('03_relevel'))
    if prior_relevel is not None:
        retained = prior_relevel / 'retained_users.json'
        if retained.exists():
            try:
                data = json.loads(retained.read_text())
                retained_users = set(map(str, data.get('retained_users', [])))
                eligible_users = [u for u in eligible_users if u in retained_users]
            except Exception:
                pass

    if len(eligible_users) == 0:
        raise RuntimeError("No eligible users at the given min_likes_per_user threshold after filtering")

    # Split
    t0 = time.time()
    log_operation_start('Split users into train/val/holdout', 'STAGE_04_SPLIT', logger)
    import numpy as np
    rng = np.random.RandomState(int(getattr(args, 'random_seed', 42)))
    users_shuffled = eligible_users.copy()
    rng.shuffle(users_shuffled)
    holdout_ratio = float(getattr(args, 'holdout_ratio', 0.2))
    val_ratio = float(getattr(args, 'val_ratio', 0.2))

    n_holdout = int(np.floor(len(users_shuffled) * holdout_ratio))
    holdout_users = users_shuffled[:n_holdout]
    remaining = users_shuffled[n_holdout:]
    n_val = int(np.floor(len(remaining) * val_ratio))
    val_users = remaining[:n_val]
    train_users = remaining[n_val:]

    log_operation_start('Save user splits', 'STAGE_04_SPLIT', logger)
    splits = {
        'train_users': train_users,
        'val_users': val_users,
        'holdout_users': holdout_users,
        'min_likes_per_user': int(min_likes_per_user),
    }
    splits_path = out_dir / 'user_splits.json'
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)

    summary = {
        'counts': {
            'eligible_users': int(len(eligible_users)),
            'train_users': int(len(train_users)),
            'val_users': int(len(val_users)),
            'holdout_users': int(len(holdout_users)),
        }
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Stage info
    info_lines = [
        f"stage: split",
        f"runtime_seconds: {time.time()-t0:.2f}",
        f"settings: min_likes_per_user={min_likes_per_user}, holdout_ratio={holdout_ratio}, val_ratio={val_ratio}",
        f"inputs: embedding_bundle, mixtures/retained_users (optional)",
        f"N_eligible_users: {len(eligible_users)}",
        f"N_train_users: {len(train_users)}",
        f"N_val_users: {len(val_users)}",
        f"N_holdout_users: {len(holdout_users)}",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    return {
        'output_dir': out_dir,
        'artifacts': {
            'user_splits_path': str(splits_path.resolve()),
            'embedding_bundle_path': str(bundle_path.resolve()),
        }
    }


