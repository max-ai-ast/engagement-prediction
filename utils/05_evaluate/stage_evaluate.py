#!/usr/bin/env python3

"""
Stage 6: Evaluate a trained model (consolidated evaluator).

Modes:
- pairs: training-consistent pairs evaluation with metrics and plots
- matrix: full feed user×post probability matrix for holdouts
- global_unliked: probability matrix over posts unliked by anyone (per splits' holdout users)

Inputs:
- model checkpoint from Stage 5
- embedding_bundle_*.pkl from Stage 2
- user_splits.json from Stage 4

Outputs under <run_dir>/evaluate/<timestamp>_<mode>/
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd

from utils.pipeline.core import new_stage_timestamp_dir, select_prior_output, Context

# Shared helpers
from utils.helpers import (
    get_actual_feature_columns,
    create_pairs_dataset,
    get_stage_logger,
    log_operation_start,
    get_device,
)


def load_saved_model(model_path: str, device: str):
    """Load a trained model checkpoint and reconstruct the network architecture.

    This redefines the training-time architecture locally to avoid importing training modules.
    """
    import torch
    import torch.nn as nn
    ckpt = torch.load(model_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    input_dim = int(ckpt.get('input_dim')) if isinstance(ckpt, dict) and ('input_dim' in ckpt) else None
    hidden_dims = ckpt.get('hidden_dims', [64, 32, 16]) if isinstance(ckpt, dict) else [64, 32, 16]
    dropout_rate = float(ckpt.get('dropout_rate', 0.5)) if isinstance(ckpt, dict) else 0.5

    if input_dim is None:
        # Fallback: infer from first linear layer weight in state_dict
        for k, v in state_dict.items():
            if k.endswith('.0.weight') and v.ndim == 2:
                input_dim = int(v.shape[1])
                break
        if input_dim is None:
            raise RuntimeError("Unable to infer input_dim from checkpoint")

    class _EvalPredictor(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: list, dropout_rate: float):
            super().__init__()
            layers = []
            prev = int(input_dim)
            for h in list(hidden_dims):
                layers.extend([nn.Linear(prev, int(h)), nn.BatchNorm1d(int(h)), nn.GELU(), nn.Dropout(float(dropout_rate))])
                prev = int(h)
            layers.append(nn.Linear(prev, 1))
            layers.append(nn.Sigmoid())
            self.network = nn.Sequential(*layers)
        def forward(self, x):
            return self.network(x)

    model = _EvalPredictor(int(input_dim), list(hidden_dims), float(dropout_rate))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, ckpt


def load_embedding_bundle(bundle_path: str) -> Dict[str, Any]:
    """Load the embedding bundle saved by Stage 2 (save_bundle)."""
    import pickle
    with open(bundle_path, 'rb') as f:
        return pickle.load(f)

def _resolve_assets(run_dir: Path, context, args) -> Tuple[str, str, str]:
    # embedding bundle
    prior_featurize = select_prior_output(run_dir, '02_featurize', use_latest=context.use_latest, prior_path=context.prior_outputs.get('02_featurize'))
    if prior_featurize is None:
        raise FileNotFoundError("Featurize output not found.")
    bundle_candidates = sorted(prior_featurize.glob('embedding_bundle_*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not bundle_candidates:
        raise FileNotFoundError(f"No embedding_bundle_*.pkl found under {prior_featurize}")
    bundle_path = str(bundle_candidates[0].resolve())

    # user_splits
    prior_split = select_prior_output(run_dir, '04_split', use_latest=context.use_latest, prior_path=context.prior_outputs.get('04_split'))
    if prior_split is None:
        raise FileNotFoundError("Split output not found.")
    splits_path = str((prior_split / 'user_splits.json').resolve())
    if not Path(splits_path).exists():
        raise FileNotFoundError(f"user_splits.json not found under {prior_split}")

    # model
    prior_train = select_prior_output(run_dir, '05_train', use_latest=context.use_latest, prior_path=context.prior_outputs.get('05_train'))
    model_path = None
    if prior_train is not None:
        ckpts = list((prior_train / 'checkpoints').glob('*.pth')) if (prior_train / 'checkpoints').exists() else []
        if ckpts:
            ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            model_path = str(ckpts[0].resolve())
    if model_path is None:
        raise FileNotFoundError("Model checkpoint not found under prior train stage")

    return model_path, bundle_path, splits_path


def _load_holdout_users(splits_path: str) -> List[str]:
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    return list(map(str, splits.get('holdout_users', [])))


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()

    mode = str(getattr(args, 'mode', 'heterogeneity'))  # 'heterogeneity' | 'pairs' | 'matrix' | 'global_unliked'
    out_dir = new_stage_timestamp_dir(run_dir, '06_evaluate')
    # Rename directory to include mode for clarity
    mode_dir = out_dir.parent / f"{out_dir.name}_{mode}"
    out_dir.rename(mode_dir)
    out_dir = mode_dir

    # Initialize logger
    logger = get_stage_logger('STAGE_06_EVALUATE', log_file=out_dir / 'stage.log')

    log_operation_start('Resolve assets (model, bundle, splits)', 'STAGE_06_EVALUATE', logger)
    model_path, bundle_path, splits_path = _resolve_assets(run_dir, context, args)
    device = get_device(args.device)
    batch_size = int(args.eval_batch_size)
    enforce_training_config = bool(getattr(args, 'enforce_training_config', True))

    # Load model & bundle
    log_operation_start('Load model and bundle', 'STAGE_06_EVALUATE', logger)
    model, checkpoint = load_saved_model(model_path, device=device)
    bundle = load_embedding_bundle(bundle_path)
    posts_emb_df: pd.DataFrame = bundle['posts_emb_df']  # type: ignore[assignment]
    likes_df: pd.DataFrame = bundle['likes_df']          # type: ignore[assignment]
    join_like: str = str(bundle['join_like'])
    join_post: str = str(bundle['join_post'])
    embedding_dim: int = int(bundle['embedding_dim'])

    # Resolve feature_columns lazily with fallbacks (checkpoint → training_config.json → None)
    feature_columns = checkpoint['feature_columns'] if (isinstance(checkpoint, dict) and ('feature_columns' in checkpoint)) else None
    if feature_columns is None:
        # try training_config.json from prior 05_train
        prior_train = select_prior_output(run_dir, '05_train', use_latest=context.use_latest, prior_path=context.prior_outputs.get('05_train'))
        cfg_path = None
        if prior_train:
            candidates = [prior_train / 'training_config.json']
            logs_dir = prior_train / 'logs'
            if logs_dir.exists():
                for c in sorted(logs_dir.glob('training_config*.json')):
                    candidates.append(c)
            for c in candidates:
                if c.exists():
                    cfg_path = c
                    break
        if cfg_path:
            try:
                with open(cfg_path, 'r') as cf:
                    cfg = json.load(cf)
                    if 'feature_columns' in cfg:
                        feature_columns = cfg['feature_columns']
            except Exception:
                pass

    holdout_users = _load_holdout_users(splits_path)
    if not holdout_users:
        raise RuntimeError("No holdout users in splits file")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build disjoint sets per user similar to training and construct pairs (pairs/matrix helper reuse)
    import time
    from utils.helpers import build_user_feature_frame as build_user_features_shared
    from utils.helpers import get_actual_feature_columns, create_pairs_dataset

    # Eligibility computation (joinable likes vs embedded posts)
    log_operation_start('Compute eligible holdout users', 'STAGE_06_EVALUATE', logger)
    likes_hou = likes_df[likes_df['did'].isin(holdout_users)].copy()
    available_posts = set(posts_emb_df[join_post].astype(str).unique())
    likes_hou[join_like] = likes_hou[join_like].astype(str)
    likes_joinable = likes_hou[likes_hou[join_like].isin(available_posts)]
    min_likes_per_user = int(args.min_likes_per_user)
    counts = (
        likes_joinable.groupby('did', observed=True)[join_like]
        .nunique()
        .astype(int)
        .sort_values(ascending=False)
    )
    eligible_set = set(counts[counts >= min_likes_per_user].index.astype(str).tolist())
    selected_users = [u for u in holdout_users if str(u) in eligible_set]

    # Allocation for pairs/matrix
    prediction_posts_per_user = int(args.prediction_posts_per_user)
    max_embedding_posts_per_user = int(getattr(args, 'max_embedding_posts_per_user', 50))
    negatives_liked_only = bool(getattr(args, 'negatives_liked_only', False))
    cap_seed = int(args.cap_random_seed)

    # Filter likes to selected users & joinable
    likes_local = likes_df[likes_df['did'].isin(set(selected_users))].copy()
    likes_local[join_like] = likes_local[join_like].astype(str)
    likes_local = likes_local[likes_local[join_like].isin(available_posts)]

    # Allocate disjoint sets
    log_operation_start('Allocate disjoint sets per user', 'STAGE_06_EVALUATE', logger)
    embedding_likes_list = []
    prediction_likes_list = []
    for user_id, g in likes_local.groupby('did'):
        user_posts = sorted(list(set(g[join_like].astype(str).unique())))
        if len(user_posts) < max(2, prediction_posts_per_user + 1):
            continue
        posts_for_prediction = set(user_posts[-int(prediction_posts_per_user):])
        posts_for_embedding = set(user_posts[:-int(prediction_posts_per_user)])
        if len(posts_for_embedding) > int(max_embedding_posts_per_user):
            posts_for_embedding = set(sorted(list(posts_for_embedding))[:int(max_embedding_posts_per_user)])
        if posts_for_prediction & posts_for_embedding:
            continue
        if posts_for_embedding:
            embedding_likes_list.append(g[g[join_like].isin(posts_for_embedding)])
        if posts_for_prediction:
            prediction_likes_list.append(g[g[join_like].isin(posts_for_prediction)])
    embedding_likes_df = pd.concat(embedding_likes_list, ignore_index=True) if embedding_likes_list else pd.DataFrame()
    prediction_likes_df = pd.concat(prediction_likes_list, ignore_index=True) if prediction_likes_list else pd.DataFrame()

    # Build user features matching checkpoint feature_columns layout
    log_operation_start('Build user features', 'STAGE_06_EVALUATE', logger)
    user_emb_df = build_user_features_shared(
        schema='multi_centroid' if any(c.startswith('user_k') for c in feature_columns[0]) else ('topic_mixture' if any(c.startswith('user_topic_') for c in feature_columns[0]) else 'mean'),
        likes_df=embedding_likes_df if len(embedding_likes_df) else likes_local,
        posts_emb_df=posts_emb_df,
        join_like=join_like,
        join_post=join_post,
        embedding_dim=int(embedding_dim),
        selected_users=selected_users,
        feature_columns=feature_columns,
        random_seed=cap_seed,
    )

    # Create outputs depending on mode
    artifacts: Dict[str, Any] = {}
    t0 = time.time()
    log_operation_start(f'Run inference (mode={mode})', 'STAGE_06_EVALUATE', logger)
    if mode == 'heterogeneity':
        # Prefer Stage 5 holdout predictions
        # Resolve 05_train (fallback 'train')
        prior_train = select_prior_output(run_dir, '05_train', use_latest=context.use_latest, prior_path=context.prior_outputs.get('05_train'))
        if prior_train is None:
            prior_train = select_prior_output(run_dir, 'train', use_latest=context.use_latest, prior_path=context.prior_outputs.get('train'))
        he_dir = None
        if prior_train is not None and (prior_train / 'holdout_eval').exists():
            he_dir = prior_train / 'holdout_eval'
        if he_dir is not None and (he_dir / 'predictions.parquet').exists():
            preds_df = pd.read_parquet(he_dir / 'predictions.parquet')
        elif he_dir is not None and (he_dir / 'predictions.csv').exists():
            preds_df = pd.read_csv(he_dir / 'predictions.csv')
        else:
            # Fallback: compute predictions as in pairs mode, then proceed
            # Reuse pairs branch to generate y_true/y_pred and ids quickly
            # Then compute per-user stats
            # Build via pairs path below, but do not save pairs artifacts
            # (Minimal duplication for brevity)
            text_emb_cols = [c for c in posts_emb_df.columns if c.startswith('post_emb_')]
            image_emb_cols = [c for c in posts_emb_df.columns if c.startswith('image_emb_')]
            post_emb_cols = text_emb_cols + image_emb_cols
            from utils.helpers import build_user_feature_frame, get_actual_feature_columns
            feature_columns = checkpoint.get('feature_columns') if isinstance(checkpoint, dict) else None
            if feature_columns is None:
                feature_columns = get_actual_feature_columns(posts_emb_df)
            # Compute selected users (same as below)
            likes_hou = likes_df.copy()
            likes_hou[join_like] = likes_hou[join_like].astype(str)
            likes_hou = likes_hou[likes_hou[join_like].isin(set(posts_emb_df[join_post].astype(str).unique()))]
            selected_users = list(likes_hou['did'].astype(str).unique())
            # Simple per-user allocation
            embedding_likes_list = []
            prediction_likes_list = []
            for user_id, g in likes_hou.groupby('did'):
                user_posts = sorted(list(set(g[join_like].astype(str).unique())))
                if len(user_posts) < 2:
                    continue
                posts_for_prediction = set([user_posts[-1]])
                posts_for_embedding = set(user_posts[:-1])
                if posts_for_embedding:
                    embedding_likes_list.append(g[g[join_like].isin(posts_for_embedding)])
                prediction_likes_list.append(g[g[join_like].isin(posts_for_prediction)])
            embedding_likes_df = pd.concat(embedding_likes_list, ignore_index=True) if embedding_likes_list else pd.DataFrame()
            prediction_likes_df = pd.concat(prediction_likes_list, ignore_index=True) if prediction_likes_list else pd.DataFrame()
            user_emb_df = build_user_feature_frame(
                schema='multi_centroid' if any(c.startswith('user_k') for c in feature_columns[0]) else ('topic_mixture' if any(c.startswith('user_topic_') for c in feature_columns[0]) else 'mean'),
                likes_df=embedding_likes_df if len(embedding_likes_df) else likes_hou,
                posts_emb_df=posts_emb_df,
                join_like=join_like,
                join_post=join_post,
                embedding_dim=int(embedding_dim),
                selected_users=selected_users,
                feature_columns=feature_columns,
                random_seed=int(args.random_seed),
            )
            pos_df = prediction_likes_df.merge(
                posts_emb_df[[join_post] + post_emb_cols], left_on=join_like, right_on=join_post, how='inner'
            ) if len(prediction_likes_df) else pd.DataFrame(columns=[join_post]+post_emb_cols+['did'])
            if len(pos_df) > 0:
                pos_df['liked'] = 1
                prediction_pairs_df = pos_df.merge(user_emb_df, on='did', how='inner')
                # Inference
                user_emb_cols, post_emb_cols, feature_cols = feature_columns
                def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
                    return df[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
                X = np.concatenate([
                    _coerce_numeric(prediction_pairs_df, user_emb_cols).values,
                    _coerce_numeric(prediction_pairs_df, post_emb_cols).values
                ], axis=1).astype(np.float32, copy=False)
                import torch
                preds_list: List[np.ndarray] = []
                bs = max(1024, min(batch_size, 131072))
                model.eval()
                with torch.no_grad():
                    for start in range(0, X.shape[0], bs):
                        end = min(start + bs, X.shape[0])
                        xb = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)
                        pb = model(xb).squeeze().detach().cpu().numpy()
                        preds_list.append(pb)
                y_pred = np.concatenate(preds_list, axis=0).astype(np.float32, copy=False)
                preds_df = pd.DataFrame({
                    'did': prediction_pairs_df['did'].astype(str).tolist(),
                    'post_id': prediction_pairs_df[join_post].astype(str).tolist(),
                    'y_true': prediction_pairs_df['liked'].astype(np.int8).tolist(),
                    'y_pred_proba': y_pred.tolist(),
                })
            else:
                preds_df = pd.DataFrame(columns=['did','post_id','y_true','y_pred_proba'])

        # Compute per-user metrics and inequality
        rows = []
        try:
            from sklearn.metrics import roc_auc_score, accuracy_score
            for uid, g in preds_df.groupby('did'):
                acc = float(accuracy_score(g['y_true'], (g['y_pred_proba'] > 0.5).astype(int))) if len(g) else float('nan')
                auc = float(roc_auc_score(g['y_true'], g['y_pred_proba'])) if len(set(g['y_true'])) > 1 else float('nan')
                rows.append({'did': uid, 'num_samples': int(len(g)), 'accuracy': acc, 'auc_roc': auc})
        except Exception:
            for uid, g in preds_df.groupby('did'):
                acc = float(((g['y_pred_proba'] > 0.5).astype(int) == g['y_true']).mean()) if len(g) else float('nan')
                rows.append({'did': uid, 'num_samples': int(len(g)), 'accuracy': acc})
        per_user_df = pd.DataFrame(rows)
        # Inequality: Gini, Theil, CV on per-user accuracy (drop NaN)
        acc_vals = per_user_df['accuracy'].dropna().values.astype(float)
        def _gini(x: np.ndarray) -> float:
            if x.size == 0:
                return float('nan')
            diff_sum = np.abs(x[:, None] - x[None, :]).sum()
            return diff_sum / (2 * x.size * x.sum()) if x.sum() != 0 else float('nan')
        def _theil(x: np.ndarray) -> float:
            if x.size == 0:
                return float('nan')
            mu = x.mean()
            x_safe = x.copy()
            x_safe[x_safe <= 0] = 1e-12
            return float((x_safe / mu * np.log(x_safe / mu)).mean()) if mu > 0 else float('nan')
        def _cv(x: np.ndarray) -> float:
            return float(x.std() / x.mean()) if x.size and x.mean() != 0 else float('nan')
        hetero = {
            'num_users': int(len(per_user_df)),
            'gini_accuracy': _gini(acc_vals),
            'theil_accuracy': _theil(acc_vals),
            'cv_accuracy': _cv(acc_vals),
        }
        # Save artifacts
        per_user_df.to_csv(out_dir / f"per_user_metrics_{timestamp}.csv", index=False)
        with open(out_dir / f"heterogeneity_summary_{timestamp}.json", 'w') as f:
            json.dump(hetero, f, indent=2)
        artifacts.update({'per_user_metrics_csv': str((out_dir / f"per_user_metrics_{timestamp}.csv").resolve()),
                          'heterogeneity_summary_json': str((out_dir / f"heterogeneity_summary_{timestamp}.json").resolve())})

    elif mode == 'matrix':
        log_operation_start('Compute user×post probability matrix', 'STAGE_06_EVALUATE', logger)
        prob_matrix, user_ids, post_ids = predict_user_post_matrix(
            model,
            user_emb_df,
            posts_emb_df,
            join_post=join_post,
            expected_input_dim=checkpoint.get('input_dim') if isinstance(checkpoint, dict) else None,
            device=device,
            batch_size=batch_size,
            feature_columns=feature_columns,
        )
        prob_path = out_dir / f"prob_matrix_{timestamp}.npz"
        np.savez_compressed(
                prob_path,
            probs=prob_matrix.astype(np.float16),
            user_ids=np.array(user_ids, dtype=object),
            post_ids=np.array(post_ids, dtype=object),
        )
        artifacts['prob_matrix_path'] = str(prob_path)
    elif mode == 'global_unliked':
        log_operation_start('Filter posts to unliked and compute probability matrix', 'STAGE_06_EVALUATE', logger)
        # Filter posts to those unliked by anyone
        all_liked_posts = set(likes_df[join_like].astype(str).unique())
        posts_emb_df = posts_emb_df.copy()
        posts_emb_df[join_post] = posts_emb_df[join_post].astype(str)
        mask_unliked = ~posts_emb_df[join_post].isin(all_liked_posts)
        unliked_posts_df = posts_emb_df[mask_unliked].reset_index(drop=True)
        prob_matrix, user_ids, post_ids = predict_user_post_matrix(
            model,
            user_emb_df,
            unliked_posts_df,
            join_post=join_post,
            expected_input_dim=checkpoint.get('input_dim') if isinstance(checkpoint, dict) else None,
            device=device,
            batch_size=batch_size,
            feature_columns=feature_columns,
        )
        prob_path = out_dir / f"global_prob_matrix_{timestamp}.npz"
        np.savez_compressed(
                prob_path,
            probs=prob_matrix.astype(np.float16),
            user_ids=np.array(user_ids, dtype=object),
            post_ids=np.array(post_ids, dtype=object),
        )
        artifacts['global_prob_matrix_path'] = str(prob_path)
    else:
        # pairs
        log_operation_start('Create prediction pairs and run inference', 'STAGE_06_EVALUATE', logger)
        text_emb_cols = [c for c in posts_emb_df.columns if c.startswith('post_emb_')]
        image_emb_cols = [c for c in posts_emb_df.columns if c.startswith('image_emb_')]
        post_emb_cols = text_emb_cols + image_emb_cols
        if negatives_liked_only:
            pos_df = prediction_likes_df.merge(
                posts_emb_df[[join_post] + post_emb_cols], left_on=join_like, right_on=join_post, how='inner'
            )
            pos_df['liked'] = 1
            all_liked_posts = set(likes_local[join_like].unique())
            available_posts_with_embeddings = set(posts_emb_df[join_post].unique())
            liked_with_emb = all_liked_posts & available_posts_with_embeddings
            user_positive_posts = {u: set(pos_df[pos_df['did'] == u][join_post].unique()) for u in pos_df['did'].unique()}
            negative_pairs = []
            rng = np.random.RandomState(int(cap_seed))
            for u, pos_posts in user_positive_posts.items():
                embedding_posts_u = set(embedding_likes_df[embedding_likes_df['did'] == u][join_like].unique()) if len(embedding_likes_df) else set()
                candidate_neg = list(liked_with_emb - pos_posts - embedding_posts_u)
                k = len(pos_posts)
                if k > 0 and len(candidate_neg) > 0:
                    take = rng.choice(candidate_neg, size=min(k, len(candidate_neg)), replace=False)
                    negative_pairs.extend([(u, p) for p in take])
            if negative_pairs:
                neg_df = pd.DataFrame(negative_pairs, columns=['did', join_post])
                neg_df['liked'] = 0
                neg_df = neg_df.merge(posts_emb_df[[join_post] + post_emb_cols], on=join_post, how='inner')
                prediction_pairs_df = pd.concat([pos_df, neg_df], ignore_index=True)
            else:
                prediction_pairs_df = pos_df
        else:
            prediction_pairs_df = create_pairs_dataset(
                prediction_likes_df, posts_emb_df, join_like, join_post, neg_ratio=0.5,
                random_seed=cap_seed, use_parallel=True
            )
        prediction_pairs_df = prediction_pairs_df.merge(user_emb_df, on='did', how='inner')

        # Ensure feature_columns available for pairs mode
        if feature_columns is None:
            from utils.helpers import get_actual_feature_columns as _get_cols
            feature_columns = _get_cols(posts_emb_df)
        # Score pairs
        user_emb_cols, post_emb_cols, feature_cols = feature_columns
        def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
            return df[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
        user_vals = _coerce_numeric(prediction_pairs_df, user_emb_cols)
        post_vals = _coerce_numeric(prediction_pairs_df, post_emb_cols)
        X = np.concatenate([user_vals.values, post_vals.values], axis=1).astype(np.float32, copy=False)
        y_true = prediction_pairs_df['liked'].astype(np.int8).values
        # Batch inference
        preds_list: List[np.ndarray] = []
        bs = max(1024, min(batch_size, 131072))
        model.eval()
        import torch
        with torch.no_grad():
            for start in range(0, X.shape[0], bs):
                end = min(start + bs, X.shape[0])
                xb = torch.as_tensor(X[start:end], dtype=torch.float32, device=device)
                pb = model(xb).squeeze().detach().cpu().numpy()
                preds_list.append(pb)
        y_pred = np.concatenate(preds_list, axis=0).astype(np.float32, copy=False)
        pairs_npz_path = out_dir / f"pairs_eval_{timestamp}.npz"
        np.savez_compressed(
            pairs_npz_path,
            y_true=y_true,
            y_pred_proba=y_pred,
            user_ids=np.array(prediction_pairs_df['did'].astype(str).tolist(), dtype=object),
            post_ids=np.array(prediction_pairs_df[join_post].astype(str).tolist(), dtype=object),
        )
        artifacts['pairs_eval_npz'] = str(pairs_npz_path)

    # Summary
    log_operation_start('Compute metrics and save outputs', 'STAGE_06_EVALUATE', logger)
    summary = {
        'timestamp': timestamp,
        'mode': mode,
        'model_path': str(model_path),
        'embedding_bundle': str(bundle_path),
        'user_splits': str(splits_path),
        **artifacts,
    }
    with open(out_dir / f"summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Stage info
    info_lines = [
        f"stage: evaluate",
        f"runtime_seconds: {time.time()-t0:.2f}",
        f"settings: mode={mode}, batch_size={batch_size}, min_likes_per_user={min_likes_per_user}",
        f"inputs: model, embedding_bundle, user_splits",
        f"N_holdout_users_total: {len(holdout_users)}",
        f"N_selected_users: {len(selected_users)}",
        f"N_posts_emb: {len(posts_emb_df)}",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    return {
        'output_dir': out_dir,
        'artifacts': summary,
    }

