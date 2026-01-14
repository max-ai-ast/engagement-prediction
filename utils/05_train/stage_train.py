#!/usr/bin/env python3

"""
Stage 5: Train model using embedding bundle + user splits, then evaluate on holdout users.

Inputs:
- embedding_bundle_*.pkl from Stage 2
- user_splits.json from Stage 4

Outputs:
- <run_dir>/05_train/<timestamp>/{checkpoints,plots,logs,training_config.json}
- <run_dir>/05_train/<timestamp>/holdout_eval/{predictions.parquet,metrics_overall.json,metrics_per_user.csv,plots}
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from utils.pipeline.core import select_prior_output
from utils.helpers import (
    build_user_feature_frame,
    get_actual_feature_columns,
    plot_model_performance,
    create_pairs_dataset,
    get_stage_logger,
    log_operation_start,
)
import json
import numpy as np
import pandas as pd
import time
import torch


import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class EngagementDataset(Dataset):
    """Simple dataset for engagement prediction (stage-local)."""
    def __init__(self, features: np.ndarray, labels: np.ndarray, user_ids: Optional[List[str]] = None, post_ids: Optional[List[str]] = None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        self.user_ids = user_ids if user_ids else [f"user_{i}" for i in range(len(features))]
        self.post_ids = post_ids if post_ids else [f"post_{i}" for i in range(len(features))]
        print(f"📊 Dataset: {len(self)} samples, {self.features.shape[1]} features")
    def __len__(self) -> int:
        return len(self.features)
    def __getitem__(self, idx: int):
        return {
            'features': self.features[idx],
            'label': self.labels[idx],
            'user_id': self.user_ids[idx],
            'post_id': self.post_ids[idx],
        }


class EngagementPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout_rate)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        print(f"model architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> 1")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def create_data_loaders(train_dataset: Dataset, val_dataset: Dataset, batch_size: int, test_dataset: Optional[Dataset] = None):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None
    return train_loader, val_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    model_name: str = "engagement_model",
    load_best_checkpoint: bool = False,
    holdout_data: Optional[Dict[str, Any]] = None,
    checkpoints_dir: Optional[Path] = None,
    disable_progress: bool = False,
) -> Dict[str, Any]:
    try:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        import torch.optim as optim
        from sklearn.metrics import roc_auc_score
    except Exception:
        ReduceLROnPlateau = None  # type: ignore
        import torch.optim as optim  # type: ignore
        def roc_auc_score(y_true, y_score) -> Float:  # type: ignore
            return 0.5
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5) if ReduceLROnPlateau else None
    history = {'train_loss': [], 'val_loss': [], 'train_auc': [], 'val_auc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    ckpt_dir = Path(checkpoints_dir) if checkpoints_dir is not None else (Path(__file__).parent.parent.parent / 'outputs' / 'checkpoints')
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    from tqdm import tqdm as _tqdm  # use real tqdm if available
    for epoch in _tqdm(range(epochs), desc="Training epochs", disable=disable_progress):
        model.train()
        tr_loss = 0.0
        tr_preds: List[float] = []
        tr_labels: List[float] = []
        for batch in _tqdm(train_loader, desc="Training", leave=False, disable=disable_progress):
            feats = batch['features'].to(device)
            labels = batch['label'].to(device)
            optimizer.zero_grad()
            preds = model(feats).squeeze()
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tr_preds.extend(preds.detach().cpu().numpy().tolist())
            tr_labels.extend(labels.detach().cpu().numpy().tolist())
        val_loss = 0.0
        val_preds: List[float] = []
        val_labels: List[float] = []
        model.eval()
        with torch.inference_mode():
            for batch in _tqdm(val_loader, desc="Validation", leave=False, disable=disable_progress):
                feats = batch['features'].to(device)
                labels = batch['label'].to(device)
                preds = model(feats).squeeze()
                loss = criterion(preds, labels)
                val_loss += loss.item()
                val_preds.extend(preds.detach().cpu().numpy().tolist())
                val_labels.extend(labels.detach().cpu().numpy().tolist())
        tr_auc = roc_auc_score(tr_labels, tr_preds) if len(set(tr_labels)) > 1 else 0.5
        va_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.5
        history['train_loss'].append(float(tr_loss / max(1, len(train_loader))))
        history['val_loss'].append(float(val_loss / max(1, len(val_loader))))
        history['train_auc'].append(float(tr_auc))
        history['val_auc'].append(float(va_auc))
        if scheduler is not None:
            scheduler.step(va_auc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_full = ckpt_dir / f"{model_name}_best.pth"
            ckpt_weights = ckpt_dir / f"{model_name}_best_weights.pth"
            history_clean = {k: [float(x) for x in v] for k, v in history.items()}
            payload = {'epoch': int(epoch), 'model_state_dict': model.state_dict(), 'val_loss': float(val_loss), 'val_auc': float(va_auc), 'history': history_clean}
            if holdout_data is not None:
                payload['holdout_data'] = holdout_data
            torch.save(payload, ckpt_full)
            torch.save(model.state_dict(), ckpt_weights)
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f"⏹️  Early stopping at epoch {epoch+1}")
            break
    if load_best_checkpoint:
        ckpt_full = ckpt_dir / f"{model_name}_best.pth"
        if ckpt_full.exists():
            try:
                ckpt = torch.load(ckpt_full, weights_only=False)
                state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
                model.load_state_dict(state)
                if isinstance(ckpt, dict) and 'history' in ckpt:
                    history = ckpt['history']
            except Exception:
                pass
    return {'model': model, 'history': history, 'best_val_loss': best_val_loss, 'best_val_auc': max(history['val_auc']) if history['val_auc'] else 0.0}


def clear_cuda_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("🧹 CUDA Memory cleared")


def set_random_seeds(seed: int):
    print(f"🎲 Setting random seeds to {seed}")
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _log_pos_neg_balance(df: pd.DataFrame, label_col: str = 'liked', user_col: str = 'did', context: str = "") -> None:
    try:
        if df is None or len(df) == 0 or label_col not in df.columns or user_col not in df.columns:
            print(f"   ℹ️  {context} balance: unavailable (missing data)")
            return
        total = int(len(df))
        labels = df[label_col]
        if labels.dtype == object:
            labels = labels.astype(int)
        pos = int(labels.sum())
        neg = int(total - pos)
        pos_rate = (pos / total) if total else 0.0
        print(f"   🔢 {context} overall: total={total}, pos={pos}, neg={neg}, pos_rate={pos_rate:.3f}")
    except Exception as _e:
        print(f"   ⚠️  {context} balance check failed: {_e}")


def _enforce_strict_5050_balance(df: pd.DataFrame, random_seed: int, label_col: str = 'liked', context: str = "") -> pd.DataFrame:
    if df is None or len(df) == 0:
        return df
    if label_col not in df.columns:
        raise ValueError(f"Missing '{label_col}' column in {context} dataframe")
    labels = df[label_col].astype(int)
    pos_idx = df.index[labels == 1]
    neg_idx = df.index[labels == 0]
    n_pos = int(len(pos_idx))
    n_neg = int(len(neg_idx))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"{context} has zero samples for a class (pos={n_pos}, neg={n_neg})")
    if n_pos == n_neg:
        return df.reset_index(drop=True)
    rng = np.random.RandomState(random_seed)
    if n_pos > n_neg:
        drop = rng.choice(pos_idx, size=n_pos - n_neg, replace=False)
    else:
        drop = rng.choice(neg_idx, size=n_neg - n_pos, replace=False)
    balanced = df.drop(index=drop).reset_index(drop=True)
    return balanced


def create_model(input_dim: int, hidden_dims: List[int], dropout_rate: float) -> EngagementPredictor:
    print(f"🤖 Creating model: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> 1")
    return EngagementPredictor(input_dim, hidden_dims, dropout_rate)


def save_test_results(results: Dict[str, Any], test_name: str = "model_results", logs_dir: Optional[Path] = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_logs_dir = Path(logs_dir) if logs_dir is not None else (Path(__file__).parent.parent.parent / 'outputs' / 'logs')
    base_logs_dir.mkdir(parents=True, exist_ok=True)
    results_file = base_logs_dir / f"{test_name}_{timestamp}.json"
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        try:
            return str(obj)
        except Exception:
            return repr(obj)
    with open(results_file, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)
    print(f"📄 Results saved to: {results_file}")
    return results_file


def run_training_pipeline(
    min_likes_per_user: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    hidden_dims: List[int],
    dropout_rate: float,
    prediction_posts_per_user: int,
    device: str,
    random_seed: int,
    save_model: bool = True,
    generate_plots: bool = True,
    embedding_bundle: Optional[str] = None,
    user_splits: Optional[str] = None,
    schema: str = 'auto',
    negatives_liked_only: bool = False,
    user_k: int = 3,
    min_cluster_size: int = 3,
    max_embedding_posts_per_user: int = 50,
    output_dir: Optional[Path] = None,
    disable_progress: bool = False,
) -> Dict[str, Any]:
    # This is the consolidated version adapted from src/train.py
    
    # Create output dirs FIRST so we can create a logger
    ROOT = Path(__file__).resolve().parents[2]
    DEFAULT_OUTPUTS_DIR = ROOT / "outputs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is not None:
        base_dir = Path(output_dir) / "train" / timestamp
    else:
        base_dir = DEFAULT_OUTPUTS_DIR / "train" / timestamp
    checkpoints_dir = base_dir / "checkpoints"
    plots_dir = base_dir / "plots"
    logs_dir = base_dir / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Initialize logger BEFORE any potentially slow operations
    logger = get_stage_logger('STAGE_05_TRAIN', log_file=base_dir / 'stage.log')
    log_operation_start('run_training_pipeline started', 'STAGE_05_TRAIN', logger)
    
    # Now do potentially slow operations with logging
    log_operation_start('Clear CUDA memory', 'STAGE_05_TRAIN', logger)
    clear_cuda_memory()
    log_operation_start('Set random seeds', 'STAGE_05_TRAIN', logger)
    set_random_seeds(random_seed)

    # Load bundle + splits (preferred path)
    if not (embedding_bundle and user_splits):
        raise RuntimeError("run_training_pipeline requires embedding_bundle and user_splits in stage context")
    
    log_operation_start('Load embedding bundle and user splits', 'STAGE_05_TRAIN', logger)
    import pickle, json as _json
    with open(embedding_bundle, 'rb') as f:
        bundle = pickle.load(f)
    posts_emb_df = bundle['posts_emb_df']
    likes_df = bundle['likes_df']
    join_like = str(bundle['join_like'])
    join_post = str(bundle['join_post'])
    embedding_dim = int(bundle['embedding_dim'])
    with open(user_splits, 'r') as f:
        splits = _json.load(f)
    train_users = list(map(str, splits.get('train_users', [])))
    val_users = list(map(str, splits.get('val_users', [])))
    holdout_users = list(map(str, splits.get('holdout_users', [])))
    likes_local = likes_df[likes_df['did'].isin(set(train_users) | set(val_users))].copy()
    available_posts = set(posts_emb_df[join_post].astype(str).unique())
    likes_local[join_like] = likes_local[join_like].astype(str)
    likes_local = likes_local[likes_local[join_like].isin(available_posts)]

    # Allocate per-user embedding vs prediction posts
    log_operation_start('Allocate per-user embedding vs prediction posts', 'STAGE_05_TRAIN', logger)
    
    t_alloc = time.time()
    grouped = list(likes_local.groupby('did'))
    num_users_to_process = len(grouped)
    logger.info(f"Allocating posts for {num_users_to_process} users...")
    
    embedding_likes_list: List[pd.DataFrame] = []
    prediction_likes_list: List[pd.DataFrame] = []
    for user_id, g in grouped:
        user_posts = sorted(list(set(g[join_like].astype(str).unique())))
        if len(user_posts) < max(min_likes_per_user, prediction_posts_per_user + 1):
            continue
        posts_for_prediction = set(user_posts[-int(prediction_posts_per_user):])
        posts_for_embedding = set(user_posts[:-int(prediction_posts_per_user)])
        if len(posts_for_embedding) > int(max_embedding_posts_per_user):
            posts_for_embedding = set(sorted(list(posts_for_embedding))[:int(max_embedding_posts_per_user)])
        if posts_for_embedding:
            embedding_likes_list.append(g[g[join_like].isin(posts_for_embedding)])
        if posts_for_prediction:
            prediction_likes_list.append(g[g[join_like].isin(posts_for_prediction)])
    
    embedding_likes_df = pd.concat(embedding_likes_list, ignore_index=True) if embedding_likes_list else pd.DataFrame()
    prediction_likes_df = pd.concat(prediction_likes_list, ignore_index=True) if prediction_likes_list else pd.DataFrame()
    
    logger.info(f"Allocated posts in {time.time()-t_alloc:.2f}s: {len(embedding_likes_df)} embedding likes, {len(prediction_likes_df)} prediction likes")

    # Build user features (multi_centroid by default)
    effective_schema = 'multi_centroid' if schema in ('auto', 'topic_mixture', 'multi_centroid') else 'mean'
    log_operation_start(f'Build user features (schema={effective_schema})', 'STAGE_05_TRAIN', logger)
    user_emb_df = build_user_feature_frame(
        schema=effective_schema,
        likes_df=embedding_likes_df,
        posts_emb_df=posts_emb_df,
        join_like=join_like,
        join_post=join_post,
        embedding_dim=embedding_dim,
        selected_users=list(set(train_users) | set(val_users)),
        feature_columns=None,
        random_seed=random_seed,
        topic_model=None,
        pca_model=None,
        user_k=user_k,
        min_cluster_size=min_cluster_size,
        max_embedding_posts_per_user=max_embedding_posts_per_user,
    )

    # Create prediction pairs
    log_operation_start('Create prediction pairs', 'STAGE_05_TRAIN', logger)
    if negatives_liked_only:
        text_emb_cols = [c for c in posts_emb_df.columns if c.startswith('post_emb_')]
        image_emb_cols = [c for c in posts_emb_df.columns if c.startswith('image_emb_')]
        post_emb_cols = text_emb_cols + image_emb_cols
        pos_df = prediction_likes_df.merge(posts_emb_df[[join_post] + post_emb_cols], left_on=join_like, right_on=join_post, how='inner')
        pos_df['liked'] = 1
        all_liked_posts = set(likes_local[join_like].unique())
        available_posts_with_embeddings = set(posts_emb_df[join_post].unique())
        liked_with_emb = all_liked_posts & available_posts_with_embeddings
        user_positive_posts = {u: set(pos_df[pos_df['did'] == u][join_post].unique()) for u in pos_df['did'].unique()}
        negative_pairs: List[Tuple[Any, Any]] = []
        rng = np.random.RandomState(random_seed)
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
        prediction_pairs_df = create_pairs_dataset(prediction_likes_df, posts_emb_df, join_like, join_post, neg_ratio=0.5, random_seed=random_seed, use_parallel=True)

    prediction_pairs_df = prediction_pairs_df.merge(user_emb_df, on='did', how='inner')
    user_cols = [c for c in user_emb_df.columns if c != 'did']
    post_cols = [c for c in posts_emb_df.columns if c.startswith('post_emb_') or c.startswith('image_emb_')]
    feature_cols = user_cols + post_cols
    train_df = prediction_pairs_df[prediction_pairs_df['did'].isin(train_users)].copy()
    val_df = prediction_pairs_df[prediction_pairs_df['did'].isin(val_users)].copy()
    
    log_operation_start('Enforce 50/50 class balance', 'STAGE_05_TRAIN', logger)
    train_df = _enforce_strict_5050_balance(train_df, label_col='liked', random_seed=random_seed, context='Train')
    val_df = _enforce_strict_5050_balance(val_df, label_col='liked', random_seed=random_seed + 1, context='Val')
    
    # Datasets
    log_operation_start('Create datasets and data loaders', 'STAGE_05_TRAIN', logger)
    train_dataset = EngagementDataset(
        train_df[feature_cols].values, 
        train_df['liked'].to_numpy(), 
        train_df['did'].tolist(), 
        train_df[join_post].tolist()
    )
    val_dataset = EngagementDataset(
        val_df[feature_cols].values, 
        val_df['liked'].to_numpy(), 
        val_df['did'].tolist(), 
        val_df[join_post].tolist()
    )
    input_dim = train_dataset.features.shape[1]
    model = create_model(input_dim, hidden_dims, dropout_rate)
    train_loader, val_loader, _ = create_data_loaders(train_dataset, val_dataset, batch_size, test_dataset=None)
    
    log_operation_start(f'Training model (epochs={epochs}, batch_size={batch_size})', 'STAGE_05_TRAIN', logger)
    print("\n🏋️  Training model...")
    training_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        load_best_checkpoint=True,
        holdout_data=None,
        checkpoints_dir=checkpoints_dir,
        disable_progress=disable_progress,
    )
    clear_cuda_memory()
    evaluation_metrics = {'note': 'Holdout evaluation happens in Stage 6'}
    # Plots
    if generate_plots:
        log_operation_start('Generate plots', 'STAGE_05_TRAIN', logger)
        from utils.helpers import plot_training_history, plot_model_performance
        hist = training_results['history']
        try:
            best_epoch = int(np.argmin(hist.get('val_loss', []))) + 1 if hist.get('val_loss') else None
        except Exception:
            best_epoch = None
        plot_training_history(hist, plots_dir / f"training_history_{timestamp}.png", best_epoch=best_epoch)
        # Train/Val performance plots
        try:
            from torch.utils.data import DataLoader as _DL
            model.eval()
            def _collect_predictions(ds):
                loader = _DL(ds, batch_size=batch_size, shuffle=False, drop_last=False)
                ys = []
                ps = []
                with torch.inference_mode():
                    for batch in loader:
                        feats = batch['features'].to(device)
                        labels = batch['label']
                        preds = model(feats).squeeze()
                        if preds.ndim == 0:
                            ps.append(float(preds.cpu().numpy()))
                            ys.append(float(labels.cpu().numpy()))
                        else:
                            ps.extend(preds.cpu().numpy().tolist())
                            ys.extend(labels.cpu().numpy().tolist())
                return np.asarray(ys, dtype=float), np.asarray(ps, dtype=float)
            y_train, p_train = _collect_predictions(train_dataset)
            plot_model_performance(y_train, p_train, plots_dir / f"train_model_performance_{timestamp}.png")
            y_val, p_val = _collect_predictions(val_dataset)
            plot_model_performance(y_val, p_val, plots_dir / f"val_model_performance_{timestamp}.png")
        except Exception:
            pass
    # Save model
    model_path = None
    if save_model:
        log_operation_start('Save model checkpoint', 'STAGE_05_TRAIN', logger)
        model_path = checkpoints_dir / f"engagement_model_{timestamp}.pth"
        # Sanitize training_results to avoid pickling the model object
        tr_sanitized = {k: v for k, v in training_results.items() if k != 'model'}
        payload = {
            'model_state_dict': model.state_dict(),
            'input_dim': input_dim,
            'hidden_dims': hidden_dims,
            'dropout_rate': dropout_rate,
            'training_results': tr_sanitized,
            'evaluation_metrics': evaluation_metrics,
            'training_parameters': {
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'epochs': epochs,
                'patience': patience,
            },
            'data_info': {
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'holdout_users': len(holdout_users),
                'feature_dim': input_dim,
            },
            'feature_columns': (user_cols, post_cols, feature_cols),
        }
        torch.save(payload, model_path)
        print(f"💾 Model saved to: {model_path}")
    duration = datetime.now() - datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    results = {
        'model': model,
        'training_results': training_results,
        'evaluation_metrics': evaluation_metrics,
        'model_path': str(model_path) if model_path else None,
        'training_duration': str(duration),
        'data_info': {
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'holdout_users': len(holdout_users),
            'feature_dim': input_dim,
        },
        'feature_columns': (user_cols, post_cols, feature_cols),
    }
    return results


def run(context, args) -> Dict[str, Any]:

    run_dir = Path(context.run_dir).resolve()
    
    # Create a temporary logger for the stage entry point
    from datetime import datetime
    temp_log_dir = run_dir / '05_train'
    temp_log_dir.mkdir(parents=True, exist_ok=True)
    entry_logger = get_stage_logger('STAGE_05_TRAIN', log_file=temp_log_dir / 'stage_entry.log')
    
    log_operation_start('Stage 5 entry point', 'STAGE_05_TRAIN', entry_logger)

    # Locate embedding bundle
    log_operation_start('Locate embedding bundle', 'STAGE_05_TRAIN', entry_logger)
    prior_featurize = select_prior_output(run_dir, '02_featurize', use_latest=context.use_latest, prior_path=context.prior_outputs.get('02_featurize'))
    if prior_featurize is None:
        raise FileNotFoundError("Featurize output not found.")
    bundle_candidates = sorted(prior_featurize.glob('embedding_bundle_*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not bundle_candidates:
        raise FileNotFoundError(f"No embedding_bundle_*.pkl found under {prior_featurize}")
    bundle_path = bundle_candidates[0]
    entry_logger.info(f"Found bundle: {bundle_path}")

    # Locate user_splits
    log_operation_start('Locate user splits', 'STAGE_05_TRAIN', entry_logger)
    prior_split = select_prior_output(run_dir, '04_split', use_latest=context.use_latest, prior_path=context.prior_outputs.get('04_split'))
    if prior_split is None:
        raise FileNotFoundError("Split output not found.")
    splits_path = (prior_split / 'user_splits.json')
    if not splits_path.exists():
        raise FileNotFoundError(f"user_splits.json not found under {prior_split}")
    entry_logger.info(f"Found splits: {splits_path}")

    # Collect training knobs from args (fallback to defaults handled by run_training_pipeline)
    log_operation_start('Call run_training_pipeline', 'STAGE_05_TRAIN', entry_logger)
    t0 = time.time()
    results = run_training_pipeline(
        min_likes_per_user=int(args.min_likes_per_user),
        embedding_bundle=str(bundle_path.resolve()),
        user_splits=str(splits_path.resolve()),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay_mlp),
        epochs=int(args.epochs),
        patience=int(args.patience),
        hidden_dims=args.hidden_dims,
        dropout_rate=float(args.dropout_rate_mlp),
        device=str(args.device),
        random_seed=int(args.random_seed),
        save_model=not bool(args.no_save_model),
        generate_plots=not bool(args.no_plots),
        output_dir=run_dir,  # training module will create train/<ts>/ under this; we'll relocate to 05_train
        disable_progress=bool(getattr(args, 'disable_progress', False)),
        prediction_posts_per_user=int(args.prediction_posts_per_user)
    )

    model_path = results.get('model_path')
    # Prefer in-memory trained model if provided by the pipeline
    trained_model = results.get('model_obj') or results.get('model')
    if trained_model is not None:
        try:
            trained_model.to(str(args.device))
            trained_model.eval()
        except Exception:
            pass
    # Infer training subdir from model path
    training_dir = None
    if model_path:
        mp = Path(model_path).resolve()
        if mp.parent.name == 'checkpoints':
            training_dir = mp.parent.parent

    # Relocate train/<ts> → 05_train/<ts>
    final_train_dir = None
    if training_dir is not None and training_dir.parent.name == 'train':
        dest_root = training_dir.parent.parent / '05_train'
        dest_root.mkdir(parents=True, exist_ok=True)
        dest = dest_root / training_dir.name
        if not dest.exists():
            try:
                training_dir.rename(dest)
            except Exception:
                # fallback: copy tree
                import shutil
                shutil.copytree(training_dir, dest, dirs_exist_ok=True)
            # best-effort cleanup
            try:
                if not any(training_dir.iterdir()):
                    training_dir.rmdir()
            except Exception:
                pass
        final_train_dir = dest
    else:
        final_train_dir = training_dir

    # Stage info
    # At beginning, dataset sizes of bundle and splits
    import json as _json
    with open(splits_path, 'r') as _f:
        _spl = _json.load(_f)
    info_lines = [
        f"stage: train",
        f"runtime_seconds: {time.time()-t0:.2f}",
        f"settings: batch_size={args.batch_size}, lr={args.learning_rate}, epochs={args.epochs}",
        f"inputs: embedding_bundle, user_splits",
        f"N_train_users: {len(_spl.get('train_users',[]))}",
        f"N_val_users: {len(_spl.get('val_users',[]))}",
        f"N_holdout_users: {len(_spl.get('holdout_users',[]))}",
    ]
    if final_train_dir is not None:
        (final_train_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    # Automatic held-out evaluation (save predictions for Stage 6 heterogeneity)
    try:
        if final_train_dir is not None and model_path:
            # Load bundle
            import pickle
            with open(bundle_path, 'rb') as f:
                bundle = pickle.load(f)
            posts_emb_df: pd.DataFrame = bundle['posts_emb_df']
            likes_df: pd.DataFrame = bundle['likes_df']
            join_like: str = str(bundle['join_like'])
            join_post: str = str(bundle['join_post'])
            embedding_dim: int = int(bundle.get('embedding_dim', 0))

            holdout_users = list(map(str, _spl.get('holdout_users', [])))
            if holdout_users:
                # Try to read training feature_columns
                cfg = None
                cfg_candidates = sorted(final_train_dir.glob('training_config*.json'))
                if cfg_candidates:
                    with open(cfg_candidates[-1], 'r') as cf:
                        cfg = json.load(cf)
                feature_columns = None
                if cfg and 'feature_columns' in cfg:
                    feature_columns = cfg['feature_columns']
                elif isinstance(results, dict) and 'feature_columns' in results:
                    feature_columns = results['feature_columns']

                # Build holdout pairs (simple balanced pairs from holdout likes)
                likes_hou = likes_df[likes_df['did'].isin(holdout_users)].copy()
                available_posts = set(posts_emb_df[join_post].astype(str).unique())
                likes_hou[join_like] = likes_hou[join_like].astype(str)
                likes_hou = likes_hou[likes_hou[join_like].isin(available_posts)]

                # Simple allocation: last like per user for prediction, rest for embedding
                emb_list = []
                pred_list = []
                for user_id, g in likes_hou.groupby('did'):
                    user_posts = sorted(list(set(g[join_like].astype(str).tolist())))
                    if len(user_posts) < 2:
                        continue
                    posts_for_prediction = set([user_posts[-1]])
                    posts_for_embedding = set(user_posts[:-1])
                    if posts_for_embedding:
                        emb_list.append(g[g[join_like].isin(posts_for_embedding)])
                    pred_list.append(g[g[join_like].isin(posts_for_prediction)])
                embedding_likes_df = pd.concat(emb_list, ignore_index=True) if emb_list else pd.DataFrame()
                prediction_likes_df = pd.concat(pred_list, ignore_index=True) if pred_list else pd.DataFrame()

                # Build user features matching training config if available
                if feature_columns is None:
                    user_emb_cols, post_emb_cols, all_cols = get_actual_feature_columns(posts_emb_df)
                    feature_columns = [user_emb_cols, post_emb_cols, all_cols]

                user_emb_df = build_user_feature_frame(
                    schema='multi_centroid' if any(c.startswith('user_k') for c in feature_columns[0]) else ('topic_mixture' if any(c.startswith('user_topic_') for c in feature_columns[0]) else 'mean'),
                    likes_df=embedding_likes_df if len(embedding_likes_df) else likes_hou,
                    posts_emb_df=posts_emb_df,
                    join_like=join_like,
                    join_post=join_post,
                    embedding_dim=int(embedding_dim) if embedding_dim else len([c for c in posts_emb_df.columns if c.startswith('post_emb_')]),
                    selected_users=list(set(prediction_likes_df['did'].astype(str).unique())) if len(prediction_likes_df) else holdout_users,
                    feature_columns=feature_columns,
                    random_seed=int(args.random_seed),
                )

                print(f"[TRAIN] user_emb_df shape: {user_emb_df.shape}")
                # Build prediction pairs for holdout users
                text_emb_cols = [c for c in posts_emb_df.columns if c.startswith('post_emb_')]
                image_emb_cols = [c for c in posts_emb_df.columns if c.startswith('image_emb_')]
                post_emb_cols = text_emb_cols + image_emb_cols
                prediction_pairs_df = pd.DataFrame()
                if len(prediction_likes_df):
                    try:
                        prediction_pairs_df = create_pairs_dataset(
                            prediction_likes_df, posts_emb_df, join_like, join_post, neg_ratio=0.5,
                            random_seed=int(args.random_seed), use_parallel=True
                        )
                    except Exception:
                        # Fallback to positive-only merge if pair creation fails
                        pos_df = prediction_likes_df.merge(
                            posts_emb_df[[join_post] + post_emb_cols], left_on=join_like, right_on=join_post, how='inner'
                        )
                        if len(pos_df):
                            pos_df['liked'] = 1
                            prediction_pairs_df = pos_df
                # Attach user features (if any pairs exist)
                if len(prediction_pairs_df):
                    prediction_pairs_df = prediction_pairs_df.merge(user_emb_df, on='did', how='inner')
                else:
                    # No pairs to evaluate; skip silently
                    pass

                # Inference
                device = str(args.device)
                model = trained_model
                user_cols = feature_columns[0]
                post_cols = feature_columns[1]
                def _coerce_numeric(df: pd.DataFrame, cols):
                    return df[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0)
                X = np.concatenate([
                    _coerce_numeric(prediction_pairs_df, user_cols).values,
                    _coerce_numeric(prediction_pairs_df, post_cols).values
                ], axis=1) if len(prediction_pairs_df) else np.zeros((0, len(user_cols)+len(post_cols)), dtype=np.float32)
                preds_list = []
                bs = 8192
                model.eval()
                if X.shape[0] > 0:
                    with torch.no_grad():
                        for start in range(0, X.shape[0], bs):
                            end = min(start + bs, X.shape[0])
                            xb = torch.as_tensor(X[start:end], dtype=torch.float32, device=str(args.device))
                            pb = model(xb).squeeze().detach().cpu().numpy()
                            preds_list.append(pb)
                y_pred = np.concatenate(preds_list, axis=0) if preds_list else np.array([])
                y_true = prediction_pairs_df['liked'].astype(np.int8).values if len(prediction_pairs_df) else np.array([])

                # Persist predictions
                he_dir = final_train_dir / 'holdout_eval'
                (he_dir / 'plots').mkdir(parents=True, exist_ok=True)
                pred_out = he_dir / 'predictions.parquet'
                out_df = pd.DataFrame({
                    'did': prediction_pairs_df['did'].astype(str).values if len(prediction_pairs_df) else [],
                    'post_id': prediction_pairs_df[join_post].astype(str).values if len(prediction_pairs_df) else [],
                    'y_true': y_true,
                    'y_pred_proba': y_pred,
                })
                try:
                    out_df.to_parquet(pred_out, index=False)
                    print(f"Saved heldout predictions: {pred_out.resolve()} rows={len(out_df)}")
                except Exception:
                    csv_out = he_dir / 'predictions.csv'
                    out_df.to_csv(csv_out, index=False)
                    print(f"Saved heldout predictions: {csv_out.resolve()} rows={len(out_df)}")

                # Metrics
                metrics_overall: Dict[str, Any] = {
                    'total_samples': int(len(out_df)),
                    'positive': int(int(out_df['y_true'].sum()) if len(out_df) else 0),
                }
                try:
                    from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve
                    if len(out_df) and len(set(out_df['y_true'])) > 1:
                        metrics_overall['auc_roc'] = float(roc_auc_score(out_df['y_true'], out_df['y_pred_proba']))
                    if len(out_df):
                        metrics_overall['accuracy@0.5'] = float(accuracy_score(out_df['y_true'], (out_df['y_pred_proba'] > 0.5).astype(int)))
                except Exception as e:
                    pass
                with open(he_dir / 'metrics_overall.json', 'w') as mf:
                    json.dump(metrics_overall, mf, indent=2)
                print(f"Saved heldout metrics: {str((he_dir / 'metrics_overall.json').resolve())}")

                # Per-user metrics (accuracy; AUC where feasible)
                rows = []
                try:
                    from sklearn.metrics import roc_auc_score, accuracy_score
                    for uid, g in out_df.groupby('did'):
                        acc = float(accuracy_score(g['y_true'], (g['y_pred_proba'] > 0.5).astype(int))) if len(g) else float('nan')
                        auc = float(roc_auc_score(g['y_true'], g['y_pred_proba'])) if len(set(g['y_true'])) > 1 else float('nan')
                        rows.append({'did': uid, 'num_samples': int(len(g)), 'accuracy': acc, 'auc_roc': auc})
                except Exception:
                    for uid, g in out_df.groupby('did'):
                        acc = float(((g['y_pred_proba'] > 0.5).astype(int) == g['y_true']).mean()) if len(g) else float('nan')
                        rows.append({'did': uid, 'num_samples': int(len(g)), 'accuracy': acc})
                pd.DataFrame(rows).to_csv(he_dir / 'metrics_per_user.csv', index=False)
                print(f"Saved heldout per-user metrics: {str((he_dir / 'metrics_per_user.csv').resolve())}")

                # Holdout performance plot under standard plots dir
                plots_dir = final_train_dir / 'plots'
                plots_dir.mkdir(parents=True, exist_ok=True)
                try:
                    ts = final_train_dir.name
                    plot_path = plots_dir / f"holdout_model_performance_{ts}.png"
                    if len(out_df) and len(set(out_df['y_true'])) > 1:
                        plot_model_performance(out_df['y_true'].to_numpy(), out_df['y_pred_proba'].to_numpy(), plot_path)
                        print(f"Saved heldout plot: {plot_path.resolve()}")
                    else:
                        print("Skipped heldout plot (insufficient class diversity or no data).")
                except Exception as e:
                    pass

                # Append holdout info to stage_info
                try:
                    si = final_train_dir / 'stage_info.txt'
                    with open(si, 'a') as fh:
                        fh.write(f"holdout_total_samples: {len(out_df)}\n")
                        fh.write(f"holdout_predictions_path: {str((he_dir / 'predictions.parquet').resolve()) if (he_dir / 'predictions.parquet').exists() else str((he_dir / 'predictions.csv').resolve())}\n")
                        fh.write(f"holdout_metrics_overall_path: {str((he_dir / 'metrics_overall.json').resolve())}\n")
                        fh.write(f"holdout_metrics_per_user_path: {str((he_dir / 'metrics_per_user.csv').resolve())}\n")
                except Exception:
                    pass
    except Exception:
        # Non-fatal; held-out evaluation best-effort
        pass

    return {
        'output_dir': final_train_dir if final_train_dir is not None else (run_dir / '05_train'),
        'artifacts': {
            'model_path': str(model_path) if model_path else None,
            'embedding_bundle_path': str(bundle_path.resolve()),
            'user_splits_path': str(splits_path.resolve()),
        }
    }
