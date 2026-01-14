#!/usr/bin/env python3

"""
Stage 5 (Alternative): Two-Tower Engagement Prediction Model

This module implements a two-tower architecture for engagement prediction as an
alternative to the MLP-based stage_train.py. It features:

- User Tower: Encodes user preferences from their liked post history via self-attention
- Post Tower: Projects post embeddings to shared space
- Training: BCE loss with explicit positive/negative pairs

Inputs:
- embedding_bundle_*.pkl from Stage 2
- user_splits.json from Stage 4

Outputs:
- <run_dir>/05_train/<timestamp>/{checkpoints,plots,logs,training_config.json}
- <run_dir>/05_train/<timestamp>/holdout_eval/{predictions.parquet,metrics_overall.json,...}
"""

from __future__ import annotations

import json
import math
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.pipeline.core import select_prior_output
from utils.helpers import get_stage_logger, log_operation_start, get_device


# =============================================================================
# User History Encoder
# =============================================================================

class UserHistoryEncoder(nn.Module):
    """
    Encodes a variable-length sequence of liked post embeddings into a fixed user representation.
    
    Architecture:
    1. Project each post embedding to internal dimension
    2. Add learnable positional encodings (recency-aware, flipped so recent = high weight)
    3. Apply self-attention layers to capture interest patterns
    4. Aggregate via attention-weighted pooling + mean pooling (dual representation)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_attention_heads: int = 4,
        num_attention_layers: int = 2,
        max_seq_len: int = 50,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        
        # Project input embeddings to hidden dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        
        # Learnable positional embeddings (will be flipped for recency weighting)
        self.positional_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # Self-attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_attention_layers,
        )
        
        # Attention pooling query (learnable)
        self.attention_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Output projection (from dual pooling: attention + mean)
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
    
    def forward(
        self,
        history_embeddings: torch.Tensor,  # [B, seq_len, input_dim]
        history_mask: Optional[torch.Tensor] = None,  # [B, seq_len] True = valid
    ) -> torch.Tensor:
        """
        Args:
            history_embeddings: Batch of user history embeddings [B, seq_len, input_dim]
            history_mask: Boolean mask where True indicates valid positions [B, seq_len]
        
        Returns:
            User embeddings [B, output_dim]
        """
        B, seq_len, _ = history_embeddings.shape
        device = history_embeddings.device
        
        # Create default mask if not provided
        if history_mask is None:
            history_mask = torch.ones(B, seq_len, dtype=torch.bool, device=device)
        
        # Project inputs
        x = self.input_projection(history_embeddings)  # [B, seq_len, hidden_dim]
        
        # Add positional embeddings (flipped for recency: position 0 = most recent)
        positions = torch.arange(seq_len, device=device)
        # Flip positions so most recent (idx 0) gets highest position value
        positions = (self.max_seq_len - 1) - positions.clamp(max=self.max_seq_len - 1)
        pos_emb = self.positional_embedding(positions)  # [seq_len, hidden_dim]
        x = x + pos_emb.unsqueeze(0)  # [B, seq_len, hidden_dim]
        
        # Create attention mask for transformer (True = ignore)
        # PyTorch transformer uses inverted mask convention
        attn_mask = ~history_mask  # [B, seq_len]
        
        # Apply self-attention
        x = self.transformer_encoder(x, src_key_padding_mask=attn_mask)  # [B, seq_len, hidden_dim]
        
        # Attention-weighted pooling
        # Expand query for batch
        query = self.attention_query.expand(B, -1, -1)  # [B, 1, hidden_dim]
        attn_scores = torch.bmm(query, x.transpose(1, 2))  # [B, 1, seq_len]
        
        # Mask invalid positions with large negative value
        attn_scores = attn_scores.masked_fill(
            attn_mask.unsqueeze(1),  # [B, 1, seq_len]
            float('-inf')
        )
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 1, seq_len]
        
        # Handle all-masked case (replace NaN with uniform)
        attn_weights = torch.nan_to_num(attn_weights, nan=1.0 / max(seq_len, 1))
        
        attention_pooled = torch.bmm(attn_weights, x).squeeze(1)  # [B, hidden_dim]
        
        # Mean pooling (masked)
        mask_expanded = history_mask.unsqueeze(-1).float()  # [B, seq_len, 1]
        masked_x = x * mask_expanded
        sum_x = masked_x.sum(dim=1)  # [B, hidden_dim]
        count = mask_expanded.sum(dim=1).clamp(min=1)  # [B, 1]
        mean_pooled = sum_x / count  # [B, hidden_dim]
        
        # Combine dual representations
        combined = torch.cat([attention_pooled, mean_pooled], dim=-1)  # [B, hidden_dim * 2]
        
        # Project to output dimension
        output = self.output_projection(combined)  # [B, output_dim]
        
        return output


# =============================================================================
# Post Tower
# =============================================================================

class PostTower(nn.Module):
    """
    Projects post embeddings (text + optional image) to shared embedding space.
    
    Simple but effective 2-layer MLP with LayerNorm and dropout.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, post_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            post_embeddings: [B, input_dim] concatenated text + image embeddings
        
        Returns:
            Projected embeddings [B, output_dim]
        """
        return self.network(post_embeddings)


# =============================================================================
# Two-Tower Engagement Model
# =============================================================================

class TwoTowerEngagement(nn.Module):
    """
    Two-tower model for engagement prediction.
    
    User tower: Encodes user preferences from liked post history
    Post tower: Projects post embeddings to shared space
    
    Training: BCE loss with explicit positive/negative pairs.
    """
    
    def __init__(
        self,
        post_embedding_dim: int,
        shared_dim: int = 128,
        user_hidden_dim: int = 256,
        post_hidden_dim: int = 256,
        num_attention_heads: int = 4,
        num_attention_layers: int = 2,
        max_history_len: int = 50,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.post_embedding_dim = post_embedding_dim
        
        # User tower (history encoder)
        self.user_tower = UserHistoryEncoder(
            input_dim=post_embedding_dim,  # User history uses same post embeddings
            hidden_dim=user_hidden_dim,
            output_dim=shared_dim,
            num_attention_heads=num_attention_heads,
            num_attention_layers=num_attention_layers,
            max_seq_len=max_history_len,
            dropout_rate=dropout_rate,
        )
        
        # Post tower
        self.post_tower = PostTower(
            input_dim=post_embedding_dim,
            hidden_dim=post_hidden_dim,
            output_dim=shared_dim,
            dropout_rate=dropout_rate,
        )
    
    def encode_user(
        self,
        history_embeddings: torch.Tensor,
        history_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode user from their liked post history."""
        return self.user_tower(history_embeddings, history_mask)
    
    def encode_post(self, post_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode post to shared space."""
        return self.post_tower(post_embeddings)
    
    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: Optional[torch.Tensor],
        post_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute engagement scores for user-post pairs.
        
        Returns:
            Scores [B] (dot product similarity)
        """
        user_emb = self.encode_user(history_embeddings, history_mask)  # [B, D]
        post_emb = self.encode_post(post_embeddings)  # [B, D]
        
        # Dot product scores
        scores = (user_emb * post_emb).sum(dim=-1)  # [B]
        return scores
    
    def train_forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: Optional[torch.Tensor],
        post_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass with BCE loss.
        
        Returns:
            (loss, scores)
        """
        user_emb = self.encode_user(history_embeddings, history_mask)
        post_emb = self.encode_post(post_embeddings)
        
        # Dot product scores
        scores = (user_emb * post_emb).sum(dim=-1)
        probs = torch.sigmoid(scores)
        
        # BCE loss
        loss = F.binary_cross_entropy(probs, labels.float())
        
        return loss, scores


# =============================================================================
# Dataset
# =============================================================================

class UserHistoryItem:
    """Represents a user's history and a target post for training."""
    
    def __init__(
        self,
        user_id: str,
        history_embeddings: np.ndarray,  # [seq_len, D]
        history_mask: np.ndarray,  # [seq_len] bool
        target_post_embedding: np.ndarray,  # [D]
        target_post_id: str,
        label: int,  # 1 = liked, 0 = not liked
    ):
        self.user_id = user_id
        self.history_embeddings = history_embeddings
        self.history_mask = history_mask
        self.target_post_embedding = target_post_embedding
        self.target_post_id = target_post_id
        self.label = label


class TwoTowerDataset(Dataset):
    """Dataset for two-tower training with user history sequences."""
    
    def __init__(self, items: List[UserHistoryItem]):
        self.items = items
    
    def __len__(self) -> int:
        return len(self.items)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.items[idx]
        return {
            'history_embeddings': torch.FloatTensor(item.history_embeddings),
            'history_mask': torch.BoolTensor(item.history_mask),
            'target_post_embedding': torch.FloatTensor(item.target_post_embedding),
            'label': torch.FloatTensor([item.label]),
            'user_id': item.user_id,
            'post_id': item.target_post_id,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """Collate batch of variable-length histories."""
    return {
        'history_embeddings': torch.stack([b['history_embeddings'] for b in batch]),
        'history_mask': torch.stack([b['history_mask'] for b in batch]),
        'target_post_embedding': torch.stack([b['target_post_embedding'] for b in batch]),
        'label': torch.cat([b['label'] for b in batch]),
        'user_ids': [b['user_id'] for b in batch],
        'post_ids': [b['post_id'] for b in batch],
    }


# =============================================================================
# Data Preparation
# =============================================================================

def _process_single_user_history(
    user_id: str,
    user_likes_df: pd.DataFrame,
    post_emb_lookup: Dict,
    emb_cols: List[str],
    join_like: str,
    max_history_len: int,
    prediction_posts_per_user: int,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Process a single user's history (for parallel execution)."""
    liked_posts = user_likes_df[join_like].unique().tolist()
    
    # Need at least prediction_posts_per_user + 1 posts
    if len(liked_posts) < prediction_posts_per_user + 1:
        return None
    
    # Split: last N for prediction, rest for history
    prediction_posts = liked_posts[-prediction_posts_per_user:]
    history_posts = liked_posts[:-prediction_posts_per_user]
    
    # Cap history length
    if len(history_posts) > max_history_len:
        history_posts = history_posts[-max_history_len:]  # Keep most recent
    
    # Get embeddings for history posts
    history_embeddings = []
    valid_history_posts = []
    for pid in history_posts:
        if pid in post_emb_lookup:
            emb = np.array([post_emb_lookup[pid][c] for c in emb_cols], dtype=np.float32)
            history_embeddings.append(emb)
            valid_history_posts.append(pid)
    
    if len(history_embeddings) == 0:
        return None
    
    return (user_id, {
        'history_embeddings': np.stack(history_embeddings),  # [seq_len, D]
        'history_post_ids': valid_history_posts,
        'prediction_post_ids': prediction_posts,
    })


def build_user_history_sequences(
    likes_df: pd.DataFrame,
    posts_emb_df: pd.DataFrame,
    join_like: str,
    join_post: str,
    user_ids: List[str],
    max_history_len: int = 20,
    prediction_posts_per_user: int = 1,
    n_jobs: int = -1,
) -> Dict[str, Dict[str, Any]]:
    """
    Build user history sequences from likes data (parallelized).
    
    For each user:
    - Reserve the last `prediction_posts_per_user` likes for prediction targets
    - Use the remaining likes (up to max_history_len) as history
    
    Args:
        n_jobs: Number of parallel jobs. -1 = use all CPUs, 1 = sequential
    
    Returns:
        Dict mapping user_id -> {
            'history_embeddings': np.ndarray [seq_len, D],
            'history_post_ids': List[str],
            'prediction_post_ids': List[str],
        }
    """
    import time
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    t0 = time.time()
    print(f"Building history sequences for {len(user_ids)} users...")
    
    # Get post embedding columns
    emb_cols = [c for c in posts_emb_df.columns if c.startswith('post_emb_') or c.startswith('image_emb_')]
    embedding_dim = len(emb_cols)
    
    # Create lookup for post embeddings
    posts_emb_df = posts_emb_df.copy()
    posts_emb_df[join_post] = posts_emb_df[join_post].astype(str)
    post_emb_lookup = posts_emb_df.set_index(join_post)[emb_cols].to_dict('index')
    available_posts = set(posts_emb_df[join_post].unique())
    
    # Filter likes to available posts and target users
    likes_local = likes_df[likes_df['did'].isin(user_ids)].copy()
    likes_local[join_like] = likes_local[join_like].astype(str)
    likes_local = likes_local[likes_local[join_like].isin(available_posts)]
    
    print(f"Processing {len(likes_local)} likes from target users...")
    
    # Group by user
    grouped = list(likes_local.groupby('did'))
    num_users = len(grouped)
    
    # Determine number of workers
    if n_jobs == -1:
        n_workers = os.cpu_count() or 4
    elif n_jobs == 1:
        n_workers = 1
    else:
        n_workers = max(1, min(n_jobs, os.cpu_count() or 4))
    
    # Use sequential processing for small datasets
    if num_users < 100 or n_workers == 1:
        print(f"Processing {num_users} users sequentially...")
        user_histories = {}
        for user_id, user_likes in grouped:
            result = _process_single_user_history(
                str(user_id), user_likes, post_emb_lookup, emb_cols,
                join_like, max_history_len, prediction_posts_per_user
            )
            if result:
                user_histories[result[0]] = result[1]
    else:
        # Parallel processing
        print(f"Processing {num_users} users with {n_workers} workers...")
        user_histories = {}
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for user_id, user_likes in grouped:
                future = executor.submit(
                    _process_single_user_history,
                    str(user_id), user_likes, post_emb_lookup, emb_cols,
                    join_like, max_history_len, prediction_posts_per_user
                )
                futures.append(future)
            
            # Collect results with progress
            from tqdm import tqdm
            for future in tqdm(as_completed(futures), total=len(futures), desc="Building histories"):
                result = future.result()
                if result:
                    user_histories[result[0]] = result[1]
    
    elapsed = time.time() - t0
    rate = len(user_histories) / elapsed if elapsed > 0 else 0
    print(f"Built {len(user_histories)} user histories in {elapsed:.2f}s ({rate:.1f} users/sec)")
    return user_histories


def _create_items_for_single_user(
    user_id: str,
    user_data: Dict[str, Any],
    post_emb_lookup: Dict,
    negative_candidate_pool: List[str],
    emb_cols: List[str],
    embedding_dim: int,
    max_history_len: int,
    neg_ratio: float,
    user_seed: int,
) -> List[UserHistoryItem]:
    """Create training items for a single user using pre-sampled negative pool."""
    rng = np.random.RandomState(user_seed)
    items = []
    
    history_emb = user_data['history_embeddings']
    history_len = len(history_emb)
    
    # Pad history to max_history_len
    if history_len < max_history_len:
        padding = np.zeros((max_history_len - history_len, embedding_dim), dtype=np.float32)
        padded_history = np.concatenate([history_emb, padding], axis=0)
        mask = np.concatenate([np.ones(history_len, dtype=bool), np.zeros(max_history_len - history_len, dtype=bool)])
    else:
        padded_history = history_emb[:max_history_len]
        mask = np.ones(max_history_len, dtype=bool)
    
    # Create positive items (prediction targets the user actually liked)
    for target_pid in user_data['prediction_post_ids']:
        if target_pid not in post_emb_lookup:
            continue
        
        target_emb = np.array([post_emb_lookup[target_pid][c] for c in emb_cols], dtype=np.float32)
        
        items.append(UserHistoryItem(
            user_id=user_id,
            history_embeddings=padded_history,
            history_mask=mask,
            target_post_embedding=target_emb,
            target_post_id=target_pid,
            label=1,
        ))
    
    # Create negative items - OPTIMIZED: Use global pre-sampled pool
    user_liked = set(user_data['history_post_ids'] + user_data['prediction_post_ids'])
    n_negatives = int(len(user_data['prediction_post_ids']) * neg_ratio)
    
    if n_negatives > 0:
        # Filter global pool to exclude user's liked posts (fast set operation)
        available_negatives = [p for p in negative_candidate_pool if p not in user_liked]
        
        # Sample what we need from the filtered pool
        if len(available_negatives) > 0:
            neg_samples = rng.choice(
                available_negatives,
                size=min(n_negatives, len(available_negatives)),
                replace=False
            )
            
            for neg_pid in neg_samples:
                neg_emb = np.array([post_emb_lookup[neg_pid][c] for c in emb_cols], dtype=np.float32)
                
                items.append(UserHistoryItem(
                    user_id=user_id,
                    history_embeddings=padded_history,
                    history_mask=mask,
                    target_post_embedding=neg_emb,
                    target_post_id=neg_pid,
                    label=0,
                ))
    
    return items


def create_training_items(
    user_histories: Dict[str, Dict[str, Any]],
    posts_emb_df: pd.DataFrame,
    join_post: str,
    max_history_len: int = 20,
    neg_ratio: float = 1.0,
    random_seed: int = 42,
    n_jobs: int = -1,
    max_negative_candidates: int = 10000,
) -> List[UserHistoryItem]:
    """
    Create training items (user history + target post pairs) - parallelized.
    
    Creates both positive and negative pairs for BCE loss training.
    
    Args:
        n_jobs: Number of parallel jobs. -1 = use all CPUs, 1 = sequential
        max_negative_candidates: Max posts to sample for negatives per user (reduces compute cost)
    """
    import time
    import os
    
    t0 = time.time()
    print(f"Creating training items for {len(user_histories)} users...")
    
    # Get embedding columns and lookup
    emb_cols = [c for c in posts_emb_df.columns if c.startswith('post_emb_') or c.startswith('image_emb_')]
    embedding_dim = len(emb_cols)
    
    posts_emb_df = posts_emb_df.copy()
    posts_emb_df[join_post] = posts_emb_df[join_post].astype(str)
    post_emb_lookup = posts_emb_df.set_index(join_post)[emb_cols].to_dict('index')
    all_post_ids_list = list(post_emb_lookup.keys())
    
    print(f"Total posts available: {len(all_post_ids_list)}, embedding dim: {embedding_dim}")
    
    # GLOBAL pre-sampling: Do this ONCE for all users!
    negative_candidate_pool = all_post_ids_list
    if len(all_post_ids_list) > max_negative_candidates:
        speedup = len(all_post_ids_list) // max_negative_candidates
        print(f"✨ Global pre-sampling optimization:")
        print(f"   Creating ONE shared pool of {max_negative_candidates:,} candidates (from {len(all_post_ids_list):,} total posts)")
        print(f"   All {len(user_histories)} users will sample negatives from this shared pool")
        print(f"   Expected speedup: ~{speedup}x faster!")
        
        # Do the ONE expensive sampling operation
        rng_global = np.random.RandomState(random_seed)
        negative_candidate_pool = rng_global.choice(
            all_post_ids_list, 
            size=max_negative_candidates, 
            replace=False
        ).tolist()
        print(f"   ✓ Global pool created with {len(negative_candidate_pool):,} posts")
    
    # Convert to list for indexing
    user_items = list(user_histories.items())
    num_users = len(user_items)
    
    # Sequential processing (now very fast with global pre-sampled pool!)
    print(f"Processing {num_users} users with shared negative pool...")
    from tqdm import tqdm
    items = []
    for idx, (user_id, user_data) in enumerate(tqdm(user_items, desc="Creating pairs")):
        user_seed = random_seed + idx
        user_items_list = _create_items_for_single_user(
            user_id, user_data, post_emb_lookup, negative_candidate_pool,
            emb_cols, embedding_dim, max_history_len, neg_ratio, user_seed
        )
        items.extend(user_items_list)
    
    elapsed = time.time() - t0
    rate = len(items) / elapsed if elapsed > 0 else 0
    print(f"Created {len(items)} training items in {elapsed:.2f}s ({rate:.1f} items/sec)")
    return items


# =============================================================================
# Training Loop
# =============================================================================

def train_two_tower_model(
    model: TwoTowerEngagement,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    patience: int = 20,
    checkpoints_dir: Optional[Path] = None,
    disable_progress: bool = False,
) -> Dict[str, Any]:
    """Train the two-tower model with early stopping."""
    from tqdm import tqdm
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_auc': [],
        'val_auc': [],
    }
    
    best_val_auc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    best_state_dict = None
    
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        def roc_auc_score(y_true, y_score):
            return 0.5
    
    for epoch in tqdm(range(epochs), desc="Training epochs", disable=disable_progress):
        # Training
        model.train()
        train_losses = []
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc="Training", leave=False, disable=disable_progress):
            history_emb = batch['history_embeddings'].to(device)
            history_mask = batch['history_mask'].to(device)
            target_emb = batch['target_post_embedding'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            loss, scores = model.train_forward(
                history_emb, history_mask, target_emb, labels
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            train_preds.extend(torch.sigmoid(scores).detach().cpu().numpy().tolist())
            train_labels.extend(labels.detach().cpu().numpy().tolist())
        
        # Validation
        model.eval()
        val_losses = []
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False, disable=disable_progress):
                history_emb = batch['history_embeddings'].to(device)
                history_mask = batch['history_mask'].to(device)
                target_emb = batch['target_post_embedding'].to(device)
                labels = batch['label'].to(device)
                
                loss, scores = model.train_forward(
                    history_emb, history_mask, target_emb, labels
                )
                
                val_losses.append(loss.item())
                val_preds.extend(torch.sigmoid(scores).detach().cpu().numpy().tolist())
                val_labels.extend(labels.detach().cpu().numpy().tolist())
        
        # Compute metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        
        train_auc = roc_auc_score(train_labels, train_preds) if len(set(train_labels)) > 1 else 0.5
        val_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.5
        
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['train_auc'].append(float(train_auc))
        history['val_auc'].append(float(val_auc))
        
        scheduler.step(val_auc)
        
        # Check for improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            
            # Save checkpoint
            if checkpoints_dir is not None:
                ckpt_path = checkpoints_dir / "two_tower_best.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_state_dict,
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                    'history': history,
                }, ckpt_path)
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    
    return {
        'model': model,
        'history': history,
        'best_val_loss': best_val_loss,
        'best_val_auc': best_val_auc,
    }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_model(
    model: TwoTowerEngagement,
    data_loader: DataLoader,
    device: str,
) -> Dict[str, Any]:
    """Evaluate model and return metrics."""
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_curve, average_precision_score
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    all_user_ids = []
    all_post_ids = []
    
    with torch.no_grad():
        for batch in data_loader:
            history_emb = batch['history_embeddings'].to(device)
            history_mask = batch['history_mask'].to(device)
            target_emb = batch['target_post_embedding'].to(device)
            labels = batch['label']
            
            scores = model(history_emb, history_mask, target_emb)
            probs = torch.sigmoid(scores)
            
            all_preds.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_user_ids.extend(batch['user_ids'])
            all_post_ids.extend(batch['post_ids'])
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    
    metrics = {
        'total_samples': len(y_true),
        'positive_samples': int(y_true.sum()),
        'negative_samples': int(len(y_true) - y_true.sum()),
    }
    
    if len(set(y_true)) > 1:
        metrics['auc_roc'] = float(roc_auc_score(y_true, y_pred))
        metrics['average_precision'] = float(average_precision_score(y_true, y_pred))
    
    metrics['accuracy_at_0.5'] = float(accuracy_score(y_true, (y_pred > 0.5).astype(int)))
    
    return {
        'metrics': metrics,
        'predictions': {
            'user_ids': all_user_ids,
            'post_ids': all_post_ids,
            'y_true': y_true,
            'y_pred': y_pred,
        },
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Path,
):
    """Plot training history."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Two-Tower - Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['train_auc'], 'b-', label='Train AUC', linewidth=2)
    ax2.plot(epochs, history['val_auc'], 'r-', label='Val AUC', linewidth=2)
    ax2.set_title('Two-Tower - AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
    title_suffix: str = '',
):
    """Plot model performance (ROC, PR curve, confusion matrix, distribution)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.5
    axes[0, 0].plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title(f'ROC Curve {title_suffix}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    axes[0, 1].plot(recall, precision)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title(f'Precision-Recall Curve {title_suffix}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion Matrix
    y_pred_binary = (y_pred > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    axes[1, 0].imshow(cm, cmap='Blues')
    for (i, j), val in np.ndenumerate(cm):
        axes[1, 0].text(j, i, int(val), ha='center', va='center', fontsize=14)
    axes[1, 0].set_title(f'Confusion Matrix {title_suffix}')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_yticks([0, 1])
    
    # Prediction Distribution
    axes[1, 1].hist(y_pred[y_true == 0], bins=50, alpha=0.7, label='Negative', color='blue')
    axes[1, 1].hist(y_pred[y_true == 1], bins=50, alpha=0.7, label='Positive', color='orange')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Prediction Distribution {title_suffix}')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# Main Pipeline
# =============================================================================

def run_two_tower_pipeline(
    embedding_bundle: str,
    user_splits: str,
    device: str,
    shared_dim: int = 128,
    user_hidden_dim: int = 256,
    post_hidden_dim: int = 256,
    num_attention_heads: int = 4,
    num_attention_layers: int = 2,
    max_history_len: int = 20,
    prediction_posts_per_user: int = 1,
    dropout_rate: float = 0.1,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    epochs: int = 100,
    patience: int = 20,
    random_seed: int = 42,
    output_dir: Optional[Path] = None,
    disable_progress: bool = False,
) -> Dict[str, Any]:
    """
    Run the two-tower training pipeline with BCE loss.
    """
    # Create output directories FIRST so we can create a logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path(__file__).resolve().parents[2] / "outputs"
    base_dir = Path(output_dir) / "train" / timestamp
    checkpoints_dir = base_dir / "checkpoints"
    plots_dir = base_dir / "plots"
    logs_dir = base_dir / "logs"
    
    for d in [checkpoints_dir, plots_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger BEFORE any potentially slow operations
    logger = get_stage_logger('STAGE_05_TRAIN_TWO_TOWER', log_file=base_dir / 'stage.log')
    log_operation_start('run_two_tower_pipeline started', 'STAGE_05_TRAIN_TWO_TOWER', logger)
    
    # Set seeds
    log_operation_start('Set random seeds', 'STAGE_05_TRAIN_TWO_TOWER', logger)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    # Load data
    log_operation_start('Load embedding bundle and user splits', 'STAGE_05_TRAIN_TWO_TOWER', logger)
    print("Loading embedding bundle and user splits...")
    with open(embedding_bundle, 'rb') as f:
        bundle = pickle.load(f)
    
    with open(user_splits, 'r') as f:
        splits = json.load(f)
    
    posts_emb_df = bundle['posts_emb_df']
    likes_df = bundle['likes_df']
    join_like = str(bundle['join_like'])
    join_post = str(bundle['join_post'])
    
    train_users = list(map(str, splits.get('train_users', [])))
    val_users = list(map(str, splits.get('val_users', [])))
    holdout_users = list(map(str, splits.get('holdout_users', [])))
    
    print(f"Train users: {len(train_users)}, Val users: {len(val_users)}, Holdout users: {len(holdout_users)}")
    
    # Get embedding dimension
    emb_cols = [c for c in posts_emb_df.columns if c.startswith('post_emb_') or c.startswith('image_emb_')]
    post_embedding_dim = len(emb_cols)
    print(f"Post embedding dimension: {post_embedding_dim}")
    
    # Build user histories
    log_operation_start('Build user history sequences', 'STAGE_05_TRAIN_TWO_TOWER', logger)
    print("Building user history sequences...")
    train_histories = build_user_history_sequences(
        likes_df, posts_emb_df, join_like, join_post,
        train_users, max_history_len, prediction_posts_per_user
    )
    val_histories = build_user_history_sequences(
        likes_df, posts_emb_df, join_like, join_post,
        val_users, max_history_len, prediction_posts_per_user
    )
    holdout_histories = build_user_history_sequences(
        likes_df, posts_emb_df, join_like, join_post,
        holdout_users, max_history_len, prediction_posts_per_user
    )
    
    print(f"Built histories - Train: {len(train_histories)}, Val: {len(val_histories)}, Holdout: {len(holdout_histories)}")
    
    print(f"\n{'='*60}")
    print(f"Training Two-Tower model")
    print(f"{'='*60}")
    
    # Create training items
    log_operation_start('Create training items (pairs)', 'STAGE_05_TRAIN_TWO_TOWER', logger)
    train_items = create_training_items(
        train_histories, posts_emb_df, join_post,
        max_history_len, neg_ratio=1.0, random_seed=random_seed
    )
    val_items = create_training_items(
        val_histories, posts_emb_df, join_post,
        max_history_len, neg_ratio=1.0, random_seed=random_seed + 1
    )
    
    print(f"Training items: {len(train_items)}, Validation items: {len(val_items)}")
    # Log class balance
    train_pos = sum(1 for item in train_items if item.label == 1)
    train_neg = len(train_items) - train_pos
    val_pos = sum(1 for item in val_items if item.label == 1)
    val_neg = len(val_items) - val_pos
    print(f"  Train: {train_pos} pos, {train_neg} neg | Val: {val_pos} pos, {val_neg} neg")
    
    # Create datasets and loaders
    train_dataset = TwoTowerDataset(train_items)
    val_dataset = TwoTowerDataset(val_items)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create model
    log_operation_start('Create two-tower model', 'STAGE_05_TRAIN_TWO_TOWER', logger)
    model = TwoTowerEngagement(
        post_embedding_dim=post_embedding_dim,
        shared_dim=shared_dim,
        user_hidden_dim=user_hidden_dim,
        post_hidden_dim=post_hidden_dim,
        num_attention_heads=num_attention_heads,
        num_attention_layers=num_attention_layers,
        max_history_len=max_history_len,
        dropout_rate=dropout_rate,
    )
    
    # Train
    log_operation_start(f'Train two-tower model (epochs={epochs}, batch_size={batch_size})', 'STAGE_05_TRAIN_TWO_TOWER', logger)
    training_result = train_two_tower_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        checkpoints_dir=checkpoints_dir,
        disable_progress=disable_progress,
    )
    
    # Plot training history
    plot_training_history(
        training_result['history'],
        plots_dir / f"training_history_{timestamp}.png",
    )
    
    # Evaluate on validation set
    val_eval = evaluate_model(model, val_loader, device)
    print(f"\nValidation metrics: {val_eval['metrics']}")
    
    # Plot validation performance
    plot_model_performance(
        val_eval['predictions']['y_true'],
        val_eval['predictions']['y_pred'],
        plots_dir / f"val_performance_{timestamp}.png",
        title_suffix=""
    )
    
    # Save model
    model_path = checkpoints_dir / f"two_tower_{timestamp}.pth"
    config = {
        'model_type': 'two_tower',
        'post_embedding_dim': post_embedding_dim,
        'shared_dim': shared_dim,
        'user_hidden_dim': user_hidden_dim,
        'post_hidden_dim': post_hidden_dim,
        'num_attention_heads': num_attention_heads,
        'num_attention_layers': num_attention_layers,
        'max_history_len': max_history_len,
        'dropout_rate': dropout_rate,
    }
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'training_history': training_result['history'],
        'best_val_auc': training_result['best_val_auc'],
        'best_val_loss': training_result['best_val_loss'],
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    results = {}
    
    # Holdout evaluation
    if len(holdout_histories) > 0:
        print(f"\nEvaluating on holdout users...")
        holdout_items = create_training_items(
            holdout_histories, posts_emb_df, join_post,
            max_history_len, neg_ratio=1.0, random_seed=random_seed + 2
        )
        holdout_dataset = TwoTowerDataset(holdout_items)
        holdout_loader = DataLoader(
            holdout_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn
        )
        
        holdout_eval = evaluate_model(model, holdout_loader, device)
        print(f"Holdout metrics: {holdout_eval['metrics']}")
        
        # Plot holdout performance
        plot_model_performance(
            holdout_eval['predictions']['y_true'],
            holdout_eval['predictions']['y_pred'],
            plots_dir / f"holdout_performance_{timestamp}.png",
            title_suffix="(Holdout)"
        )
        
        # Save holdout predictions
        holdout_dir = base_dir / "holdout_eval"
        holdout_dir.mkdir(parents=True, exist_ok=True)
        
        pred_df = pd.DataFrame({
            'did': holdout_eval['predictions']['user_ids'],
            'post_id': holdout_eval['predictions']['post_ids'],
            'y_true': holdout_eval['predictions']['y_true'],
            'y_pred_proba': holdout_eval['predictions']['y_pred'],
        })
        pred_df.to_parquet(holdout_dir / 'predictions.parquet', index=False)
        
        with open(holdout_dir / 'metrics_overall.json', 'w') as f:
            json.dump(holdout_eval['metrics'], f, indent=2)
        
        results = {
            'training_result': {k: v for k, v in training_result.items() if k != 'model'},
            'val_metrics': val_eval['metrics'],
            'holdout_metrics': holdout_eval['metrics'],
            'model_path': str(model_path),
        }
    else:
        results = {
            'training_result': {k: v for k, v in training_result.items() if k != 'model'},
            'val_metrics': val_eval['metrics'],
            'model_path': str(model_path),
        }
    
    # Save training config
    training_config = {
        'model_type': 'two_tower',
        'post_embedding_dim': post_embedding_dim,
        'shared_dim': shared_dim,
        'user_hidden_dim': user_hidden_dim,
        'post_hidden_dim': post_hidden_dim,
        'num_attention_heads': num_attention_heads,
        'num_attention_layers': num_attention_layers,
        'max_history_len': max_history_len,
        'prediction_posts_per_user': prediction_posts_per_user,
        'dropout_rate': dropout_rate,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'epochs': epochs,
        'patience': patience,
        'random_seed': random_seed,
        'train_users': len(train_users),
        'val_users': len(val_users),
        'holdout_users': len(holdout_users),
        'results': results,
    }
    
    with open(base_dir / 'training_config.json', 'w') as f:
        json.dump(training_config, f, indent=2)
    
    # Write stage info
    stage_info = [
        f"stage: train_two_tower",
        f"timestamp: {timestamp}",
        f"train_users: {len(train_users)}",
        f"val_users: {len(val_users)}",
        f"holdout_users: {len(holdout_users)}",
        f"val_auc: {results.get('val_metrics', {}).get('auc_roc', 'N/A')}",
    ]
    if 'holdout_metrics' in results:
        stage_info.append(f"holdout_auc: {results['holdout_metrics'].get('auc_roc', 'N/A')}")
    
    (base_dir / 'stage_info.txt').write_text('\n'.join(stage_info) + '\n')
    
    return {
        'output_dir': base_dir,
        'results': results,
        'training_config': training_config,
    }


# =============================================================================
# Pipeline Entry Point
# =============================================================================

def run(context, args) -> Dict[str, Any]:
    """
    Pipeline entry point (same interface as stage_train.py).
    
    Called by the pipeline registry when --model-type two-tower is specified.
    """
    run_dir = Path(context.run_dir).resolve()
    
    # Create a temporary logger for the stage entry point
    temp_log_dir = run_dir / '05_train'
    temp_log_dir.mkdir(parents=True, exist_ok=True)
    entry_logger = get_stage_logger('STAGE_05_TRAIN_TWO_TOWER', log_file=temp_log_dir / 'stage_entry.log')
    
    log_operation_start('Stage 5 entry point (two-tower)', 'STAGE_05_TRAIN_TWO_TOWER', entry_logger)
    
    # Locate embedding bundle
    log_operation_start('Locate embedding bundle', 'STAGE_05_TRAIN_TWO_TOWER', entry_logger)
    prior_featurize = select_prior_output(
        run_dir, '02_featurize',
        use_latest=context.use_latest,
        prior_path=context.prior_outputs.get('02_featurize')
    )
    if prior_featurize is None:
        raise FileNotFoundError("Featurize output not found.")
    
    bundle_candidates = sorted(
        prior_featurize.glob('embedding_bundle_*.pkl'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not bundle_candidates:
        raise FileNotFoundError(f"No embedding_bundle_*.pkl found under {prior_featurize}")
    bundle_path = bundle_candidates[0]
    entry_logger.info(f"Found bundle: {bundle_path}")
    
    # Locate user_splits
    log_operation_start('Locate user splits', 'STAGE_05_TRAIN_TWO_TOWER', entry_logger)
    prior_split = select_prior_output(
        run_dir, '04_split',
        use_latest=context.use_latest,
        prior_path=context.prior_outputs.get('04_split')
    )
    if prior_split is None:
        raise FileNotFoundError("Split output not found.")
    
    splits_path = prior_split / 'user_splits.json'
    if not splits_path.exists():
        raise FileNotFoundError(f"user_splits.json not found under {prior_split}")
    entry_logger.info(f"Found splits: {splits_path}")
    
    # Get parameters from args
    log_operation_start('Call run_two_tower_pipeline', 'STAGE_05_TRAIN_TWO_TOWER', entry_logger)
    t0 = time.time()
    device = get_device(args.device)
    results = run_two_tower_pipeline(
        embedding_bundle=str(bundle_path.resolve()),
        user_splits=str(splits_path.resolve()),
        shared_dim=int(args.shared_dim),
        user_hidden_dim=int(args.user_hidden_dim),
        post_hidden_dim=int(args.post_hidden_dim),
        num_attention_heads=int(args.num_attention_heads),
        num_attention_layers=int(args.num_attention_layers),
        max_history_len=int(args.max_history_len),
        prediction_posts_per_user=int(args.prediction_posts_per_user),
        dropout_rate=float(args.dropout_rate_two_tower),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay_two_tower),
        epochs=int(args.epochs),
        patience=int(args.patience),
        device=device,
        random_seed=int(args.random_seed),
        output_dir=run_dir,
        disable_progress=bool(getattr(args, 'disable_progress', False)),
    )
    
    training_dir = results['output_dir']
    
    # Relocate train/<ts> → 05_train/<ts> if needed
    final_train_dir = training_dir
    if training_dir.parent.name == 'train':
        dest_root = training_dir.parent.parent / '05_train'
        dest_root.mkdir(parents=True, exist_ok=True)
        dest = dest_root / training_dir.name
        if not dest.exists():
            try:
                training_dir.rename(dest)
            except Exception:
                import shutil
                shutil.copytree(training_dir, dest, dirs_exist_ok=True)
        final_train_dir = dest
    
    print(f"\nTwo-Tower training completed in {time.time() - t0:.2f}s")
    print(f"Output directory: {final_train_dir}")
    
    return {
        'output_dir': final_train_dir,
        'artifacts': {
            'training_config': str(final_train_dir / 'training_config.json'),
            'embedding_bundle_path': str(bundle_path.resolve()),
            'user_splits_path': str(splits_path.resolve()),
        },
    }


if __name__ == '__main__':
    # Simple CLI for standalone testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Two-Tower Training')
    parser.add_argument('--embedding-bundle', required=True)
    parser.add_argument('--user-splits', required=True)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    results = run_two_tower_pipeline(
        embedding_bundle=args.embedding_bundle,
        user_splits=args.user_splits,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    print(f"\nResults: {json.dumps(results['training_config'], indent=2)}")
