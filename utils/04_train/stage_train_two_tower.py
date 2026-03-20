#!/usr/bin/env python3

"""
Stage 4 (Two-Tower): Train two-tower engagement prediction models with flexible user encoders.

This stage trains Two-Tower models, a popular architecture for large-scale recommendation
systems that separates user and item (post) representations into independent "towers",
enabling efficient candidate retrieval and ranking.

═══════════════════════════════════════════════════════════════════════════════
TWO-TOWER ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

Core concept: Project users and posts into a SHARED EMBEDDING SPACE where
engagement is predicted by computing similarity (dot product) between representations.

    User Tower:  history_sequence -> UserEncoder -> user_vector [shared_dim]
                (or "summarized": user_vector is provided by the dataset)
    Post Tower:  post_embedding -> (optional) PostTower -> post_vector [shared_dim]
                (or identity when use_post_encoder=False)
    Prediction:  sigmoid(user_vector · post_vector) -> engagement_probability

Benefits:
    ✓ Decoupled representations: User and post towers can be independently cached
    ✓ Efficient retrieval: Pre-compute all post_vectors, then find top-K by
      dot product similarity (can use approximate nearest neighbor search)
    ✓ Scalable: Avoids expensive cross-feature interactions until final dot product

═══════════════════════════════════════════════════════════════════════════════
USER ENCODER OPTIONS
═══════════════════════════════════════════════════════════════════════════════

This stage supports THREE user-history encoders, selected via `--user-encoder`:

0. **"summarized"** - Hand-crafted summarization (no trainable user encoder)
   ───────────────────────────────────────────────────────────────────────────
   Uses `SummarizedEngagementDataset` to pre-compute a fixed-size user summary
   vector (e.g. mean / EMA / linear recency). In this mode the "user tower" is
   effectively the identity function at training time because the dataset has
   already produced the user vector.

1. **"full_transformer"** - TransformerDualPoolingEncoder (Full Transformer Self-Attention)
   ───────────────────────────────────────────────────────────────────────────
   Uses transformer encoder with multi-head self-attention to capture complex
   inter-post relationships in user history. Best modeling capacity but highest
   computational cost.
   
   Architecture: Input projection -> Positional encoding -> Transformer encoder
                 layers -> Dual pooling (attention + mean) -> Output projection


2. **"cross_attention"** - CrossAttentionPoolingEncoder (Single-Query Cross-Attention)
   ───────────────────────────────────────────────────────────────────────────
   Skips expensive self-attention layers, using only a single learned-query
   cross-attention for aggregation. Significantly faster with fewer parameters.
   
   Architecture: Input projection -> Positional encoding -> Cross-attention
                 pooling (single query) + Mean pooling -> Output projection

═══════════════════════════════════════════════════════════════════════════════
TRAINING DETAILS
═══════════════════════════════════════════════════════════════════════════════

Loss:       Binary cross-entropy on sigmoid(dot_product)
Sampling:   Balanced positive/negative pairs (1:1 ratio)
Optimizer:  AdamW with weight decay for regularization
Scheduling: ReduceLROnPlateau based on validation AUC
Regularization: Gradient clipping, dropout, early stopping

The model is trained to maximize dot product for engaged pairs and minimize for
non-engaged pairs, learning a metric space where similar preferences cluster.

═══════════════════════════════════════════════════════════════════════════════

Inputs (from prior pipeline stages):
    - embeddings_*.npy memmap from 01_get_data
    - target_posts_*.parquet from 02_target_posts
    - history_posts_*.parquet from 03_user_history

Outputs under <run_dir>/04_train/<timestamp>/:
    - checkpoints/two_tower_best.pth (best-by-validation checkpoint during training)
    - checkpoints/two_tower_<timestamp>.pth (final model checkpoint)
    - plots/training_history_*.png (loss and AUC curves)
    - plots/{train,val,holdout}_performance_*.png (precision-recall, ROC curves)
    - logs/ (training logs)
    - training_config.json (hyperparameters and configuration)
    - stage_info.txt (pipeline metadata)
    - predictions/train.parquet (per-row predictions on the training set)
    - predictions/val.parquet (per-row predictions on the validation set)
    - predictions/holdout_unseen_users.parquet (predictions for user-split holdout)
    - predictions/holdout_seen_users.parquet (predictions for temporal holdout, if configured)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
from torch.utils.data import DataLoader, Dataset

from utils.pipeline.core import Context
from utils.helpers import (
    get_stage_logger,
    log_operation_start,
    get_device,
    plot_model_performance,
    plot_training_history,
    clear_cuda_memory,
    set_random_seeds,
)
from utils.dataloaders import (
    load_training_data,
    SequenceEngagementDataset,
    SummarizedEngagementDataset,
    SummarizedUserTower,
    TransformerDualPoolingEncoder,
    CrossAttentionPoolingEncoder,
    create_data_loaders,
    get_summarizer,
)

STAGE_LOG_NAME = "STAGE_04_TRAIN_TWO_TOWER"


# =============================================================================
# Post Tower
# =============================================================================

class PostTower(nn.Module):
    """Post tower: Projects post embeddings to shared embedding space.
    
    This is the "item tower" of the two-tower architecture. It maps raw post
    embeddings into the same latent space as user representations, enabling
    dot-product similarity scoring.
    
    Architecture:
        Input: Post embedding [input_dim]
        Hidden: Linear -> LayerNorm -> GELU -> Dropout
        Output: Linear -> Shared space representation [output_dim]
    
    Design choices:
        - LayerNorm (not BatchNorm): Post embeddings don't have batch-level
          statistics during inference (we process posts independently)
        - GELU activation: Smooth gradients for better optimization
        - Single hidden layer: Posts are already semantic embeddings from a
          pre-trained model, so minimal transformation is often sufficient
    
    Args:
        input_dim: Dimensionality of input post embeddings (e.g., 384 for all-MiniLM)
        hidden_dim: Internal hidden layer size
        output_dim: Shared space dimensionality
        dropout_rate: Dropout probability for regularization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float,
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
        """Initialize weights using Xavier uniform for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, post_embeddings: torch.Tensor) -> torch.Tensor:
        """Project post embeddings to shared space.
        
        Args:
            post_embeddings: Batch of post embeddings [batch, input_dim]
        
        Returns:
            Shared space representations [batch, output_dim]
        """
        return self.network(post_embeddings)


# =============================================================================
# Two-Tower Engagement Model
# =============================================================================
class TwoTowerModel(nn.Module):
    """Two-tower engagement prediction model with pluggable user encoders.
    
    Implements the two-tower architecture where user and post representations
    are independently computed and combined via dot product similarity. This
    architecture is particularly well-suited for large-scale retrieval and
    ranking systems.
    
    Architecture:
        User Tower:
            - "full_transformer": TransformerDualPoolingEncoder(history_sequence, mask) -> user_vector [shared_dim]
            - "cross_attention": CrossAttentionPoolingEncoder(history_sequence, mask) -> user_vector [shared_dim]
            - "summarized": user_vector is provided by SummarizedEngagementDataset
              (the model treats the dataset-provided user summary as the user embedding)
        
        Post Tower:
            - use_post_encoder=True:  PostTower(post_embedding) -> post_vector [shared_dim]
            - use_post_encoder=False: post_vector is the raw post embedding (identity)
        
        Scoring: dot_product(user_vector, post_vector) -> raw score
                 sigmoid(raw_score) -> engagement_probability
    
    Key characteristics:
        - Shared embedding space: Both towers output the same dimensionality for dot product
        - Independent computation: Towers never exchange information (until final dot product)
        - Modular encoders: User tower can be "summarized", "full_transformer", or "cross_attention"
    
    Deployment pattern:
        1. Pre-compute post_vectors for all candidate posts
        2. At inference, encode user history once -> user_vector
        3. Find top-K posts by dot product (can use ANN for scale)
        4. Return ranked candidates
    
    Args:
        post_embedding_dim: Dimensionality of input post embeddings
        shared_dim: Output dimension for both towers
        user_hidden_dim: User tower internal hidden size
        post_hidden_dim: Post tower internal hidden size
        num_attention_heads: Attention heads for TransformerDualPoolingEncoder
        num_attention_layers: Transformer layers for TransformerDualPoolingEncoder
        max_history_len: Maximum history sequence length
        dropout_rate: Dropout probability
        user_encoder_type: User tower architecture - "summarized", "full_transformer", or "cross_attention"
        use_post_encoder: If True, learn a post projection (PostTower). If False, use raw post embeddings.
    """

    def __init__(
        self,
        post_embedding_dim: int,
        shared_dim: int,
        user_hidden_dim: int,
        post_hidden_dim: int,
        num_attention_heads: int,
        num_attention_layers: int,
        max_history_len: int,
        dropout_rate: float,
        user_encoder_type: str,
        use_post_encoder: bool,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.post_embedding_dim = post_embedding_dim
        self.user_encoder_type = user_encoder_type
        self.use_post_encoder = use_post_encoder

        # Instantiate user tower based on encoder type
        if user_encoder_type == "cross_attention":
            self.user_tower = CrossAttentionPoolingEncoder(
                input_dim=post_embedding_dim,
                hidden_dim=user_hidden_dim,
                output_dim=shared_dim,
                max_seq_len=max_history_len,
                dropout_rate=dropout_rate,
            )
        elif user_encoder_type == "full_transformer":
            self.user_tower = TransformerDualPoolingEncoder(
                input_dim=post_embedding_dim,
                hidden_dim=user_hidden_dim,
                output_dim=shared_dim,
                num_attention_heads=num_attention_heads,
                num_attention_layers=num_attention_layers,
                max_seq_len=max_history_len,
                dropout_rate=dropout_rate,
            )
        elif user_encoder_type == "summarized":
            # In "summarized" mode, the dataset provides the user vector directly
            # (e.g., mean/EMA/linear-recency summary). The model treats the
            # history input as an already-encoded user embedding (placed at
            # position 0 in a padded sequence for a consistent forward() signature).
            self.user_tower = SummarizedUserTower(embed_dim=post_embedding_dim)

            # If we still learn a post projection (use_post_encoder=True), its output
            # dimension must match the dataset-provided user embedding dimension.
            if use_post_encoder and (shared_dim != post_embedding_dim):
                raise ValueError(f"--shared-dim ({shared_dim}) and post embedding dim ({post_embedding_dim}) do not match! They must match for two tower with user summarization.")
        else:
            raise ValueError(
                f"Unknown user_encoder_type '{user_encoder_type}'. "
                "Choose 'summarized', 'full_transformer' or 'cross_attention'."
            )

        # If we don't project posts, then `post_embeddings` stays in the raw embedding
        # space and must match the user embedding dim for dot product scoring.
        #
        # In summarized mode, the user embedding is also in the raw embedding space.
        if (not use_post_encoder) and (user_encoder_type != "summarized") and (shared_dim != post_embedding_dim):
            raise ValueError(
                f"use_post_encoder=False requires --shared-dim ({shared_dim}) to equal post embedding dim ({post_embedding_dim}) "
                f"(or set use_post_encoder=True to project posts to shared_dim)."
            )

        if use_post_encoder:
            # Post tower is the same regardless of user encoder type
            self.post_tower = PostTower(
                input_dim=post_embedding_dim,
                hidden_dim=post_hidden_dim,
                output_dim=shared_dim,
                dropout_rate=dropout_rate,
            )
        else:
            self.post_tower = nn.Identity()


    def encode_user(self, history_embeddings: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        """Encode user engagement history into shared space representation.
        
        Args:
            history_embeddings: Padded history sequences [batch, seq_len, input_dim]
            history_mask: Boolean mask [batch, seq_len], True = valid position
        
        Returns:
            User vectors in shared space [batch, shared_dim]
        """
        return self.user_tower(history_embeddings, history_mask)

    def encode_post(self, post_embeddings: torch.Tensor) -> torch.Tensor:
        """Encode post embeddings into shared space representation.
        
        Args:
            post_embeddings: Raw post embeddings [batch, input_dim]
        
        Returns:
            Post vectors for dot product scoring.
                - use_post_encoder=True: [batch, shared_dim]
                - otherwise: [batch, post_embedding_dim] (identity)
        """
        return self.post_tower(post_embeddings)

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        post_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute engagement scores via dot product in shared space.
        
        This is the core two-tower computation: encode user and post independently,
        then measure similarity via dot product. Higher dot product = higher
        predicted engagement probability.
        
        Args:
            history_embeddings: User history input.
                - all modes: padded history sequences [batch, seq_len, input_dim]
                  In "summarized" mode, the user summary is expected to be placed
                  at position 0 (and optionally padded to seq_len > 1).
            history_mask: History validity mask [batch, seq_len] (optional in summarized mode)
            post_embeddings: Target post embeddings [batch, input_dim]
        
        Returns:
            Raw engagement scores [batch] (logits before sigmoid)
        """
        user_emb = self.encode_user(history_embeddings, history_mask)
        post_emb = self.encode_post(post_embeddings)
        
        # Dot product: element-wise multiply then sum over shared_dim
        return (user_emb * post_emb).sum(dim=-1)

    def compute_loss_and_preds(
        self,
        batch: Dict[str, Any],
        device: str,
        embed_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss and predictions for a batch.
        
        This method provides a unified interface for training, validation, and
        inference loops. It computes raw scores (dot products), applies sigmoid
        to get probabilities, and calculates binary cross-entropy loss.
        
        Args:
            batch: Batch dictionary. Expected keys depend on `user_encoder_type`:
                - "summarized": {"features", "label"} where features is
                  [B, 2*embed_dim] concatenated [user_summary || post_embedding]
                - otherwise: {"history_embeddings", "history_mask", "target_post_embedding", "label"}
            device: Device string (e.g. "cpu" or "cuda")
            embed_dim: Post embedding dimensionality D (used only to split "features" in summarized mode)
        
        Returns:
            Tuple of (loss, scores):
                - loss: Scalar BCE loss tensor
                - scores: Raw dot product scores [batch] (before sigmoid)
        
        Note:
            Returns raw scores (not probabilities) for flexibility in evaluation.
            Apply sigmoid(scores) to get probabilities.
        """
        # unpack inputs
        if self.user_encoder_type == "summarized":
            features = batch["features"].to(device) # [B, embed_dim*2]
            user_summary = features[:, :embed_dim] # [B, embed_dim]
            history_embeddings = user_summary.unsqueeze(1)  # [B, 1, embed_dim] (summary token at position 0)
            post_embeddings = features[:, embed_dim:] # [B, embed_dim]
            # Cold-start handling: empty histories have an all-zero summary sentinel.
            # Use the mask to indicate whether the summary came from at least one
            # history item so the summarized user tower can inject a learnable
            # cold-start embedding.
            has_history = user_summary.abs().sum(dim=1) > 0 # [B]
            history_mask = has_history.unsqueeze(1).to(device=device, dtype=torch.bool) # [B, 1]
            assert history_embeddings.shape[-1] == post_embeddings.shape[-1]
        else:
            history_embeddings = batch["history_embeddings"].to(device) # [B, seq_len, embed_dim]
            history_mask = batch["history_mask"].to(device) # [B, seq_len]
            post_embeddings = batch["target_post_embedding"].to(device) # [B, embed_dim]
        labels = batch["label"].to(device)

        scores = self.forward(history_embeddings, history_mask, post_embeddings)
        probs = torch.sigmoid(scores)
        loss = F.binary_cross_entropy(probs, labels.float())
        return loss, scores


# =============================================================================
# Training Loop
# =============================================================================

def train_two_tower_model(
    model: TwoTowerModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    checkpoints_dir: Optional[Path],
    disable_progress: bool,
    lr_scheduler_factor: float,
    lr_scheduler_patience: int,
    gradient_clip_max_norm: float,
    embed_dim: int
) -> Dict[str, Any]:
    from tqdm import tqdm

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_scheduler_factor, patience=lr_scheduler_patience
    )

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    best_val_auc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    for epoch in tqdm(range(epochs), desc="Training epochs", disable=disable_progress):
        # --- Training ---
        model.train()
        train_losses: List[float] = []
        train_preds: List[float] = []
        train_labels: List[float] = []

        for batch in tqdm(train_loader, desc="Training", leave=False, disable=disable_progress):
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            loss, scores = model.compute_loss_and_preds(batch, device, embed_dim)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_max_norm)
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(torch.sigmoid(scores).detach().cpu().numpy().tolist())
            train_labels.extend(labels.cpu().numpy().tolist())

        # --- Validation ---
        model.eval()
        val_losses: List[float] = []
        val_preds: List[float] = []
        val_labels: List[float] = []

        with torch.inference_mode():
            for batch in tqdm(val_loader, desc="Validation", leave=False, disable=disable_progress):
                labels = batch["label"].to(device)

                loss, scores = model.compute_loss_and_preds(batch, device, embed_dim)

                val_losses.append(loss.item())
                val_preds.extend(torch.sigmoid(scores).detach().cpu().numpy().tolist())
                val_labels.extend(labels.cpu().numpy().tolist())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        train_auc = roc_auc_score(train_labels, train_preds) if len(set(train_labels)) > 1 else 0.5
        val_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.5

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auc"].append(float(train_auc))
        history["val_auc"].append(float(val_auc))

        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0

            if checkpoints_dir is not None:
                torch.save(
                    {"epoch": epoch, "model_state_dict": best_state_dict, "val_loss": val_loss, "val_auc": val_auc, "history": history},
                    checkpoints_dir / "two_tower_best.pth",
                )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_auc": best_val_auc,
    }


# =============================================================================
# Evaluation
# =============================================================================

def _evaluate_two_tower_model(
    model: TwoTowerModel,
    data_loader: DataLoader,
    device: str,
    embed_dim: int,
) -> Dict[str, Any]:
    """Evaluate two-tower model and return metrics + predictions."""
    model = model.to(device)
    model.eval()

    all_preds: List[float] = []
    all_labels: List[float] = []
    all_user_ids: List[str] = []
    all_post_ids: List[str] = []

    with torch.inference_mode():
        for batch in data_loader:
            labels = batch["label"]

            _, scores = model.compute_loss_and_preds(batch, device, embed_dim)
            probs = torch.sigmoid(scores)

            all_preds.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_user_ids.extend(batch["user_id"])
            all_post_ids.extend(batch["post_id"])

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metrics: Dict[str, Any] = {
        "total_samples": len(y_true),
        "positive_samples": int(y_true.sum()),
        "negative_samples": int(len(y_true) - y_true.sum()),
    }

    if len(set(y_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred))
        metrics["average_precision"] = float(average_precision_score(y_true, y_pred))

    metrics["accuracy_at_0.5"] = float(accuracy_score(y_true, (y_pred > 0.5).astype(int)))

    return {
        "metrics": metrics,
        "predictions": {
            "user_id": all_user_ids,
            "post_id": all_post_ids,
            "y_true": y_true,
            "y_pred": y_pred,
        },
    }


# =============================================================================
# Plotting
# =============================================================================

# =============================================================================
# Pipeline entry point
# =============================================================================

def run(context: Context, args) -> Dict[str, Any]:
    """Pipeline entry point for two-tower training."""
    device = get_device(args.device)
    timestamp = context.run_timestamp

    # --- output dirs ---
    run_tag = args.run_tag or ""
    out_dir = context.new_stage_dir("04_train", tag=run_tag)
    checkpoints_dir = out_dir / "checkpoints"
    plots_dir = out_dir / "plots"
    logs_dir = out_dir / "logs"
    for d in (checkpoints_dir, plots_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger = get_stage_logger(STAGE_LOG_NAME, log_file=out_dir / "stage.log")
    log_operation_start("Stage 4 Two-Tower training", STAGE_LOG_NAME, logger)
    t0 = time.time()

    # --- seeds & cuda ---
    clear_cuda_memory()
    random_seed = int(args.random_seed)
    set_random_seeds(random_seed)

    # --- load data from prior stages ---
    log_operation_start("Load training data from prior stages", STAGE_LOG_NAME, logger)
    embeddings_mmap, target_posts_df, history_df, embed_dim = load_training_data(
        context, logger=logger,
    )

    # --- hyperparams (extract all args once, use locals everywhere below) ---
    max_history_len = int(args.max_history_len)
    shared_dim = int(args.shared_dim)
    user_hidden_dim = int(args.user_hidden_dim)
    post_hidden_dim = int(args.post_hidden_dim)
    num_attention_heads = int(args.num_attention_heads)
    num_attention_layers = int(args.num_attention_layers)
    dropout_rate = float(args.dropout_rate_two_tower)
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.weight_decay_two_tower)
    epochs = int(args.epochs)
    patience = int(args.patience)
    disable_progress = bool(args.disable_progress)
    user_encoder_type = args.user_encoder
    use_post_encoder = args.use_post_encoder
    generate_plots = not bool(args.no_plots)
    save_model = not bool(args.no_save_model)
    lr_scheduler_factor = float(args.lr_scheduler_factor)
    lr_scheduler_patience = int(args.lr_scheduler_patience)
    gradient_clip_max_norm = float(args.gradient_clip_max_norm)
    eval_holdout_type = str(args.eval_holdout_type)

    # Worker settings
    num_workers = int(args.num_dataloader_workers)
    pin_memory = bool(args.dataloader_pin_memory)
    persistent_workers = bool(args.dataloader_persistent_workers)
    prefetch_factor = int(args.dataloader_prefetch_factor)

    # --- datasets ---
    log_operation_start("Create datasets", STAGE_LOG_NAME, logger)
    if user_encoder_type == "summarized":
        # get summarizer
        summarizer_name = args.user_summarization
        ema_alpha = float(args.ema_alpha)
        summarizer = get_summarizer(summarizer_name, ema_alpha=ema_alpha)
        
        train_dataset = SummarizedEngagementDataset(
            embeddings_mmap, target_posts_df, history_df, split="train", 
            summarizer=summarizer, embed_dim=embed_dim, logger=logger,
        )
        val_dataset = SummarizedEngagementDataset(
            embeddings_mmap, target_posts_df, history_df, split="val", 
            summarizer=summarizer, embed_dim=embed_dim, logger=logger,
        )
    else:
        train_dataset = SequenceEngagementDataset(
            embeddings_mmap, target_posts_df, history_df, split="train",
            max_history_len=max_history_len, embed_dim=embed_dim, logger=logger,
        )
        val_dataset = SequenceEngagementDataset(
            embeddings_mmap, target_posts_df, history_df, split="val",
            max_history_len=max_history_len, embed_dim=embed_dim, logger=logger,
        )

    # Create data loaders using centralized helper
    train_loader, val_loader, _ = create_data_loaders(
        train_dataset, val_dataset, batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    logger.info(f"Post embedding dim: {embed_dim}")
    logger.info(f"Train items: {len(train_dataset)}, Val items: {len(val_dataset)}")

    # --- create model ---
    log_operation_start(f"Create two-tower model (user_encoder={user_encoder_type})", STAGE_LOG_NAME, logger)
    model = TwoTowerModel(
        post_embedding_dim=embed_dim,
        shared_dim=shared_dim,
        user_hidden_dim=user_hidden_dim,
        post_hidden_dim=post_hidden_dim,
        num_attention_heads=num_attention_heads,
        num_attention_layers=num_attention_layers,
        max_history_len=max_history_len,
        dropout_rate=dropout_rate,
        user_encoder_type=user_encoder_type,
        use_post_encoder=use_post_encoder,
    )

    if (not use_post_encoder) and (user_encoder_type == "summarized"):
        logger.info(f"Creating a simple dot-product model, no need for training")
        trained_model: TwoTowerModel = model
        training_results = None
    else:
        # --- train ---
        log_operation_start(f"Train two-tower (epochs={epochs}, batch_size={batch_size})", STAGE_LOG_NAME, logger)
        training_results = train_two_tower_model(
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
            lr_scheduler_factor=lr_scheduler_factor,
            lr_scheduler_patience=lr_scheduler_patience,
            gradient_clip_max_norm=gradient_clip_max_norm,
            embed_dim=embed_dim,
        )
        trained_model: TwoTowerModel = training_results["model"]
        clear_cuda_memory()

        # --- plots & evaluation ---
        hist = training_results["history"]

        # experiment tracker scalars (always logged, regardless of --no-plots)
        for e in range(len(hist["train_loss"])):
            context.tracker.log_scalar(title="Training Loss History", series="Train Loss", value=hist["train_loss"][e], iteration=e + 1)
            context.tracker.log_scalar(title="Training Loss History", series="Validation Loss", value=hist["val_loss"][e], iteration=e + 1)
            context.tracker.log_scalar(title="Training AUC History", series="Train AUC", value=hist["train_auc"][e], iteration=e + 1)
            context.tracker.log_scalar(title="Training AUC History", series="Validation AUC", value=hist["val_auc"][e], iteration=e + 1)

        if generate_plots:
            try:
                best_epoch = int(np.argmax(hist.get("val_auc", []))) + 1 if hist.get("val_auc") and len(hist.get("val_auc")) > 0 else None
            except Exception as e:
                logger.warning(f"Could not determine best epoch from training history: {e}")
                best_epoch = None
            plot_training_history(hist, plots_dir / f"training_history_{timestamp}.png", best_epoch=best_epoch)

    # Collect train + val predictions for performance plots & metrics
    train_eval = _evaluate_two_tower_model(trained_model, train_loader, device, embed_dim)
    val_eval = _evaluate_two_tower_model(trained_model, val_loader, device, embed_dim)
    logger.info(f"Train metrics: {train_eval['metrics']}")
    logger.info(f"Validation metrics: {val_eval['metrics']}")

    # calculate the best auc in the case that we didn't train a model (simple dot product version)
    if training_results is not None:
        best_val_auc = training_results["best_val_auc"]
    else:
        best_train_auc = roc_auc_score(train_eval["predictions"]["y_true"], train_eval["predictions"]["y_pred"])
        best_val_auc = roc_auc_score(val_eval["predictions"]["y_true"], val_eval["predictions"]["y_pred"])
        context.tracker.log_scalar(title="Training AUC History", series="Train AUC", value=float(best_train_auc), iteration=0)
        context.tracker.log_scalar(title="Training AUC History", series="Validation AUC", value=float(best_val_auc), iteration=0)

    if generate_plots:
        try:
            plot_model_performance(
                train_eval["predictions"]["y_true"],
                train_eval["predictions"]["y_pred"],
                plots_dir / f"train_performance_{timestamp}.png",
                title_suffix="(Train)",
            )
        except Exception as plot_exc:
            logger.warning(f"Train performance plotting failed: {plot_exc}")
        try:
            plot_model_performance(
                val_eval["predictions"]["y_true"],
                val_eval["predictions"]["y_pred"],
                plots_dir / f"val_performance_{timestamp}.png",
                title_suffix="(Validation)",
            )
        except Exception as plot_exc:
            logger.warning(f"Validation performance plotting failed: {plot_exc}")

    # --- save model ---
    model_path = None
    config = {
        "model_type": "two_tower",
        "user_encoder_type": user_encoder_type,
        "use_post_encoder": use_post_encoder,
        "post_embedding_dim": embed_dim,
        "shared_dim": shared_dim,
        "user_hidden_dim": user_hidden_dim,
        "post_hidden_dim": post_hidden_dim,
        "num_attention_heads": num_attention_heads,
        "num_attention_layers": num_attention_layers,
        "max_history_len": max_history_len,
        "dropout_rate": dropout_rate,
    }
    if save_model:
        model_path = checkpoints_dir / f"two_tower_{timestamp}.pth"
        torch.save(
            {
                "model_state_dict": trained_model.state_dict(),
                "config": config,
                "training_history": training_results["history"] if training_results is not None else None,
                "best_val_auc": best_val_auc,
                "best_val_loss": training_results["best_val_loss"] if training_results is not None else None,
            },
            model_path,
        )
        logger.info(f"Model saved to: {model_path}")

        # Save TorchScript file, which is the format needed for ClearML serving
        # Save the post and user towers separately
        torchscript_user_name = "engagement_user_tower"
        torchscript_user_path = checkpoints_dir / f"{torchscript_user_name}.pt"
        torch.jit.script(trained_model.user_tower.cpu()).save(torchscript_user_path)
        user_model_id = context.tracker.log_artifact(name=f"{torchscript_user_name}", path=torchscript_user_path)
        logger.info(f"User tower model id: {user_model_id}")

        torchscript_post_name = "engagement_post_tower"
        torchscript_post_path = checkpoints_dir / f"{torchscript_post_name}.pt"
        torch.jit.script(trained_model.post_tower.cpu()).save(torchscript_post_path)
        post_model_id = context.tracker.log_artifact(name=f"{torchscript_post_name}", path=torchscript_post_path)
        logger.info(f"Post tower model id: {post_model_id}")

    # --- save predictions ---
    predictions_dir = out_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    pl.DataFrame({
        "did": train_eval["predictions"]["user_id"],
        "post_id": train_eval["predictions"]["post_id"],
        "y_true": train_eval["predictions"]["y_true"],
        "y_pred_proba": train_eval["predictions"]["y_pred"],
    }).write_parquet(predictions_dir / "train.parquet")

    pl.DataFrame({
        "did": val_eval["predictions"]["user_id"],
        "post_id": val_eval["predictions"]["post_id"],
        "y_true": val_eval["predictions"]["y_true"],
        "y_pred_proba": val_eval["predictions"]["y_pred"],
    }).write_parquet(predictions_dir / "val.parquet")

    # --- holdout evaluation ---
    holdout_metrics: Dict[str, Any] = {}
    for holdout_type in ["unseen_users", "seen_users"]:
        split_name = f"holdout_{holdout_type}"
        try:
            if user_encoder_type == "summarized":
                holdout_dataset = SummarizedEngagementDataset(
                    embeddings_mmap, target_posts_df, history_df, split=split_name,
                    summarizer=summarizer, embed_dim=embed_dim, logger=logger,
                )
            else:
                holdout_dataset = SequenceEngagementDataset(
                    embeddings_mmap, target_posts_df, history_df, split=split_name,
                    max_history_len=max_history_len, embed_dim=embed_dim, logger=logger,
                )
            if len(holdout_dataset) == 0:
                logger.info(f"No rows for split '{split_name}', skipping.")
                continue
            log_operation_start(f"Holdout evaluation ({holdout_type})", STAGE_LOG_NAME, logger)
            _, _, holdout_loader = create_data_loaders(
                train_dataset, train_dataset, batch_size,  # train/val loaders unused here
                holdout_dataset=holdout_dataset,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
            holdout_eval = _evaluate_two_tower_model(trained_model, holdout_loader, device, embed_dim)
            split_metrics = holdout_eval["metrics"]
            logger.info(f"Holdout metrics ({holdout_type}): {split_metrics}")
            if holdout_type == eval_holdout_type:
                holdout_metrics = split_metrics

            pl.DataFrame({
                "did": holdout_eval["predictions"]["user_id"],
                "post_id": holdout_eval["predictions"]["post_id"],
                "y_true": holdout_eval["predictions"]["y_true"],
                "y_pred_proba": holdout_eval["predictions"]["y_pred"],
            }).write_parquet(predictions_dir / f"{split_name}.parquet")

            if generate_plots and holdout_type == eval_holdout_type:
                try:
                    plot_model_performance(
                        holdout_eval["predictions"]["y_true"],
                        holdout_eval["predictions"]["y_pred"],
                        plots_dir / f"holdout_performance_{timestamp}.png",
                        title_suffix="(Holdout)",
                    )
                except Exception as plot_exc:
                    logger.warning(f"Holdout performance plotting failed: {plot_exc}")
        except Exception as exc:
            logger.warning(f"Holdout evaluation ({holdout_type}) failed (non-fatal): {exc}")

    # --- training config ---
    training_config = {
        **config,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": patience,
        "random_seed": random_seed,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_metrics": train_eval["metrics"],
        "val_metrics": val_eval["metrics"],
        "holdout_metrics": holdout_metrics,
        "best_val_auc": best_val_auc,
    }
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    # --- stage info ---
    runtime = time.time() - t0
    info_lines = [
        f"stage: train_two_tower",
        f"timestamp: {timestamp}",
        f"runtime_seconds: {runtime:.2f}",
        f"settings: batch_size={batch_size}, lr={learning_rate}, epochs={epochs}, user_encoder={user_encoder_type}",
        f"train_samples: {len(train_dataset)}",
        f"val_samples: {len(val_dataset)}",
        f"best_val_auc: {best_val_auc:.4f}",
    ]
    if holdout_metrics.get("auc_roc"):
        info_lines.append(f"holdout_auc: {holdout_metrics['auc_roc']:.4f}")
    (out_dir / "stage_info.txt").write_text("\n".join(info_lines) + "\n")

    logger.info(f"Two-Tower training completed in {runtime:.2f}s")

    return {
        "output_dir": out_dir,
        "artifacts": {
            "model_path": str(model_path) if model_path else None,
            "training_config": str(out_dir / "training_config.json"),
        },
    }
