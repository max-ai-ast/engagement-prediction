#!/usr/bin/env python3

"""
Stage 4 (MLP): Train MLP engagement prediction models with flexible user-history representation.

This stage trains Multi-Layer Perceptron (MLP) models for binary engagement prediction
(will user engage with post?). It supports TWO modular approaches for representing
user engagement history:

═══════════════════════════════════════════════════════════════════════════════
APPROACH 1: MLP + HAND-CRAFTED SUMMARIZATION (SummarizedMLP)
═══════════════════════════════════════════════════════════════════════════════

Uses SummarizedEngagementDataset with pluggable summarization strategies (mean,
EMA, linear recency) to reduce variable-length history to fixed-size vectors.

Architecture:
    Input: [user_summary || post_embedding] concatenated vector
    Hidden: Stack of Linear -> BatchNorm -> GELU -> Dropout layers
    Output: Single sigmoid-activated probability

═══════════════════════════════════════════════════════════════════════════════
APPROACH 2: MLP + LEARNED ATTENTION ENCODER (AttentionMLP)
═══════════════════════════════════════════════════════════════════════════════

Uses SequenceEngagementDataset with TransformerDualPoolingEncoder (transformer self-attention)
to LEARN optimal history aggregation end-to-end.

Architecture:
    User "tower": TransformerDualPoolingEncoder(history_sequence) -> user_vector
    Concat: [user_vector || post_embedding]
    MLP head: Stack of Linear -> BatchNorm -> GELU -> Dropout -> sigmoid

═══════════════════════════════════════════════════════════════════════════════

Both models are trained with:
    - Binary cross-entropy loss
    - Balanced positive/negative sampling (1:1 ratio)
    - AdamW optimizer with learning rate scheduling
    - Early stopping based on validation loss
    - Comprehensive metrics tracking (loss, AUC, precision, recall)

Inputs (from prior pipeline stages):
    - embeddings_*.npy memmap from 01_get_data
    - target_posts_*.parquet from 02_target_posts
    - history_posts_*.parquet from 03_user_history

Outputs under <run_dir>/04_train/<timestamp>/:
    - checkpoints/engagement_model_*.pth (full checkpoint with training state)
    - checkpoints/engagement_model_*_weights.pth (model weights only)
    - plots/training_history_*.png (loss and AUC curves)
    - plots/{train,val,holdout}_performance_*.png (precision-recall, ROC curves)
    - logs/ (training logs)
    - training_config.json (hyperparameters and configuration)
    - stage_info.txt (pipeline metadata)
    - holdout_eval/metrics_overall.json (final test set metrics)
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.utils.data import DataLoader, Dataset

from utils.pipeline.core import new_stage_timestamp_dir, Context
from utils.helpers import (
    get_stage_logger,
    log_operation_start,
    get_device,
    plot_model_performance,
    clear_cuda_memory,
    set_random_seeds,
)
from utils.dataloaders import (
    load_training_data,
    get_summarizer,
    SummarizedEngagementDataset,
    SequenceEngagementDataset,
    sequence_collate_fn,
    TransformerDualPoolingEncoder,
)

STAGE_LOG_NAME = "STAGE_04_TRAIN_MLP"


# =============================================================================
# Model Architectures
# =============================================================================

class SummarizedMLP(nn.Module):
    """MLP engagement predictor using hand-crafted user history summarization.
    
    This is the simpler of two MLP architectures, accepting fixed-size concatenated
    [user_summary || post_embedding] vectors from SummarizedEngagementDataset.
    The user summary is computed by hand-crafted aggregation strategies (mean, EMA,
    linear recency) applied during dataset initialization.
    
    The name "SummarizedMLP" reflects that this model works with pre-computed
    user history SUMMARIES (via Summarizers like MeanSummarizer, EMASummarizer)
    rather than learned ENCODERS (like TransformerDualPoolingEncoder).
    
    Architecture:
        Input layer: Concatenated [user_summary || post_embedding] vector [2*D]
        Hidden layers: Configurable stack of (Linear -> BatchNorm -> GELU -> Dropout)
        Output layer: Linear -> Sigmoid producing engagement probability [0, 1]
    
    Design choices:
        - BatchNorm for stable training with varying input distributions
        - GELU activation for smooth gradients and better optimization
        - Dropout for regularization (prevents overfitting to training users)
        - Xavier initialization for faster convergence
    
    Args:
        input_dim: Dimension of concatenated input (2 * embedding_dim)
        hidden_dims: List of hidden layer sizes (e.g., [512, 256, 128])
        dropout_rate: Dropout probability applied after each hidden layer
    
    Example:
        >>> model = SummarizedMLP(input_dim=768, hidden_dims=[512, 256], dropout_rate=0.3)
        >>> # Input: [batch, 768] -> Hidden: [batch, 512] -> [batch, 256] -> Output: [batch, 1]
    """

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float):
        super().__init__()
        # Build sequential network: repeated (Linear -> BatchNorm -> GELU -> Dropout) blocks
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ])
            prev = h
        # Final layer: Linear -> Sigmoid for binary classification
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
        
        # Initialize weights for stable training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        print(f"model architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> 1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.
        
        Args:
            x: Input tensor [batch, input_dim]
        
        Returns:
            Engagement probabilities [batch] in range [0, 1]
        """
        return self.network(x)

    def compute_loss_and_preds(self, batch: Dict[str, torch.Tensor], device: str):
        """Compute loss and predictions for a batch.
        
        This method provides a unified interface for training, validation, and
        inference loops. It handles device transfer and loss computation.
        
        Args:
            batch: Dictionary with "features" and "label" keys from SummarizedEngagementDataset
            device: Device to run computation on ("cpu" or "cuda")
        
        Returns:
            Tuple of (loss, predictions):
                - loss: Scalar BCE loss tensor
                - predictions: Predicted probabilities [batch]
        """
        feats = batch["features"].to(device)
        labels = batch["label"].to(device)
        preds = self.forward(feats).squeeze(-1)
        loss = F.binary_cross_entropy(preds, labels)
        return loss, preds


class AttentionMLP(nn.Module):
    """MLP engagement predictor with learned attention-based user history encoder.
    
    This is the more sophisticated of two MLP architectures, using a trainable
    TransformerDualPoolingEncoder (transformer self-attention) to learn optimal history
    aggregation from raw embedding sequences. The encoder and MLP head are
    trained jointly end-to-end. NOTE: THIS MODEL DOES NOT ACTUALLY PERFORM BETTER YET -- SOMETHING TO WORK ON.
    
    Architecture:
        User encoder: TransformerDualPoolingEncoder(history_sequence, mask) -> user_vector [user_output_dim]
        Concatenation: [user_vector || post_embedding] -> [user_output_dim + embed_dim]
        MLP head: Stack of (Linear -> BatchNorm -> GELU -> Dropout) -> Linear -> Sigmoid
    
    Key differences from SummarizedMLP:
        ✓ LEARNS history aggregation via attention (not hand-crafted)
        ✓ Consumes SequenceEngagementDataset (not SummarizedEngagementDataset)
        ✓ Can capture complex temporal patterns and inter-post relationships
        ✗ More parameters to train (~2M for encoder + MLP)
        ✗ Slower training and inference
    
    When to use:
        - You have sufficient training data (>50K samples)
        - Hand-crafted summarizers underperform
        - You want the model to discover optimal aggregation
        - Computational resources allow transformer training
    
    Args:
        embed_dim: Dimensionality of post embeddings
        hidden_dims: MLP head hidden layer sizes
        dropout_rate: Dropout for MLP layers (not encoder, which has its own)
        user_hidden_dim: TransformerDualPoolingEncoder internal hidden dimension
        user_output_dim: TransformerDualPoolingEncoder output dimension (user vector size)
        num_attention_heads: Number of attention heads in transformer
        num_attention_layers: Number of transformer encoder layers
        max_history_len: Maximum sequence length for positional embeddings
        attention_dropout: Dropout rate for the TransformerDualPoolingEncoder
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dims: List[int],
        dropout_rate: float,
        user_hidden_dim: int,
        user_output_dim: int,
        num_attention_heads: int,
        num_attention_layers: int,
        max_history_len: int,
        attention_dropout: float,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.user_output_dim = user_output_dim

        # User "tower": Learned attention-based encoder over engagement history (can't actually be used separately like a true 2-tower)
        self.user_encoder = TransformerDualPoolingEncoder(
            input_dim=embed_dim,
            hidden_dim=user_hidden_dim,
            output_dim=user_output_dim,
            num_attention_heads=num_attention_heads,
            num_attention_layers=num_attention_layers,
            max_seq_len=max_history_len,
            dropout_rate=attention_dropout,
        )

        # MLP head: Processes concatenated [user_vec || post_emb] -> engagement probability
        mlp_input_dim = user_output_dim + embed_dim
        layers: List[nn.Module] = []
        prev = mlp_input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout_rate),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.mlp_head = nn.Sequential(*layers)

        # Initialize MLP head weights (user_encoder handles its own initialization)
        for m in self.mlp_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        print(
            f"AttentionMLP architecture: user_encoder({embed_dim}->{user_output_dim}) "
            f"+ post({embed_dim}) -> MLP({mlp_input_dim} -> "
            f"{' -> '.join(map(str, hidden_dims))} -> 1)"
        )

    def forward(
        self,
        history_embeddings: torch.Tensor,  # [B, seq_len, embed_dim]
        history_mask: torch.Tensor,         # [B, seq_len]
        post_embedding: torch.Tensor,       # [B, embed_dim]
    ) -> torch.Tensor:
        """Forward pass through user encoder and MLP head.
        
        Args:
            history_embeddings: Padded user history sequences [batch, seq_len, embed_dim]
            history_mask: Boolean mask [batch, seq_len], True = valid position
            post_embedding: Target post embeddings [batch, embed_dim]
        
        Returns:
            Engagement probabilities [batch] in range [0, 1]
        """
        # Encode user history into fixed-size representation
        user_vec = self.user_encoder(history_embeddings, history_mask)
        # Concatenate user and post representations
        x = torch.cat([user_vec, post_embedding], dim=-1)
        # Pass through MLP head
        return self.mlp_head(x)

    def compute_loss_and_preds(self, batch: Dict[str, torch.Tensor], device: str):
        """Compute loss and predictions for a batch.
        
        This method provides a unified interface for training, validation, and
        inference loops. It handles device transfer and loss computation.
        
        Args:
            batch: Dictionary from SequenceEngagementDataset with keys:
                   - "history_embeddings": [batch, seq_len, embed_dim]
                   - "history_mask": [batch, seq_len]
                   - "target_post_embedding": [batch, embed_dim]
                   - "label": [batch]
            device: Device to run computation on ("cpu" or "cuda")
        
        Returns:
            Tuple of (loss, predictions):
                - loss: Scalar BCE loss tensor
                - predictions: Predicted probabilities [batch]
        """
        history_emb = batch["history_embeddings"].to(device)
        history_mask = batch["history_mask"].to(device)
        target_emb = batch["target_post_embedding"].to(device)
        labels = batch["label"].to(device)
        preds = self.forward(history_emb, history_mask, target_emb).squeeze(-1)
        loss = F.binary_cross_entropy(preds, labels)
        return loss, preds


# =============================================================================
# Helper Functions
# =============================================================================

def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    holdout_dataset: Optional[Dataset] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    collate_fn = None,
):
    """Create PyTorch DataLoaders for training, validation, and optionally holdout sets.
    
    Configures efficient data loading with multi-worker parallelism and GPU pinning.
    
    DataLoader configuration rationale:
        - num_workers > 0: Parallel data loading prevents GPU starvation
        - pin_memory: Speeds up CPU->GPU transfer by using page-locked memory
        - persistent_workers: Avoids worker process respawn overhead between epochs
        - prefetch_factor: Workers pre-load batches to hide data loading latency
        - drop_last=True for training: Ensures consistent batch sizes for BatchNorm
        
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Number of samples per batch
        holdout_dataset: Optional holdout dataset for final evaluation
        num_workers: Number of parallel data loading workers (0 = main process only)
        pin_memory: Use pinned (page-locked) memory for faster GPU transfer
        persistent_workers: Keep workers alive between epochs
        prefetch_factor: Number of batches to prefetch per worker
        collate_fn: Optional custom collate function for batching (e.g., sequence_collate_fn)
    
    Returns:
        Tuple of (train_loader, val_loader, holdout_loader).
        holdout_loader is None if holdout_dataset is not provided.
    
    Note:
        With SummarizedEngagementDataset (pre-computed tensors), workers just do
        index lookups and collation, so even a few workers eliminate CPU bottlenecks.
        With SequenceEngagementDataset (on-the-fly memmap loading), workers pipeline
        the I/O to keep GPUs fed during training.
    """
    # Base worker configuration
    worker_kw: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    # Add worker-specific options only when using multiple workers
    if num_workers > 0:
        worker_kw.update(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    # Create DataLoaders with appropriate settings for each split
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Shuffle for stochastic training
        drop_last=True,  # Drop incomplete final batch for BatchNorm stability
        collate_fn=collate_fn,
        **worker_kw
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # No shuffle for validation (deterministic evaluation)
        collate_fn=collate_fn,
        **worker_kw
    )
    holdout_loader = DataLoader(
        holdout_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn,
        **worker_kw
    ) if holdout_dataset else None
    return train_loader, val_loader, holdout_loader


# =============================================================================
# Training loop
# =============================================================================

def train_mlp_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    lr_scheduler_factor: float,
    lr_scheduler_patience: int,
    model_name: str = "engagement_model",
    load_best_checkpoint: bool = False,
    checkpoints_dir: Optional[Path] = None,
    disable_progress: bool = False,
    gradient_clip_max_norm: float = 1.0,
) -> Dict[str, Any]:
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import torch.optim as optim

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=lr_scheduler_factor, patience=lr_scheduler_patience)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    best_val_auc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_dir = Path(checkpoints_dir) if checkpoints_dir is not None else (Path(__file__).resolve().parents[2] / "outputs" / "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm as _tqdm

    for epoch in _tqdm(range(epochs), desc="Training epochs", disable=disable_progress):
        model.train()
        train_loss = 0.0
        train_preds: List[float] = []
        train_labels: List[float] = []
        for batch in _tqdm(train_loader, desc="Training", leave=False, disable=disable_progress):
            optimizer.zero_grad()
            loss, preds = model.compute_loss_and_preds(batch, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_max_norm)
            optimizer.step()
            train_loss += loss.item()
            train_preds.extend(preds.detach().cpu().numpy().tolist())
            train_labels.extend(batch["label"].numpy().tolist())

        val_loss = 0.0
        val_preds: List[float] = []
        val_labels: List[float] = []
        model.eval()
        with torch.inference_mode():
            for batch in _tqdm(val_loader, desc="Validation", leave=False, disable=disable_progress):
                loss, preds = model.compute_loss_and_preds(batch, device)
                val_loss += loss.item()
                val_preds.extend(preds.detach().cpu().numpy().tolist())
                val_labels.extend(batch["label"].numpy().tolist())

        train_auc = roc_auc_score(train_labels, train_preds) if len(set(train_labels)) > 1 else 0.5
        val_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.5
        avg_train_loss = float(train_loss / max(1, len(train_loader)))
        avg_val_loss = float(val_loss / max(1, len(val_loader)))
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_auc"].append(float(train_auc))
        history["val_auc"].append(float(val_auc))
        scheduler.step(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = avg_val_loss
            checkpoint_full = checkpoint_dir / f"{model_name}_best.pth"
            checkpoint_weights = checkpoint_dir / f"{model_name}_best_weights.pth"
            history_clean = {k: [float(x) for x in v] for k, v in history.items()}
            torch.save(
                {"epoch": int(epoch), "model_state_dict": model.state_dict(), "val_loss": avg_val_loss, "val_auc": float(val_auc), "history": history_clean},
                checkpoint_full,
            )
            torch.save(model.state_dict(), checkpoint_weights)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if load_best_checkpoint:
        checkpoint_full = checkpoint_dir / f"{model_name}_best.pth"
        if checkpoint_full.exists():
            try:
                checkpoint = torch.load(checkpoint_full, weights_only=False)
                state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
                model.load_state_dict(state)
                if isinstance(checkpoint, dict) and "history" in checkpoint:
                    history = checkpoint["history"]
            except Exception as exc:
                # If loading the best checkpoint fails (e.g., file is missing or corrupted),
                # continue using the current in-memory model but emit a warning for visibility.
                print(f"Warning: failed to load best checkpoint from {checkpoint_full}: {exc}")

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_auc": best_val_auc,
    }

# =============================================================================
# Pipeline entry point
# =============================================================================

def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    device = get_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- output dirs ---
    run_tag = args.run_tag or ""
    out_dir = new_stage_timestamp_dir(run_dir, "04_train", tag=run_tag)
    checkpoints_dir = out_dir / "checkpoints"
    plots_dir = out_dir / "plots"
    logs_dir = out_dir / "logs"
    for d in (checkpoints_dir, plots_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger = get_stage_logger(STAGE_LOG_NAME, log_file=out_dir / "stage.log")
    log_operation_start("Stage 4 MLP training", STAGE_LOG_NAME, logger)
    t0 = time.time()

    # --- seeds & cuda ---
    clear_cuda_memory()
    random_seed = int(args.random_seed)
    set_random_seeds(random_seed)

    # --- load data from prior stages ---
    log_operation_start("Load training data from prior stages", STAGE_LOG_NAME, logger)
    embeddings_mmap, target_posts_df, history_df, embed_dim = load_training_data(
        run_dir, context, logger=logger,
    )

    # --- hyperparams (extract all args once, use locals everywhere below) ---
    user_encoder = args.user_encoder
    batch_size = int(args.batch_size)
    hidden_dims = list(args.hidden_dims)
    dropout_rate = float(args.dropout_rate_mlp)
    epochs = int(args.epochs)
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.weight_decay_mlp)
    patience = int(args.patience)
    disable_progress = bool(args.disable_progress)
    generate_plots = not bool(args.no_plots)
    save_model = not bool(args.no_save_model)
    lr_scheduler_factor = float(args.lr_scheduler_factor)
    lr_scheduler_patience = int(args.lr_scheduler_patience)
    gradient_clip_max_norm = float(args.gradient_clip_max_norm)
    collate_fn = None  # non-None only for sequence datasets

    # Worker settings (shared by all encoder types)
    num_workers = int(args.num_dataloader_workers)
    pin_memory = bool(args.dataloader_pin_memory)
    persistent_workers = bool(args.dataloader_persistent_workers)
    prefetch_factor = int(args.dataloader_prefetch_factor)

    if user_encoder == "summarized":
        # Classic MLP path: deterministic user summary + post embedding
        summarizer_name = args.user_summarization
        ema_alpha = float(args.ema_alpha)
        summarizer = get_summarizer(summarizer_name, ema_alpha=ema_alpha)
        logger.info(f"User encoder: summarized ({summarizer_name}, ema_alpha={ema_alpha})")

        log_operation_start("Create datasets (summarized)", STAGE_LOG_NAME, logger)
        train_dataset = SummarizedEngagementDataset(
            embeddings_mmap, target_posts_df, history_df, split="train",
            summarizer=summarizer, embed_dim=embed_dim, logger=logger,
        )
        val_dataset = SummarizedEngagementDataset(
            embeddings_mmap, target_posts_df, history_df, split="val",
            summarizer=summarizer, embed_dim=embed_dim, logger=logger,
        )
        input_dim = 2 * embed_dim  # [user_summary || post_embedding]
        model = SummarizedMLP(input_dim, hidden_dims, dropout_rate)
        
        train_loader, val_loader, _ = create_data_loaders(
            train_dataset, val_dataset, batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    elif user_encoder == "attention":
        # Attention MLP path: sequence dataset + AttentionMLP with learned encoder
        logger.info("User encoder: attention (TransformerDualPoolingEncoder + MLP)")
        max_history_len = int(args.max_history_len)
        user_hidden_dim = int(args.user_hidden_dim)
        user_output_dim = int(args.user_output_dim)
        num_attention_heads = int(args.num_attention_heads)
        num_attention_layers = int(args.num_attention_layers)
        attention_dropout = float(args.attention_dropout)
        summarizer_name = "attention"  # for config logging
        ema_alpha = 0.0

        log_operation_start("Create datasets (sequence)", STAGE_LOG_NAME, logger)
        train_dataset = SequenceEngagementDataset(
            embeddings_mmap, target_posts_df, history_df, split="train",
            max_history_len=max_history_len, embed_dim=embed_dim, logger=logger,
        )
        val_dataset = SequenceEngagementDataset(
            embeddings_mmap, target_posts_df, history_df, split="val",
            max_history_len=max_history_len, embed_dim=embed_dim, logger=logger,
        )

        # AttentionMLP uses user_output_dim (not shared_dim) since user vector is 
        # concatenated with post embedding, not compared via dot product like TwoTower
        model = AttentionMLP(
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            user_hidden_dim=user_hidden_dim,
            user_output_dim=user_output_dim,
            num_attention_heads=num_attention_heads,
            num_attention_layers=num_attention_layers,
            max_history_len=max_history_len,
            attention_dropout=attention_dropout,
        )

        collate_fn = sequence_collate_fn
        train_loader, val_loader, _ = create_data_loaders(
            train_dataset, val_dataset, batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=collate_fn,
        )
        input_dim = user_output_dim + embed_dim  # for config logging
    else:
        raise ValueError(
            f"Unknown user_encoder '{user_encoder}' for MLP. "
            "Choose 'summarized' or 'attention'."
        )

    # --- train ---
    log_operation_start(f"Training MLP (epochs={epochs}, batch_size={batch_size})", STAGE_LOG_NAME, logger)
    training_results = train_mlp_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        load_best_checkpoint=True,
        checkpoints_dir=checkpoints_dir,
        disable_progress=disable_progress,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        gradient_clip_max_norm=gradient_clip_max_norm,
    )
    trained_model: nn.Module = training_results["model"]
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
        log_operation_start("Generate plots", STAGE_LOG_NAME, logger)
        from utils.helpers import plot_training_history

        try:
            best_epoch = int(np.argmin(hist.get("val_loss", []))) + 1 if hist.get("val_loss") else None
        except Exception as e:
            logger.warning(f"Could not determine best epoch from training history: {e}")
            best_epoch = None
        plot_training_history(hist, plots_dir / f"training_history_{timestamp}.png", best_epoch=best_epoch)

    # Collect train + val predictions for performance plots & metrics
    def _collect_predictions(ds: Dataset, collate_fn_=None) -> tuple:
        loader_kw_: Dict[str, Any] = dict(
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        if num_workers > 0:
            loader_kw_.update(
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
            )
        if collate_fn_ is not None:
            loader_kw_["collate_fn"] = collate_fn_
        loader = DataLoader(ds, **loader_kw_)
        ys, ps = [], []
        trained_model.eval()
        with torch.inference_mode():
            for batch in loader:
                _, preds = trained_model.compute_loss_and_preds(batch, device)
                if preds.ndim == 0:
                    ps.append(float(preds.cpu()))
                    ys.append(float(batch["label"].cpu()))
                else:
                    ps.extend(preds.cpu().numpy().tolist())
                    ys.extend(batch["label"].numpy().tolist())
        return np.asarray(ys), np.asarray(ps)

    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        m: Dict[str, Any] = {"total_samples": len(y_true), "positive_samples": int(y_true.sum())}
        if len(set(y_true)) > 1:
            m["auc_roc"] = float(roc_auc_score(y_true, y_pred))
        m["accuracy@0.5"] = float(accuracy_score(y_true, (y_pred > 0.5).astype(int)))
        return m

    y_train, p_train = _collect_predictions(train_dataset, collate_fn_=collate_fn)
    y_val, p_val = _collect_predictions(val_dataset, collate_fn_=collate_fn)
    train_metrics = _compute_metrics(y_train, p_train)
    val_metrics = _compute_metrics(y_val, p_val)
    logger.info(f"Train metrics: {train_metrics}")
    logger.info(f"Validation metrics: {val_metrics}")

    if generate_plots:
        try:
            plot_model_performance(y_train, p_train, plots_dir / f"train_performance_{timestamp}.png", title_suffix="(Train)")
        except Exception as e:
            logger.warning(f"Train performance plotting failed: {e}")
        try:
            plot_model_performance(y_val, p_val, plots_dir / f"val_performance_{timestamp}.png", title_suffix="(Validation)")
        except Exception as e:
            logger.warning(f"Validation performance plotting failed: {e}")

    # --- save model ---
    model_path = None
    if save_model:
        log_operation_start("Save model checkpoint", STAGE_LOG_NAME, logger)
        model_path = checkpoints_dir / f"engagement_model_{timestamp}.pth"
        tr_sanitized = {k: v for k, v in training_results.items() if k != "model"}
        torch.save(
            {
                "model_state_dict": trained_model.state_dict(),
                "model_type": "mlp",
                "user_encoder": user_encoder,
                "input_dim": input_dim,
                "hidden_dims": hidden_dims,
                "dropout_rate": dropout_rate,
                "embed_dim": embed_dim,
                "user_summarization": summarizer_name,
                "ema_alpha": ema_alpha,
                "training_results": tr_sanitized,
                "training_parameters": {
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                    "patience": patience,
                },
                "data_info": {
                    "train_samples": len(train_dataset),
                    "val_samples": len(val_dataset),
                    "feature_dim": input_dim,
                },
            },
            model_path,
        )
        logger.info(f"Model saved to: {model_path}")
        context.tracker.log_artifact(name="trained_model_mlp", path=model_path)

    # --- holdout eval ---
    holdout_metrics: Dict[str, Any] = {}
    try:
        if user_encoder == "summarized":
            holdout_dataset = SummarizedEngagementDataset(
                embeddings_mmap, target_posts_df, history_df, split="holdout",
                summarizer=summarizer, embed_dim=embed_dim, logger=logger,
            )
        else:
            holdout_dataset = SequenceEngagementDataset(
                embeddings_mmap, target_posts_df, history_df, split="holdout",
                max_history_len=max_history_len, embed_dim=embed_dim, logger=logger,
            )
        if len(holdout_dataset) > 0:
            log_operation_start("Holdout evaluation", STAGE_LOG_NAME, logger)
            y_holdout, p_holdout = _collect_predictions(holdout_dataset, collate_fn_=collate_fn)
            holdout_metrics = _compute_metrics(y_holdout, p_holdout)
            logger.info(f"Holdout metrics: {holdout_metrics}")

            he_dir = out_dir / "holdout_eval"
            he_dir.mkdir(parents=True, exist_ok=True)
            with open(he_dir / "metrics_overall.json", "w") as f:
                json.dump(holdout_metrics, f, indent=2)

            if generate_plots:
                try:
                    plot_model_performance(
                        y_holdout, p_holdout,
                        plots_dir / f"holdout_performance_{timestamp}.png",
                        title_suffix="(Holdout)",
                    )
                except Exception as e:
                    logger.warning(f"Holdout performance plotting failed: {e}")
    except Exception as exc:
        logger.warning(f"Holdout evaluation failed (non-fatal): {exc}")

    # --- training config ---
    training_config = {
        "model_type": "mlp",
        "user_encoder": user_encoder,
        "embed_dim": embed_dim,
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "dropout_rate": dropout_rate,
        "user_summarization": summarizer_name,
        "ema_alpha": ema_alpha,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": patience,
        "random_seed": random_seed,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "holdout_metrics": holdout_metrics,
        "best_val_auc": training_results["best_val_auc"],
    }
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    # --- stage info ---
    runtime = time.time() - t0
    info_lines = [
        f"stage: train_mlp",
        f"timestamp: {timestamp}",
        f"runtime_seconds: {runtime:.2f}",
        f"settings: batch_size={batch_size}, lr={learning_rate}, epochs={epochs}, user_encoder={user_encoder}, summarizer={summarizer_name}",
        f"inputs: embeddings memmap, target_posts, user_history",
        f"train_samples: {len(train_dataset)}",
        f"val_samples: {len(val_dataset)}",
        f"best_val_auc: {training_results['best_val_auc']:.4f}",
    ]
    if holdout_metrics.get("auc_roc"):
        info_lines.append(f"holdout_auc: {holdout_metrics['auc_roc']:.4f}")
    (out_dir / "stage_info.txt").write_text("\n".join(info_lines) + "\n")

    logger.info(f"MLP training completed in {runtime:.2f}s")

    return {
        "output_dir": out_dir,
        "artifacts": {
            "model_path": str(model_path) if model_path else None,
            "training_config": str(out_dir / "training_config.json"),
        },
    }
