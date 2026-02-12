#!/usr/bin/env python3

"""
Stage 4 (MLP): Train an MLP engagement predictor using on-the-fly
user-history summarization backed by memmap embeddings.

Inputs (from prior pipeline stages):
- embeddings_*.npy memmap from 01_get_data
- target_posts_*.parquet from 02_target_posts
- history_posts_*.parquet from 03_user_history

Outputs under <run_dir>/04_train/<timestamp>/:
- checkpoints/engagement_model_*.pth
- plots/training_history_*.png, train_performance_*.png, val_performance_*.png, holdout_performance_*.png
- logs/
- training_config.json
- stage_info.txt
- holdout_eval/metrics_overall.json
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
from torch.utils.data import DataLoader, Dataset

from utils.pipeline.core import new_stage_timestamp_dir, Context
from utils.helpers import (
    get_stage_logger,
    log_operation_start,
    get_device,
    plot_model_performance,
)
from utils.dataloaders import (
    load_training_data,
    get_summarizer,
    SummarizedEngagementDataset,
    SequenceEngagementDataset,
    sequence_collate_fn,
    UserHistoryEncoder,
)

STAGE_LOG_NAME = "STAGE_04_TRAIN_MLP"


# =============================================================================
# Model
# =============================================================================

class EngagementPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout_rate: float):
        super().__init__()
        layers: List[nn.Module] = []
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


    def train_step(self, batch: Dict[str, torch.Tensor], device: str):
        """Unified training step: unpack batch, forward, compute loss.

        Returns (loss, predictions_tensor).
        """
        feats = batch["features"].to(device)
        labels = batch["label"].to(device)
        preds = self.forward(feats).squeeze()
        loss = F.binary_cross_entropy(preds, labels)
        return loss, preds


class AttentionMLP(nn.Module):
    """MLP engagement predictor with a learned attention encoder over user history.

    Wraps :class:`UserHistoryEncoder` (from ``utils.dataloaders``) to produce a
    fixed-size user vector, concatenates it with the target post embedding, and
    feeds the result through MLP layers.  The attention encoder is trained
    end-to-end as part of this model.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dims: List[int],
        dropout_rate: float,
        user_hidden_dim: int = 256,
        user_output_dim: int = 128,
        num_attention_heads: int = 4,
        num_attention_layers: int = 2,
        max_history_len: int = 50,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.user_output_dim = user_output_dim

        self.user_encoder = UserHistoryEncoder(
            input_dim=embed_dim,
            hidden_dim=user_hidden_dim,
            output_dim=user_output_dim,
            num_attention_heads=num_attention_heads,
            num_attention_layers=num_attention_layers,
            max_seq_len=max_history_len,
            dropout_rate=attention_dropout,
        )

        # MLP head: [user_vec || post_emb] -> binary prediction
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

        # Init MLP head weights (user_encoder does its own init)
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
        user_vec = self.user_encoder(history_embeddings, history_mask)
        x = torch.cat([user_vec, post_embedding], dim=-1)
        return self.mlp_head(x)

    def train_step(self, batch: Dict[str, torch.Tensor], device: str):
        """Unified training step: unpack sequence batch, forward, compute loss."""
        history_emb = batch["history_embeddings"].to(device)
        history_mask = batch["history_mask"].to(device)
        target_emb = batch["target_post_embedding"].to(device)
        labels = batch["label"].to(device)
        preds = self.forward(history_emb, history_mask, target_emb).squeeze()
        loss = F.binary_cross_entropy(preds, labels)
        return loss, preds


# =============================================================================
# Helpers
# =============================================================================

def create_model(input_dim: int, hidden_dims: List[int], dropout_rate: float) -> EngagementPredictor:
    return EngagementPredictor(input_dim, hidden_dims, dropout_rate)


def create_data_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    test_dataset: Optional[Dataset] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
):
    # With pre-computed tensors the workers just do index lookups + collation,
    # so even a few workers eliminate any remaining CPU-side bottleneck and
    # keep the GPU continuously fed.
    worker_kw: Dict[str, Any] = {}
    if num_workers > 0:
        worker_kw = dict(
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **worker_kw)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **worker_kw)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **worker_kw) if test_dataset else None
    return train_loader, val_loader, test_loader


def clear_cuda_memory():
    import gc
    gc.collect()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def set_random_seeds(seed: int):
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Training loop
# =============================================================================

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
    checkpoints_dir: Optional[Path] = None,
    disable_progress: bool = False,
    lr_scheduler_mode: str = "max",
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 5,
) -> Dict[str, Any]:
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import torch.optim as optim
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        def roc_auc_score(y_true, y_score):  # type: ignore[misc]
            return 0.5

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # Cast mode to satisfy type checker (value is validated by argparse choices)
    mode: Any = lr_scheduler_mode
    scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=lr_scheduler_factor, patience=lr_scheduler_patience)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    best_val_loss = float("inf")
    patience_counter = 0
    ckpt_dir = Path(checkpoints_dir) if checkpoints_dir is not None else (Path(__file__).resolve().parents[2] / "outputs" / "checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    from tqdm import tqdm as _tqdm

    for epoch in _tqdm(range(epochs), desc="Training epochs", disable=disable_progress):
        model.train()
        tr_loss = 0.0
        tr_preds: List[float] = []
        tr_labels: List[float] = []
        for batch in _tqdm(train_loader, desc="Training", leave=False, disable=disable_progress):
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss, preds = model.train_step(batch, device)
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
                labels = batch["label"].to(device)
                loss, preds = model.train_step(batch, device)
                val_loss += loss.item()
                val_preds.extend(preds.detach().cpu().numpy().tolist())
                val_labels.extend(labels.detach().cpu().numpy().tolist())

        tr_auc = roc_auc_score(tr_labels, tr_preds) if len(set(tr_labels)) > 1 else 0.5
        va_auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else 0.5
        history["train_loss"].append(float(tr_loss / max(1, len(train_loader))))
        history["val_loss"].append(float(val_loss / max(1, len(val_loader))))
        history["train_auc"].append(float(tr_auc))
        history["val_auc"].append(float(va_auc))
        scheduler.step(va_auc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_full = ckpt_dir / f"{model_name}_best.pth"
            ckpt_weights = ckpt_dir / f"{model_name}_best_weights.pth"
            history_clean = {k: [float(x) for x in v] for k, v in history.items()}
            torch.save(
                {"epoch": int(epoch), "model_state_dict": model.state_dict(), "val_loss": float(val_loss), "val_auc": float(va_auc), "history": history_clean},
                ckpt_full,
            )
            torch.save(model.state_dict(), ckpt_weights)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if load_best_checkpoint:
        ckpt_full = ckpt_dir / f"{model_name}_best.pth"
        if ckpt_full.exists():
            try:
                ckpt = torch.load(ckpt_full, weights_only=False)
                state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
                model.load_state_dict(state)
                if isinstance(ckpt, dict) and "history" in ckpt:
                    history = ckpt["history"]
            except Exception:
                pass

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_val_auc": max(history["val_auc"]) if history["val_auc"] else 0.0,
    }


# =============================================================================
# Lightweight holdout evaluation
# =============================================================================

# =============================================================================
# Pipeline entry point
# =============================================================================

def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    device = get_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- output dirs ---
    run_tag = getattr(args, "run_tag", "") or ""
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
    set_random_seeds(int(args.random_seed))

    # --- load data from prior stages ---
    log_operation_start("Load training data from prior stages", STAGE_LOG_NAME, logger)
    embeddings_mmap, target_posts_df, history_df, embed_dim = load_training_data(
        run_dir, context, logger=logger,
    )

    # --- user encoder selection ---
    user_encoder = getattr(args, "user_encoder", "summarized")
    batch_size = int(args.batch_size)
    hidden_dims = list(args.hidden_dims)
    dropout_rate = float(args.dropout_rate_mlp)
    collate_fn = None  # non-None only for sequence datasets
    
    # Get worker settings from args (shared by all encoder types)
    num_workers = int(getattr(args, "num_dataloader_workers", 4))
    pin_memory = bool(getattr(args, "dataloader_pin_memory", True))
    persistent_workers = bool(getattr(args, "dataloader_persistent_workers", True))
    prefetch_factor = int(getattr(args, "dataloader_prefetch_factor", 2))

    if user_encoder == "summarized":
        # Classic MLP path: deterministic user summary + post embedding
        summarizer_name = getattr(args, "user_summarization", "mean")
        ema_alpha = float(getattr(args, "ema_alpha", 0.1))
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
        model = create_model(input_dim, hidden_dims, dropout_rate)
        
        train_loader, val_loader, _ = create_data_loaders(
            train_dataset, val_dataset, batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    elif user_encoder == "attention":
        # Attention MLP path: sequence dataset + AttentionMLP with learned encoder
        logger.info("User encoder: attention (UserHistoryEncoder + MLP)")
        max_history_len = int(args.max_history_len)
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

        model = AttentionMLP(
            embed_dim=embed_dim,
            hidden_dims=hidden_dims,
            dropout_rate=dropout_rate,
            user_hidden_dim=int(args.user_hidden_dim),
            user_output_dim=int(args.shared_dim),
            num_attention_heads=int(args.num_attention_heads),
            num_attention_layers=int(args.num_attention_layers),
            max_history_len=max_history_len,
            attention_dropout=float(args.dropout_rate_two_tower),
        )

        collate_fn = sequence_collate_fn
        _worker_kw: Dict[str, Any] = dict(
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, drop_last=True, **_worker_kw,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, **_worker_kw,
        )
        input_dim = int(args.shared_dim) + embed_dim  # for config logging
    else:
        raise ValueError(
            f"Unknown user_encoder '{user_encoder}' for MLP. "
            "Choose 'summarized' or 'attention'."
        )

    # --- train ---
    log_operation_start(f"Training MLP (epochs={args.epochs}, batch_size={batch_size})", STAGE_LOG_NAME, logger)
    disable_progress = bool(getattr(args, "disable_progress", False))
    
    # Get scheduler settings from args
    lr_scheduler_mode = str(getattr(args, "lr_scheduler_mode", "max"))
    lr_scheduler_factor = float(getattr(args, "lr_scheduler_factor", 0.5))
    lr_scheduler_patience = int(getattr(args, "lr_scheduler_patience", 5))
    
    training_results = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay_mlp),
        patience=int(args.patience),
        load_best_checkpoint=True,
        checkpoints_dir=checkpoints_dir,
        disable_progress=disable_progress,
        lr_scheduler_mode=lr_scheduler_mode,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
    )
    trained_model: nn.Module = training_results["model"]
    clear_cuda_memory()

    # --- plots & evaluation ---
    generate_plots = not bool(args.no_plots)
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
        except Exception:
            best_epoch = None
        plot_training_history(hist, plots_dir / f"training_history_{timestamp}.png", best_epoch=best_epoch)

    # Collect train + val predictions for performance plots & metrics
    def _collect_predictions(ds: Dataset, collate_fn_=None) -> tuple:
        loader_kw_ = dict(
            batch_size=batch_size, shuffle=False, drop_last=False,
            num_workers=num_workers, pin_memory=pin_memory, 
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
        )
        if collate_fn_ is not None:
            loader_kw_["collate_fn"] = collate_fn_
        loader = DataLoader(ds, **loader_kw_)
        ys, ps = [], []
        trained_model.eval()
        with torch.inference_mode():
            for batch in loader:
                _, preds = trained_model.train_step(batch, device)
                if preds.ndim == 0:
                    ps.append(float(preds.cpu()))
                    ys.append(float(batch["label"].cpu()))
                else:
                    ps.extend(preds.cpu().numpy().tolist())
                    ys.extend(batch["label"].numpy().tolist())
        return np.asarray(ys), np.asarray(ps)

    try:
        from sklearn.metrics import roc_auc_score, accuracy_score
        _have_sklearn = True
    except ImportError:
        _have_sklearn = False

    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        m: Dict[str, Any] = {"total_samples": len(y_true), "positive_samples": int(y_true.sum())}
        if _have_sklearn and len(set(y_true)) > 1:
            m["auc_roc"] = float(roc_auc_score(y_true, y_pred))
        if _have_sklearn:
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
        except Exception:
            pass
        try:
            plot_model_performance(y_val, p_val, plots_dir / f"val_performance_{timestamp}.png", title_suffix="(Validation)")
        except Exception:
            pass

    # --- save model ---
    model_path = None
    if not bool(args.no_save_model):
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
                    "learning_rate": float(args.learning_rate),
                    "weight_decay": float(args.weight_decay_mlp),
                    "epochs": int(args.epochs),
                    "patience": int(args.patience),
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
                max_history_len=int(args.max_history_len), embed_dim=embed_dim, logger=logger,
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
                except Exception:
                    pass
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
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay_mlp),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "random_seed": int(args.random_seed),
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
        f"settings: batch_size={batch_size}, lr={args.learning_rate}, epochs={args.epochs}, user_encoder={user_encoder}, summarizer={summarizer_name}",
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
