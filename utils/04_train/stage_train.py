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
- plots/training_history_*.png, train_model_performance_*.png, val_model_performance_*.png
- logs/
- training_config.json
- stage_info.txt
- holdout_eval/metrics_overall.json  (lightweight)
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
):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) if test_dataset else None
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
) -> Dict[str, Any]:
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    import torch.optim as optim
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        def roc_auc_score(y_true, y_score):  # type: ignore[misc]
            return 0.5

    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
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
            feats = batch["features"].to(device)
            labels = batch["label"].to(device)
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
                feats = batch["features"].to(device)
                labels = batch["label"].to(device)
                preds = model(feats).squeeze()
                loss = criterion(preds, labels)
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

def _holdout_auc(
    model: nn.Module,
    holdout_dataset: SummarizedEngagementDataset,
    device: str,
    batch_size: int,
) -> Dict[str, Any]:
    """Compute AUC + accuracy on the holdout split (best-effort)."""
    try:
        from sklearn.metrics import roc_auc_score, accuracy_score
    except ImportError:
        return {"note": "sklearn not available"}

    if len(holdout_dataset) == 0:
        return {"note": "no holdout samples"}

    loader = DataLoader(holdout_dataset, batch_size=batch_size, shuffle=False)
    all_preds: List[float] = []
    all_labels: List[float] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            feats = batch["features"].to(device)
            preds = model(feats).squeeze()
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch["label"].numpy().tolist())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    metrics: Dict[str, Any] = {"total_samples": len(y_true), "positive": int(y_true.sum())}
    if len(set(y_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred))
    metrics["accuracy@0.5"] = float(accuracy_score(y_true, (y_pred > 0.5).astype(int)))
    return metrics


# =============================================================================
# Pipeline entry point
# =============================================================================

def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    device = get_device(args.device)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- output dirs ---
    out_dir = new_stage_timestamp_dir(run_dir, "04_train")
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

    # --- summarizer ---
    summarizer_name = getattr(args, "user_summarization", "mean")
    ema_alpha = float(getattr(args, "ema_alpha", 0.1))
    summarizer = get_summarizer(summarizer_name, ema_alpha=ema_alpha)
    logger.info(f"User summarization: {summarizer_name} (ema_alpha={ema_alpha})")

    # --- datasets ---
    log_operation_start("Create datasets", STAGE_LOG_NAME, logger)
    train_dataset = SummarizedEngagementDataset(
        embeddings_mmap, target_posts_df, history_df, split="train",
        summarizer=summarizer, embed_dim=embed_dim, logger=logger,
    )
    val_dataset = SummarizedEngagementDataset(
        embeddings_mmap, target_posts_df, history_df, split="val",
        summarizer=summarizer, embed_dim=embed_dim, logger=logger,
    )

    input_dim = 2 * embed_dim  # [user_summary || post_embedding]
    batch_size = int(args.batch_size)
    hidden_dims = list(args.hidden_dims)
    dropout_rate = float(args.dropout_rate_mlp)

    model = create_model(input_dim, hidden_dims, dropout_rate)
    train_loader, val_loader, _ = create_data_loaders(train_dataset, val_dataset, batch_size)

    # --- train ---
    log_operation_start(f"Training MLP (epochs={args.epochs}, batch_size={batch_size})", STAGE_LOG_NAME, logger)
    disable_progress = bool(getattr(args, "disable_progress", False))
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
    )
    trained_model: nn.Module = training_results["model"]
    clear_cuda_memory()

    # --- plots ---
    generate_plots = not bool(args.no_plots)
    if generate_plots:
        log_operation_start("Generate plots", STAGE_LOG_NAME, logger)
        from utils.helpers import plot_training_history

        hist = training_results["history"]
        try:
            best_epoch = int(np.argmin(hist.get("val_loss", []))) + 1 if hist.get("val_loss") else None
        except Exception:
            best_epoch = None
        plot_training_history(hist, plots_dir / f"training_history_{timestamp}.png", best_epoch=best_epoch)

        # experiment tracker scalars
        for e in range(len(hist["train_loss"])):
            context.tracker.log_scalar(title="Training Loss History", series="Train Loss", value=hist["train_loss"][e], iteration=e + 1)
            context.tracker.log_scalar(title="Training Loss History", series="Validation Loss", value=hist["val_loss"][e], iteration=e + 1)
            context.tracker.log_scalar(title="Training AUC History", series="Train AUC", value=hist["train_auc"][e], iteration=e + 1)
            context.tracker.log_scalar(title="Training AUC History", series="Validation AUC", value=hist["val_auc"][e], iteration=e + 1)

        # performance plots (train + val)
        try:
            trained_model.eval()

            def _collect_predictions(ds: Dataset) -> tuple:
                loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
                ys, ps = [], []
                with torch.inference_mode():
                    for batch in loader:
                        feats = batch["features"].to(device)
                        preds = trained_model(feats).squeeze()
                        if preds.ndim == 0:
                            ps.append(float(preds.cpu()))
                            ys.append(float(batch["label"].cpu()))
                        else:
                            ps.extend(preds.cpu().numpy().tolist())
                            ys.extend(batch["label"].numpy().tolist())
                return np.asarray(ys), np.asarray(ps)

            y_train, p_train = _collect_predictions(train_dataset)
            plot_model_performance(y_train, p_train, plots_dir / f"train_model_performance_{timestamp}.png")
            y_val, p_val = _collect_predictions(val_dataset)
            plot_model_performance(y_val, p_val, plots_dir / f"val_model_performance_{timestamp}.png")
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

    # --- lightweight holdout eval ---
    holdout_metrics: Dict[str, Any] = {}
    try:
        holdout_dataset = SummarizedEngagementDataset(
            embeddings_mmap, target_posts_df, history_df, split="holdout",
            summarizer=summarizer, embed_dim=embed_dim, logger=logger,
        )
        if len(holdout_dataset) > 0:
            log_operation_start("Holdout evaluation", STAGE_LOG_NAME, logger)
            holdout_metrics = _holdout_auc(trained_model, holdout_dataset, device, batch_size)
            logger.info(f"Holdout metrics: {holdout_metrics}")
            he_dir = out_dir / "holdout_eval"
            he_dir.mkdir(parents=True, exist_ok=True)
            with open(he_dir / "metrics_overall.json", "w") as f:
                json.dump(holdout_metrics, f, indent=2)
    except Exception as exc:
        logger.warning(f"Holdout evaluation failed (non-fatal): {exc}")

    # --- training config ---
    training_config = {
        "model_type": "mlp",
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
        "best_val_auc": training_results["best_val_auc"],
        "holdout_metrics": holdout_metrics,
    }
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    # --- stage info ---
    runtime = time.time() - t0
    info_lines = [
        f"stage: train_mlp",
        f"runtime_seconds: {runtime:.2f}",
        f"settings: batch_size={batch_size}, lr={args.learning_rate}, epochs={args.epochs}, summarizer={summarizer_name}",
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
        },
    }
