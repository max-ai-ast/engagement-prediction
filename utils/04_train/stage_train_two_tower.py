#!/usr/bin/env python3

"""
Stage 4 (Two-Tower): Train a two-tower engagement prediction model using
on-the-fly embedding sequence retrieval from memmap.

User Tower: Self-attention encoder over liked-post history sequence
Post Tower: MLP projection of post embeddings to shared space
Training: BCE loss with explicit positive/negative pairs

Inputs (from prior pipeline stages):
- embeddings_*.npy memmap from 01_get_data
- target_posts_*.parquet from 02_target_posts
- history_posts_*.parquet from 03_user_history

Outputs under <run_dir>/04_train/<timestamp>/:
- checkpoints/two_tower_*.pth
- plots/training_history_*.png, train_performance_*.png, val_performance_*.png, holdout_performance_*.png
- logs/
- training_config.json
- stage_info.txt
- holdout_eval/metrics_overall.json, predictions.parquet
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from utils.pipeline.core import new_stage_timestamp_dir, Context
from utils.helpers import get_stage_logger, log_operation_start, get_device, plot_model_performance
from utils.dataloaders import (
    load_training_data,
    SequenceEngagementDataset,
    sequence_collate_fn,
    UserHistoryEncoder,
    LightweightAttentionEncoder,
)

STAGE_LOG_NAME = "STAGE_04_TRAIN_TWO_TOWER"


# =============================================================================
# Post Tower
# =============================================================================

class PostTower(nn.Module):
    """Projects post embeddings to shared embedding space."""

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
        return self.network(post_embeddings)


# =============================================================================
# Two-Tower Engagement Model
# =============================================================================

class TwoTowerEngagement(nn.Module):
    """Two-tower model: user tower (attention or lightweight encoder) + post tower (MLP)."""

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
        user_encoder_type: str = "attention",
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.post_embedding_dim = post_embedding_dim
        self.user_encoder_type = user_encoder_type

        if user_encoder_type == "lightweight":
            self.user_tower = LightweightAttentionEncoder(
                input_dim=post_embedding_dim,
                hidden_dim=user_hidden_dim,
                output_dim=shared_dim,
                max_seq_len=max_history_len,
                dropout_rate=dropout_rate,
            )
        elif user_encoder_type == "attention":
            self.user_tower = UserHistoryEncoder(
                input_dim=post_embedding_dim,
                hidden_dim=user_hidden_dim,
                output_dim=shared_dim,
                num_attention_heads=num_attention_heads,
                num_attention_layers=num_attention_layers,
                max_seq_len=max_history_len,
                dropout_rate=dropout_rate,
            )
        else:
            raise ValueError(
                f"Unknown user_encoder_type '{user_encoder_type}'. "
                "Choose 'attention' or 'lightweight'."
            )

        self.post_tower = PostTower(
            input_dim=post_embedding_dim,
            hidden_dim=post_hidden_dim,
            output_dim=shared_dim,
            dropout_rate=dropout_rate,
        )

    def encode_user(self, history_embeddings: torch.Tensor, history_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.user_tower(history_embeddings, history_mask)

    def encode_post(self, post_embeddings: torch.Tensor) -> torch.Tensor:
        return self.post_tower(post_embeddings)

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: Optional[torch.Tensor],
        post_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        user_emb = self.encode_user(history_embeddings, history_mask)
        post_emb = self.encode_post(post_embeddings)
        return (user_emb * post_emb).sum(dim=-1)

    def train_forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: Optional[torch.Tensor],
        post_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        user_emb = self.encode_user(history_embeddings, history_mask)
        post_emb = self.encode_post(post_embeddings)
        scores = (user_emb * post_emb).sum(dim=-1)
        probs = torch.sigmoid(scores)
        loss = F.binary_cross_entropy(probs, labels.float())
        return loss, scores


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
    lr_scheduler_mode: str = "max",
    lr_scheduler_factor: float = 0.5,
    lr_scheduler_patience: int = 5,
    gradient_clip_max_norm: float = 1.0,
) -> Dict[str, Any]:
    from tqdm import tqdm

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=lr_scheduler_mode, factor=lr_scheduler_factor, patience=lr_scheduler_patience
    )

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "train_auc": [], "val_auc": []}
    best_val_auc = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        def roc_auc_score(y_true, y_score):  # type: ignore[misc]
            return 0.5

    for epoch in tqdm(range(epochs), desc="Training epochs", disable=disable_progress):
        # --- Training ---
        model.train()
        train_losses: List[float] = []
        train_preds: List[float] = []
        train_labels: List[float] = []

        for batch in tqdm(train_loader, desc="Training", leave=False, disable=disable_progress):
            history_emb = batch["history_embeddings"].to(device)
            history_mask = batch["history_mask"].to(device)
            target_emb = batch["target_post_embedding"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            loss, scores = model.train_forward(history_emb, history_mask, target_emb, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_max_norm)
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(torch.sigmoid(scores).detach().cpu().numpy().tolist())
            train_labels.extend(labels.detach().cpu().numpy().tolist())

        # --- Validation ---
        model.eval()
        val_losses: List[float] = []
        val_preds: List[float] = []
        val_labels: List[float] = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", leave=False, disable=disable_progress):
                history_emb = batch["history_embeddings"].to(device)
                history_mask = batch["history_mask"].to(device)
                target_emb = batch["target_post_embedding"].to(device)
                labels = batch["label"].to(device)

                loss, scores = model.train_forward(history_emb, history_mask, target_emb, labels)

                val_losses.append(loss.item())
                val_preds.extend(torch.sigmoid(scores).detach().cpu().numpy().tolist())
                val_labels.extend(labels.detach().cpu().numpy().tolist())

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

def evaluate_model(
    model: TwoTowerEngagement,
    data_loader: DataLoader,
    device: str,
) -> Dict[str, Any]:
    """Evaluate model and return metrics + predictions."""
    try:
        from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
    except ImportError:
        def roc_auc_score(y_true, y_score):  # type: ignore[misc]
            return 0.5
        accuracy_score = None  # type: ignore[assignment]
        average_precision_score = None  # type: ignore[assignment]

    model = model.to(device)
    model.eval()

    all_preds: List[float] = []
    all_labels: List[float] = []
    all_user_ids: List[str] = []
    all_post_ids: List[str] = []

    with torch.no_grad():
        for batch in data_loader:
            history_emb = batch["history_embeddings"].to(device)
            history_mask = batch["history_mask"].to(device)
            target_emb = batch["target_post_embedding"].to(device)
            labels = batch["label"]

            scores = model(history_emb, history_mask, target_emb)
            probs = torch.sigmoid(scores)

            all_preds.extend(probs.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
            all_user_ids.extend(batch["user_ids"])
            all_post_ids.extend(batch["post_ids"])

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    metrics: Dict[str, Any] = {
        "total_samples": len(y_true),
        "positive_samples": int(y_true.sum()),
        "negative_samples": int(len(y_true) - y_true.sum()),
    }

    if len(set(y_true)) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred))
        if average_precision_score is not None:
            metrics["average_precision"] = float(average_precision_score(y_true, y_pred))

    if accuracy_score is not None:
        metrics["accuracy_at_0.5"] = float(accuracy_score(y_true, (y_pred > 0.5).astype(int)))

    return {
        "metrics": metrics,
        "predictions": {
            "user_ids": all_user_ids,
            "post_ids": all_post_ids,
            "y_true": y_true,
            "y_pred": y_pred,
        },
    }


# =============================================================================
# Plotting
# =============================================================================

def plot_training_history(history: Dict[str, List[float]], save_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax1.set_title("Two-Tower - Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_auc"], "b-", label="Train AUC", linewidth=2)
    ax2.plot(epochs, history["val_auc"], "r-", label="Val AUC", linewidth=2)
    ax2.set_title("Two-Tower - AUC")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("AUC")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# Pipeline entry point
# =============================================================================

def run(context: Context, args) -> Dict[str, Any]:
    """Pipeline entry point for two-tower training."""
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
    log_operation_start("Stage 4 Two-Tower training", STAGE_LOG_NAME, logger)
    t0 = time.time()

    # --- seeds ---
    random_seed = int(args.random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # --- load data from prior stages ---
    log_operation_start("Load training data from prior stages", STAGE_LOG_NAME, logger)
    embeddings_mmap, target_posts_df, history_df, embed_dim = load_training_data(
        run_dir, context, logger=logger,
    )

    # --- hyperparams ---
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
    disable_progress = bool(getattr(args, "disable_progress", False))

    # Get worker settings from args
    num_workers = int(getattr(args, "num_dataloader_workers", 4))
    pin_memory = bool(getattr(args, "dataloader_pin_memory", True))
    persistent_workers = bool(getattr(args, "dataloader_persistent_workers", True))
    prefetch_factor = int(getattr(args, "dataloader_prefetch_factor", 2))

    # --- datasets ---
    log_operation_start("Create datasets", STAGE_LOG_NAME, logger)
    train_dataset = SequenceEngagementDataset(
        embeddings_mmap, target_posts_df, history_df, split="train",
        max_history_len=max_history_len, embed_dim=embed_dim, logger=logger,
    )
    val_dataset = SequenceEngagementDataset(
        embeddings_mmap, target_posts_df, history_df, split="val",
        max_history_len=max_history_len, embed_dim=embed_dim, logger=logger,
    )

    # With pre-computed tensors, workers just do index lookups + collation.
    _worker_kw: Dict[str, Any] = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=sequence_collate_fn, drop_last=True, **_worker_kw,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=sequence_collate_fn, **_worker_kw,
    )

    logger.info(f"Post embedding dim: {embed_dim}")
    logger.info(f"Train items: {len(train_dataset)}, Val items: {len(val_dataset)}")

    # --- create model ---
    user_encoder_type = getattr(args, "user_encoder", "attention")
    log_operation_start(f"Create two-tower model (user_encoder={user_encoder_type})", STAGE_LOG_NAME, logger)
    model = TwoTowerEngagement(
        post_embedding_dim=embed_dim,
        shared_dim=shared_dim,
        user_hidden_dim=user_hidden_dim,
        post_hidden_dim=post_hidden_dim,
        num_attention_heads=num_attention_heads,
        num_attention_layers=num_attention_layers,
        max_history_len=max_history_len,
        dropout_rate=dropout_rate,
        user_encoder_type=user_encoder_type,
    )

    # --- train ---
    log_operation_start(f"Train two-tower (epochs={epochs}, batch_size={batch_size})", STAGE_LOG_NAME, logger)
    
    # Get scheduler and training optimization settings from args
    lr_scheduler_mode = str(getattr(args, "lr_scheduler_mode", "max"))
    lr_scheduler_factor = float(getattr(args, "lr_scheduler_factor", 0.5))
    lr_scheduler_patience = int(getattr(args, "lr_scheduler_patience", 5))
    gradient_clip_max_norm = float(getattr(args, "gradient_clip_max_norm", 1.0))
    
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
        lr_scheduler_mode=lr_scheduler_mode,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        gradient_clip_max_norm=gradient_clip_max_norm,
    )
    trained_model: TwoTowerEngagement = training_result["model"]

    # --- plots & evaluation ---
    generate_plots = not bool(getattr(args, "no_plots", False))

    hist = training_result["history"]

    # experiment tracker scalars (always logged, regardless of --no-plots)
    for e in range(len(hist["train_loss"])):
        context.tracker.log_scalar(title="Training Loss History", series="Train Loss", value=hist["train_loss"][e], iteration=e + 1)
        context.tracker.log_scalar(title="Training Loss History", series="Validation Loss", value=hist["val_loss"][e], iteration=e + 1)
        context.tracker.log_scalar(title="Training AUC History", series="Train AUC", value=hist["train_auc"][e], iteration=e + 1)
        context.tracker.log_scalar(title="Training AUC History", series="Validation AUC", value=hist["val_auc"][e], iteration=e + 1)

    if generate_plots:
        plot_training_history(hist, plots_dir / f"training_history_{timestamp}.png")

    # Collect train + val predictions for performance plots & metrics
    train_eval = evaluate_model(trained_model, train_loader, device)
    val_eval = evaluate_model(trained_model, val_loader, device)
    logger.info(f"Train metrics: {train_eval['metrics']}")
    logger.info(f"Validation metrics: {val_eval['metrics']}")

    if generate_plots:
        try:
            plot_model_performance(
                train_eval["predictions"]["y_true"],
                train_eval["predictions"]["y_pred"],
                plots_dir / f"train_performance_{timestamp}.png",
                title_suffix="(Train)",
            )
        except Exception:
            pass
        try:
            plot_model_performance(
                val_eval["predictions"]["y_true"],
                val_eval["predictions"]["y_pred"],
                plots_dir / f"val_performance_{timestamp}.png",
                title_suffix="(Validation)",
            )
        except Exception:
            pass

    # --- save model ---
    model_path = checkpoints_dir / f"two_tower_{timestamp}.pth"
    config = {
        "model_type": "two_tower",
        "user_encoder_type": user_encoder_type,
        "post_embedding_dim": embed_dim,
        "shared_dim": shared_dim,
        "user_hidden_dim": user_hidden_dim,
        "post_hidden_dim": post_hidden_dim,
        "num_attention_heads": num_attention_heads,
        "num_attention_layers": num_attention_layers,
        "max_history_len": max_history_len,
        "dropout_rate": dropout_rate,
    }
    torch.save(
        {
            "model_state_dict": trained_model.state_dict(),
            "config": config,
            "training_history": training_result["history"],
            "best_val_auc": training_result["best_val_auc"],
            "best_val_loss": training_result["best_val_loss"],
        },
        model_path,
    )
    logger.info(f"Model saved to: {model_path}")
    context.tracker.log_artifact(name="trained_model_two_tower", path=model_path)

    # --- holdout evaluation ---
    holdout_metrics: Dict[str, Any] = {}
    holdout_dir = out_dir / "holdout_eval"
    try:
        holdout_dataset = SequenceEngagementDataset(
            embeddings_mmap, target_posts_df, history_df, split="holdout",
            max_history_len=max_history_len, embed_dim=embed_dim, logger=logger,
        )
        if len(holdout_dataset) > 0:
            log_operation_start("Holdout evaluation", STAGE_LOG_NAME, logger)
            holdout_loader = DataLoader(
                holdout_dataset, batch_size=batch_size, shuffle=False,
                collate_fn=sequence_collate_fn, **_worker_kw,
            )
            holdout_eval = evaluate_model(trained_model, holdout_loader, device)
            holdout_metrics = holdout_eval["metrics"]
            logger.info(f"Holdout metrics: {holdout_metrics}")

            holdout_dir.mkdir(parents=True, exist_ok=True)

            # Save predictions
            import pandas as pd
            pred_df = pd.DataFrame({
                "did": holdout_eval["predictions"]["user_ids"],
                "post_id": holdout_eval["predictions"]["post_ids"],
                "y_true": holdout_eval["predictions"]["y_true"],
                "y_pred_proba": holdout_eval["predictions"]["y_pred"],
            })
            pred_df.to_parquet(holdout_dir / "predictions.parquet", index=False)

            with open(holdout_dir / "metrics_overall.json", "w") as f:
                json.dump(holdout_metrics, f, indent=2)

            if generate_plots:
                try:
                    plot_model_performance(
                        holdout_eval["predictions"]["y_true"],
                        holdout_eval["predictions"]["y_pred"],
                        plots_dir / f"holdout_performance_{timestamp}.png",
                        title_suffix="(Holdout)",
                    )
                except Exception:
                    pass
    except Exception as exc:
        logger.warning(f"Holdout evaluation failed (non-fatal): {exc}")

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
        "best_val_auc": training_result["best_val_auc"],
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
        f"best_val_auc: {training_result['best_val_auc']:.4f}",
    ]
    if holdout_metrics.get("auc_roc"):
        info_lines.append(f"holdout_auc: {holdout_metrics['auc_roc']:.4f}")
    (out_dir / "stage_info.txt").write_text("\n".join(info_lines) + "\n")

    logger.info(f"Two-Tower training completed in {runtime:.2f}s")

    return {
        "output_dir": out_dir,
        "artifacts": {
            "model_path": str(model_path),
            "training_config": str(out_dir / "training_config.json"),
        },
    }
