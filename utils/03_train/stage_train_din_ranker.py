#!/usr/bin/env python3

"""Stage 3 model components and training for a DIN ranker."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from shared.input_data_helpers import AUTHOR_PAD_IDX, AUTHOR_UNK_IDX
from utils.dataloaders import (
    BucketedEngagementDataset,
    create_bucketed_data_loaders,
    get_author_table_num_rows,
    load_bucketed_training_data,
)
from utils.helpers import (
    clear_cuda_memory,
    find_author_idx_artifact_path,
    get_device,
    get_stage_logger,
    log_operation_start,
    log_prior_stage_inputs,
    plot_training_history,
    set_random_seeds,
)
from utils.matrix_ranking import (
    MatrixBatchScores,
    empty_rank_metric_sums,
    evaluate_matrix_scorer,
    rank_metric_sums_for_batch,
    stage_info_metric_lines,
)
from utils.pipeline.core import Context
from utils.ranker_utilities import BSTPostAuthorFeatureEncoder, LinearPredictionHead


STAGE_LOG_NAME = "STAGE_03_TRAIN_DIN_RANKER"


class DINRanker(nn.Module):
    """Deep Interest Network ranker with candidate-conditioned history attention."""

    def __init__(
        self,
        post_embedding_dim: int,
        author_table_num_rows: int,
        author_embedding_dim: int,
        content_projection_dim: int,
        author_projection_dim: int,
        model_dim: int,
        attention_hidden_dims: Sequence[int],
        prediction_hidden_dims: Sequence[int],
        dropout_rate: float,
        author_unknown_dropout_rate: float,
    ):
        super().__init__()
        if post_embedding_dim <= 0:
            raise ValueError("post_embedding_dim must be positive")
        if model_dim <= 0:
            raise ValueError("model_dim must be positive")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be in [0, 1]")

        self.post_embedding_dim = int(post_embedding_dim)
        self.content_projection_dim = int(content_projection_dim)
        self.author_projection_dim = int(author_projection_dim)
        self.model_dim = int(model_dim)
        self.post_feature_encoder = BSTPostAuthorFeatureEncoder(
            post_embedding_dim=post_embedding_dim,
            author_table_num_rows=author_table_num_rows,
            author_embedding_dim=author_embedding_dim,
            content_projection_dim=content_projection_dim,
            author_projection_dim=author_projection_dim,
            model_dim=model_dim,
            author_unknown_dropout_rate=author_unknown_dropout_rate,
        )
        self.attention_unit = LinearPredictionHead(
            input_dim=4 * self.model_dim,
            hidden_dims=attention_hidden_dims,
            dropout_rate=dropout_rate,
        )
        self.prediction_head = LinearPredictionHead(
            input_dim=4 * self.model_dim,
            hidden_dims=prediction_hidden_dims,
            dropout_rate=dropout_rate,
        )

    def _attention_features(
        self,
        history_vectors: torch.Tensor,
        candidate_vectors: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                history_vectors,
                candidate_vectors,
                history_vectors * candidate_vectors,
                history_vectors - candidate_vectors,
            ],
            dim=-1,
        )

    def _pair_features(
        self,
        candidate_vectors: torch.Tensor,
        interest_vectors: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                candidate_vectors,
                interest_vectors,
                candidate_vectors * interest_vectors,
                candidate_vectors - interest_vectors,
            ],
            dim=-1,
        )

    def _attend_pairs(
        self,
        history_vectors: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_vectors: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, max_history_len, model_dim = history_vectors.shape
        if candidate_vectors.shape != (batch_size, model_dim):
            raise ValueError("candidate_vectors must have shape [B, model_dim]")
        history_mask = history_mask.to(device=history_vectors.device, dtype=torch.bool)
        if history_mask.shape != (batch_size, max_history_len):
            raise ValueError("history_mask must have shape [B, H]")
        if max_history_len == 0:
            return torch.zeros((batch_size, model_dim), device=history_vectors.device, dtype=history_vectors.dtype)

        candidate_expanded = candidate_vectors.unsqueeze(1).expand(-1, max_history_len, -1)
        attention_logits = self.attention_unit(
            self._attention_features(history_vectors, candidate_expanded)
        )
        attention_logits = attention_logits.masked_fill(~history_mask, -1.0e9)
        attention_weights = torch.softmax(attention_logits, dim=1)
        attention_weights = attention_weights.masked_fill(~history_mask, 0.0)
        return (attention_weights.unsqueeze(-1) * history_vectors).sum(dim=1)

    def _attend_matrix(
        self,
        history_vectors: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_vectors: torch.Tensor,
    ) -> torch.Tensor:
        num_users, max_history_len, model_dim = history_vectors.shape
        num_candidates = int(candidate_vectors.size(0))
        if candidate_vectors.shape != (num_candidates, model_dim):
            raise ValueError("candidate_vectors must have shape [C, model_dim]")
        history_mask = history_mask.to(device=history_vectors.device, dtype=torch.bool)
        if history_mask.shape != (num_users, max_history_len):
            raise ValueError("history_mask must have shape [U, H]")
        if max_history_len == 0:
            return torch.zeros(
                (num_users, num_candidates, model_dim),
                device=history_vectors.device,
                dtype=history_vectors.dtype,
            )

        history_expanded = history_vectors.unsqueeze(2).expand(-1, -1, num_candidates, -1)
        candidate_expanded = candidate_vectors.unsqueeze(0).unsqueeze(0).expand(num_users, max_history_len, -1, -1)
        attention_logits = self.attention_unit(
            self._attention_features(history_expanded, candidate_expanded)
        )
        attention_mask = history_mask.unsqueeze(-1)
        attention_logits = attention_logits.masked_fill(~attention_mask, -1.0e9)
        attention_weights = torch.softmax(attention_logits, dim=1)
        attention_weights = attention_weights.masked_fill(~attention_mask, 0.0)
        return torch.einsum("uhc,uhm->ucm", attention_weights, history_vectors)

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_post_embeddings: torch.Tensor,
        history_author_indices: torch.Tensor,
        candidate_post_author_idx: torch.Tensor,
    ) -> torch.Tensor:
        if history_embeddings.dim() != 3:
            raise ValueError("history_embeddings must have shape [B, H, D]")
        if candidate_post_embeddings.dim() != 2:
            raise ValueError("candidate_post_embeddings must have shape [B, D]")
        batch_size, max_history_len, embed_dim = history_embeddings.shape
        if embed_dim != self.post_embedding_dim:
            raise ValueError(
                f"history_embeddings last dimension ({embed_dim}) must match post_embedding_dim ({self.post_embedding_dim})"
            )
        if candidate_post_embeddings.shape != (batch_size, self.post_embedding_dim):
            raise ValueError("candidate_post_embeddings must have shape [B, post_embedding_dim]")
        if history_author_indices.shape != (batch_size, max_history_len):
            raise ValueError("history_author_indices must have shape [B, H]")
        if candidate_post_author_idx.shape != (batch_size,):
            raise ValueError("candidate_post_author_idx must have shape [B]")

        device = history_embeddings.device
        history_mask = history_mask.to(device=device, dtype=torch.bool)
        history_author_indices = history_author_indices.to(device=device, dtype=torch.long)
        candidate_post_author_idx = candidate_post_author_idx.to(device=device, dtype=torch.long)

        history_vectors = self.post_feature_encoder(history_embeddings, history_author_indices)
        candidate_vectors = self.post_feature_encoder(candidate_post_embeddings, candidate_post_author_idx)
        interest_vectors = self._attend_pairs(history_vectors, history_mask, candidate_vectors)
        logits = self.prediction_head(self._pair_features(candidate_vectors, interest_vectors))
        if logits.dim() == 2 and logits.shape == (batch_size, 1):
            logits = logits.squeeze(-1)
        if logits.shape != (batch_size,):
            raise RuntimeError("prediction_head must return logits with shape [B] or [B, 1]")
        return logits

    def score_candidate_matrix(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_post_embeddings: torch.Tensor,
        history_author_indices: torch.Tensor,
        candidate_post_author_idx: torch.Tensor,
    ) -> torch.Tensor:
        if history_embeddings.dim() != 3:
            raise ValueError("history_embeddings must have shape [U, H, D]")
        if candidate_post_embeddings.dim() != 2:
            raise ValueError("candidate_post_embeddings must have shape [C, D]")
        num_users, max_history_len, embed_dim = history_embeddings.shape
        num_candidates = int(candidate_post_embeddings.size(0))
        if embed_dim != self.post_embedding_dim:
            raise ValueError(
                f"history_embeddings last dimension ({embed_dim}) must match post_embedding_dim ({self.post_embedding_dim})"
            )
        if candidate_post_embeddings.shape != (num_candidates, self.post_embedding_dim):
            raise ValueError("candidate_post_embeddings must have shape [C, post_embedding_dim]")
        if history_author_indices.shape != (num_users, max_history_len):
            raise ValueError("history_author_indices must have shape [U, H]")
        if candidate_post_author_idx.shape != (num_candidates,):
            raise ValueError("candidate_post_author_idx must have shape [C]")

        device = history_embeddings.device
        history_mask = history_mask.to(device=device, dtype=torch.bool)
        history_author_indices = history_author_indices.to(device=device, dtype=torch.long)
        candidate_post_author_idx = candidate_post_author_idx.to(device=device, dtype=torch.long)

        history_vectors = self.post_feature_encoder(history_embeddings, history_author_indices)
        candidate_vectors = self.post_feature_encoder(candidate_post_embeddings, candidate_post_author_idx)
        interest_vectors = self._attend_matrix(history_vectors, history_mask, candidate_vectors)
        candidate_expanded = candidate_vectors.unsqueeze(0).expand(num_users, -1, -1)
        logits = self.prediction_head(
            self._pair_features(candidate_expanded, interest_vectors).reshape(num_users * num_candidates, 4 * self.model_dim)
        )
        if logits.dim() == 2 and logits.shape == (num_users * num_candidates, 1):
            logits = logits.squeeze(-1)
        if logits.shape != (num_users * num_candidates,):
            raise RuntimeError("prediction_head must return logits with shape [U*C] or [U*C, 1]")
        return logits.reshape(num_users, num_candidates)


def _compute_din_listwise_loss_and_preds(
    model: DINRanker,
    batch: Dict[str, Any],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history_embeddings = batch["history_embeddings"].to(device, non_blocking=True)
    history_mask = batch["history_mask"].to(device, non_blocking=True)
    candidate_post_embeddings = batch["candidate_post_embeddings"].to(device, non_blocking=True)
    labels = batch["label_matrix"].to(device, dtype=torch.float32, non_blocking=True)
    if "history_author_indices" not in batch or "candidate_post_author_idx" not in batch:
        raise RuntimeError("DIN ranker batches must include author index tensors")
    history_author_indices = batch["history_author_indices"].to(device, dtype=torch.long, non_blocking=True)
    candidate_post_author_idx = batch["candidate_post_author_idx"].to(device, dtype=torch.long, non_blocking=True)

    scores = model.score_candidate_matrix(
        history_embeddings=history_embeddings,
        history_mask=history_mask,
        candidate_post_embeddings=candidate_post_embeddings,
        history_author_indices=history_author_indices,
        candidate_post_author_idx=candidate_post_author_idx,
    )
    if scores.shape != labels.shape:
        raise RuntimeError("Expected DIN scores and label_matrix to have matching [num_users, num_candidates] shapes")
    positive_counts = labels.sum(dim=1, keepdim=True)
    if torch.any(positive_counts <= 0):
        raise RuntimeError("Each user row in label_matrix must contain at least one positive candidate")

    targets = labels / positive_counts
    loss_per_user = -(targets * F.log_softmax(scores, dim=1)).sum(dim=1)
    return loss_per_user.mean(), scores, labels


def run_din_epoch(
    *,
    train: bool,
    split_name: str,
    model: DINRanker,
    device: str,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    disable_progress: bool,
    gradient_clip_max_norm: float,
    metrics_top_ks: List[int],
    max_batches: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    if train:
        if optimizer is None:
            raise ValueError("optimizer is required when train=True")
        model.train()
    else:
        model.eval()

    loss_sum = torch.zeros((), device=device)
    batches = 0
    metric_sums = empty_rank_metric_sums(metrics_top_ks)
    metric_user_count = 0

    with nullcontext() if train else torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=split_name, leave=False, disable=disable_progress)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if train and optimizer is not None:
                optimizer.zero_grad()

            loss, scores, labels = _compute_din_listwise_loss_and_preds(model, batch, device)
            ranked_indices = torch.argsort(scores.detach(), dim=1, descending=True)
            ranked_labels = torch.gather(labels, dim=1, index=ranked_indices)
            batch_metric_sums, batch_metric_user_count = rank_metric_sums_for_batch(
                ranked_labels,
                metrics_top_ks,
            )

            if train and optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_max_norm)
                optimizer.step()

            loss_sum += loss.detach()
            batches += 1
            metric_user_count += batch_metric_user_count
            for key, value in batch_metric_sums.items():
                metric_sums[key] += value

    loss = (loss_sum / max(batches, 1)).item()
    metrics: Dict[str, Any] = {
        key: value / metric_user_count if metric_user_count > 0 else 0.0
        for key, value in metric_sums.items()
    }
    metrics["loss"] = loss
    metrics["rank_metric_user_count"] = metric_user_count
    return loss, metrics


class DINRankerMatrixScorer:
    """Matrix-ranking scorer for an in-memory DIN model."""

    def __init__(self, model: DINRanker):
        self.model = model

    def prepare_for_eval(self, device: str) -> None:
        self.model = self.model.to(device)
        self.model.eval()

    def score_batch(self, batch: Dict[str, Any], device: str) -> MatrixBatchScores:
        loss, scores, _ = _compute_din_listwise_loss_and_preds(self.model, batch, device)
        return MatrixBatchScores(scores=scores, loss=loss)


def _history_metric_names(metrics_top_ks: List[int]) -> List[str]:
    names: List[str] = []
    for k in metrics_top_ks:
        names.extend([f"ndcg@{k}", f"recall@{k}"])
    names.append("mean_average_precision")
    return names


def _append_split_metrics_to_history(
    history: Dict[str, List[float]],
    split_name: str,
    metrics: Dict[str, Any],
    metric_names: List[str],
) -> None:
    for metric_name in metric_names:
        key = f"{split_name}_{metric_name}"
        metric_value = metrics.get(metric_name)
        history.setdefault(key, []).append(float(metric_value) if metric_value is not None else float("nan"))


def _log_din_epoch_metrics(
    experiment_tracker: Optional[Any],
    iteration: int,
    train_loss: float,
    val_loss: float,
    val_unseen_loss: float,
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    val_unseen_metrics: Dict[str, Any],
    metrics_top_ks: List[int],
    primary_metric_name: str,
) -> None:
    if experiment_tracker is None:
        return
    experiment_tracker.log_scalar("Training Loss History", "Train Loss", float(train_loss), iteration)
    experiment_tracker.log_scalar("Training Loss History", "Validation Loss", float(val_loss), iteration)
    experiment_tracker.log_scalar("Training Loss History", "Validation Unseen Users Loss", float(val_unseen_loss), iteration)
    primary_metric_key = primary_metric_name.replace("val_unseen_", "", 1)
    primary_metric_value = val_unseen_metrics.get(primary_metric_key)
    if primary_metric_value is not None:
        experiment_tracker.log_scalar(
            f"Primary Ranking Metric ({primary_metric_key})",
            f"Validation Unseen Users {primary_metric_key}",
            float(primary_metric_value),
            iteration,
        )
    for k in metrics_top_ks:
        for metric_name, metric_label in ((f"ndcg@{k}", f"NDCG@{k}"), (f"recall@{k}", f"Recall@{k}")):
            for split_label, metrics in (
                ("Train", train_metrics),
                ("Validation", val_metrics),
                ("Validation Unseen Users", val_unseen_metrics),
            ):
                metric_value = metrics.get(metric_name)
                if metric_value is None:
                    continue
                experiment_tracker.log_scalar(metric_label, f"{split_label} {metric_label}", float(metric_value), iteration)


def train_din_ranker_model(
    model: DINRanker,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_unseen_loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    early_stopping_min_delta: float,
    checkpoints_dir: Optional[Path],
    disable_progress: bool,
    lr_scheduler_factor: float,
    lr_scheduler_patience: int,
    gradient_clip_max_norm: float,
    metrics_top_ks: Optional[List[int]] = None,
    max_train_batches_per_epoch: Optional[int] = None,
    experiment_tracker: Optional[Any] = None,
) -> Dict[str, Any]:
    metrics_top_ks = list(metrics_top_ks or [30])
    if not metrics_top_ks:
        raise ValueError("metrics_top_ks must contain at least one value")
    if max_train_batches_per_epoch is not None and max_train_batches_per_epoch <= 0:
        raise ValueError("max_train_batches_per_epoch must be positive when provided")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_scheduler_factor, patience=lr_scheduler_patience
    )

    primary_metric_name = f"val_unseen_ndcg@{metrics_top_ks[0]}"
    metric_names = _history_metric_names(metrics_top_ks)
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_unseen_loss": [],
    }
    for split_name in ("train", "val", "val_unseen"):
        for metric_name in metric_names:
            history[f"{split_name}_{metric_name}"] = []

    best_val_metric = float("-inf")
    best_reset_val_metric = float("-inf")
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    for epoch in tqdm(range(epochs), desc="Training epochs", disable=disable_progress):
        train_loss, train_metrics = run_din_epoch(
            train=True,
            split_name="Train",
            model=model,
            device=device,
            dataloader=train_loader,
            optimizer=optimizer,
            disable_progress=disable_progress,
            gradient_clip_max_norm=gradient_clip_max_norm,
            metrics_top_ks=metrics_top_ks,
            max_batches=max_train_batches_per_epoch,
        )
        val_loss, val_metrics = run_din_epoch(
            train=False,
            split_name="Validation",
            model=model,
            device=device,
            dataloader=val_loader,
            optimizer=None,
            disable_progress=disable_progress,
            gradient_clip_max_norm=gradient_clip_max_norm,
            metrics_top_ks=metrics_top_ks,
        )
        val_unseen_loss, val_unseen_metrics = run_din_epoch(
            train=False,
            split_name="Validation Unseen Users",
            model=model,
            device=device,
            dataloader=val_unseen_loader,
            optimizer=None,
            disable_progress=disable_progress,
            gradient_clip_max_norm=gradient_clip_max_norm,
            metrics_top_ks=metrics_top_ks,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_unseen_loss"].append(val_unseen_loss)
        _append_split_metrics_to_history(history, "train", train_metrics, metric_names)
        _append_split_metrics_to_history(history, "val", val_metrics, metric_names)
        _append_split_metrics_to_history(history, "val_unseen", val_unseen_metrics, metric_names)
        _log_din_epoch_metrics(
            experiment_tracker,
            epoch + 1,
            train_loss,
            val_loss,
            val_unseen_loss,
            train_metrics,
            val_metrics,
            val_unseen_metrics,
            metrics_top_ks,
            primary_metric_name,
        )

        primary_metric_key = primary_metric_name.replace("val_unseen_", "", 1)
        primary_metric_value = val_unseen_metrics.get(primary_metric_key)
        primary_metric = float(primary_metric_value) if primary_metric_value is not None else None
        scheduler.step(primary_metric if primary_metric is not None else float("-inf"))

        if primary_metric is not None and primary_metric > best_val_metric:
            best_val_metric = primary_metric
            best_val_loss = val_unseen_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if checkpoints_dir is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_state_dict,
                        "val_unseen_loss": val_unseen_loss,
                        "primary_metric_name": primary_metric_name,
                        "val_unseen_primary_metric": primary_metric,
                        "history": history,
                    },
                    checkpoints_dir / "din_ranker_best.pth",
                )

        significant_improvement = (
            primary_metric is not None
            and primary_metric > best_reset_val_metric
            and (primary_metric - best_reset_val_metric) >= early_stopping_min_delta
        )
        if primary_metric is not None and significant_improvement:
            best_reset_val_metric = primary_metric
            patience_counter = 0
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
        "best_val_metric": best_val_metric,
        "primary_metric_name": primary_metric_name,
    }


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    device = get_device(args.device)
    timestamp = context.run_timestamp

    run_tag = args.run_tag or ""
    out_dir = context.new_stage_dir("03_train", tag=run_tag)
    checkpoints_dir = out_dir / "checkpoints"
    plots_dir = out_dir / "plots"
    logs_dir = out_dir / "logs"
    for directory in (checkpoints_dir, plots_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    logger = get_stage_logger(STAGE_LOG_NAME, log_file=out_dir / "stage.log")
    log_operation_start("Stage 3 DIN ranker training", STAGE_LOG_NAME, logger)
    t0 = time.time()

    clear_cuda_memory()
    random_seed = int(args.random_seed)
    set_random_seeds(random_seed)

    log_operation_start("Load training data from prior stages", STAGE_LOG_NAME, logger)
    embeddings_mmap, likes_core_df, posts_core_df, history_df, author_idx_mapping_df, embed_dim = load_bucketed_training_data(
        context, logger=logger,
    )
    log_prior_stage_inputs(context, logger)

    max_history_len = int(args.max_history_len)
    model_dim = int(args.din_model_dim)
    content_projection_dim = int(args.content_projection_dim)
    author_projection_dim = int(args.author_projection_dim)
    attention_hidden_dims = tuple(int(v) for v in args.din_attention_hidden_dims)
    prediction_hidden_dims = tuple(int(v) for v in args.prediction_hidden_dims)
    dropout_rate = float(args.din_dropout_rate)
    batch_size = int(args.batch_size)
    candidate_sample_size = int(args.candidate_sample_size)
    max_train_batches_per_epoch = getattr(args, "din_max_train_batches_per_epoch", None)
    if max_train_batches_per_epoch is not None:
        max_train_batches_per_epoch = int(max_train_batches_per_epoch)
    metrics_top_ks = list(args.metrics_top_ks)
    if not metrics_top_ks:
        raise ValueError("metrics_top_ks must contain at least one value")
    use_author_embedding_table = bool(args.use_author_embedding_table)
    author_embedding_dim = int(args.author_embedding_dim)
    author_unknown_dropout_rate = float(args.author_unknown_dropout_rate)
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.din_weight_decay)
    epochs = int(args.epochs)
    patience = int(args.patience)
    early_stopping_min_delta = float(args.early_stopping_min_delta)
    disable_progress = bool(args.disable_progress)
    generate_plots = not bool(args.no_plots)
    save_model = not bool(args.no_save_model)
    lr_scheduler_factor = float(args.lr_scheduler_factor)
    lr_scheduler_patience = int(args.lr_scheduler_patience)
    gradient_clip_max_norm = float(args.gradient_clip_max_norm)
    primary_metric_name = f"val_unseen_ndcg@{metrics_top_ks[0]}"

    if not use_author_embedding_table:
        raise ValueError("DIN ranker v1 requires use_author_embedding_table=True")
    if author_idx_mapping_df is None:
        raise FileNotFoundError(
            "author_idx artifact was not found in 01_get_data output, but use_author_embedding_table was enabled."
        )
    author_table_num_rows = get_author_table_num_rows(author_idx_mapping_df)
    logger.info(
        "Author embedding table enabled: "
        f"author_embedding_dim={author_embedding_dim}, "
        f"author_table_num_rows={author_table_num_rows}"
    )
    author_idx_artifact_path = find_author_idx_artifact_path(context)
    if author_idx_artifact_path is None:
        logger.warning("Author embedding table enabled, but no author_idx parquet path was found to log")
    elif context.tracker is not None:
        author_idx_artifact_id = context.tracker.log_file_artifact(
            name="author_idx_mapping",
            path=author_idx_artifact_path,
        )
        logger.info(f"Author index mapping artifact id: {author_idx_artifact_id}")

    num_workers = int(args.num_dataloader_workers)
    pin_memory = bool(args.dataloader_pin_memory)
    persistent_workers = bool(args.dataloader_persistent_workers)
    prefetch_factor = int(args.dataloader_prefetch_factor)

    config = {
        "model_type": "din-ranker",
        "post_embedding_dim": embed_dim,
        "model_dim": model_dim,
        "content_projection_dim": content_projection_dim,
        "author_projection_dim": author_projection_dim,
        "attention_hidden_dims": list(attention_hidden_dims),
        "prediction_hidden_dims": list(prediction_hidden_dims),
        "dropout_rate": dropout_rate,
        "max_history_len": max_history_len,
        "use_author_embedding_table": use_author_embedding_table,
        "author_embedding_dim": author_embedding_dim,
        "author_unknown_dropout_rate": author_unknown_dropout_rate,
        "author_table_num_rows": author_table_num_rows,
        "author_pad_idx": AUTHOR_PAD_IDX,
        "author_unk_idx": AUTHOR_UNK_IDX,
        "candidate_sample_size": candidate_sample_size,
    }
    training_config = {
        **config,
        "batch_size": batch_size,
        "din_max_train_batches_per_epoch": max_train_batches_per_epoch,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "random_seed": random_seed,
        "lr_scheduler_factor": lr_scheduler_factor,
        "lr_scheduler_patience": lr_scheduler_patience,
        "gradient_clip_max_norm": gradient_clip_max_norm,
        "primary_metric_name": primary_metric_name,
        "metrics_top_ks": metrics_top_ks,
        "num_dataloader_workers": num_workers,
        "dataloader_pin_memory": pin_memory,
        "dataloader_persistent_workers": persistent_workers,
        "dataloader_prefetch_factor": prefetch_factor,
        "save_model": save_model,
        "generate_plots": generate_plots,
    }
    training_config_path = out_dir / "training_config.json"
    with open(training_config_path, "w") as f:
        json.dump(training_config, f, indent=2)
    logger.info(f"Training config written to: {training_config_path}")

    log_operation_start("Create capped bucketed DIN datasets", STAGE_LOG_NAME, logger)
    train_dataset = BucketedEngagementDataset(
        embeddings_mmap=embeddings_mmap,
        likes_core_df=likes_core_df,
        posts_core_df=posts_core_df,
        history_df=history_df,
        split="train",
        max_history_len=max_history_len,
        embed_dim=embed_dim,
        use_author_embedding_table=use_author_embedding_table,
        candidate_sample_size=candidate_sample_size,
        seed=random_seed,
        logger=logger,
    )
    val_dataset = BucketedEngagementDataset(
        embeddings_mmap=embeddings_mmap,
        likes_core_df=likes_core_df,
        posts_core_df=posts_core_df,
        history_df=history_df,
        split="val",
        max_history_len=max_history_len,
        embed_dim=embed_dim,
        use_author_embedding_table=use_author_embedding_table,
        candidate_sample_size=candidate_sample_size,
        seed=random_seed,
        logger=logger,
    )
    val_unseen_dataset = BucketedEngagementDataset(
        embeddings_mmap=embeddings_mmap,
        likes_core_df=likes_core_df,
        posts_core_df=posts_core_df,
        history_df=history_df,
        split="val_unseen_users",
        max_history_len=max_history_len,
        embed_dim=embed_dim,
        use_author_embedding_table=use_author_embedding_table,
        candidate_sample_size=candidate_sample_size,
        seed=random_seed,
        logger=logger,
    )
    train_loader, val_loader, val_unseen_loader, _ = create_bucketed_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        val_unseen_dataset=val_unseen_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        seed=random_seed,
        train_resample_candidates_each_epoch=True,
    )
    del likes_core_df, posts_core_df, history_df, author_idx_mapping_df

    log_operation_start("Create DIN ranker model", STAGE_LOG_NAME, logger)
    model = DINRanker(
        post_embedding_dim=embed_dim,
        author_table_num_rows=author_table_num_rows,
        author_embedding_dim=author_embedding_dim,
        content_projection_dim=content_projection_dim,
        author_projection_dim=author_projection_dim,
        model_dim=model_dim,
        attention_hidden_dims=attention_hidden_dims,
        prediction_hidden_dims=prediction_hidden_dims,
        dropout_rate=dropout_rate,
        author_unknown_dropout_rate=author_unknown_dropout_rate,
    )

    log_operation_start(f"Train DIN ranker (epochs={epochs}, batch_size={batch_size})", STAGE_LOG_NAME, logger)
    training_results = train_din_ranker_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_unseen_loader=val_unseen_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        patience=patience,
        early_stopping_min_delta=early_stopping_min_delta,
        checkpoints_dir=checkpoints_dir,
        disable_progress=disable_progress,
        lr_scheduler_factor=lr_scheduler_factor,
        lr_scheduler_patience=lr_scheduler_patience,
        gradient_clip_max_norm=gradient_clip_max_norm,
        metrics_top_ks=metrics_top_ks,
        max_train_batches_per_epoch=max_train_batches_per_epoch,
        experiment_tracker=context.tracker,
    )
    trained_model: DINRanker = training_results["model"]
    clear_cuda_memory()

    if generate_plots:
        hist = training_results["history"]
        try:
            primary_metric_name = training_results["primary_metric_name"]
            val_unseen_metric_history = hist.get(primary_metric_name, [])
            valid_metrics = [
                (idx + 1, float(value))
                for idx, value in enumerate(val_unseen_metric_history)
                if float(value) == float(value)
            ]
            best_epoch = max(valid_metrics, key=lambda item: item[1])[0] if valid_metrics else None
        except Exception as exc:
            logger.warning(f"Could not determine best epoch from DIN training history: {exc}")
            best_epoch = None
        plot_training_history(hist, plots_dir / f"training_history_{timestamp}.png", best_epoch=best_epoch)

    din_matrix_scorer = DINRankerMatrixScorer(trained_model)
    train_eval = evaluate_matrix_scorer(
        din_matrix_scorer,
        train_loader,
        device=device,
        metrics_top_ks=metrics_top_ks,
        progress_desc="Evaluate train",
        disable_progress=disable_progress,
        max_batches=max_train_batches_per_epoch,
    )
    val_eval = evaluate_matrix_scorer(
        din_matrix_scorer,
        val_loader,
        device=device,
        metrics_top_ks=metrics_top_ks,
        progress_desc="Evaluate validation",
        disable_progress=disable_progress,
    )
    val_unseen_eval = evaluate_matrix_scorer(
        din_matrix_scorer,
        val_unseen_loader,
        device=device,
        metrics_top_ks=metrics_top_ks,
        progress_desc="Evaluate validation unseen users",
        disable_progress=disable_progress,
    )
    train_metrics = train_eval["metrics"]
    val_metrics = val_eval["metrics"]
    val_unseen_metrics = val_unseen_eval["metrics"]
    train_loss = float(train_metrics["loss"]) if train_metrics.get("loss") is not None else 0.0
    val_loss = float(val_metrics["loss"]) if val_metrics.get("loss") is not None else 0.0
    val_unseen_loss = float(val_unseen_metrics["loss"]) if val_unseen_metrics.get("loss") is not None else 0.0
    logger.info(f"Train metrics: {train_metrics}")
    logger.info(f"Validation metrics: {val_metrics}")
    logger.info(f"Validation unseen users metrics: {val_unseen_metrics}")

    model_path = None
    if save_model:
        log_operation_start("Save DIN ranker checkpoint", STAGE_LOG_NAME, logger)
        model_path = checkpoints_dir / f"din_ranker_{timestamp}.pth"
        torch.save(
            {
                "model_state_dict": trained_model.state_dict(),
                "config": config,
                "training_history": training_results["history"],
                "primary_metric_name": training_results["primary_metric_name"],
                "best_val_metric": training_results["best_val_metric"],
                "best_val_loss": training_results["best_val_loss"],
            },
            model_path,
        )
        logger.info(f"Model saved to: {model_path}")

    final_split_metrics: Dict[str, Dict[str, Any]] = {
        "train": train_metrics,
        "val": val_metrics,
        "val_unseen_users": val_unseen_metrics,
    }
    runtime = time.time() - t0
    training_results_path = out_dir / "training_results.json"
    end_of_training_values = {
        "runtime_seconds": runtime,
        "model_path": str(model_path) if model_path else None,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "val_unseen_samples": len(val_unseen_dataset),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_unseen_loss": val_unseen_loss,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "val_unseen_metrics": val_unseen_metrics,
        "primary_metric_name": training_results["primary_metric_name"],
        "best_val_metric": training_results["best_val_metric"],
        "best_val_loss": training_results["best_val_loss"],
        "training_history": training_results["history"],
    }
    with open(training_results_path, "w") as f:
        json.dump(end_of_training_values, f, indent=2)
    logger.info(f"Training results written to: {training_results_path}")

    info_lines = [
        "stage: train_din_ranker",
        f"timestamp: {timestamp}",
        f"runtime_seconds: {runtime:.2f}",
        f"settings: batch_size={batch_size}, candidate_sample_size={candidate_sample_size}, lr={learning_rate}, epochs={epochs}, max_history_len={max_history_len}, early_stopping_min_delta={early_stopping_min_delta}",
        f"train_samples: {len(train_dataset)}",
        f"val_samples: {len(val_dataset)}",
        f"val_unseen_samples: {len(val_unseen_dataset)}",
        f"primary_metric_name: {training_results['primary_metric_name']}",
        f"best_val_metric: {training_results['best_val_metric']:.4f}",
    ]
    info_lines.extend(stage_info_metric_lines(final_split_metrics))
    (out_dir / "stage_info.txt").write_text("\n".join(info_lines) + "\n")

    logger.info(f"DIN ranker training completed in {runtime:.2f}s")

    return {
        "output_dir": out_dir,
        "artifacts": {
            "model_path": str(model_path) if model_path else None,
            "training_config": str(training_config_path),
            "training_results": str(training_results_path),
        },
    }
