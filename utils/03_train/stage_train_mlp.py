#!/usr/bin/env python3

"""Stage 3 (MLP): Train bucketed matrix-ranker engagement models.

The MLP stage uses the same Stage 1/2 bucketed data contract as Two-Tower:
each batch contains a set of user-hour rows and same-hour candidate posts. The
model scores the full user x candidate matrix and trains with a row-wise
multi-positive softmax loss.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.pipeline.core import Context
from utils.helpers import (
    get_stage_logger,
    log_operation_start,
    log_prior_stage_inputs,
    get_device,
    plot_training_history,
    clear_cuda_memory,
    set_random_seeds,
    find_author_idx_artifact_path,
)
from shared.input_data_helpers import AUTHOR_PAD_IDX, AUTHOR_UNK_IDX
from utils.dataloaders import (
    BucketedEngagementDataset,
    TransformerDualPoolingEncoder,
    CrossAttentionPoolingEncoder,
    create_bucketed_data_loaders,
    get_author_table_num_rows,
    load_bucketed_training_data,
)
from utils.author_features import PostAuthorFeatureEncoder
from utils.matrix_ranking import (
    evaluate_matrix_model,
    log_final_classification_metrics,
    run_matrix_epoch,
    stage_info_metric_lines,
    write_ranking_rows,
)

STAGE_LOG_NAME = "STAGE_03_TRAIN_MLP"


class SummarizedHistoryEncoder(nn.Module):
    """Hand-crafted masked history summarizer with a learned cold-start vector."""

    def __init__(self, embed_dim: int, user_summarization: str, ema_alpha: float):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.user_summarization = str(user_summarization)
        self.ema_alpha = float(ema_alpha)
        if self.user_summarization not in ("mean", "ema", "linear_recency"):
            raise ValueError("user_summarization must be one of: mean, ema, linear_recency")
        if not 0.0 < self.ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        self.empty_user_embedding = nn.Parameter(torch.randn(self.embed_dim) * 0.02)

    def forward(self, history_embeddings: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        history_mask = history_mask.to(device=history_embeddings.device, dtype=torch.bool)
        weights = history_mask.to(dtype=history_embeddings.dtype)
        if self.user_summarization == "ema":
            seq_len = history_embeddings.size(1)
            positions = torch.arange(seq_len, device=history_embeddings.device, dtype=history_embeddings.dtype)
            recency_weights = self.ema_alpha * ((1.0 - self.ema_alpha) ** positions)
            weights = weights * recency_weights.unsqueeze(0)
        elif self.user_summarization == "linear_recency":
            seq_len = history_embeddings.size(1)
            recency_weights = torch.arange(seq_len, 0, -1, device=history_embeddings.device, dtype=history_embeddings.dtype)
            weights = weights * recency_weights.unsqueeze(0)

        denom = weights.sum(dim=1, keepdim=True).clamp(min=1.0e-12)
        summary = (history_embeddings * weights.unsqueeze(-1)).sum(dim=1) / denom
        has_history = history_mask.any(dim=1)
        has_history_f = has_history.to(dtype=summary.dtype).unsqueeze(1)
        empty = self.empty_user_embedding.unsqueeze(0)
        return summary * has_history_f + empty * (1.0 - has_history_f)


class MLPModel(nn.Module):
    """MLP matrix ranker with a pluggable user-history encoder."""

    def __init__(
        self,
        post_embedding_dim: int,
        hidden_dims: List[int],
        dropout_rate: float,
        user_hidden_dim: int,
        user_output_dim: int,
        num_attention_heads: int,
        num_attention_layers: int,
        max_history_len: int,
        attention_dropout: float,
        user_encoder_type: str,
        user_summarization: str,
        ema_alpha: float,
        use_author_embedding_table: bool = False,
        author_table_num_rows: Optional[int] = None,
        author_embedding_dim: Optional[int] = None,
        author_unknown_dropout_rate: Optional[float] = None,
    ):
        super().__init__()
        self.post_embedding_dim = int(post_embedding_dim)
        self.user_output_dim = int(user_output_dim)
        self.user_encoder_type = str(user_encoder_type)
        self.user_summarization = str(user_summarization)
        self.ema_alpha = float(ema_alpha)
        self.use_author_embedding_table = bool(use_author_embedding_table)
        self.post_author_feature_encoder = None
        if self.use_author_embedding_table:
            if author_table_num_rows is None or author_table_num_rows < 2:
                raise ValueError("author_table_num_rows must be provided and >= 2 when use_author_embedding_table is True")
            if author_embedding_dim is None or author_embedding_dim <= 0:
                raise ValueError("author_embedding_dim must be provided and positive when use_author_embedding_table is True")
            if author_unknown_dropout_rate is None or author_unknown_dropout_rate < 0.0 or author_unknown_dropout_rate > 1.0:
                raise ValueError("author_unknown_dropout_rate must be provided when use_author_embedding_table is True")
            self.post_author_feature_encoder = PostAuthorFeatureEncoder(
                post_embedding_dim=post_embedding_dim,
                author_table_num_rows=author_table_num_rows,
                author_embedding_dim=author_embedding_dim,
                author_unknown_dropout_rate=author_unknown_dropout_rate,
            )

        if self.user_encoder_type == "cross_attention":
            self.user_encoder = CrossAttentionPoolingEncoder(
                input_dim=post_embedding_dim,
                hidden_dim=user_hidden_dim,
                output_dim=user_output_dim,
                max_seq_len=max_history_len,
                dropout_rate=attention_dropout,
            )
        elif self.user_encoder_type == "full_transformer":
            self.user_encoder = TransformerDualPoolingEncoder(
                input_dim=post_embedding_dim,
                hidden_dim=user_hidden_dim,
                output_dim=user_output_dim,
                num_attention_heads=num_attention_heads,
                num_attention_layers=num_attention_layers,
                max_seq_len=max_history_len,
                dropout_rate=attention_dropout,
            )
        elif self.user_encoder_type == "summarized":
            if user_output_dim != post_embedding_dim:
                raise ValueError(
                    f"user_encoder_type='summarized' requires user_output_dim ({user_output_dim}) == post_embedding_dim ({post_embedding_dim})"
                )
            self.user_encoder = SummarizedHistoryEncoder(
                embed_dim=post_embedding_dim,
                user_summarization=self.user_summarization,
                ema_alpha=self.ema_alpha,
            )
        else:
            raise ValueError(
                f"Unknown user_encoder_type '{user_encoder_type}' for MLPModel. "
                "Choose 'summarized', 'full_transformer' or 'cross_attention'."
            )

        mlp_input_dim = user_output_dim + post_embedding_dim
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
        self.mlp_head = nn.Sequential(*layers)

        for m in self.mlp_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_user(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_author_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_author_embedding_table:
            if history_author_indices is None:
                raise RuntimeError("history_author_indices are required when use_author_embedding_table is True")
            if self.post_author_feature_encoder is None:
                raise RuntimeError("post_author_feature_encoder must be initialized when author embeddings are enabled")
            history_embeddings = self.post_author_feature_encoder(history_embeddings, history_author_indices)
            history_embeddings = history_embeddings.masked_fill(~history_mask.unsqueeze(-1), 0.0)
        return self.user_encoder(history_embeddings, history_mask)

    def encode_post(
        self,
        post_embeddings: torch.Tensor,
        target_author_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_author_embedding_table:
            if target_author_indices is None:
                raise RuntimeError("target_author_indices are required when use_author_embedding_table is True")
            if self.post_author_feature_encoder is None:
                raise RuntimeError("post_author_feature_encoder must be initialized when author embeddings are enabled")
            return self.post_author_feature_encoder(post_embeddings, target_author_indices)
        return post_embeddings

    def score_matrix(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_post_embeddings: torch.Tensor,
        history_author_indices: Optional[torch.Tensor] = None,
        candidate_post_author_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        user_vec = self.encode_user(history_embeddings, history_mask, history_author_indices)
        candidate_post_embeddings = self.encode_post(candidate_post_embeddings, candidate_post_author_idx)
        num_users = user_vec.size(0)
        num_candidates = candidate_post_embeddings.size(0)
        user_features = user_vec.unsqueeze(1).expand(num_users, num_candidates, user_vec.size(-1))
        post_features = candidate_post_embeddings.unsqueeze(0).expand(num_users, num_candidates, candidate_post_embeddings.size(-1))
        pair_features = torch.cat([user_features, post_features], dim=-1).reshape(num_users * num_candidates, -1)
        logits = self.mlp_head(pair_features).reshape(num_users, num_candidates)
        return logits

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        post_embedding: torch.Tensor,
        history_author_indices: Optional[torch.Tensor] = None,
        target_author_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.score_matrix(
            history_embeddings,
            history_mask,
            post_embedding,
            history_author_indices,
            target_author_indices,
        )
        return torch.sigmoid(logits)

    def compute_loss_and_preds(
        self,
        batch: Dict[str, Any],
        device: str,
        embed_dim: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        history_embeddings = batch["history_embeddings"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        post_embeddings = batch["candidate_post_embeddings"].to(device, non_blocking=True)
        label_matrix = batch["label_matrix"].to(device, dtype=torch.float32, non_blocking=True)
        history_author_indices = (
            batch["history_author_indices"].to(device, dtype=torch.long, non_blocking=True)
            if "history_author_indices" in batch
            else None
        )
        candidate_post_author_idx = (
            batch["candidate_post_author_idx"].to(device, dtype=torch.long, non_blocking=True)
            if "candidate_post_author_idx" in batch
            else None
        )

        scores = self.score_matrix(
            history_embeddings,
            history_mask,
            post_embeddings,
            history_author_indices,
            candidate_post_author_idx,
        )
        if scores.shape != label_matrix.shape:
            raise RuntimeError("Expected scores and label_matrix to have matching [num_users, num_candidates] shapes")
        positive_counts = label_matrix.sum(dim=1, keepdim=True)
        if torch.any(positive_counts <= 0):
            raise RuntimeError("Each user row in label_matrix must contain at least one positive candidate")

        targets = label_matrix / positive_counts
        loss_per_user = -(targets * F.log_softmax(scores, dim=1)).sum(dim=1)
        loss = loss_per_user.mean()
        return loss, scores


def _log_epoch_metrics(
    experiment_tracker: Optional[Any],
    iteration: int,
    primary_metric_name: str,
    metrics_top_ks: list[int],
    train_loss: float,
    val_loss: float,
    val_unseen_loss: float,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    val_unseen_metrics: Dict[str, float],
    train_baseline_metrics: Dict[str, float],
    val_baseline_metrics: Dict[str, float],
    val_unseen_baseline_metrics: Dict[str, float],
    calc_baseline_metrics: bool,
) -> None:
    if experiment_tracker is None:
        return
    experiment_tracker.log_scalar("Training Loss History", "Train Loss", float(train_loss), iteration)
    experiment_tracker.log_scalar("Training Loss History", "Validation Loss", float(val_loss), iteration)
    experiment_tracker.log_scalar("Training Loss History", "Validation Unseen Users Loss", float(val_unseen_loss), iteration)
    experiment_tracker.log_scalar(f"Primary Ranking Metric ({primary_metric_name})", f"Train {primary_metric_name}", float(train_metrics[primary_metric_name]), iteration)
    experiment_tracker.log_scalar(f"Primary Ranking Metric ({primary_metric_name})", f"Validation {primary_metric_name}", float(val_metrics[primary_metric_name]), iteration)
    experiment_tracker.log_scalar(f"Primary Ranking Metric ({primary_metric_name})", f"Validation Unseen Users {primary_metric_name}", float(val_unseen_metrics[primary_metric_name]), iteration)
    for k in metrics_top_ks:
        experiment_tracker.log_scalar(f"NDCG@{k}", f"Train NDCG@{k}", float(train_metrics[f"ndcg@{k}"]), iteration)
        experiment_tracker.log_scalar(f"NDCG@{k}", f"Validation NDCG@{k}", float(val_metrics[f"ndcg@{k}"]), iteration)
        experiment_tracker.log_scalar(f"NDCG@{k}", f"Validation Unseen Users NDCG@{k}", float(val_unseen_metrics[f"ndcg@{k}"]), iteration)
        experiment_tracker.log_scalar(f"Recall@{k}", f"Train Recall@{k}", float(train_metrics[f"recall@{k}"]), iteration)
        experiment_tracker.log_scalar(f"Recall@{k}", f"Validation Recall@{k}", float(val_metrics[f"recall@{k}"]), iteration)
        experiment_tracker.log_scalar(f"Recall@{k}", f"Validation Unseen Users Recall@{k}", float(val_unseen_metrics[f"recall@{k}"]), iteration)
        if calc_baseline_metrics:
            experiment_tracker.log_scalar(f"Baseline NDCG@{k}", f"Train Baseline NDCG@{k}", float(train_baseline_metrics[f"ndcg@{k}"]), iteration)
            experiment_tracker.log_scalar(f"Baseline NDCG@{k}", f"Validation Baseline NDCG@{k}", float(val_baseline_metrics[f"ndcg@{k}"]), iteration)
            experiment_tracker.log_scalar(f"Baseline NDCG@{k}", f"Validation Unseen Users Baseline NDCG@{k}", float(val_unseen_baseline_metrics[f"ndcg@{k}"]), iteration)
            experiment_tracker.log_scalar(f"Baseline Recall@{k}", f"Train Baseline Recall@{k}", float(train_baseline_metrics[f"recall@{k}"]), iteration)
            experiment_tracker.log_scalar(f"Baseline Recall@{k}", f"Validation Baseline Recall@{k}", float(val_baseline_metrics[f"recall@{k}"]), iteration)
            experiment_tracker.log_scalar(f"Baseline Recall@{k}", f"Validation Unseen Users Baseline Recall@{k}", float(val_unseen_baseline_metrics[f"recall@{k}"]), iteration)


def train_mlp_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_unseen_loader: DataLoader,
    device: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    early_stopping_min_delta: float,
    lr_scheduler_factor: float,
    lr_scheduler_patience: int,
    checkpoints_dir: Optional[Path],
    disable_progress: bool,
    gradient_clip_max_norm: float,
    embed_dim: int,
    metrics_top_ks: list[int],
    experiment_tracker: Optional[Any] = None,
) -> Dict[str, Any]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_scheduler_factor, patience=lr_scheduler_patience
    )

    if not metrics_top_ks:
        raise ValueError("metrics_top_ks must contain at least one value")
    primary_metric_name = f"ndcg@{metrics_top_ks[0]}"
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        f"train_{primary_metric_name}": [],
        f"val_{primary_metric_name}": [],
    }
    best_val_metric = float("-inf")
    best_reset_val_metric = float("-inf")
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    for epoch in tqdm(range(epochs), desc="Training epochs", disable=disable_progress):
        calc_baseline_metrics = epoch == 0
        train_loss, train_metrics, train_baseline_metrics = run_matrix_epoch(
            train=True,
            split_name="Train",
            model=model,
            device=device,
            dataloader=train_loader,
            optimizer=optimizer,
            disable_progress=disable_progress,
            embed_dim=embed_dim,
            gradient_clip_max_norm=gradient_clip_max_norm,
            metrics_top_ks=metrics_top_ks,
            calc_baseline_metrics=calc_baseline_metrics,
        )
        val_loss, val_metrics, val_baseline_metrics = run_matrix_epoch(
            train=False,
            split_name="Validation",
            model=model,
            device=device,
            dataloader=val_loader,
            optimizer=optimizer,
            disable_progress=disable_progress,
            embed_dim=embed_dim,
            gradient_clip_max_norm=gradient_clip_max_norm,
            metrics_top_ks=metrics_top_ks,
            calc_baseline_metrics=calc_baseline_metrics,
        )
        val_unseen_loss, val_unseen_metrics, val_unseen_baseline_metrics = run_matrix_epoch(
            train=False,
            split_name="Validation Unseen Users",
            model=model,
            device=device,
            dataloader=val_unseen_loader,
            optimizer=optimizer,
            disable_progress=disable_progress,
            embed_dim=embed_dim,
            gradient_clip_max_norm=gradient_clip_max_norm,
            metrics_top_ks=metrics_top_ks,
            calc_baseline_metrics=calc_baseline_metrics,
        )

        train_primary_metric = float(train_metrics[primary_metric_name])
        val_primary_metric = float(val_metrics[primary_metric_name])
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history[f"train_{primary_metric_name}"].append(train_primary_metric)
        history[f"val_{primary_metric_name}"].append(val_primary_metric)

        _log_epoch_metrics(
            experiment_tracker,
            epoch + 1,
            primary_metric_name,
            metrics_top_ks,
            train_loss,
            val_loss,
            val_unseen_loss,
            train_metrics,
            val_metrics,
            val_unseen_metrics,
            train_baseline_metrics,
            val_baseline_metrics,
            val_unseen_baseline_metrics,
            calc_baseline_metrics,
        )

        scheduler.step(val_primary_metric)

        if val_primary_metric > best_val_metric:
            best_val_metric = val_primary_metric
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if checkpoints_dir is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_state_dict,
                        "val_loss": val_loss,
                        "primary_metric_name": primary_metric_name,
                        "val_primary_metric": val_primary_metric,
                        "history": history,
                    },
                    checkpoints_dir / "mlp_best.pth",
                )

        significant_improvement = (
            val_primary_metric > best_reset_val_metric
            and (val_primary_metric - best_reset_val_metric) >= early_stopping_min_delta
        )
        if significant_improvement:
            best_reset_val_metric = val_primary_metric
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
    eval_dir = out_dir / "eval"
    for d in (checkpoints_dir, plots_dir, logs_dir, eval_dir):
        d.mkdir(parents=True, exist_ok=True)

    logger = get_stage_logger(STAGE_LOG_NAME, log_file=out_dir / "stage.log")
    log_operation_start("Stage 3 MLP training", STAGE_LOG_NAME, logger)
    t0 = time.time()

    clear_cuda_memory()
    random_seed = int(args.random_seed)
    set_random_seeds(random_seed)

    log_operation_start("Load training data from prior stages", STAGE_LOG_NAME, logger)
    embeddings_mmap, likes_core_df, posts_core_df, history_df, author_idx_mapping_df, embed_dim = load_bucketed_training_data(
        context, logger=logger,
    )
    log_prior_stage_inputs(context, logger)
    num_total_likes_by_user = {
        str(row["did"]): int(row["num_total_likes"])
        for row in (
            likes_core_df
            .group_by("did")
            .agg(pl.col("subject_uri").n_unique().alias("num_total_likes"))
            .iter_rows(named=True)
        )
    }

    user_encoder = args.user_encoder
    batch_size = int(args.batch_size)
    hidden_dims = list(args.hidden_dims)
    dropout_rate = float(args.dropout_rate_mlp)
    epochs = int(args.epochs)
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.weight_decay_mlp)
    patience = int(args.patience)
    early_stopping_min_delta = float(args.early_stopping_min_delta)
    disable_progress = bool(args.disable_progress)
    generate_plots = not bool(args.no_plots)
    save_model = not bool(args.no_save_model)
    lr_scheduler_factor = float(args.lr_scheduler_factor)
    lr_scheduler_patience = int(args.lr_scheduler_patience)
    gradient_clip_max_norm = float(args.gradient_clip_max_norm)
    eval_holdout_type = str(args.eval_holdout_type)
    max_history_len = int(args.max_history_len)
    user_hidden_dim = int(args.user_hidden_dim)
    user_output_dim = int(args.user_output_dim)
    num_attention_heads = int(args.num_attention_heads)
    num_attention_layers = int(args.num_attention_layers)
    attention_dropout = float(args.attention_dropout)
    summarizer_name = str(args.user_summarization)
    ema_alpha = float(args.ema_alpha)
    use_author_embedding_table = bool(args.use_author_embedding_table)
    author_embedding_dim = int(args.author_embedding_dim)
    author_unknown_dropout_rate = float(args.author_unknown_dropout_rate)
    metrics_top_ks = list(args.metrics_top_ks)
    if not metrics_top_ks:
        raise ValueError("metrics_top_ks must contain at least one value")

    if user_encoder == "summarized":
        effective_user_output_dim = embed_dim
    elif user_encoder in ("full_transformer", "cross_attention"):
        effective_user_output_dim = user_output_dim
    else:
        raise ValueError(
            f"Unknown user_encoder '{user_encoder}' for MLP. "
            "Choose 'summarized', 'full_transformer' or 'cross_attention'."
        )
    primary_metric_name = f"ndcg@{metrics_top_ks[0]}"
    if use_author_embedding_table and author_idx_mapping_df is None:
        raise FileNotFoundError(
            "author_idx artifact was not found in 01_get_data output, but --use-author-embedding-table was enabled."
        )
    author_table_num_rows = 0
    if use_author_embedding_table:
        if author_idx_mapping_df is None:
            raise FileNotFoundError("author_idx_mapping_df is required when use_author_embedding_table is True")
        author_table_num_rows = get_author_table_num_rows(author_idx_mapping_df)
        logger.info(
            "Author embedding table enabled: "
            f"author_embedding_dim={author_embedding_dim}, "
            f"author_table_num_rows={author_table_num_rows}"
        )
        author_idx_artifact_path = find_author_idx_artifact_path(context)
        if author_idx_artifact_path is None:
            logger.warning("Author embedding table enabled, but no author_idx parquet path was found to log")
        else:
            author_idx_artifact_id = context.tracker.log_file_artifact(
                name="author_idx_mapping",
                path=author_idx_artifact_path,
            )
            logger.info(f"Author index mapping artifact id: {author_idx_artifact_id}")

    num_workers = int(args.num_dataloader_workers)
    pin_memory = bool(args.dataloader_pin_memory)
    persistent_workers = bool(args.dataloader_persistent_workers)
    prefetch_factor = int(args.dataloader_prefetch_factor)

    log_operation_start("Create bucketed datasets", STAGE_LOG_NAME, logger)
    train_dataset = BucketedEngagementDataset(
        embeddings_mmap, likes_core_df, posts_core_df, history_df, split="train",
        max_history_len=max_history_len, embed_dim=embed_dim,
        use_author_embedding_table=use_author_embedding_table,
        logger=logger,
    )
    val_dataset = BucketedEngagementDataset(
        embeddings_mmap, likes_core_df, posts_core_df, history_df, split="val",
        max_history_len=max_history_len, embed_dim=embed_dim,
        use_author_embedding_table=use_author_embedding_table,
        logger=logger,
    )
    val_unseen_dataset = BucketedEngagementDataset(
        embeddings_mmap, likes_core_df, posts_core_df, history_df, split="val_unseen_users",
        max_history_len=max_history_len, embed_dim=embed_dim,
        use_author_embedding_table=use_author_embedding_table,
        logger=logger,
    )

    train_loader, val_loader, val_unseen_loader, _ = create_bucketed_data_loaders(
        train_dataset, val_dataset, val_unseen_dataset, batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        seed=random_seed,
    )

    log_operation_start(f"Create MLP model (user_encoder={user_encoder})", STAGE_LOG_NAME, logger)
    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        user_hidden_dim=user_hidden_dim,
        user_output_dim=effective_user_output_dim,
        num_attention_heads=num_attention_heads,
        num_attention_layers=num_attention_layers,
        max_history_len=max_history_len,
        attention_dropout=attention_dropout,
        user_encoder_type=user_encoder,
        user_summarization=summarizer_name,
        ema_alpha=ema_alpha,
        use_author_embedding_table=use_author_embedding_table,
        author_table_num_rows=author_table_num_rows if use_author_embedding_table else None,
        author_embedding_dim=author_embedding_dim if use_author_embedding_table else None,
        author_unknown_dropout_rate=author_unknown_dropout_rate,
    )

    log_operation_start(f"Training MLP (epochs={epochs}, batch_size={batch_size})", STAGE_LOG_NAME, logger)
    training_results = train_mlp_model(
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
        embed_dim=embed_dim,
        metrics_top_ks=metrics_top_ks,
        experiment_tracker=context.tracker,
    )
    trained_model: nn.Module = training_results["model"]
    clear_cuda_memory()

    if generate_plots:
        hist = training_results["history"]
        try:
            val_metric_history = hist.get(f"val_{training_results['primary_metric_name']}", [])
            best_epoch = val_metric_history.index(max(val_metric_history)) + 1 if val_metric_history else None
        except Exception as e:
            logger.warning(f"Could not determine best epoch from training history: {e}")
            best_epoch = None
        plot_training_history(hist, plots_dir / f"training_history_{timestamp}.png", best_epoch=best_epoch)

    train_eval = evaluate_matrix_model(
        trained_model, train_loader, device, embed_dim, metrics_top_ks,
        progress_desc="Evaluate train",
        disable_progress=disable_progress,
    )
    val_eval = evaluate_matrix_model(
        trained_model, val_loader, device, embed_dim, metrics_top_ks,
        progress_desc="Evaluate validation",
        disable_progress=disable_progress,
    )
    val_unseen_eval = evaluate_matrix_model(
        trained_model, val_unseen_loader, device, embed_dim, metrics_top_ks,
        progress_desc="Evaluate validation unseen users",
        disable_progress=disable_progress,
    )
    logger.info(f"Train metrics: {train_eval['metrics']}")
    logger.info(f"Validation metrics: {val_eval['metrics']}")
    logger.info(f"Validation unseen users metrics: {val_unseen_eval['metrics']}")

    model_path = None
    config = {
        "model_type": "mlp",
        "user_encoder": user_encoder,
        "post_embedding_dim": embed_dim,
        "input_dim": effective_user_output_dim + embed_dim,
        "hidden_dims": hidden_dims,
        "dropout_rate": dropout_rate,
        "user_hidden_dim": user_hidden_dim,
        "user_output_dim": effective_user_output_dim,
        "num_attention_heads": num_attention_heads,
        "num_attention_layers": num_attention_layers,
        "max_history_len": max_history_len,
        "attention_dropout": attention_dropout,
        "user_summarization": summarizer_name,
        "ema_alpha": ema_alpha,
        "use_author_embedding_table": use_author_embedding_table,
        "author_embedding_dim": author_embedding_dim if use_author_embedding_table else None,
        "author_unknown_dropout_rate": author_unknown_dropout_rate if use_author_embedding_table else None,
        "author_table_num_rows": author_table_num_rows if use_author_embedding_table else None,
        "author_pad_idx": AUTHOR_PAD_IDX,
        "author_unk_idx": AUTHOR_UNK_IDX,
    }
    if save_model:
        log_operation_start("Save model checkpoint", STAGE_LOG_NAME, logger)
        model_path = checkpoints_dir / f"mlp_{timestamp}.pth"
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

        trained_model = trained_model.cpu()
        torchscript_name = "engagement_mlp"
        torchscript_path = checkpoints_dir / f"{torchscript_name}.pt"
        torch.jit.script(trained_model).save(torchscript_path)
        mlp_model_id = context.tracker.log_artifact(name=torchscript_name, path=torchscript_path)
        logger.info(f"MLP model id: {mlp_model_id}")

    holdout_metrics: Dict[str, Any] = {}
    all_holdout_metrics: Dict[str, Dict[str, Any]] = {}
    for holdout_type in ["unseen_users", "seen_users"]:
        split_name = f"holdout_{holdout_type}"
        try:
            holdout_dataset = BucketedEngagementDataset(
                embeddings_mmap, likes_core_df, posts_core_df, history_df, split=split_name,
                max_history_len=max_history_len, embed_dim=embed_dim,
                use_author_embedding_table=use_author_embedding_table,
                logger=logger,
            )
            if len(holdout_dataset) == 0:
                logger.info(f"No rows for split '{split_name}', skipping.")
                continue
            log_operation_start(f"Holdout evaluation ({holdout_type})", STAGE_LOG_NAME, logger)
            _, _, _, holdout_loader = create_bucketed_data_loaders(
                train_dataset, val_dataset, val_unseen_dataset, batch_size,
                holdout_dataset=holdout_dataset,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                prefetch_factor=prefetch_factor,
                seed=random_seed,
            )
            if holdout_loader is None:
                logger.info(f"No holdout loader created for split '{split_name}', skipping.")
                continue
            holdout_eval = evaluate_matrix_model(
                trained_model,
                holdout_loader,
                device,
                embed_dim,
                metrics_top_ks,
                collect_ranking_rows=True,
                progress_desc=f"Evaluate holdout {holdout_type}",
                disable_progress=disable_progress,
            )
            split_metrics = holdout_eval["metrics"]
            logger.info(f"Holdout metrics ({holdout_type}): {split_metrics}")
            all_holdout_metrics[split_name] = split_metrics

            ranking_rows_path = eval_dir / f"{split_name}_ranking_rows.parquet"
            write_ranking_rows(
                holdout_eval["ranking_rows"],
                ranking_rows_path,
                split_name,
                num_total_likes_by_user,
            )
            logger.info(f"Saved holdout ranking rows ({holdout_type}): {ranking_rows_path}")

            if holdout_type == eval_holdout_type:
                holdout_metrics = split_metrics
        except Exception as exc:
            logger.warning(f"Holdout evaluation ({holdout_type}) failed (non-fatal): {exc}")

    final_split_metrics: Dict[str, Dict[str, Any]] = {
        "train": train_eval["metrics"],
        "val": val_eval["metrics"],
        "val_unseen_users": val_unseen_eval["metrics"],
        **all_holdout_metrics,
    }
    final_metric_iteration = len(training_results["history"]["train_loss"])
    log_final_classification_metrics(
        context.tracker,
        final_split_metrics,
        final_metric_iteration,
    )

    training_config = {
        **config,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "patience": patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "random_seed": random_seed,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "val_unseen_samples": len(val_unseen_dataset),
        "train_metrics": train_eval["metrics"],
        "val_metrics": val_eval["metrics"],
        "val_unseen_metrics": val_unseen_eval["metrics"],
        "holdout_metrics": holdout_metrics,
        "all_holdout_metrics": all_holdout_metrics,
        "primary_metric_name": training_results["primary_metric_name"],
        "best_val_metric": training_results["best_val_metric"],
    }
    with open(out_dir / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)

    runtime = time.time() - t0
    info_lines = [
        f"stage: train_mlp",
        f"timestamp: {timestamp}",
        f"runtime_seconds: {runtime:.2f}",
        f"settings: batch_size={batch_size}, lr={learning_rate}, epochs={epochs}, user_encoder={user_encoder}, summarizer={summarizer_name}, early_stopping_min_delta={early_stopping_min_delta}",
        f"inputs: embeddings memmap, likes_core, posts_core, user_history",
        f"train_samples: {len(train_dataset)}",
        f"val_samples: {len(val_dataset)}",
        f"val_unseen_samples: {len(val_unseen_dataset)}",
        f"primary_metric_name: {primary_metric_name}",
        f"best_val_metric: {training_results['best_val_metric']:.4f}",
    ]
    info_lines.extend(stage_info_metric_lines(final_split_metrics))
    if holdout_metrics.get(primary_metric_name) is not None:
        info_lines.append(f"holdout_{primary_metric_name}: {holdout_metrics[primary_metric_name]:.4f}")
    (out_dir / "stage_info.txt").write_text("\n".join(info_lines) + "\n")

    logger.info(f"MLP training completed in {runtime:.2f}s")

    return {
        "output_dir": out_dir,
        "artifacts": {
            "model_path": str(model_path) if model_path else None,
            "training_config": str(out_dir / "training_config.json"),
        },
    }
