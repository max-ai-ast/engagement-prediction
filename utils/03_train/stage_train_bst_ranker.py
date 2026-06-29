#!/usr/bin/env python3

"""Stage 3 model components and training for a BST heavy ranker."""

from __future__ import annotations

import argparse
import copy
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


STAGE_LOG_NAME = "STAGE_03_TRAIN_BST_RANKER"


def _validate_time_delta_bucket_boundaries(boundaries_hours: Sequence[float]) -> tuple[float, ...]:
    boundaries = tuple(float(boundary) for boundary in boundaries_hours)
    if len(boundaries) == 0:
        raise ValueError("time delta bucket boundaries must not be empty")
    previous = 0.0
    for boundary in boundaries:
        if boundary <= 0.0:
            raise ValueError("time delta bucket boundaries must be positive")
        if boundary <= previous:
            raise ValueError("time delta bucket boundaries must be strictly increasing")
        previous = boundary
    return boundaries


class BSTRanker(nn.Module):
    """Behavior Sequence Transformer encoder for one user-history/candidate pair."""

    def __init__(
        self,
        post_embedding_dim: int,
        author_table_num_rows: int,
        author_embedding_dim: int,
        content_projection_dim: int,
        author_projection_dim: int,
        model_dim: int,
        time_embedding_dim: int,
        num_attention_heads: int,
        num_transformer_layers: int,
        transformer_ff_dim: int,
        dropout_rate: float,
        author_unknown_dropout_rate: float,
        norm_first: bool,
        time_delta_bucket_boundaries_hours: List[float],
        prediction_hidden_dims: List[int],
    ):
        super().__init__()
        if time_embedding_dim <= 0:
            raise ValueError("time_embedding_dim must be positive")
        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if num_transformer_layers <= 0:
            raise ValueError("num_transformer_layers must be positive")
        if transformer_ff_dim <= 0:
            raise ValueError("transformer_ff_dim must be positive")
        if not 0.0 <= dropout_rate <= 1.0:
            raise ValueError("dropout_rate must be in [0, 1]")

        self.post_embedding_dim = int(post_embedding_dim)
        self.content_projection_dim = int(content_projection_dim)
        self.author_projection_dim = int(author_projection_dim)
        self.model_dim = int(model_dim)
        self.time_embedding_dim = int(time_embedding_dim)
        self.dropout_rate = float(dropout_rate)
        self.time_delta_bucket_boundaries_hours = _validate_time_delta_bucket_boundaries(
            time_delta_bucket_boundaries_hours
        )
        self.register_buffer(
            "_time_delta_bucket_boundaries_tensor",
            torch.tensor(self.time_delta_bucket_boundaries_hours, dtype=torch.float32),
            persistent=False,
        )
        self.num_time_delta_buckets = len(self.time_delta_bucket_boundaries_hours) + 2
        self.transformer_input_dim = self.model_dim + self.time_embedding_dim
        if self.transformer_input_dim % int(num_attention_heads) != 0:
            raise ValueError("model_dim + time_embedding_dim must be divisible by num_attention_heads")

        self.post_feature_encoder = BSTPostAuthorFeatureEncoder(
            post_embedding_dim=post_embedding_dim,
            author_table_num_rows=author_table_num_rows,
            author_embedding_dim=author_embedding_dim,
            content_projection_dim=content_projection_dim,
            author_projection_dim=author_projection_dim,
            model_dim=model_dim,
            author_unknown_dropout_rate=author_unknown_dropout_rate,
        )
        self.time_delta_embedding = nn.Embedding(
            num_embeddings=self.num_time_delta_buckets,
            embedding_dim=self.time_embedding_dim,
        )
        nn.init.xavier_uniform_(self.time_delta_embedding.weight)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_input_dim,
            nhead=int(num_attention_heads),
            dim_feedforward=int(transformer_ff_dim),
            dropout=float(dropout_rate),
            activation="gelu",
            batch_first=True,
            norm_first=bool(norm_first),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=int(num_transformer_layers),
            enable_nested_tensor=False,
        )
        self.prediction_head = LinearPredictionHead(
            input_dim=self.transformer_input_dim,
            hidden_dims=prediction_hidden_dims,
            dropout_rate=dropout_rate,
        )

    def _bucketize_time_deltas_hours(self, time_deltas_hours: torch.Tensor) -> torch.Tensor:
        deltas = time_deltas_hours
        if not torch.is_floating_point(deltas):
            deltas = deltas.to(dtype=torch.float32)
        deltas = torch.clamp(deltas, min=0.0)
        boundary_tensor = self._time_delta_bucket_boundaries_tensor.to(
            device=deltas.device,
            dtype=deltas.dtype,
        )
        positive_bucket_ids = torch.bucketize(deltas, boundary_tensor, right=False) + 1
        zero_bucket_ids = torch.zeros_like(positive_bucket_ids)
        return torch.where(deltas <= 0.0, zero_bucket_ids, positive_bucket_ids).to(dtype=torch.long)

    def _forward_transformer(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
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
        if history_mask.shape != (batch_size, max_history_len):
            raise ValueError("history_mask must have shape [B, H]")
        if history_time_deltas_hours.shape != (batch_size, max_history_len):
            raise ValueError("history_time_deltas_hours must have shape [B, H]")
        if history_author_indices.shape != (batch_size, max_history_len):
            raise ValueError("history_author_indices must have shape [B, H]")
        if candidate_post_author_idx.shape != (batch_size,):
            raise ValueError("candidate_post_author_idx must have shape [B]")

        device = history_embeddings.device
        history_mask = history_mask.to(device=device, dtype=torch.bool)
        history_time_deltas_hours = history_time_deltas_hours.to(device=device)
        candidate_post_embeddings = candidate_post_embeddings.to(device=device)
        history_author_indices = history_author_indices.to(device=device, dtype=torch.long)
        candidate_post_author_idx = candidate_post_author_idx.to(device=device, dtype=torch.long)

        history_post_vectors = self.post_feature_encoder(history_embeddings, history_author_indices)
        candidate_post_vector = self.post_feature_encoder(
            candidate_post_embeddings,
            candidate_post_author_idx,
        ).unsqueeze(1)
        post_sequence = torch.cat([history_post_vectors, candidate_post_vector], dim=1)

        candidate_time_delta = torch.zeros((batch_size, 1), device=device, dtype=history_time_deltas_hours.dtype)
        sequence_time_deltas = torch.cat([history_time_deltas_hours, candidate_time_delta], dim=1)
        time_bucket_ids = self._bucketize_time_deltas_hours(sequence_time_deltas)
        time_embeddings = self.time_delta_embedding(time_bucket_ids)
        transformer_input = torch.cat([post_sequence, time_embeddings], dim=-1)

        candidate_is_not_padding = torch.zeros((batch_size, 1), device=device, dtype=torch.bool)
        src_key_padding_mask = torch.cat([~history_mask, candidate_is_not_padding], dim=1)
        encoded_sequence = self.transformer_encoder(
            transformer_input,
            src_key_padding_mask=src_key_padding_mask,
        )
        return encoded_sequence[:, -1, :]

    def _validate_one_layer_matrix_scorer(self) -> nn.TransformerEncoderLayer:
        layers = getattr(self.transformer_encoder, "layers", None)
        if layers is None or len(layers) != 1:
            raise RuntimeError("score_candidate_matrix_one_layer requires exactly one transformer layer")
        layer = layers[0]
        if not isinstance(layer, nn.TransformerEncoderLayer):
            raise RuntimeError("score_candidate_matrix_one_layer requires a standard TransformerEncoderLayer")
        self_attn = layer.self_attn
        if (
            not isinstance(self_attn, nn.MultiheadAttention)
            or not self_attn.batch_first
            or not getattr(self_attn, "_qkv_same_embed_dim", False)
            or self_attn.in_proj_weight is None
            or self_attn.in_proj_bias is None
            or self_attn.out_proj is None
        ):
            raise RuntimeError("score_candidate_matrix_one_layer requires packed batch-first self-attention projections")
        if self_attn.embed_dim != self.transformer_input_dim:
            raise RuntimeError("score_candidate_matrix_one_layer found a transformer dimension mismatch")
        return layer

    def _candidate_token_self_attention(
        self,
        layer: nn.TransformerEncoderLayer,
        history_input: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_input: torch.Tensor,
    ) -> torch.Tensor:
        self_attn = layer.self_attn
        num_users, max_history_len, embed_dim = history_input.shape
        num_candidates = int(candidate_input.size(0))
        num_heads = int(self_attn.num_heads)
        head_dim = embed_dim // num_heads
        scale = float(head_dim) ** -0.5

        in_proj_weight = self_attn.in_proj_weight
        in_proj_bias = self_attn.in_proj_bias
        if in_proj_weight is None or in_proj_bias is None:
            raise RuntimeError("score_candidate_matrix requires packed self-attention projections")
        q_weight, k_weight, v_weight = in_proj_weight.chunk(3, dim=0)
        q_bias, k_bias, v_bias = in_proj_bias.chunk(3, dim=0)
        query = F.linear(candidate_input, q_weight, q_bias).view(num_candidates, num_heads, head_dim)
        history_key = F.linear(history_input, k_weight, k_bias).view(num_users, max_history_len, num_heads, head_dim)
        history_value = F.linear(history_input, v_weight, v_bias).view(num_users, max_history_len, num_heads, head_dim)
        candidate_key = F.linear(candidate_input, k_weight, k_bias).view(num_candidates, num_heads, head_dim)
        candidate_value = F.linear(candidate_input, v_weight, v_bias).view(num_candidates, num_heads, head_dim)

        history_scores = torch.einsum("cnd,uhnd->unch", query, history_key) * scale
        history_scores = history_scores.masked_fill(~history_mask[:, None, None, :], float("-inf"))
        candidate_scores = (query * candidate_key).sum(dim=-1).transpose(0, 1) * scale
        candidate_scores = candidate_scores.unsqueeze(0).unsqueeze(-1).expand(num_users, -1, -1, -1)
        attention_scores = torch.cat([history_scores, candidate_scores], dim=-1)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = F.dropout(attention_weights, p=self.dropout_rate, training=self.training)

        if max_history_len == 0:
            history_context = torch.zeros(
                (num_users, num_candidates, num_heads, head_dim),
                device=history_input.device,
                dtype=history_input.dtype,
            )
        else:
            history_context = torch.einsum(
                "unch,uhnd->ucnd",
                attention_weights[..., :max_history_len],
                history_value,
            )
        candidate_context = (
            attention_weights[..., max_history_len].permute(0, 2, 1).unsqueeze(-1)
            * candidate_value.unsqueeze(0)
        )
        attention_output = (history_context + candidate_context).reshape(num_users, num_candidates, embed_dim)
        return F.linear(attention_output, self_attn.out_proj.weight, self_attn.out_proj.bias)

    def _candidate_token_feed_forward(
        self,
        layer: nn.TransformerEncoderLayer,
        candidate_state: torch.Tensor,
    ) -> torch.Tensor:
        hidden = F.linear(candidate_state, layer.linear1.weight, layer.linear1.bias)
        hidden = F.gelu(hidden)
        hidden = F.dropout(hidden, p=self.dropout_rate, training=self.training)
        hidden = F.linear(hidden, layer.linear2.weight, layer.linear2.bias)
        return F.dropout(hidden, p=self.dropout_rate, training=self.training)

    def score_candidate_matrix_one_layer(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
        candidate_post_embeddings: torch.Tensor,
        history_author_indices: torch.Tensor,
        candidate_post_author_idx: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_one_layer_matrix_scorer()
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
        if history_mask.shape != (num_users, max_history_len):
            raise ValueError("history_mask must have shape [U, H]")
        if history_time_deltas_hours.shape != (num_users, max_history_len):
            raise ValueError("history_time_deltas_hours must have shape [U, H]")
        if history_author_indices.shape != (num_users, max_history_len):
            raise ValueError("history_author_indices must have shape [U, H]")
        if candidate_post_author_idx.shape != (num_candidates,):
            raise ValueError("candidate_post_author_idx must have shape [C]")

        return self.score_candidate_matrix(
            history_embeddings=history_embeddings,
            history_mask=history_mask,
            history_time_deltas_hours=history_time_deltas_hours,
            candidate_post_embeddings=candidate_post_embeddings,
            history_author_indices=history_author_indices,
            candidate_post_author_idx=candidate_post_author_idx,
        )

    @torch.jit.export
    def score_candidate_matrix(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
        candidate_post_embeddings: torch.Tensor,
        history_author_indices: torch.Tensor,
        candidate_post_author_idx: torch.Tensor,
    ) -> torch.Tensor:
        if len(self.transformer_encoder.layers) != 1:
            raise RuntimeError("score_candidate_matrix requires exactly one transformer layer")
        layer = self.transformer_encoder.layers[0]

        num_users = int(history_embeddings.size(0))
        num_candidates = int(candidate_post_embeddings.size(0))
        device = history_embeddings.device
        history_mask = history_mask.to(device=device, dtype=torch.bool)
        history_time_deltas_hours = history_time_deltas_hours.to(device=device)
        candidate_post_embeddings = candidate_post_embeddings.to(device=device)
        history_author_indices = history_author_indices.to(device=device, dtype=torch.long)
        candidate_post_author_idx = candidate_post_author_idx.to(device=device, dtype=torch.long)

        history_post_vectors = self.post_feature_encoder(history_embeddings, history_author_indices)
        candidate_post_vectors = self.post_feature_encoder(
            candidate_post_embeddings,
            candidate_post_author_idx,
        )
        history_time_bucket_ids = self._bucketize_time_deltas_hours(history_time_deltas_hours)
        history_time_embeddings = self.time_delta_embedding(history_time_bucket_ids)
        candidate_time_bucket_ids = torch.zeros((num_candidates,), device=device, dtype=torch.long)
        candidate_time_embeddings = self.time_delta_embedding(candidate_time_bucket_ids)
        history_input = torch.cat([history_post_vectors, history_time_embeddings], dim=-1)
        candidate_input = torch.cat([candidate_post_vectors, candidate_time_embeddings], dim=-1)

        if layer.norm_first:
            normed_history_input = F.layer_norm(
                history_input,
                [self.transformer_input_dim],
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
            )
            normed_candidate_input = F.layer_norm(
                candidate_input,
                [self.transformer_input_dim],
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
            )
            attention_output = F.dropout(
                self._candidate_token_self_attention(
                    layer,
                    normed_history_input,
                    history_mask,
                    normed_candidate_input,
                ),
                p=self.dropout_rate,
                training=self.training,
            )
            candidate_state = candidate_input.unsqueeze(0) + attention_output
            normed_candidate_state = F.layer_norm(
                candidate_state,
                [self.transformer_input_dim],
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps,
            )
            candidate_state = candidate_state + self._candidate_token_feed_forward(
                layer,
                normed_candidate_state,
            )
        else:
            attention_output = F.dropout(
                self._candidate_token_self_attention(layer, history_input, history_mask, candidate_input),
                p=self.dropout_rate,
                training=self.training,
            )
            candidate_state = F.layer_norm(
                candidate_input.unsqueeze(0) + attention_output,
                [self.transformer_input_dim],
                layer.norm1.weight,
                layer.norm1.bias,
                layer.norm1.eps,
            )
            candidate_state = F.layer_norm(
                candidate_state + self._candidate_token_feed_forward(layer, candidate_state),
                [self.transformer_input_dim],
                layer.norm2.weight,
                layer.norm2.bias,
                layer.norm2.eps,
            )

        logits = self.prediction_head(candidate_state.reshape(num_users * num_candidates, self.transformer_input_dim))
        if logits.dim() == 2 and logits.shape == (num_users * num_candidates, 1):
            logits = logits.squeeze(-1)
        if logits.shape != (num_users * num_candidates,):
            raise RuntimeError("prediction_head must return logits with shape [U*C] or [U*C, 1]")
        return logits.reshape(num_users, num_candidates)

    def forward(
        self,
        history_embeddings: torch.Tensor,
        history_mask: torch.Tensor,
        history_time_deltas_hours: torch.Tensor,
        candidate_post_embeddings: torch.Tensor,
        history_author_indices: torch.Tensor,
        candidate_post_author_idx: torch.Tensor,
    ) -> torch.Tensor:
        transformer_output = self._forward_transformer(
            history_embeddings=history_embeddings,
            history_mask=history_mask,
            history_time_deltas_hours=history_time_deltas_hours,
            candidate_post_embeddings=candidate_post_embeddings,
            history_author_indices=history_author_indices,
            candidate_post_author_idx=candidate_post_author_idx,
        )
        logits = self.prediction_head(transformer_output)
        if logits.dim() == 2 and logits.shape == (transformer_output.size(0), 1):
            logits = logits.squeeze(-1)
        if logits.shape != (transformer_output.size(0),):
            raise RuntimeError("prediction_head must return logits with shape [B] or [B, 1]")
        return logits

def _compute_bst_listwise_loss_and_preds(
    model: BSTRanker,
    batch: Dict[str, Any],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history_embeddings = batch["history_embeddings"].to(device, non_blocking=True)
    history_mask = batch["history_mask"].to(device, non_blocking=True)
    history_time_deltas_hours = batch["history_time_deltas_hours"].to(device, non_blocking=True)
    candidate_post_embeddings = batch["candidate_post_embeddings"].to(device, non_blocking=True)
    labels = batch["label_matrix"].to(device, dtype=torch.float32, non_blocking=True)
    if "history_author_indices" not in batch or "candidate_post_author_idx" not in batch:
        raise RuntimeError("BST listwise batches must include author index tensors")
    history_author_indices = batch["history_author_indices"].to(device, dtype=torch.long, non_blocking=True)
    candidate_post_author_idx = batch["candidate_post_author_idx"].to(device, dtype=torch.long, non_blocking=True)

    scores = model.score_candidate_matrix_one_layer(
        history_embeddings=history_embeddings,
        history_mask=history_mask,
        history_time_deltas_hours=history_time_deltas_hours,
        candidate_post_embeddings=candidate_post_embeddings,
        history_author_indices=history_author_indices,
        candidate_post_author_idx=candidate_post_author_idx,
    )
    if scores.shape != labels.shape:
        raise RuntimeError("Expected BST scores and label_matrix to have matching [num_users, num_candidates] shapes")
    positive_counts = labels.sum(dim=1, keepdim=True)
    if torch.any(positive_counts <= 0):
        raise RuntimeError("Each user row in label_matrix must contain at least one positive candidate")

    targets = labels / positive_counts
    loss_per_user = -(targets * F.log_softmax(scores, dim=1)).sum(dim=1)
    return loss_per_user.mean(), scores, labels


def run_bst_listwise_epoch(
    *,
    train: bool,
    split_name: str,
    model: BSTRanker,
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

            loss, scores, labels = _compute_bst_listwise_loss_and_preds(model, batch, device)
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


class BSTRankerMatrixScorer:
    """Matrix-ranking scorer for an in-memory BST model."""

    def __init__(self, model: BSTRanker):
        self.model = model

    def prepare_for_eval(self, device: str) -> None:
        self.model = self.model.to(device)
        self.model.eval()

    def score_batch(self, batch: Dict[str, Any], device: str) -> MatrixBatchScores:
        loss, scores, _ = _compute_bst_listwise_loss_and_preds(self.model, batch, device)
        return MatrixBatchScores(scores=scores, loss=loss)


def _log_bst_epoch_metrics(
    experiment_tracker: Optional[Any],
    iteration: int,
    train_loss: float,
    val_loss: float,
    val_unseen_loss: float,
) -> None:
    if experiment_tracker is None:
        return
    experiment_tracker.log_scalar("Training Loss History", "Train Loss", float(train_loss), iteration)
    experiment_tracker.log_scalar("Training Loss History", "Validation Loss", float(val_loss), iteration)
    experiment_tracker.log_scalar("Training Loss History", "Validation Unseen Users Loss", float(val_unseen_loss), iteration)


def _listwise_history_metric_names(metrics_top_ks: List[int]) -> List[str]:
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


def _log_bst_listwise_epoch_metrics(
    experiment_tracker: Optional[Any],
    iteration: int,
    train_metrics: Dict[str, Any],
    val_metrics: Dict[str, Any],
    val_unseen_metrics: Dict[str, Any],
    metrics_top_ks: List[int],
    primary_metric_name: str,
) -> None:
    if experiment_tracker is None:
        return
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


def train_bst_ranker_model(
    model: BSTRanker,
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
    bst_max_train_batches_per_epoch: Optional[int] = None,
    experiment_tracker: Optional[Any] = None,
) -> Dict[str, Any]:
    metrics_top_ks = list(metrics_top_ks or [30])
    if not metrics_top_ks:
        raise ValueError("metrics_top_ks must contain at least one value")
    if bst_max_train_batches_per_epoch is not None and bst_max_train_batches_per_epoch <= 0:
        raise ValueError("bst_max_train_batches_per_epoch must be positive when provided")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_scheduler_factor, patience=lr_scheduler_patience
    )

    primary_metric_name = f"val_unseen_ndcg@{metrics_top_ks[0]}"
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_unseen_loss": [],
    }
    listwise_metric_names = _listwise_history_metric_names(metrics_top_ks)
    for split_name in ("train", "val", "val_unseen"):
        for metric_name in listwise_metric_names:
            history[f"{split_name}_{metric_name}"] = []
    best_val_metric = float("-inf")
    best_reset_val_metric = float("-inf")
    best_val_loss = float("inf")
    patience_counter = 0
    best_state_dict = None

    for epoch in tqdm(range(epochs), desc="Training epochs", disable=disable_progress):
        train_loss, train_metrics = run_bst_listwise_epoch(
            train=True,
            split_name="Train",
            model=model,
            device=device,
            dataloader=train_loader,
            optimizer=optimizer,
            disable_progress=disable_progress,
            gradient_clip_max_norm=gradient_clip_max_norm,
            metrics_top_ks=metrics_top_ks,
            max_batches=bst_max_train_batches_per_epoch,
        )
        val_loss, val_metrics = run_bst_listwise_epoch(
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
        val_unseen_loss, val_unseen_metrics = run_bst_listwise_epoch(
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
        _append_split_metrics_to_history(history, "train", train_metrics, listwise_metric_names)
        _append_split_metrics_to_history(history, "val", val_metrics, listwise_metric_names)
        _append_split_metrics_to_history(history, "val_unseen", val_unseen_metrics, listwise_metric_names)

        _log_bst_epoch_metrics(
            experiment_tracker,
            epoch + 1,
            train_loss,
            val_loss,
            val_unseen_loss,
        )
        _log_bst_listwise_epoch_metrics(
            experiment_tracker,
            epoch + 1,
            train_metrics,
            val_metrics,
            val_unseen_metrics,
            metrics_top_ks,
            primary_metric_name,
        )

        primary_metric_key = primary_metric_name.replace("val_unseen_", "", 1)
        primary_metric_value = val_unseen_metrics.get(primary_metric_key)
        primary_metric = float(primary_metric_value) if primary_metric_value is not None else None

        if primary_metric is not None:
            scheduler.step(primary_metric)
        else:
            scheduler.step(float("-inf"))

        better_than_best = (
            primary_metric is not None
            and primary_metric > best_val_metric
        )
        if better_than_best and primary_metric is not None:
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
                    checkpoints_dir / "bst_ranker_best.pth",
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
    log_operation_start("Stage 3 BST ranker training", STAGE_LOG_NAME, logger)
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
    model_dim = int(args.bst_model_dim)
    content_projection_dim = int(args.content_projection_dim)
    author_projection_dim = int(args.author_projection_dim)
    time_embedding_dim = int(args.bst_time_embedding_dim)
    num_attention_heads = int(args.bst_num_attention_heads)
    num_transformer_layers = int(args.bst_num_transformer_layers)
    transformer_ff_dim = int(args.bst_transformer_ff_dim)
    dropout_rate = float(args.bst_dropout_rate)
    norm_first = bool(args.bst_norm_first)
    time_delta_bucket_boundaries_hours = [float(v) for v in args.bst_time_delta_bucket_boundaries_hours]
    if args.prediction_hidden_dims is None:
        raise ValueError("prediction_hidden_dims is required for BST ranker training")
    prediction_hidden_dims = [int(v) for v in args.prediction_hidden_dims]
    use_author_embedding_table = bool(args.use_author_embedding_table)
    author_embedding_dim = int(args.author_embedding_dim)
    author_unknown_dropout_rate = float(args.author_unknown_dropout_rate)
    batch_size = int(args.batch_size)
    bst_additional_batch_negatives = int(args.bst_additional_batch_negatives)
    bst_max_train_batches_per_epoch = getattr(args, "bst_max_train_batches_per_epoch", None)
    if bst_max_train_batches_per_epoch is not None:
        bst_max_train_batches_per_epoch = int(bst_max_train_batches_per_epoch)
    metrics_top_ks = list(args.metrics_top_ks)
    if not metrics_top_ks:
        raise ValueError("metrics_top_ks must contain at least one value")
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.bst_weight_decay)
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

    if num_transformer_layers != 1:
        raise ValueError("BST ranker requires bst_num_transformer_layers=1")

    if not use_author_embedding_table:
        raise ValueError("BST ranker v1 requires use_author_embedding_table=True")
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
        "model_type": "bst-ranker",
        "post_embedding_dim": embed_dim,
        "model_dim": model_dim,
        "content_projection_dim": content_projection_dim,
        "author_projection_dim": author_projection_dim,
        "time_embedding_dim": time_embedding_dim,
        "num_attention_heads": num_attention_heads,
        "num_transformer_layers": num_transformer_layers,
        "transformer_ff_dim": transformer_ff_dim,
        "dropout_rate": dropout_rate,
        "norm_first": norm_first,
        "time_delta_bucket_boundaries_hours": list(time_delta_bucket_boundaries_hours),
        "prediction_hidden_dims": list(prediction_hidden_dims),
        "max_history_len": max_history_len,
        "use_author_embedding_table": use_author_embedding_table,
        "author_embedding_dim": author_embedding_dim,
        "author_unknown_dropout_rate": author_unknown_dropout_rate,
        "author_table_num_rows": author_table_num_rows,
        "author_pad_idx": AUTHOR_PAD_IDX,
        "author_unk_idx": AUTHOR_UNK_IDX,
        "bst_additional_batch_negatives": bst_additional_batch_negatives,
    }
    training_config = {
        **config,
        "batch_size": batch_size,
        "bst_max_train_batches_per_epoch": bst_max_train_batches_per_epoch,
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

    log_operation_start("Create bucketed BST datasets", STAGE_LOG_NAME, logger)
    train_dataset = BucketedEngagementDataset(
        embeddings_mmap=embeddings_mmap,
        likes_core_df=likes_core_df,
        posts_core_df=posts_core_df,
        history_df=history_df,
        split="train",
        max_history_len=max_history_len,
        embed_dim=embed_dim,
        use_author_embedding_table=use_author_embedding_table,
        bst_additional_batch_negatives=bst_additional_batch_negatives,
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
        bst_additional_batch_negatives=bst_additional_batch_negatives,
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
        bst_additional_batch_negatives=bst_additional_batch_negatives,
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

    log_operation_start("Create BST ranker model", STAGE_LOG_NAME, logger)
    model = BSTRanker(
        post_embedding_dim=embed_dim,
        author_table_num_rows=author_table_num_rows,
        author_embedding_dim=author_embedding_dim,
        content_projection_dim=content_projection_dim,
        author_projection_dim=author_projection_dim,
        model_dim=model_dim,
        time_embedding_dim=time_embedding_dim,
        num_attention_heads=num_attention_heads,
        num_transformer_layers=num_transformer_layers,
        transformer_ff_dim=transformer_ff_dim,
        dropout_rate=dropout_rate,
        author_unknown_dropout_rate=author_unknown_dropout_rate,
        norm_first=norm_first,
        time_delta_bucket_boundaries_hours=time_delta_bucket_boundaries_hours,
        prediction_hidden_dims=prediction_hidden_dims,
    )

    log_operation_start(f"Train BST ranker (epochs={epochs}, batch_size={batch_size})", STAGE_LOG_NAME, logger)
    training_results = train_bst_ranker_model(
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
        bst_max_train_batches_per_epoch=bst_max_train_batches_per_epoch,
        experiment_tracker=context.tracker,
    )
    trained_model: BSTRanker = training_results["model"]
    clear_cuda_memory()

    model_path = None
    if save_model:
        log_operation_start("Save BST ranker checkpoint", STAGE_LOG_NAME, logger)
        model_path = checkpoints_dir / f"bst_ranker_{timestamp}.pth"
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

        torchscript_model = copy.deepcopy(trained_model).cpu().eval()
        torchscript_name = "ranker"
        torchscript_path = checkpoints_dir / f"{torchscript_name}.pt"
        torch.jit.script(torchscript_model).save(torchscript_path)
        ranker_model_metadata = context.tracker.log_artifact(name=torchscript_name, path=torchscript_path)
        ranker_model_id = ranker_model_metadata.get("model_id", "")
        ranker_uri = ranker_model_metadata.get("uri", "")
        logger.info(f"Ranker model id: {ranker_model_id}")
        logger.info(f"Ranker model URI: {ranker_uri}")

        manifest = {
            "ranker_clearml_model_id": ranker_model_id,
            "ranker_uri": ranker_uri,
            "clearml_task_id": getattr(context.tracker, "id", ""),
        }
        manifest_path = checkpoints_dir / "ranker_serving_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        try:
            manifest_uri = context.tracker.log_file_artifact("ranker_serving_manifest", manifest_path)
            logger.info(f"Ranker serving manifest artifact id: {manifest_uri}")
        except Exception:
            logger.exception("Failed to upload ranker serving manifest; continuing without manifest artifact.")

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
            if primary_metric_name.endswith("_loss"):
                best_epoch = min(valid_metrics, key=lambda item: item[1])[0] if valid_metrics else None
            else:
                best_epoch = max(valid_metrics, key=lambda item: item[1])[0] if valid_metrics else None
        except Exception as exc:
            logger.warning(f"Could not determine best epoch from BST training history: {exc}")
            best_epoch = None
        plot_training_history(hist, plots_dir / f"training_history_{timestamp}.png", best_epoch=best_epoch)

    bst_matrix_scorer = BSTRankerMatrixScorer(trained_model)
    train_eval = evaluate_matrix_scorer(
        bst_matrix_scorer,
        train_loader,
        device=device,
        metrics_top_ks=metrics_top_ks,
        progress_desc="Evaluate train",
        disable_progress=disable_progress,
        max_batches=bst_max_train_batches_per_epoch,
    )
    val_eval = evaluate_matrix_scorer(
        bst_matrix_scorer,
        val_loader,
        device=device,
        metrics_top_ks=metrics_top_ks,
        progress_desc="Evaluate validation",
        disable_progress=disable_progress,
    )
    val_unseen_eval = evaluate_matrix_scorer(
        bst_matrix_scorer,
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
        "stage: train_bst_ranker",
        f"timestamp: {timestamp}",
        f"runtime_seconds: {runtime:.2f}",
        f"settings: batch_size={batch_size}, bst_additional_batch_negatives={bst_additional_batch_negatives}, lr={learning_rate}, epochs={epochs}, max_history_len={max_history_len}, early_stopping_min_delta={early_stopping_min_delta}",
        f"train_samples: {len(train_dataset)}",
        f"val_samples: {len(val_dataset)}",
        f"val_unseen_samples: {len(val_unseen_dataset)}",
        f"primary_metric_name: {training_results['primary_metric_name']}",
        f"best_val_metric: {training_results['best_val_metric']:.4f}",
    ]
    info_lines.extend(stage_info_metric_lines(final_split_metrics))
    (out_dir / "stage_info.txt").write_text("\n".join(info_lines) + "\n")

    logger.info(f"BST ranker training completed in {runtime:.2f}s")

    return {
        "output_dir": out_dir,
        "artifacts": {
            "model_path": str(model_path) if model_path else None,
            "training_config": str(training_config_path),
            "training_results": str(training_results_path),
        },
    }
