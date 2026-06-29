"""Shared training/evaluation helpers for bucketed user-candidate ranking."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import polars as pl
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

DEFAULT_MAX_CLASSIFICATION_METRIC_PAIRS = 2_000_000
FINAL_CLASSIFICATION_METRICS = ("auc_roc", "classification_average_precision")
CLASSIFICATION_METRIC_ALIASES = {
    "classification_average_precision": ("classification_average_precision", "average_precision"),
}


@dataclass
class MatrixBatchScores:
    scores: torch.Tensor
    loss: Optional[torch.Tensor] = None


class MatrixRankingScorer(Protocol):
    def prepare_for_eval(self, device: str) -> None:
        ...

    def score_batch(self, batch: Dict[str, Any], device: str) -> MatrixBatchScores:
        ...


class TorchMatrixModelScorer:
    def __init__(self, model: torch.nn.Module, embed_dim: int):
        self.model = model
        self.embed_dim = int(embed_dim)

    def prepare_for_eval(self, device: str) -> None:
        self.model = self.model.to(device)
        self.model.eval()

    def score_batch(self, batch: Dict[str, Any], device: str) -> MatrixBatchScores:
        loss, scores = self.model.compute_loss_and_preds(batch, device, self.embed_dim)
        return MatrixBatchScores(scores=scores, loss=loss)


def candidate_valid_mask_for_batch(batch: Dict[str, Any], labels: torch.Tensor) -> torch.Tensor:
    if "candidate_valid_mask" not in batch:
        raise RuntimeError("Batch is missing required candidate_valid_mask")
    mask_value = batch["candidate_valid_mask"]
    if mask_value is None:
        raise RuntimeError("Batch is missing required candidate_valid_mask")
    if isinstance(mask_value, torch.Tensor):
        candidate_valid_mask = mask_value.to(device=labels.device, dtype=torch.bool, non_blocking=True)
    else:
        candidate_valid_mask = torch.as_tensor(mask_value, device=labels.device, dtype=torch.bool)
    if candidate_valid_mask.shape != labels.shape:
        raise RuntimeError("Expected candidate_valid_mask and label_matrix to have matching [num_users, num_candidates] shapes")
    return candidate_valid_mask | (labels > 0)


def mask_scores_for_valid_candidates(scores: torch.Tensor, candidate_valid_mask: torch.Tensor) -> torch.Tensor:
    if scores.shape != candidate_valid_mask.shape:
        raise RuntimeError("Expected scores and candidate_valid_mask to have matching [num_users, num_candidates] shapes")
    return scores.masked_fill(~candidate_valid_mask, -1.0e9)


def empty_rank_metric_sums(metrics_top_ks: list[int]) -> Dict[str, float]:
    metric_sums = {f"dcg@{k}": 0.0 for k in metrics_top_ks}
    metric_sums.update({f"ndcg@{k}": 0.0 for k in metrics_top_ks})
    metric_sums.update({f"recall@{k}": 0.0 for k in metrics_top_ks})
    metric_sums["mean_average_precision"] = 0.0
    return metric_sums


def calc_baseline_rank_metrics_for_batch(
    unranked_labels: torch.Tensor,
    metrics_top_ks: list[int],
    candidate_valid_mask: torch.Tensor,
) -> Tuple[Dict[str, float], int]:
    """Calculate expected rank metrics for a uniformly random candidate order."""
    if unranked_labels.dim() != 2:
        raise RuntimeError("unranked_labels must have shape [num_users, num_candidates]")

    with torch.no_grad():
        labels = unranked_labels.to(dtype=torch.float32)
        candidate_valid_mask = candidate_valid_mask.to(device=labels.device, dtype=torch.bool)
        if candidate_valid_mask.shape != labels.shape:
            raise RuntimeError("Expected candidate_valid_mask and unranked_labels to have matching [num_users, num_candidates] shapes")
        candidate_valid_mask = candidate_valid_mask | (labels > 0)
        metric_sums = empty_rank_metric_sums(metrics_top_ks)
        eligible_count = 0
        max_requested_k = max(metrics_top_ks)
        for row_idx in range(labels.size(0)):
            row_labels = labels[row_idx][candidate_valid_mask[row_idx]]
            total_relevant = row_labels.sum()
            if total_relevant <= 0:
                continue
            num_candidates = int(row_labels.numel())
            if num_candidates <= 0:
                continue
            eligible_count += 1
            max_k = min(max_requested_k, num_candidates)
            discounts = 1.0 / torch.log2(
                torch.arange(max_k, device=labels.device, dtype=torch.float32) + 2.0
            )
            cumulative_discounts = discounts.cumsum(dim=0)
            relevant_probability = total_relevant / float(num_candidates)
            positions = torch.arange(1, num_candidates + 1, device=labels.device, dtype=torch.float32)
            if num_candidates == 1:
                expected_average_precision = torch.ones_like(total_relevant)
            else:
                harmonic_sum = (1.0 / positions).sum()
                tail_sum = ((positions - 1.0) / positions).sum()
                expected_average_precision = (
                    harmonic_sum + tail_sum * ((total_relevant - 1.0) / float(num_candidates - 1))
                ) / float(num_candidates)
            metric_sums["mean_average_precision"] += float(expected_average_precision.item())

            for k in metrics_top_ks:
                k_eff = min(k, num_candidates)
                discount_sum = cumulative_discounts[k_eff - 1]
                dcg = relevant_probability * discount_sum
                ideal_count = int(min(float(total_relevant.item()), float(k_eff)))
                idcg = cumulative_discounts[ideal_count - 1].clamp(min=1.0e-12)
                recall = float(k_eff) / float(num_candidates)

                metric_sums[f"dcg@{k}"] += float(dcg.item())
                metric_sums[f"ndcg@{k}"] += float((dcg / idcg).item())
                metric_sums[f"recall@{k}"] += recall

        return metric_sums, eligible_count


def rank_metric_sums_for_batch(
    ranked_labels: torch.Tensor,
    metrics_top_ks: list[int],
    ranked_valid_mask: torch.Tensor,
) -> Tuple[Dict[str, float], int]:
    """Return summed per-user rank metrics for one [users, ranked_candidates] batch."""
    if ranked_labels.dim() != 2:
        raise RuntimeError("ranked_labels must have shape [num_users, num_candidates]")

    with torch.no_grad():
        ranked_labels = ranked_labels.to(dtype=torch.float32)
        ranked_valid_mask = ranked_valid_mask.to(device=ranked_labels.device, dtype=torch.bool)
        if ranked_valid_mask.shape != ranked_labels.shape:
            raise RuntimeError("Expected ranked_valid_mask and ranked_labels to have matching [num_users, num_candidates] shapes")
        ranked_valid_mask = ranked_valid_mask | (ranked_labels > 0)
        metric_sums = empty_rank_metric_sums(metrics_top_ks)
        eligible_count = 0
        max_requested_k = max(metrics_top_ks)
        for row_idx in range(ranked_labels.size(0)):
            row_labels = ranked_labels[row_idx][ranked_valid_mask[row_idx]]
            total_relevant = row_labels.sum()
            if total_relevant <= 0:
                continue
            eligible_count += 1
            positions = torch.arange(
                1,
                row_labels.numel() + 1,
                device=ranked_labels.device,
                dtype=torch.float32,
            )
            cumulative_relevant = row_labels.cumsum(dim=0)
            precision_at_rank = cumulative_relevant / positions
            average_precision = (precision_at_rank * row_labels).sum() / total_relevant
            metric_sums["mean_average_precision"] += float(average_precision.item())

            max_k = min(max_requested_k, row_labels.numel())
            discounts = 1.0 / torch.log2(
                torch.arange(max_k, device=ranked_labels.device, dtype=torch.float32) + 2.0
            )
            cumulative_discounts = discounts.cumsum(dim=0)
            for k in metrics_top_ks:
                k_eff = min(k, row_labels.numel())
                top_labels = row_labels[:k_eff]
                dcg = (top_labels * discounts[:k_eff]).sum()
                ideal_count = int(min(float(total_relevant.item()), float(k_eff)))
                idcg = cumulative_discounts[ideal_count - 1].clamp(min=1.0e-12)
                recall = top_labels.sum() / total_relevant

                metric_sums[f"dcg@{k}"] += float(dcg.item())
                metric_sums[f"ndcg@{k}"] += float((dcg / idcg).item())
                metric_sums[f"recall@{k}"] += float(recall.item())

        return metric_sums, eligible_count


def finalize_rank_metrics(metric_sums: Dict[str, float], user_count: int) -> Dict[str, float]:
    if user_count <= 0:
        return {key: 0.0 for key in metric_sums}
    return {
        key: value / user_count
        for key, value in metric_sums.items()
    }


def ranking_rows_for_batch(
    batch: Dict[str, Any],
    scores: torch.Tensor,
    labels: torch.Tensor,
    metrics_top_ks: list[int],
    candidate_valid_mask: torch.Tensor,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    user_ids = batch["user_id"]
    bucket = batch["bucket"]
    history_mask = batch["history_mask"].detach().cpu()
    labels_cpu = labels.detach().cpu()
    scores_cpu = scores.detach().cpu()
    valid_mask_cpu = candidate_valid_mask.detach().cpu().to(dtype=torch.bool)
    if valid_mask_cpu.shape != labels_cpu.shape:
        raise RuntimeError("Expected candidate_valid_mask and labels to have matching [num_users, num_candidates] shapes")
    valid_mask_cpu = valid_mask_cpu | (labels_cpu > 0)

    for user_idx, user_id in enumerate(user_ids):
        row_valid_mask = valid_mask_cpu[user_idx]
        row_labels = labels_cpu[user_idx][row_valid_mask].to(dtype=torch.float32)
        row_scores = scores_cpu[user_idx][row_valid_mask].to(dtype=torch.float32)
        ranked_indices = torch.argsort(row_scores, descending=True)
        ranked_labels = row_labels[ranked_indices]
        positive_count = int(row_labels.sum().item())
        candidate_count = int(row_labels.numel())

        row: Dict[str, Any] = {
            "did": str(user_id),
            "like_hour_bucket": bucket,
            "num_embedding_likes": int(history_mask[user_idx].sum().item()),
            "candidate_count": candidate_count,
            "positive_count": positive_count,
        }

        if positive_count > 0:
            max_k = min(max(metrics_top_ks), candidate_count)
            discounts = 1.0 / torch.log2(torch.arange(max_k, dtype=torch.float32) + 2.0)
            cumulative_discounts = discounts.cumsum(dim=0)
            positive_rank_positions = torch.nonzero(ranked_labels > 0, as_tuple=False).flatten() + 1
            positive_ranks = positive_rank_positions.to(dtype=torch.float32)
            row["positive_rank_min"] = float(positive_ranks.min().item())
            row["positive_rank_mean"] = float(positive_ranks.mean().item())
            row["positive_rank_max"] = float(positive_ranks.max().item())

            for k in metrics_top_ks:
                k_eff = min(k, candidate_count)
                top_labels = ranked_labels[:k_eff]
                dcg = float((top_labels * discounts[:k_eff]).sum().item())
                ideal_count = min(positive_count, k_eff)
                idcg = float(cumulative_discounts[ideal_count - 1].item()) if ideal_count > 0 else 0.0
                row[f"dcg@{k}"] = dcg
                row[f"ndcg@{k}"] = dcg / max(idcg, 1.0e-12)
                row[f"recall@{k}"] = float(top_labels.sum().item()) / positive_count

            y_true = row_labels.numpy()
            y_score = row_scores.numpy()
            row["average_precision"] = float(average_precision_score(y_true, y_score))
            row["auc_roc"] = float(roc_auc_score(y_true, y_score)) if np.unique(y_true).size > 1 else None
        else:
            row["positive_rank_min"] = None
            row["positive_rank_mean"] = None
            row["positive_rank_max"] = None
            for k in metrics_top_ks:
                row[f"dcg@{k}"] = 0.0
                row[f"ndcg@{k}"] = 0.0
                row[f"recall@{k}"] = 0.0
            row["average_precision"] = None
            row["auc_roc"] = None

        rows.append(row)

    return rows


def run_matrix_epoch(
    train: bool,
    split_name: str,
    model: torch.nn.Module,
    device: str,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    disable_progress: bool,
    embed_dim: int,
    gradient_clip_max_norm: float,
    metrics_top_ks: list[int],
    calc_baseline_metrics: bool,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    if train:
        if optimizer is None:
            raise ValueError("optimizer is required when train=True")
        model.train()
    else:
        model.eval()

    loss_sum = torch.zeros((), device=device)
    batches = 0
    baseline_metric_sums = empty_rank_metric_sums(metrics_top_ks)
    baseline_metric_user_count = 0
    metric_sums = empty_rank_metric_sums(metrics_top_ks)
    metric_user_count = 0

    with nullcontext() if train else torch.inference_mode():
        for batch in tqdm(dataloader, desc=split_name, leave=False, disable=disable_progress):
            if train and optimizer is not None:
                optimizer.zero_grad()

            loss, scores = model.compute_loss_and_preds(batch, device, embed_dim)
            labels = batch["label_matrix"].to(device, dtype=torch.float32, non_blocking=True)
            candidate_valid_mask = candidate_valid_mask_for_batch(batch, labels)

            if calc_baseline_metrics:
                baseline_batch_metric_sums, baseline_batch_metric_user_count = calc_baseline_rank_metrics_for_batch(
                    labels,
                    metrics_top_ks,
                    candidate_valid_mask,
                )
                baseline_metric_user_count += baseline_batch_metric_user_count
                for key, value in baseline_batch_metric_sums.items():
                    baseline_metric_sums[key] += value

            ranked_indices = torch.argsort(scores.detach(), dim=1, descending=True)
            ranked_labels = torch.gather(labels, dim=1, index=ranked_indices)
            ranked_valid_mask = torch.gather(candidate_valid_mask, dim=1, index=ranked_indices)
            batch_metric_sums, batch_metric_user_count = rank_metric_sums_for_batch(
                ranked_labels,
                metrics_top_ks,
                ranked_valid_mask,
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
    baseline_metrics_dict = finalize_rank_metrics(baseline_metric_sums, baseline_metric_user_count)
    metrics_dict = finalize_rank_metrics(metric_sums, metric_user_count)
    return loss, metrics_dict, baseline_metrics_dict


def evaluate_matrix_scorer(
    scorer: MatrixRankingScorer,
    data_loader: DataLoader,
    device: str,
    metrics_top_ks: list[int],
    max_classification_metric_pairs: Optional[int] = DEFAULT_MAX_CLASSIFICATION_METRIC_PAIRS,
    collect_ranking_rows: bool = False,
    progress_desc: Optional[str] = None,
    disable_progress: bool = True,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a matrix-ranking scorer with streamed rank metrics and sampled AUC/AP."""
    scorer.prepare_for_eval(device)

    loss_sum = torch.zeros((), device=device)
    loss_batches = 0
    metric_sums = empty_rank_metric_sums(metrics_top_ks)
    metric_user_count = 0
    classification_pair_count = 0
    classification_positive_count = 0
    metric_labels: Optional[np.ndarray] = None
    metric_scores: Optional[np.ndarray] = None
    metric_priorities: Optional[np.ndarray] = None
    ranking_rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(0)

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc=progress_desc, leave=False, disable=disable_progress)):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch_scores = scorer.score_batch(batch, device)
            scores = batch_scores.scores.to(device)
            labels = batch["label_matrix"].to(device, dtype=torch.float32, non_blocking=True)
            if scores.shape != labels.shape:
                raise RuntimeError("Expected scores and label_matrix to have matching [num_users, num_candidates] shapes")
            candidate_valid_mask = candidate_valid_mask_for_batch(batch, labels)

            ranked_indices = torch.argsort(scores, dim=1, descending=True)
            ranked_labels = torch.gather(labels, dim=1, index=ranked_indices)
            ranked_valid_mask = torch.gather(candidate_valid_mask, dim=1, index=ranked_indices)
            batch_metric_sums, batch_metric_user_count = rank_metric_sums_for_batch(
                ranked_labels,
                metrics_top_ks,
                ranked_valid_mask,
            )

            if batch_scores.loss is not None:
                loss_sum += batch_scores.loss.detach().to(device)
                loss_batches += 1
            metric_user_count += batch_metric_user_count
            for key, value in batch_metric_sums.items():
                metric_sums[key] += value

            if collect_ranking_rows:
                ranking_rows.extend(ranking_rows_for_batch(batch, scores, labels, metrics_top_ks, candidate_valid_mask))

            flat_labels = labels[candidate_valid_mask].detach().cpu().numpy().astype(np.int8, copy=False)
            flat_scores = scores[candidate_valid_mask].detach().cpu().numpy().astype(np.float64, copy=False)
            classification_pair_count += int(flat_labels.size)
            classification_positive_count += int(flat_labels.sum())
            if max_classification_metric_pairs is None:
                if metric_labels is None or metric_scores is None:
                    metric_labels = flat_labels
                    metric_scores = flat_scores
                else:
                    metric_labels = np.concatenate([metric_labels, flat_labels])
                    metric_scores = np.concatenate([metric_scores, flat_scores])
            elif max_classification_metric_pairs > 0:
                flat_priorities = rng.random(flat_labels.size)
                if metric_labels is None or metric_scores is None or metric_priorities is None:
                    metric_labels = flat_labels
                    metric_scores = flat_scores
                    metric_priorities = flat_priorities
                else:
                    metric_labels = np.concatenate([metric_labels, flat_labels])
                    metric_scores = np.concatenate([metric_scores, flat_scores])
                    metric_priorities = np.concatenate([metric_priorities, flat_priorities])
                if metric_labels is not None and metric_scores is not None and metric_priorities is not None: 
                    if metric_labels.size > max_classification_metric_pairs:
                        keep_idx = np.argpartition(metric_priorities, max_classification_metric_pairs - 1)[:max_classification_metric_pairs]
                        metric_labels = metric_labels[keep_idx]
                        metric_scores = metric_scores[keep_idx]
                        metric_priorities = metric_priorities[keep_idx]

    metrics: Dict[str, Any] = finalize_rank_metrics(metric_sums, metric_user_count)
    metrics["loss"] = (loss_sum / loss_batches).item() if loss_batches > 0 else None
    metrics["rank_metric_user_count"] = metric_user_count
    metrics["classification_metric_pair_count"] = classification_pair_count
    metrics["classification_metric_positive_count"] = classification_positive_count
    metrics["classification_metric_sampled_pair_count"] = int(metric_labels.size) if metric_labels is not None else 0
    metrics["classification_metric_sampled"] = (
        max_classification_metric_pairs is not None
        and classification_pair_count > max_classification_metric_pairs
    )
    if metric_labels is not None and np.unique(metric_labels).size > 1 and metric_scores is not None:
        metrics["auc_roc"] = float(roc_auc_score(metric_labels, metric_scores))
    else:
        metrics["auc_roc"] = None
    if metric_labels is not None and int(metric_labels.sum()) > 0 and metric_scores is not None:
        metrics["classification_average_precision"] = float(average_precision_score(metric_labels, metric_scores))
    else:
        metrics["classification_average_precision"] = None

    return {
        "metrics": metrics,
        "ranking_rows": ranking_rows,
    }


def evaluate_matrix_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
    embed_dim: int,
    metrics_top_ks: list[int],
    max_classification_metric_pairs: Optional[int] = DEFAULT_MAX_CLASSIFICATION_METRIC_PAIRS,
    collect_ranking_rows: bool = False,
    progress_desc: Optional[str] = None,
    disable_progress: bool = True,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate a matrix-ranking model with streamed rank metrics and sampled AUC/AP."""
    return evaluate_matrix_scorer(
        TorchMatrixModelScorer(model, embed_dim),
        data_loader,
        device,
        metrics_top_ks,
        max_classification_metric_pairs=max_classification_metric_pairs,
        collect_ranking_rows=collect_ranking_rows,
        progress_desc=progress_desc,
        disable_progress=disable_progress,
        max_batches=max_batches,
    )


def optional_float_metric(value: Any) -> Optional[float]:
    if value is None:
        return None
    metric_value = float(value)
    if math.isnan(metric_value):
        return None
    return metric_value


def optional_metric_value(metrics: Dict[str, Any], metric_name: str) -> Optional[float]:
    for key in CLASSIFICATION_METRIC_ALIASES.get(metric_name, (metric_name,)):
        metric_value = optional_float_metric(metrics.get(key))
        if metric_value is not None:
            return metric_value
    return None


def split_metric_label(split_name: str) -> str:
    return split_name.replace("_", " ").title()


def clearml_metric_label(metric_name: str) -> str:
    return {
        "auc_roc": "AUC-ROC",
        "classification_average_precision": "Classification Average Precision",
    }.get(metric_name, metric_name.replace("_", " ").title())


def log_final_classification_metrics(
    experiment_tracker: Optional[Any],
    split_metrics: Dict[str, Dict[str, Any]],
    iteration: int,
) -> None:
    if experiment_tracker is None:
        return
    for split_name, metrics in split_metrics.items():
        for metric_name in FINAL_CLASSIFICATION_METRICS:
            metric_value = optional_metric_value(metrics, metric_name)
            if metric_value is None:
                continue
            metric_label = clearml_metric_label(metric_name)
            experiment_tracker.log_scalar(
                title=f"Final {metric_label} by Split",
                series=f"{split_metric_label(split_name)} {metric_label}",
                value=metric_value,
                iteration=iteration,
            )


def stage_info_metric_lines(split_metrics: Dict[str, Dict[str, Any]]) -> List[str]:
    lines = []
    for split_name, metrics in split_metrics.items():
        for metric_name in FINAL_CLASSIFICATION_METRICS:
            metric_value = optional_metric_value(metrics, metric_name)
            if metric_value is not None:
                lines.append(f"{split_name}_{metric_name}: {metric_value:.4f}")
    return lines


def write_ranking_rows(
    rows: List[Dict[str, Any]],
    output_path: Path,
    split_name: str,
    num_total_likes_by_user: Dict[str, int],
) -> None:
    enriched_rows = []
    for row in rows:
        enriched = dict(row)
        user_id = str(enriched["did"])
        enriched["split"] = split_name
        enriched["num_total_likes"] = int(num_total_likes_by_user.get(user_id, 0))
        enriched_rows.append(enriched)
    pl.DataFrame(enriched_rows).write_parquet(output_path)
