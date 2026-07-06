"""Checkpoint-backed matrix-ranking adapters for cross-model evaluation."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from utils.matrix_ranking import MatrixBatchScores


def _load_checkpoint(checkpoint_path: Path) -> Dict[str, Any]:
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint at {checkpoint_path} to contain a dictionary")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint at {checkpoint_path} is missing model_state_dict")
    return checkpoint


def _load_checkpoint_config(checkpoint: Dict[str, Any], checkpoint_path: Path) -> Dict[str, Any]:
    config = checkpoint.get("config")
    if isinstance(config, dict):
        return config

    training_config_path = checkpoint_path.parent.parent / "training_config.json"
    if training_config_path.exists():
        loaded_config = json.loads(training_config_path.read_text())
        if isinstance(loaded_config, dict):
            return loaded_config

    raise ValueError(
        f"Checkpoint at {checkpoint_path} is missing config, and no training_config.json "
        f"was found at {training_config_path}"
    )


def load_checkpoint_config(checkpoint_path: str | Path) -> Dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path)
    return _load_checkpoint_config(checkpoint, checkpoint_path)


def _require_config(config: Dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"Model config is missing required key: {key}")
    return config[key]


class TwoTowerPthAdapter:
    """Matrix-ranking scorer for saved TwoTowerModel .pth checkpoints."""

    def __init__(self, checkpoint_path: str | Path, config_overrides: Optional[Dict[str, Any]] = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.config_overrides = dict(config_overrides or {})
        self.model: Optional[torch.nn.Module] = None
        self.config: Optional[Dict[str, Any]] = None

    def prepare_for_eval(self, device: str) -> None:
        if self.model is None:
            checkpoint = _load_checkpoint(self.checkpoint_path)
            config = _load_checkpoint_config(checkpoint, self.checkpoint_path)
            config = {**config, **self.config_overrides}
            if config.get("model_type") not in ("two_tower", "two-tower"):
                raise ValueError(f"Expected two-tower checkpoint, got model_type={config.get('model_type')!r}")

            stage_train_two_tower = importlib.import_module("utils.03_train.stage_train_two_tower")
            model = stage_train_two_tower.TwoTowerModel(
                post_embedding_dim=int(_require_config(config, "post_embedding_dim")),
                shared_dim=int(_require_config(config, "shared_dim")),
                user_hidden_dim=int(_require_config(config, "user_hidden_dim")),
                post_hidden_dim=int(_require_config(config, "post_hidden_dim")),
                num_attention_heads=int(_require_config(config, "num_attention_heads")),
                num_attention_layers=int(_require_config(config, "num_attention_layers")),
                max_history_len=int(_require_config(config, "max_history_len")),
                dropout_rate=float(_require_config(config, "dropout_rate")),
                l2_normalize_embeddings=bool(_require_config(config, "l2_normalize_embeddings")),
                similarity_temperature=float(_require_config(config, "similarity_temperature")),
                user_encoder_type=str(_require_config(config, "user_encoder_type")),
                use_post_encoder=bool(_require_config(config, "use_post_encoder")),
                use_author_embedding_table=bool(config.get("use_author_embedding_table", False)),
                author_table_num_rows=(
                    int(config["author_table_num_rows"])
                    if config.get("use_author_embedding_table")
                    else None
                ),
                author_embedding_dim=(
                    int(config["author_embedding_dim"])
                    if config.get("use_author_embedding_table")
                    else None
                ),
                content_projection_dim=(
                    int(_require_config(config, "content_projection_dim"))
                    if config.get("use_author_embedding_table")
                    else None
                ),
                author_projection_dim=(
                    int(_require_config(config, "author_projection_dim"))
                    if config.get("use_author_embedding_table")
                    else None
                ),
                author_unknown_dropout_rate=float(config.get("author_unknown_dropout_rate") or 0.0),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            self.model = model
            self.config = config

        if self.model is None:
            raise ValueError("Problem loading Two Tower model!")
        self.model = self.model.to(device)
        self.model.eval()

    def score_batch(self, batch: Dict[str, Any], device: str) -> MatrixBatchScores:
        if self.model is None:
            self.prepare_for_eval(device)
        if self.model is None:
            raise RuntimeError("TwoTowerPthAdapter model was not initialized")

        history_author_indices = None
        candidate_post_author_idx = None
        history_embeddings = batch["history_embeddings"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        candidate_post_embeddings = batch["candidate_post_embeddings"].to(device, non_blocking=True)
        if bool(getattr(self.model, "use_author_embedding_table", False)):
            if "history_author_indices" not in batch or "candidate_post_author_idx" not in batch:
                raise RuntimeError("TwoTowerPthAdapter requires author tensors for this checkpoint")
            history_author_indices = batch["history_author_indices"].to(device, dtype=torch.long, non_blocking=True)
            candidate_post_author_idx = batch["candidate_post_author_idx"].to(device, dtype=torch.long, non_blocking=True)

        scores = self.model(
            history_embeddings,
            history_mask,
            candidate_post_embeddings,
            history_author_indices,
            candidate_post_author_idx,
        )
        return MatrixBatchScores(scores=scores)


class BstPthAdapter:
    """Matrix-ranking scorer for saved BSTRanker .pth checkpoints."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        candidate_chunk_size: int,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        if candidate_chunk_size <= 0:
            raise ValueError("candidate_chunk_size must be positive")
        self.checkpoint_path = Path(checkpoint_path)
        self.candidate_chunk_size = int(candidate_chunk_size)
        self.config_overrides = dict(config_overrides or {})
        self.model: Optional[torch.nn.Module] = None
        self.config: Optional[Dict[str, Any]] = None

    def prepare_for_eval(self, device: str) -> None:
        if self.model is None:
            checkpoint = _load_checkpoint(self.checkpoint_path)
            config = _load_checkpoint_config(checkpoint, self.checkpoint_path)
            config = {**config, **self.config_overrides}
            if config.get("model_type") != "bst-ranker":
                raise ValueError(f"Expected bst-ranker checkpoint, got model_type={config.get('model_type')!r}")
            if not bool(config.get("use_author_embedding_table", False)):
                raise ValueError("BstPthAdapter requires a BST checkpoint trained with author embeddings")
            use_popularity_feature = bool(config.get("bst_use_popularity_feature", False))

            stage_train_bst_ranker = importlib.import_module("utils.03_train.stage_train_bst_ranker")
            model = stage_train_bst_ranker.BSTRanker(
                post_embedding_dim=int(_require_config(config, "post_embedding_dim")),
                author_table_num_rows=int(_require_config(config, "author_table_num_rows")),
                author_embedding_dim=int(_require_config(config, "author_embedding_dim")),
                content_projection_dim=int(_require_config(config, "content_projection_dim")),
                author_projection_dim=int(_require_config(config, "author_projection_dim")),
                model_dim=int(_require_config(config, "model_dim")),
                time_embedding_dim=int(_require_config(config, "time_embedding_dim")),
                num_attention_heads=int(_require_config(config, "num_attention_heads")),
                num_transformer_layers=int(_require_config(config, "num_transformer_layers")),
                transformer_ff_dim=int(_require_config(config, "transformer_ff_dim")),
                dropout_rate=float(_require_config(config, "dropout_rate")),
                author_unknown_dropout_rate=float(config.get("author_unknown_dropout_rate") or 0.0),
                norm_first=bool(_require_config(config, "norm_first")),
                time_delta_bucket_boundaries_hours=list(_require_config(config, "time_delta_bucket_boundaries_hours")),
                prediction_hidden_dims=list(_require_config(config, "prediction_hidden_dims")),
                use_popularity_feature=use_popularity_feature,
                popularity_projection_dim=int(config.get("bst_popularity_projection_dim") or 0),
                popularity_log_mean=float(config.get("bst_popularity_log_mean") or 0.0),
                popularity_log_std=float(config.get("bst_popularity_log_std") or 1.0),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            self.model = model
            self.config = config

        if self.model is None:
            raise ValueError("Problem loading BST model!")
        self.model = self.model.to(device)
        self.model.eval()

    def score_batch(self, batch: Dict[str, Any], device: str) -> MatrixBatchScores:
        if self.model is None:
            self.prepare_for_eval(device)
        if self.model is None:
            raise RuntimeError("BstPthAdapter model was not initialized")

        required_fields = (
            "history_embeddings",
            "history_mask",
            "history_time_deltas_hours",
            "candidate_post_embeddings",
            "history_author_indices",
            "candidate_post_author_idx",
        )
        if self.model.use_popularity_feature:
            required_fields = required_fields + (
                "history_prior_cumulative_likes",
                "candidate_prior_cumulative_likes",
            )
        missing = [field for field in required_fields if field not in batch]
        if missing:
            raise RuntimeError(f"BstPthAdapter batch is missing required fields: {', '.join(missing)}")

        history_embeddings = batch["history_embeddings"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        history_time_deltas_hours = batch["history_time_deltas_hours"].to(device, non_blocking=True)
        candidate_post_embeddings = batch["candidate_post_embeddings"].to(device, non_blocking=True)
        history_author_indices = batch["history_author_indices"].to(device, dtype=torch.long, non_blocking=True)
        candidate_post_author_idx = batch["candidate_post_author_idx"].to(device, dtype=torch.long, non_blocking=True)
        history_prior_cumulative_likes = None
        candidate_prior_cumulative_likes = None
        if self.model.use_popularity_feature:
            history_prior_cumulative_likes = batch["history_prior_cumulative_likes"].to(device, dtype=torch.float32, non_blocking=True)
            candidate_prior_cumulative_likes = batch["candidate_prior_cumulative_likes"].to(device, dtype=torch.float32, non_blocking=True)

        num_candidates = int(candidate_post_embeddings.size(0))
        score_chunks = []
        for start in range(0, num_candidates, self.candidate_chunk_size):
            end = min(start + self.candidate_chunk_size, num_candidates)
            candidate_embeddings_chunk = candidate_post_embeddings[start:end]
            candidate_author_chunk = candidate_post_author_idx[start:end]
            candidate_popularity_chunk = (
                candidate_prior_cumulative_likes[start:end]
                if candidate_prior_cumulative_likes is not None
                else None
            )
            logits = self.model.score_candidate_matrix_one_layer(
                history_embeddings=history_embeddings,
                history_mask=history_mask,
                history_time_deltas_hours=history_time_deltas_hours,
                candidate_post_embeddings=candidate_embeddings_chunk,
                history_author_indices=history_author_indices,
                candidate_post_author_idx=candidate_author_chunk,
                history_prior_cumulative_likes=history_prior_cumulative_likes,
                candidate_prior_cumulative_likes=candidate_popularity_chunk,
            )
            score_chunks.append(logits)

        return MatrixBatchScores(scores=torch.cat(score_chunks, dim=1))
