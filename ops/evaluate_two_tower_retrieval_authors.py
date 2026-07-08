#!/usr/bin/env python3
"""Qualitative two-tower retrieval author concentration check.

This script runs local TorchScript user/post towers over a recent Ingex post
window and reports which authors dominate the top-K retrieved posts for one
user DID.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import json
import os
import re
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import httpx
import numpy as np
import polars as pl
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.input_data_helpers import (
    AUTHOR_UNK_IDX,
    get_padded_author_indices,
    get_padded_embedding_history_and_mask,
)
from utils.helpers import apply_time_filter, parse_one_ts
from utils.pipeline.core import generate_run_timestamp


DEFAULT_TRAIN_ARTIFACTS_DIR = Path("/mnt/data/dave/outputs/artifacts/03_train")
DEFAULT_OUTPUT_ROOT = Path("/mnt/data/dave/outputs/artifacts/two_tower_retrieval_eval")
DEFAULT_GCS_BUCKET = "greenearth-471522-ingex-extract-prod"
DEFAULT_EMBEDDING_MODEL = "all_MiniLM_L12_v2"
DEFAULT_ES_HOST = "https://localhost:9200"
POST_BLOB_PREFIX = "bsky_posts"
LIKES_INDEX = "likes"
POSTS_INDEX = "posts"
TIMESTAMP_COL_NAME = "record_created_at"
POST_TEXT_COLUMNS = ("record_text", "content")
AT_URI_POST_RE = re.compile(r"^at://([^/]+)/app\.bsky\.feed\.post/([^/?#]+)/?(?:[?#].*)?$")


@dataclass(frozen=True)
class CandidatePost:
    at_uri: str
    author_did: str
    record_created_at: str
    embedding: list[float]
    content: Optional[str] = None


@dataclass(frozen=True)
class HistoryPost:
    rank: int
    at_uri: str
    author_did: Optional[str]
    embedding: Optional[list[float]]

    @property
    def embedding_present(self) -> bool:
        return self.embedding is not None


@dataclass(frozen=True)
class TopPost:
    model_run_id: str
    rank: int
    score: float
    at_uri: str
    author_did: str
    record_created_at: str
    content: Optional[str] = None


@dataclass
class ModelBundle:
    run_id: str
    train_dir: Path
    training_config: dict[str, Any]
    manifest: dict[str, Any]
    user_tower: Any
    post_tower: Any
    author_idx_by_did: Optional[dict[str, int]]
    load_source: str

    @property
    def use_author_embedding_table(self) -> bool:
        return bool(self.training_config.get("use_author_embedding_table"))

    @property
    def max_history_len(self) -> int:
        return int(self.training_config["max_history_len"])

    @property
    def post_embedding_dim(self) -> int:
        return int(self.training_config["post_embedding_dim"])


class TopKAccumulator:
    def __init__(self, k: int):
        if k <= 0:
            raise ValueError("top_k must be positive")
        self.k = int(k)
        self._items: list[tuple[float, CandidatePost]] = []

    def add(self, scores: torch.Tensor, posts: list[CandidatePost]) -> None:
        if len(posts) == 0:
            return
        if scores.numel() != len(posts):
            raise ValueError("scores and posts must have the same length")
        scores_cpu = scores.detach().flatten().to("cpu")
        local_k = min(self.k, scores_cpu.numel())
        top_scores, top_indices = torch.topk(scores_cpu, k=local_k)
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            self._items.append((float(score), posts[int(idx)]))
        self._items.sort(key=lambda item: item[0], reverse=True)
        del self._items[self.k :]

    def ranked(self, model_run_id: str) -> list[TopPost]:
        return [
            TopPost(
                model_run_id=model_run_id,
                rank=rank,
                score=score,
                at_uri=post.at_uri,
                author_did=post.author_did,
                record_created_at=post.record_created_at,
                content=post.content,
            )
            for rank, (score, post) in enumerate(self._items, start=1)
        ]


def get_post_id(at_uri: str) -> str:
    match = AT_URI_POST_RE.match(at_uri)
    if not match:
        raise ValueError(f"Could not find post ID in AT URI: {at_uri}")
    return match.group(2)


def at_uri_to_url(at_uri: str) -> str:
    match = AT_URI_POST_RE.match(at_uri)
    if not match:
        raise ValueError(f"Could not parse AT URI: {at_uri}")
    author_did, post_id = match.groups()
    return f"https://bsky.app/profile/{author_did}/post/{post_id}"


def safe_at_uri_to_url(at_uri: str) -> Optional[str]:
    try:
        return at_uri_to_url(at_uri)
    except ValueError:
        return None


def default_date_range(today: Optional[date] = None) -> tuple[str, str]:
    today = today or datetime.now(timezone.utc).date()
    start = today - timedelta(days=2)
    end = today
    return start.isoformat(), end.isoformat()


def normalize_es_host(host: str) -> str:
    host = host.strip().rstrip("/")
    if not host:
        raise ValueError("es_host must not be empty")
    if "://" not in host:
        host = f"https://{host}"
    return host


def is_local_https(host: str) -> bool:
    normalized = normalize_es_host(host)
    return normalized.startswith("https://localhost:") or normalized.startswith("https://127.0.0.1:")


def create_output_dir(
    output_dir: Optional[Path],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    timestamp: Optional[str] = None,
    short_uuid: Optional[str] = None,
) -> Path:
    if output_dir is not None:
        path = Path(output_dir)
    else:
        ts = timestamp or generate_run_timestamp()
        suffix = short_uuid or uuid.uuid4().hex[:8]
        path = Path(output_root) / f"{ts}_{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def load_stage_get_data_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "utils/01_get_data/stage_get_data.py"
    spec = importlib.util.spec_from_file_location("stage_get_data_for_retrieval_eval", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load Stage 1 helper module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_gcs_bucket_and_embedding_model(first_manifest: dict[str, Any]) -> tuple[str, str]:
    get_data_dir = Path(first_manifest.get("inputs", {}).get("01_get_data", ""))
    resolved_config_path = get_data_dir / "resolved_config.json"
    if not resolved_config_path.exists():
        return DEFAULT_GCS_BUCKET, DEFAULT_EMBEDDING_MODEL
    resolved_config = _read_json(resolved_config_path)
    return (
        str(resolved_config.get("gcs_bucket") or DEFAULT_GCS_BUCKET),
        str(resolved_config.get("embedding_model") or DEFAULT_EMBEDDING_MODEL),
    )


def resolve_model_paths(train_artifacts_dir: Path, run_id: str) -> dict[str, Path]:
    train_dir = Path(train_artifacts_dir) / run_id
    paths = {
        "train_dir": train_dir,
        "user_tower": train_dir / "checkpoints/engagement_user_tower.pt",
        "post_tower": train_dir / "checkpoints/engagement_post_tower.pt",
        "best_user_tower": train_dir / "checkpoints/engagement_user_tower_best.pt",
        "best_post_tower": train_dir / "checkpoints/engagement_post_tower_best.pt",
        "best_checkpoint": train_dir / "checkpoints/two_tower_best.pth",
        "training_config": train_dir / "training_config.json",
        "manifest": train_dir / "manifest.json",
        "partial_manifest": train_dir / "manifest.partial.json",
    }
    missing = [
        str(paths[key])
        for key in ("training_config",)
        if not paths[key].exists()
    ]
    if missing:
        raise FileNotFoundError(f"Model run {run_id} is missing required artifact(s): {', '.join(missing)}")
    if not has_model_load_source(paths):
        raise FileNotFoundError(
            f"Model run {run_id} has no supported model artifacts. Expected one of: "
            f"{paths['user_tower']} + {paths['post_tower']}, "
            f"{paths['best_user_tower']} + {paths['best_post_tower']}, "
            f"or {paths['best_checkpoint']}"
        )
    return paths


def load_train_manifest(paths: dict[str, Path]) -> dict[str, Any]:
    if paths["manifest"].exists():
        return _read_json(paths["manifest"])
    if paths["partial_manifest"].exists():
        return _read_json(paths["partial_manifest"])
    return {}


def has_model_load_source(paths: dict[str, Path]) -> bool:
    return (
        paths["user_tower"].exists()
        and paths["post_tower"].exists()
    ) or (
        paths["best_user_tower"].exists()
        and paths["best_post_tower"].exists()
    ) or paths["best_checkpoint"].exists()


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Expected checkpoint at {checkpoint_path} to contain a dictionary")
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint at {checkpoint_path} is missing model_state_dict")
    return checkpoint


def _require_config(config: dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"Model config is missing required key: {key}")
    return config[key]


def reconstruct_model_from_checkpoint(
    checkpoint_path: Path,
    training_config: dict[str, Any],
    device: torch.device,
) -> Any:
    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint_config = checkpoint.get("config")
    config = {**training_config, **checkpoint_config} if isinstance(checkpoint_config, dict) else training_config
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
    model = model.to(device)
    model.eval()
    return model


def load_towers_from_paths(
    paths: dict[str, Path],
    training_config: dict[str, Any],
    device: torch.device,
) -> tuple[Any, Any, str]:
    if paths["user_tower"].exists() and paths["post_tower"].exists():
        user_tower = torch.jit.load(str(paths["user_tower"]), map_location=device).eval()
        post_tower = torch.jit.load(str(paths["post_tower"]), map_location=device).eval()
        return user_tower, post_tower, "torchscript_final"

    if paths["best_user_tower"].exists() and paths["best_post_tower"].exists():
        user_tower = torch.jit.load(str(paths["best_user_tower"]), map_location=device).eval()
        post_tower = torch.jit.load(str(paths["best_post_tower"]), map_location=device).eval()
        return user_tower, post_tower, "torchscript_best"

    model = reconstruct_model_from_checkpoint(paths["best_checkpoint"], training_config, device)
    return model.user_tower, model.post_tower, "pth_best"


def resolve_author_idx_path(get_data_dir: Path) -> Path:
    matches = sorted(Path(get_data_dir).glob("author_idx_*.parquet"))
    if not matches:
        raise FileNotFoundError(f"No author_idx_*.parquet found under {get_data_dir}")
    return matches[0]


def load_author_idx_map(get_data_dir: Path) -> dict[str, int]:
    author_idx_path = resolve_author_idx_path(get_data_dir)
    df = pl.read_parquet(author_idx_path, columns=["author_did", "author_idx"])
    return {
        str(row["author_did"]): int(row["author_idx"])
        for row in df.iter_rows(named=True)
        if row.get("author_did") is not None and row.get("author_idx") is not None
    }


def load_model_bundle(train_artifacts_dir: Path, run_id: str, device: torch.device) -> ModelBundle:
    paths = resolve_model_paths(train_artifacts_dir, run_id)
    training_config = _read_json(paths["training_config"])
    manifest = load_train_manifest(paths)
    user_tower, post_tower, load_source = load_towers_from_paths(paths, training_config, device)

    author_idx_by_did = None
    if bool(training_config.get("use_author_embedding_table")):
        get_data_dir = Path(manifest.get("inputs", {}).get("01_get_data", ""))
        if not get_data_dir.exists():
            raise FileNotFoundError(
                f"Model run {run_id} uses author embeddings, but manifest input 01_get_data was not found: {get_data_dir}. "
                "A manifest with the training 01_get_data path is required to map author DIDs to author_idx values."
            )
        author_idx_by_did = load_author_idx_map(get_data_dir)

    return ModelBundle(
        run_id=run_id,
        train_dir=paths["train_dir"],
        training_config=training_config,
        manifest=manifest,
        user_tower=user_tower,
        post_tower=post_tower,
        author_idx_by_did=author_idx_by_did,
        load_source=load_source,
    )


def validate_model_compatibility(models: list[ModelBundle]) -> int:
    if not models:
        raise ValueError("At least one model run id is required")
    embed_dims = {model.post_embedding_dim for model in models}
    if len(embed_dims) != 1:
        by_model = {model.run_id: model.post_embedding_dim for model in models}
        raise ValueError(f"All models must use the same post_embedding_dim; got {by_model}")
    return next(iter(embed_dims))


def author_idx_for_did(model: ModelBundle, author_did: Optional[str]) -> int:
    if not author_did or model.author_idx_by_did is None:
        return AUTHOR_UNK_IDX
    return int(model.author_idx_by_did.get(author_did, AUTHOR_UNK_IDX))


def history_tensors_for_model(
    model: ModelBundle,
    history_posts: list[HistoryPost],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    embeddings = [
        post.embedding
        for post in history_posts
        if post.embedding is not None and len(post.embedding) == model.post_embedding_dim
    ]
    padded, mask = get_padded_embedding_history_and_mask(
        embeddings,
        model.max_history_len,
        model.post_embedding_dim,
    )
    history_embeddings = torch.from_numpy(padded).unsqueeze(0).to(device=device, dtype=torch.float32)
    history_mask = torch.from_numpy(mask).unsqueeze(0).to(device=device, dtype=torch.bool)

    if not model.use_author_embedding_table:
        return history_embeddings, history_mask, None

    author_indices = [
        author_idx_for_did(model, post.author_did)
        for post in history_posts
        if post.embedding is not None and len(post.embedding) == model.post_embedding_dim
    ]
    padded_author_indices = get_padded_author_indices(author_indices, model.max_history_len)
    history_author_indices = torch.from_numpy(padded_author_indices).unsqueeze(0).to(device=device, dtype=torch.long)
    return history_embeddings, history_mask, history_author_indices


def compute_user_embedding_for_model(
    model: ModelBundle,
    history_posts: list[HistoryPost],
    device: torch.device,
) -> torch.Tensor:
    history_embeddings, history_mask, history_author_indices = history_tensors_for_model(model, history_posts, device)
    with torch.no_grad():
        if model.use_author_embedding_table:
            if history_author_indices is None:
                raise RuntimeError("history_author_indices are required for author-aware model")
            output = model.user_tower(history_embeddings, history_mask, history_author_indices)
        else:
            output = model.user_tower(history_embeddings, history_mask)
    return output.squeeze(0)


def score_candidate_batch_for_model(
    model: ModelBundle,
    user_embedding: torch.Tensor,
    candidates: list[CandidatePost],
    device: torch.device,
) -> torch.Tensor:
    if not candidates:
        return torch.empty(0, dtype=torch.float32)
    post_embeddings_np = np.asarray([post.embedding for post in candidates], dtype=np.float32)
    post_embeddings = torch.from_numpy(post_embeddings_np).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        if model.use_author_embedding_table:
            author_indices = [
                author_idx_for_did(model, post.author_did)
                for post in candidates
            ]
            target_author_indices = torch.tensor(author_indices, device=device, dtype=torch.long)
            post_vectors = model.post_tower(post_embeddings, target_author_indices)
        else:
            post_vectors = model.post_tower(post_embeddings)
        return post_vectors.matmul(user_embedding.to(device=device))


async def es_search_json(
    client: httpx.AsyncClient,
    *,
    es_host: str,
    index: str,
    body: dict[str, Any],
    api_key: Optional[str],
) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"ApiKey {api_key}"
    resp = await client.post(f"{es_host}/{index}/_search", json=body, headers=headers)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected ES response for index {index}: expected object")
    return payload


async def fetch_recent_liked_uris(
    client: httpx.AsyncClient,
    *,
    es_host: str,
    user_did: str,
    limit: int,
    api_key: Optional[str],
) -> list[str]:
    payload = await es_search_json(
        client,
        es_host=es_host,
        index=LIKES_INDEX,
        api_key=api_key,
        body={
            "_source": ["subject_uri"],
            "query": {"bool": {"filter": [{"term": {"author_did": user_did}}]}},
            "sort": [{"created_at": "desc"}],
            "size": limit,
        },
    )
    liked_uris = []
    for hit in payload.get("hits", {}).get("hits", []):
        uri = (hit.get("_source") or {}).get("subject_uri")
        if uri:
            liked_uris.append(str(uri))
    return liked_uris


def extract_embedding_from_es_source(source: dict[str, Any], embedding_model: str) -> Optional[list[float]]:
    embeddings = source.get("embeddings")
    if isinstance(embeddings, dict):
        vec = embeddings.get(embedding_model)
        if isinstance(vec, list):
            return vec
    vec = source.get(f"embeddings.{embedding_model}")
    if isinstance(vec, list):
        return vec
    return None


def build_history_posts_from_es(
    liked_uris: list[str],
    posts_by_uri: dict[str, dict[str, Any]],
    *,
    embedding_model: str,
    embed_dim: int,
) -> list[HistoryPost]:
    history_posts: list[HistoryPost] = []
    for rank, uri in enumerate(liked_uris, start=1):
        source = posts_by_uri.get(uri, {})
        embedding = extract_embedding_from_es_source(source, embedding_model) if source else None
        if embedding is not None and len(embedding) != embed_dim:
            embedding = None
        author_did = source.get("author_did") if source else None
        history_posts.append(
            HistoryPost(
                rank=rank,
                at_uri=uri,
                author_did=str(author_did) if author_did else None,
                embedding=embedding,
            )
        )
    return history_posts


async def fetch_history_posts(
    client: httpx.AsyncClient,
    *,
    es_host: str,
    user_did: str,
    history_limit: int,
    api_key: Optional[str],
    embedding_model: str,
    embed_dim: int,
) -> list[HistoryPost]:
    liked_uris = await fetch_recent_liked_uris(
        client,
        es_host=es_host,
        user_did=user_did,
        limit=history_limit,
        api_key=api_key,
    )
    if not liked_uris:
        return []
    payload = await es_search_json(
        client,
        es_host=es_host,
        index=POSTS_INDEX,
        api_key=api_key,
        body={
            "_source": ["at_uri", "author_did", f"embeddings.{embedding_model}"],
            "query": {"terms": {"at_uri": liked_uris}},
            "size": len(liked_uris),
        },
    )
    posts_by_uri = {}
    for hit in payload.get("hits", {}).get("hits", []):
        source = hit.get("_source") or {}
        at_uri = source.get("at_uri")
        if at_uri:
            posts_by_uri[str(at_uri)] = source
    return build_history_posts_from_es(
        liked_uris,
        posts_by_uri,
        embedding_model=embedding_model,
        embed_dim=embed_dim,
    )


async def fetch_handle_for_did(client: httpx.AsyncClient, author_did: str) -> Optional[str]:
    resp = await client.get(
        "https://public.api.bsky.app/xrpc/app.bsky.actor.getProfile",
        params={"actor": author_did},
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        return None
    handle = payload.get("handle")
    return str(handle) if handle else None


async def resolve_author_handles(
    author_dids: Iterable[str],
    *,
    fetch_profile: Optional[Callable[[httpx.AsyncClient, str], Any]] = None,
) -> dict[str, str]:
    unique_dids = sorted({did for did in author_dids if did})
    handles: dict[str, str] = {}
    fetch_profile = fetch_profile or fetch_handle_for_did
    async with httpx.AsyncClient(timeout=10) as client:
        for did in tqdm(unique_dids, desc="Resolving author handles", unit="author"):
            try:
                handle = await fetch_profile(client, did)
            except Exception:
                handle = None
            handles[did] = str(handle) if handle else did
    return handles


def list_gcs_post_paths(
    *,
    gcs_bucket: str,
    start_date: str,
    end_date: str,
    stage_get_data_module: Any,
) -> list[str]:
    start_dt = parse_one_ts(start_date)
    end_dt = parse_one_ts(end_date)
    client = stage_get_data_module.storage.Client()
    paths_with_ts: list[tuple[datetime, str]] = []
    blobs = client.list_blobs(gcs_bucket, prefix=POST_BLOB_PREFIX)
    for blob in tqdm(blobs, desc="Listing GCS post parquets", unit="blob"):
        if not blob.name.endswith(".parquet"):
            continue
        ts = stage_get_data_module._parse_ts_from_name_ingex_gcs(blob.name, POST_BLOB_PREFIX)
        if ts is None:
            continue
        if start_dt is not None and ts < start_dt:
            continue
        if end_dt is not None and ts >= end_dt:
            continue
        paths_with_ts.append((ts, f"gs://{gcs_bucket}/{blob.name}"))
    paths_with_ts.sort(key=lambda item: item[0])
    return [path for _, path in paths_with_ts]


def _record_created_at_to_str(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def scan_candidate_file(
    path: str,
    *,
    start_date: str,
    end_date: str,
    embedding_model: str,
    embed_dim: int,
    stage_get_data_module: Any,
) -> list[CandidatePost]:
    lf = pl.scan_parquet(path)
    lf = apply_time_filter(lf, start_date, end_date)
    schema = lf.collect_schema()
    text_column = next((col for col in POST_TEXT_COLUMNS if col in schema), None)
    select_columns = ["at_uri", "did", TIMESTAMP_COL_NAME, "embeddings"]
    if text_column is not None:
        select_columns.append(text_column)
    lf = lf.select(select_columns)
    if text_column is not None:
        lf = lf.with_columns(pl.col(text_column).cast(pl.String).alias("_content"))
    else:
        lf = lf.with_columns(pl.lit(None).cast(pl.String).alias("_content"))
    lf = stage_get_data_module.get_embeddings_list_col_polars(lf, embedding_model)
    df = (
        lf.select(["at_uri", "did", TIMESTAMP_COL_NAME, "_content", "_emb_vec"])
        .filter(pl.col("_emb_vec").is_not_null())
        .collect(engine="streaming")
    )
    candidates: list[CandidatePost] = []
    for row in df.iter_rows(named=True):
        emb = row["_emb_vec"]
        if emb is None or len(emb) != embed_dim:
            continue
        at_uri = row.get("at_uri")
        if not at_uri:
            continue
        candidates.append(
            CandidatePost(
                at_uri=str(at_uri),
                author_did=str(row.get("did") or ""),
                record_created_at=_record_created_at_to_str(row.get(TIMESTAMP_COL_NAME)),
                embedding=list(emb),
                content=row.get("_content"),
            )
        )
    return candidates


def iter_candidate_batches(
    paths: list[str],
    *,
    start_date: str,
    end_date: str,
    embedding_model: str,
    embed_dim: int,
    batch_size: int,
    stage_get_data_module: Any,
) -> Iterable[list[CandidatePost]]:
    if batch_size <= 0:
        raise ValueError("candidate_batch_size must be positive")
    seen_uris: set[str] = set()
    batch: list[CandidatePost] = []
    for path in tqdm(paths, desc="Scanning candidate parquet files", unit="file"):
        for candidate in scan_candidate_file(
            path,
            start_date=start_date,
            end_date=end_date,
            embedding_model=embedding_model,
            embed_dim=embed_dim,
            stage_get_data_module=stage_get_data_module,
        ):
            if candidate.at_uri in seen_uris:
                continue
            seen_uris.add(candidate.at_uri)
            batch.append(candidate)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def build_author_counts(top_posts: list[TopPost], handles_by_did: dict[str, str]) -> list[dict[str, Any]]:
    counts = Counter(post.author_did for post in top_posts)
    best_rank_by_author: dict[str, int] = {}
    for post in top_posts:
        best_rank_by_author.setdefault(post.author_did, post.rank)
    rows = [
        {
            "author_did": author_did,
            "handle": handles_by_did.get(author_did, author_did),
            "count": count,
            "best_rank": best_rank_by_author[author_did],
        }
        for author_did, count in counts.items()
    ]
    rows.sort(key=lambda row: (-int(row["count"]), int(row["best_rank"]), str(row["author_did"])))
    return rows


def history_posts_json(history_posts: list[HistoryPost], handles_by_did: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "rank": post.rank,
            "at_uri": post.at_uri,
            "author_did": post.author_did,
            "handle": handles_by_did.get(post.author_did, post.author_did) if post.author_did else None,
            "embedding_present": post.embedding_present,
        }
        for post in history_posts
    ]


def top_k_posts_json(top_posts_by_model: dict[str, list[TopPost]], handles_by_did: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_run_id, top_posts in top_posts_by_model.items():
        for post in top_posts:
            rows.append(
                {
                    "model_run_id": model_run_id,
                    "rank": post.rank,
                    "score": post.score,
                    "at_uri": post.at_uri,
                    "url": safe_at_uri_to_url(post.at_uri),
                    "author_did": post.author_did,
                    "handle": handles_by_did.get(post.author_did, post.author_did),
                    "record_created_at": post.record_created_at,
                    "content": post.content,
                }
            )
    return rows


def build_summary(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    models: list[ModelBundle],
    author_counts_by_model: dict[str, list[dict[str, Any]]],
    top_posts_by_model: dict[str, list[TopPost]],
    n_candidate_posts_scanned: int,
    n_history_likes: int,
    n_embeddable_history_posts: int,
    gcs_bucket: str,
    embedding_model: str,
) -> dict[str, Any]:
    model_summaries = {}
    for model in models:
        top_posts = top_posts_by_model.get(model.run_id, [])
        scores = [post.score for post in top_posts]
        model_summaries[model.run_id] = {
            "train_dir": str(model.train_dir),
            "load_source": model.load_source,
            "use_author_embedding_table": model.use_author_embedding_table,
            "author_counts": author_counts_by_model.get(model.run_id, []),
            "top_k_returned": len(top_posts),
            "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None,
        }
    return {
        "run_config": {
            "train_run_ids": list(args.train_run_ids),
            "train_artifacts_dir": str(args.train_artifacts_dir),
            "user_did": args.user_did,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "top_k": args.top_k,
            "gcs_bucket": gcs_bucket,
            "embedding_model": embedding_model,
            "es_host": args.es_host,
            "history_limit": args.history_limit,
            "candidate_batch_size": args.candidate_batch_size,
            "output_dir": str(output_dir),
        },
        "diagnostics": {
            "n_models": len(models),
            "n_candidate_posts_scanned": n_candidate_posts_scanned,
            "n_history_likes": n_history_likes,
            "n_embeddable_history_posts": n_embeddable_history_posts,
        },
        "models": model_summaries,
    }


def write_output_artifacts(
    *,
    output_dir: Path,
    summary: dict[str, Any],
    history_rows: list[dict[str, Any]],
    top_k_rows: list[dict[str, Any]],
) -> None:
    for name, payload in tqdm(
        [
            ("summary.json", summary),
            ("history_posts.json", history_rows),
            ("top_k_posts.json", top_k_rows),
        ],
        desc="Writing output artifacts",
        unit="file",
    ):
        _write_json(output_dir / name, payload)


def print_console_summary(
    *,
    output_dir: Path,
    summary: dict[str, Any],
) -> None:
    print(f"\nOutput directory: {output_dir}")
    diagnostics = summary["diagnostics"]
    print(
        "Candidates scanned: {n_candidate_posts_scanned} | History likes: {n_history_likes} | "
        "Embeddable history posts: {n_embeddable_history_posts}".format(**diagnostics)
    )
    for model_run_id, model_summary in summary["models"].items():
        print(f"\n=== {model_run_id} ===")
        print(
            f"top_k_returned={model_summary['top_k_returned']} "
            f"score_min={model_summary['score_min']} score_max={model_summary['score_max']}"
        )
        for row in model_summary["author_counts"]:
            print(
                f"{row['count']:>3}  best_rank={row['best_rank']:>2}  "
                f"{row['handle']} ({row['author_did']})"
            )


def cleanup_empty_default_output_dir(output_dir: Path, *, is_default_output_dir: bool) -> bool:
    if not is_default_output_dir:
        return False
    try:
        output_dir.rmdir()
    except FileNotFoundError:
        return False
    except OSError:
        return False
    return True


async def run(args: argparse.Namespace) -> Path:
    output_dir = create_output_dir(args.output_dir)
    try:
        return await run_with_output_dir(args, output_dir)
    except Exception:
        cleanup_empty_default_output_dir(
            output_dir,
            is_default_output_dir=getattr(args, "output_dir", None) is None,
        )
        raise


async def run_with_output_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    device = torch.device(args.device)
    args.es_host = normalize_es_host(args.es_host)
    if args.es_insecure is None:
        args.es_insecure = is_local_https(args.es_host)
    api_key = args.es_api_key or os.getenv("GE_ELASTICSEARCH_API_KEY") or os.getenv("ES_API_KEY")

    models: list[ModelBundle] = []
    for run_id in tqdm(args.train_run_ids, desc="Resolving/loading model artifacts", unit="model"):
        models.append(load_model_bundle(args.train_artifacts_dir, run_id, device))
    embed_dim = validate_model_compatibility(models)

    inferred_bucket, inferred_embedding_model = infer_gcs_bucket_and_embedding_model(models[0].manifest)
    gcs_bucket = args.gcs_bucket or inferred_bucket
    embedding_model = args.embedding_model or inferred_embedding_model

    async with httpx.AsyncClient(timeout=600, verify=not bool(args.es_insecure)) as client:
        with tqdm(total=2, desc="Querying ES history likes/posts", unit="step") as pbar:
            liked_uris = await fetch_recent_liked_uris(
                client,
                es_host=args.es_host,
                user_did=args.user_did,
                limit=args.history_limit,
                api_key=api_key,
            )
            pbar.update(1)
            if liked_uris:
                payload = await es_search_json(
                    client,
                    es_host=args.es_host,
                    index=POSTS_INDEX,
                    api_key=api_key,
                    body={
                        "_source": ["at_uri", "author_did", f"embeddings.{embedding_model}"],
                        "query": {"terms": {"at_uri": liked_uris}},
                        "size": len(liked_uris),
                    },
                )
                posts_by_uri = {
                    str((hit.get("_source") or {}).get("at_uri")): hit.get("_source") or {}
                    for hit in payload.get("hits", {}).get("hits", [])
                    if (hit.get("_source") or {}).get("at_uri")
                }
                history_posts = build_history_posts_from_es(
                    liked_uris,
                    posts_by_uri,
                    embedding_model=embedding_model,
                    embed_dim=embed_dim,
                )
            else:
                history_posts = []
            pbar.update(1)

    user_embeddings = {
        model.run_id: compute_user_embedding_for_model(model, history_posts, device)
        for model in tqdm(models, desc="Encoding user towers", unit="model")
    }

    stage_get_data_module = load_stage_get_data_module()
    post_paths = list_gcs_post_paths(
        gcs_bucket=gcs_bucket,
        start_date=args.start_date,
        end_date=args.end_date,
        stage_get_data_module=stage_get_data_module,
    )
    if not post_paths:
        raise FileNotFoundError(f"No {POST_BLOB_PREFIX} parquet files found in gs://{gcs_bucket} for {args.start_date} to {args.end_date}")

    topk_by_model = {
        model.run_id: TopKAccumulator(args.top_k)
        for model in models
    }
    score_bars = {
        model.run_id: tqdm(desc=f"Encoding/scoring post batches ({model.run_id})", unit="post")
        for model in models
    }
    n_candidate_posts_scanned = 0
    try:
        for batch in iter_candidate_batches(
            post_paths,
            start_date=args.start_date,
            end_date=args.end_date,
            embedding_model=embedding_model,
            embed_dim=embed_dim,
            batch_size=args.candidate_batch_size,
            stage_get_data_module=stage_get_data_module,
        ):
            n_candidate_posts_scanned += len(batch)
            for model in models:
                scores = score_candidate_batch_for_model(
                    model,
                    user_embeddings[model.run_id],
                    batch,
                    device,
                )
                topk_by_model[model.run_id].add(scores, batch)
                score_bars[model.run_id].update(len(batch))
    finally:
        for bar in score_bars.values():
            bar.close()

    top_posts_by_model = {
        model.run_id: topk_by_model[model.run_id].ranked(model.run_id)
        for model in models
    }
    author_dids = {
        post.author_did
        for top_posts in top_posts_by_model.values()
        for post in top_posts
        if post.author_did
    }
    author_dids.update(
        post.author_did
        for post in history_posts
        if post.author_did
    )
    handles_by_did = await resolve_author_handles(author_dids)

    author_counts_by_model = {
        model.run_id: build_author_counts(top_posts_by_model[model.run_id], handles_by_did)
        for model in models
    }
    history_rows = history_posts_json(history_posts, handles_by_did)
    top_k_rows = top_k_posts_json(top_posts_by_model, handles_by_did)
    summary = build_summary(
        args=args,
        output_dir=output_dir,
        models=models,
        author_counts_by_model=author_counts_by_model,
        top_posts_by_model=top_posts_by_model,
        n_candidate_posts_scanned=n_candidate_posts_scanned,
        n_history_likes=len(history_posts),
        n_embeddable_history_posts=sum(1 for post in history_posts if post.embedding_present),
        gcs_bucket=gcs_bucket,
        embedding_model=embedding_model,
    )
    write_output_artifacts(
        output_dir=output_dir,
        summary=summary,
        history_rows=history_rows,
        top_k_rows=top_k_rows,
    )
    print_console_summary(output_dir=output_dir, summary=summary)
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    default_start_date, default_end_date = default_date_range()
    parser = argparse.ArgumentParser(description="Evaluate two-tower retrieval author concentration for one user DID.")
    parser.add_argument("--train-run-ids", nargs="+", help="03_train artifact subfolder names to compare")
    parser.add_argument("--train-artifacts-dir", type=Path, default=DEFAULT_TRAIN_ARTIFACTS_DIR)
    parser.add_argument("--user-did", required=True)
    parser.add_argument("--start-date", default=default_start_date)
    parser.add_argument("--end-date", default=default_end_date)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--gcs-bucket", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--es-host", default=DEFAULT_ES_HOST)
    parser.add_argument("--es-api-key", default=None)
    parser.add_argument("--es-insecure", action="store_true", default=None)
    parser.add_argument("--es-secure", dest="es_insecure", action="store_false")
    parser.set_defaults(es_insecure=None)
    parser.add_argument("--history-limit", type=int, default=50)
    parser.add_argument("--candidate-batch-size", type=int, default=8192)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
