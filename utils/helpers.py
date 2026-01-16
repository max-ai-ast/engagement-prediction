#!/usr/bin/env python3

"""
Consolidated Helpers for Engagement Prediction Pipeline

This module centralizes the shared helper functions used across pipeline stages.
Only truly cross-stage utilities live here. Stage-specific helpers should live
inside their respective stage scripts (e.g., utils/05_train/stage_train.py).
"""

from __future__ import annotations

import os
import sys
import json
import random
import tempfile
import base64
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from datetime import datetime, timedelta, timezone
from io import BytesIO
import multiprocessing as mp
from google.cloud import storage
import re
import polars as pl

import numpy as np
import pandas as pd

# Optional heavy deps: provide stubs/fallbacks to keep imports robust
try:
    import boto3  # type: ignore
    from botocore.exceptions import ClientError, NoCredentialsError  # type: ignore
except Exception:  # pragma: no cover
    boto3 = None  # type: ignore
    class ClientError(Exception):  # type: ignore
        pass
    class NoCredentialsError(Exception):  # type: ignore
        pass

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(kwargs.get('total', 0) or 0)

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    class nn:  # type: ignore
        Module = object

try:
    import torchvision.transforms as transforms  # type: ignore
    from torchvision.models import resnet18, ResNet18_Weights  # type: ignore
except Exception:  # pragma: no cover
    transforms = None  # type: ignore
    resnet18 = None  # type: ignore
    ResNet18_Weights = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore


# ----------------------------------------
# Config
# ----------------------------------------
SPACES_BUCKET = "parquet-dumps"
SPACES_REGION = "sfo3"
SPACES_HOST = f"{SPACES_REGION}.digitaloceanspaces.com"

# Avoid HF tokenizers fork warnings/deadlocks in multiprocessing contexts
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


# ----------------------------------------
# Datetime helpers
# ----------------------------------------
# For parsing GCS Ingex filenames
TIMESTAMP_SUFFIX_GCS = "_(\\d{8})_(\\d{6})\\.parquet$"

# For parsing CLI arg strings
KNOWN_TS_FORMATS = [
    "%Y-%m-%dT%H:%M:%S%z",     # 2024-02-10T13:45:00+0000
    "%Y-%m-%dT%H:%M:%S%z",     # 2024-02-10T13:45:00+00:00
    "%Y-%m-%dT%H:%M:%S",       # 2024-02-10T13:45:00
    "%Y-%m-%d",                # 2024-02-10
]

def parse_one_ts(raw_ts: Optional[str]) -> Optional[datetime]:
    """Parse a single timestamp string into a timezone-aware datetime (UTC)."""
    if raw_ts is None:
        return None
    for fmt in KNOWN_TS_FORMATS:
        try:
            dt = datetime.strptime(raw_ts, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    raise ValueError(f"Unrecognized datetime format: {raw_ts!r}")


# ----------------------------------------
# Data IO helpers (Green Earth Ingex + GCS)
# ----------------------------------------
def parse_ts_from_name_ingex_gcs(
        blob_name: str, 
        blob_prefix: str
    ) -> Optional[datetime]:
    """Parse timestamp from GCS blob name based on Ingex naming convention."""
    pattern = re.compile(blob_prefix + TIMESTAMP_SUFFIX_GCS)
    m = pattern.match(blob_name)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)

def list_files_in_range_ingex_gcs(
        gcs_bucket: str, 
        blob_prefix: str, 
        start: Optional[datetime], 
        end: Optional[datetime],
        ) -> list[str]:
    """List GCS blob URIs within specified time range based on Ingex naming convention."""
    client = storage.Client()
    blobs = client.list_blobs(gcs_bucket)
    out = []
    for b in blobs:
        ts = parse_ts_from_name_ingex_gcs(blob_name=b.name, blob_prefix=blob_prefix)
        if ts is None:
            continue
        if start is not None and ts < start:
            continue
        if end is not None and ts >= end:
            continue
        out.append(f"gs://{gcs_bucket}/{b.name}")
    return out

def load_raw_data_ingex(
        gcs_bucket: str, 
        blob_prefix: str,
        start_str: Optional[str], 
        end_str: Optional[str], 
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data from GreenEarth Ingex on GCS within specified time ranges."""
    
    start_dt: Optional[datetime] = parse_one_ts(start_str)
    end_dt: Optional[datetime] = parse_one_ts(end_str)
    
    paths = list_files_in_range_ingex_gcs(
        gcs_bucket = gcs_bucket,
        blob_prefix = blob_prefix,
        start = start_dt,
        end = end_dt,
    )

    # LazyFrame (from polars)
    lf = (
        pl
        .scan_parquet(paths)
        .with_columns(
            pl.col("inserted_at").str.to_datetime(time_zone="UTC").alias("inserted_at_dt")
        )
    )
    if start_dt is not None:
        lf = lf.filter(pl.col("inserted_at_dt") >= start_dt)
    if end_dt is not None:
        lf = lf.filter(pl.col("inserted_at_dt") < end_dt)
    pandas_df = lf.collect().to_pandas()

    return pandas_df


# ----------------------------------------
# Data IO helpers (Digital Ocean Spaces/S3 + parquet)
# ----------------------------------------
def list_recent_objects_digital_ocean(bucket: str, prefix: str, days: int) -> Tuple[List[str], List[dict]]:
    """List S3 object keys from the last `days` days within `prefix`."""
    if boto3 is None:
        return [], []
    s3 = boto3.client(
        "s3",
        region_name=SPACES_REGION,
        endpoint_url=f"https://{SPACES_HOST}",
        aws_access_key_id=os.getenv("SPACES_KEY"),
        aws_secret_access_key=os.getenv("SPACES_SECRET"),
    )
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    paginator = s3.get_paginator("list_objects_v2")

    keys: List[str] = []
    file_info: List[dict] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["LastModified"] >= cutoff:
                keys.append(obj["Key"])
                file_info.append(
                    {
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "modified": obj["LastModified"],
                    }
                )
    return keys, file_info


def list_all_objects_digital_ocean(bucket: str, prefix: str) -> Tuple[List[str], List[dict]]:
    """List all S3 object keys for a prefix (no time filter)."""
    if boto3 is None:
        return [], []
    s3 = boto3.client(
        "s3",
        region_name=SPACES_REGION,
        endpoint_url=f"https://{SPACES_HOST}",
        aws_access_key_id=os.getenv("SPACES_KEY"),
        aws_secret_access_key=os.getenv("SPACES_SECRET"),
    )
    paginator = s3.get_paginator("list_objects_v2")

    keys: List[str] = []
    file_info: List[dict] = []
    for page in tqdm(paginator.paginate(Bucket=bucket, Prefix=prefix), desc="Scanning S3 pages"):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
            file_info.append({
                "key": obj["Key"],
                "size": obj["Size"],
                "modified": obj["LastModified"],
            })
    return keys, file_info


def download_parquet_files_digital_ocean(keys: List[str], bucket: str, dest_dir: Path) -> List[Path]:
    """Download parquet files from Spaces/S3 to dest_dir; skip existing."""
    if boto3 is None:
        return []
    s3 = boto3.client(
        "s3",
        region_name=SPACES_REGION,
        endpoint_url=f"https://{SPACES_HOST}",
        aws_access_key_id=os.getenv("SPACES_KEY"),
        aws_secret_access_key=os.getenv("SPACES_SECRET"),
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded: List[Path] = []
    for key in tqdm(keys, desc="Downloading files"):
        local_path = dest_dir / Path(key).name
        if not local_path.exists():
            s3.download_file(bucket, key, str(local_path))
        downloaded.append(local_path)
    return downloaded


def load_and_combine_data_digital_ocean(datasets: Dict[str, List[Path]], drop_unliked_posts: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load parquet dataframes for posts/likes/(optional images metadata) and optionally drop unliked posts."""
    posts_dfs: List[pd.DataFrame] = []
    likes_dfs: List[pd.DataFrame] = []
    metadata_dfs: List[pd.DataFrame] = []

    for f in tqdm(datasets.get("posts", []), desc="Loading posts"):
        posts_dfs.append(pd.read_parquet(f))
    for f in tqdm(datasets.get("likes", []), desc="Loading likes"):
        likes_dfs.append(pd.read_parquet(f))
    for f in tqdm(datasets.get("images", []), desc="Loading images"):
        metadata_dfs.append(pd.read_parquet(f))

    metadata_df = (
        pd.DataFrame(columns=['commit_cid', 'embed_images'])
        if len(metadata_dfs) == 0 else pd.concat(metadata_dfs, ignore_index=True)
    )
    posts_df = pd.concat(posts_dfs, ignore_index=True) if posts_dfs else pd.DataFrame()
    likes_df = pd.concat(likes_dfs, ignore_index=True) if likes_dfs else pd.DataFrame()

    if drop_unliked_posts and not likes_df.empty and not posts_df.empty:
        posts_df = posts_df[posts_df.get("did").isin(likes_df.get("did"))]

    return posts_df, likes_df, metadata_df


# ----------------------------------------
# Join/text detection
# ----------------------------------------
def find_join_key(posts_df: pd.DataFrame, likes_df: pd.DataFrame) -> Tuple[str, str]:
    """Find joins between posts and likes with common cases and overlap fallback."""
    if "subject_cid" in likes_df.columns and "commit_cid" in posts_df.columns:
        return "subject_cid", "commit_cid"
    if "subject_uri" in likes_df.columns and "at_uri" in posts_df.columns:
        return "subject_uri", "at_uri"
    common = set(posts_df.columns) & set(likes_df.columns)
    if not common:
        raise ValueError("No common column names between likes and posts tables")
    for col in common:
        if posts_df[col].isin(likes_df[col]).any():
            return col, col
    raise ValueError("No obvious join key between likes and posts tables")


def find_text_column(posts_df: pd.DataFrame) -> str:
    """Heuristic to find the text column."""
    if "record_text" in posts_df.columns:
        return "record_text"
    text_cols = [c for c in posts_df.columns if "text" in c.lower()]
    if not text_cols:
        raise ValueError("No text column found in posts table for embedding")
    return text_cols[0]


# ----------------------------------------
# Embeddings (text + image)
# ----------------------------------------
def get_embed_col_names(dim: int) -> List[str]:
    """Generate embedding column names for given dimension."""
    return [f"post_emb_{i}" for i in range(dim)]


def compute_post_embeddings(posts_df: pd.DataFrame, text_column: str, model_name: str) -> Tuple[pd.DataFrame, int]:
    """Compute sentence-transformer embeddings for all posts."""
    import time
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not available")
    
    print(f"  Loading embedding model: {model_name}...")
    t0 = time.time()
    model = SentenceTransformer(model_name)
    
    # Check if GPU is available
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f"  Model loaded in {time.time()-t0:.2f}s (using GPU)")
    else:
        print(f"  Model loaded in {time.time()-t0:.2f}s (using CPU)")
    
    sample_text = posts_df[text_column].fillna("").astype(str).iloc[0] if len(posts_df) else ""
    emb = model.encode([sample_text])
    dim = emb.shape[1]
    
    texts = posts_df[text_column].fillna("").astype(str).tolist()
    # Use larger batch size for GPU, smaller for CPU
    batch_size = 1024 if device == 'cuda' else 256
    print(f"  Computing embeddings for {len(texts)} posts (dim={dim}, batch_size={batch_size})...")
    t1 = time.time()
    all_emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True, device=device)
    rate = len(texts) / (time.time() - t1) if time.time() - t1 > 0 else 0
    print(f"  Embeddings computed in {time.time()-t1:.2f}s ({rate:.1f} posts/sec)")
    
    emb_cols = get_embed_col_names(dim)
    emb_df = pd.DataFrame(all_emb, columns=emb_cols)
    posts_emb_df = pd.concat([posts_df.reset_index(drop=True), emb_df], axis=1)
    return posts_emb_df, dim


def embedding_loads(s: str, decompress: Optional[bool] = None) -> list[float]:
    """
    Convert an embedding from a base85-encoded string to a list of floats.

    If `decompress` is `True`, decompress with zlib and throw an error if decompression fails.

    If `decompress` is `False`, do not decompress before unpacking.

    If `decompress` is `None`, attempt decompression and silently fallback to an uncompressed string
    if decompression fails.
    """

    bs = base64.b85decode(s.encode())

    if decompress or decompress is None:
        try:
            bs = zlib.decompress(bs)
        except zlib.error:
            if decompress:
                raise

    return list(struct.unpack(f'<{int(len(bs) / 4)}f', bs))


def extract_encoded_embedding_ingex(emb_list: Optional[list[dict]], model_name: str) -> Optional[str]:
    """Extract base85-encoded embedding string from Ingex embeddings list for given model name."""
    if emb_list is None:
        return None
    for emb_dict in emb_list:
        if emb_dict['key'] == model_name:
            return emb_dict['value']
    return None


def load_embeddings_ingex(posts_df: pd.DataFrame, model_name: str) -> Tuple[pd.DataFrame, int]:
    """Load precomputed embeddings from GreenEarth Ingex."""

    # get the dimension of the embeddings by finding one example:
    embed_dim = None
    for _, row in posts_df.iterrows():
        emb_list = row['embeddings']
        if emb_list is None:
            continue
        else:
            emb_str = extract_encoded_embedding_ingex(emb_list, model_name)
            if emb_str is not None:
                sample_emb = embedding_loads(emb_str, decompress=True)
                embed_dim = len(sample_emb)
                break
    if embed_dim is None:
        raise ValueError(f"No embeddings found for model {model_name} in posts data")

    # Now load all embeddings
    # First get the string out of the list of dicts for the given model
    embed_str_col = f"embed_{model_name}"
    posts_df[embed_str_col] = posts_df['embeddings'].map(lambda x: extract_encoded_embedding_ingex(x, model_name))

    # Pre-allocate the numpy array to speed things up
    n = len(posts_df)
    arr = np.zeros((n, embed_dim), dtype=float)
    for i, x in enumerate(posts_df[embed_str_col].to_numpy()):
        if x is not None:
            arr[i] = embedding_loads(x, True)

    emb_cols = get_embed_col_names(embed_dim)

    lded_embs_df = pd.DataFrame(arr, index=posts_df.index, columns=emb_cols)
    posts_emb_df = pd.concat([posts_df, lded_embs_df], axis=1)

    return posts_emb_df, embed_dim


def _load_image_tensor(image_url: str, target_size: Tuple[int, int] = (224, 224)):
    if requests is None or Image is None or transforms is None:
        return None
    try:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        image = Image.open(BytesIO(resp.content)).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0)
    except Exception:
        return None


def compute_image_embeddings(posts_emb_df: pd.DataFrame, image_column: str, batch_size: int = 32, max_images: Optional[int] = None) -> Tuple[pd.DataFrame, int]:
    """Compute ResNet18 features for posts that have an image URL in `image_column`."""
    if resnet18 is None or torch is None:
        # Fallback: add zero image embeddings
        zero_dim = 512
        cols = [f"image_emb_{i}" for i in range(zero_dim)]
        z = np.zeros((len(posts_emb_df), zero_dim), dtype=float)
        return pd.concat([posts_emb_df.reset_index(drop=True), pd.DataFrame(z, columns=cols)], axis=1), zero_dim

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224)
        if torch.cuda.is_available():
            dummy = dummy.cuda()
        out = model(dummy)
        emb_dim = int(out.shape[1])

    df = posts_emb_df.copy()
    has_img = df[image_column].notna() & (df[image_column] != "") if image_column in df.columns else pd.Series(False, index=df.index)
    idxs = df[has_img].index.tolist()
    if max_images is not None:
        idxs = idxs[:max_images]
    all_embeddings: Dict[int, np.ndarray] = {}
    for start in tqdm(range(0, len(idxs), batch_size), desc="Processing images"):
        for idx in idxs[start:start+batch_size]:
            img_url = df.at[idx, image_column]
            tensor = _load_image_tensor(img_url)
            if tensor is None:
                all_embeddings[idx] = np.zeros((emb_dim,), dtype=float)
                continue
            with torch.no_grad():
                if torch.cuda.is_available():
                    tensor = tensor.cuda()
                emb = model(tensor).squeeze().detach().cpu().numpy()
            all_embeddings[idx] = emb
    cols = [f"image_emb_{i}" for i in range(emb_dim)]
    img_emb_df = pd.DataFrame(0.0, index=df.index, columns=cols)
    if all_embeddings:
        filled = pd.DataFrame.from_dict(all_embeddings, orient='index')
        filled.columns = cols
        img_emb_df.loc[filled.index] = filled.values
    return pd.concat([df.reset_index(drop=True), img_emb_df.reset_index(drop=True)], axis=1), emb_dim


# ----------------------------------------
# Feature column helpers
# ----------------------------------------
def get_actual_feature_columns(posts_emb_df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    text_emb_cols = [c for c in posts_emb_df.columns if c.startswith("post_emb_")]
    image_emb_cols = [c for c in posts_emb_df.columns if c.startswith("image_emb_")]
    post_cols = text_emb_cols + image_emb_cols
    user_cols = [f"user_emb_{i}" for i in range(len(post_cols))]
    all_cols = user_cols + post_cols
    return user_cols, post_cols, all_cols


# ----------------------------------------
# User feature construction (mean/multi_centroid/topic_mixture)
# ----------------------------------------
try:
    from sklearn.cluster import MiniBatchKMeans as _MBK  # type: ignore
except Exception:  # pragma: no cover
    _MBK = None  # type: ignore


def build_user_feature_frame(
    schema: str,
    likes_df: pd.DataFrame,
    posts_emb_df: pd.DataFrame,
    join_like: str,
    join_post: str,
    embedding_dim: int,
    *,
    selected_users: Optional[List[str]] = None,
    feature_columns: Optional[List[List[str]]] = None,
    random_seed: int = 42,
    topic_model: Optional[Any] = None,
    pca_model: Optional[Any] = None,
    global_topic_k: Optional[int] = None,
    user_k: int = 3,
    min_cluster_size: int = 3,
    max_embedding_posts_per_user: int = 20,
) -> pd.DataFrame:
    rng = np.random.RandomState(int(random_seed))
    likes_local = likes_df[likes_df['did'].isin(selected_users)] if selected_users is not None else likes_df.copy()
    if feature_columns is not None:
        expected_user_cols, post_cols_expected, _ = feature_columns
    else:
        expected_user_cols, post_cols_expected, _ = get_actual_feature_columns(posts_emb_df)

    available_posts = set(posts_emb_df[join_post].astype(str).unique())
    if join_like not in likes_local.columns:
        raise KeyError(f"likes_df missing join_like column: {join_like}")
    likes_local[join_like] = likes_local[join_like].astype(str)
    likes_local = likes_local[likes_local[join_like].isin(available_posts)]
    feat_cols = [c for c in posts_emb_df.columns if c.startswith('post_emb_') or c.startswith('image_emb_')]

    if schema == 'topic_mixture':
        if topic_model is None:
            raise ValueError("topic_model is required for topic_mixture schema")
        joined = likes_local.merge(posts_emb_df[[join_post] + feat_cols], left_on=join_like, right_on=join_post, how='inner')
        if len(joined) == 0:
            raise ValueError("No joinable likes to compute topic mixtures")
        X = joined[feat_cols].values.astype(np.float32, copy=False)
        if pca_model is not None and hasattr(pca_model, 'components_'):
            try:
                if X.shape[1] == int(pca_model.components_.shape[1]):
                    X = pca_model.transform(X)
            except Exception:
                pass
        topics = topic_model.predict(X)
        joined['_topic'] = topics
        counts = joined.groupby(['did', '_topic']).size().unstack(fill_value=0)
        if global_topic_k is None:
            global_topic_k = int(counts.shape[1])
        for t in range(int(global_topic_k)):
            if t not in counts.columns:
                counts[t] = 0
        counts = counts[sorted(counts.columns)]
        probs = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
        user_features_df = probs.reset_index()
        user_features_df.columns = ['did'] + [f'user_topic_{t}' for t in range(int(global_topic_k))]
        if feature_columns is not None:
            for c in expected_user_cols:
                if c not in user_features_df.columns:
                    user_features_df[c] = 0.0
            return user_features_df[['did'] + expected_user_cols].copy()
        return user_features_df

    if schema == 'multi_centroid':
        if _MBK is None:
            raise RuntimeError("scikit-learn is required for multi_centroid user features")
        if feature_columns is not None:
            # infer K and D
            import re
            k_indices: List[int] = []
            d_indices: List[int] = []
            for c in expected_user_cols:
                m_d = re.match(r'user_k(\d+)_d(\d+)$', c)
                if m_d:
                    k_indices.append(int(m_d.group(1)))
                    d_indices.append(int(m_d.group(2)))
                    continue
                m_w = re.match(r'user_k(\d+)_weight$', c)
                if m_w:
                    k_indices.append(int(m_w.group(1)))
                    continue
            K = (max(k_indices) + 1) if k_indices else int(user_k)
            D = (max(d_indices) + 1) if d_indices else None
        else:
            K, D = int(user_k), None
        joined = likes_local.merge(posts_emb_df[[join_post] + feat_cols], left_on=join_like, right_on=join_post, how='inner')
        rows: List[Dict[str, Any]] = []
        for user_id, g in joined.groupby('did'):
            Xg = g[feat_cols].values.astype(np.float32, copy=False)
            if len(Xg) == 0:
                continue
            cap = min(int(max_embedding_posts_per_user), len(Xg))
            if len(Xg) > cap:
                idx = rng.choice(len(Xg), size=cap, replace=False)
                Xg = Xg[idx]
            k_eff = int(K)
            if len(Xg) < k_eff:
                k_eff = max(1, len(Xg) // max(1, int(min_cluster_size)))
            if k_eff < 1:
                continue
            mbk = _MBK(n_clusters=k_eff, batch_size=min(256, max(16, len(Xg))), random_state=int(random_seed), n_init=5)
            labels = mbk.fit_predict(Xg)
            centroids = mbk.cluster_centers_
            counts = np.bincount(labels, minlength=k_eff).astype(np.float32)
            weights = counts / (counts.sum() if counts.sum() > 0 else 1.0)
            norms = np.linalg.norm(centroids, axis=1)
            order = np.lexsort((-norms, -weights))
            centroids = centroids[order]
            weights = weights[order]
            if D is None:
                D = centroids.shape[1]
            pad_centroids = np.zeros((int(K), int(D)), dtype=np.float32)
            pad_weights = np.zeros((int(K),), dtype=np.float32)
            pad_centroids[:k_eff, :min(int(D), centroids.shape[1])] = centroids[:, :min(int(D), centroids.shape[1])]
            pad_weights[:k_eff] = weights
            row: Dict[str, Any] = {'did': user_id, 'user_k_effective': int(k_eff)}
            for i in range(int(K)):
                for d in range(int(D)):
                    row[f'user_k{i}_d{d}'] = float(pad_centroids[i, d])
                row[f'user_k{i}_weight'] = float(pad_weights[i])
            rows.append(row)
        if not rows:
            raise ValueError("No users had sufficient embedding posts to compute multi-centroid features")
        user_df = pd.DataFrame(rows)
        if feature_columns is not None:
            for c in expected_user_cols:
                if c not in user_df.columns:
                    user_df[c] = 0.0
            return user_df[['did'] + expected_user_cols].copy()
        return user_df

    # mean embedding fallback (compat with older paths)
    text_emb_cols = [col for col in posts_emb_df.columns if col.startswith("post_emb_")]
    image_emb_cols = [col for col in posts_emb_df.columns if col.startswith("image_emb_")]
    feat_cols = text_emb_cols + image_emb_cols
    joined = likes_local.merge(posts_emb_df[[join_post] + feat_cols], left_on=join_like, right_on=join_post, how='inner')
    user_embeddings = joined.groupby("did")[feat_cols].mean().reset_index()
    user_emb_cols = [f"user_emb_{i}" for i in range(len(feat_cols))]
    user_embeddings.columns = ["did"] + user_emb_cols
    if feature_columns is not None:
        missing = [c for c in expected_user_cols if c not in user_embeddings.columns]
        if missing:
            raise ValueError("Computed mean user embeddings do not match expected schema")
        return user_embeddings[['did'] + expected_user_cols].copy()
    return user_embeddings


# ----------------------------------------
# Pairs dataset construction (shared by train/evaluate)
# ----------------------------------------
def _gen_negative_pairs_batch(args: Tuple[List[Any], Dict[Any, Set[Any]], Set[Any], Set[Tuple[Any, Any]], int, int]) -> List[Tuple[Any, Any]]:
    """Top-level worker function for multiprocessing (must be picklable)."""
    user_batch, user_posts_dict, all_posts, positive_pairs, worker_id, random_seed = args
    import random as _rnd
    seed = (hash(f"worker_{worker_id}") ^ int(random_seed)) & 0xFFFFFFFF
    _rnd.seed(seed)
    pairs: List[Tuple[Any, Any]] = []
    for u in user_batch:
        u_posts = user_posts_dict[u]
        avail = list(all_posts - u_posts)
        k = min(len(u_posts), len(avail))
        if k > 0:
            negs = _rnd.sample(avail, k)
            for p in negs:
                pair = (u, p)
                if pair not in positive_pairs:
                    pairs.append(pair)
    return pairs

def create_pairs_dataset(
    likes_df: pd.DataFrame,
    posts_emb_df: pd.DataFrame,
    join_like: str,
    join_post: str,
    neg_ratio: float = 0.5,
    random_seed: int = 42,
    use_parallel: bool = True,
) -> pd.DataFrame:
    random.seed(int(random_seed))
    text_emb_cols = [col for col in posts_emb_df.columns if col.startswith("post_emb_")]
    image_emb_cols = [col for col in posts_emb_df.columns if col.startswith("image_emb_")]
    post_emb_cols = text_emb_cols + image_emb_cols

    pos_df = likes_df.merge(posts_emb_df[[join_post] + post_emb_cols], left_on=join_like, right_on=join_post, how="inner")
    pos_df['liked'] = 1

    all_users = pos_df['did'].unique()
    all_posts = set(posts_emb_df[join_post].unique())
    user_posts_dict = {u: set(pos_df[pos_df['did'] == u][join_post].unique()) for u in all_users}
    positive_pairs = set(zip(pos_df['did'], pos_df[join_post]))

    # Parallel path when many users
    negative_pairs: List[Tuple[Any, Any]] = []
    if use_parallel and len(all_users) > 50:
        try:
            # Ensure stable start method for CUDA envs
            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn', force=True)
        except Exception:
            pass
        optimal_workers = min(max(1, mp.cpu_count()), 16, len(all_users) // 10 + 1)
        user_batches = [list(b) for b in np.array_split(all_users, optimal_workers) if len(b) > 0]
        batch_args = [
            (batch, user_posts_dict, all_posts, positive_pairs, i, int(random_seed))
            for i, batch in enumerate(user_batches)
        ]
        with mp.Pool(processes=optimal_workers) as pool:
            for pairs in tqdm(pool.imap_unordered(_gen_negative_pairs_batch, batch_args), total=len(batch_args), desc="Generating negative samples (parallel)"):
                negative_pairs.extend(pairs)
        # Deduplicate and drop positives
        seen: Set[Tuple[Any, Any]] = set()
        negative_pairs = [pair for pair in negative_pairs if (pair not in positive_pairs and not (pair in seen or seen.add(pair)))]
    else:
        seen: Set[Tuple[Any, Any]] = set()
        for u in tqdm(all_users, desc="Generating negative samples"):
            u_posts = user_posts_dict[u]
            avail = list(all_posts - u_posts)
            k = min(len(u_posts), len(avail))
            if k > 0:
                negs = random.sample(avail, k)
                for p in negs:
                    pair = (u, p)
                    if pair not in seen and pair not in positive_pairs:
                        seen.add(pair)
                        negative_pairs.append(pair)

    if negative_pairs:
        neg_df = pd.DataFrame(negative_pairs, columns=['did', join_post])
        neg_df['liked'] = 0
        neg_df = neg_df.merge(posts_emb_df[[join_post] + post_emb_cols], on=join_post, how='inner')
        final_df = pd.concat([pos_df, neg_df], ignore_index=True)
    else:
        final_df = pos_df
    return final_df


# ----------------------------------------
# Data integrity validation (shared)
# ----------------------------------------
def validate_data_integrity(data_dict: Dict) -> bool:
    required_keys = ['train_df', 'embedding_dim', 'join_post', 'join_like']
    for key in required_keys:
        if key not in data_dict:
            print(f"❌ Missing required key: {key}")
            return False
    if len(data_dict['train_df']) == 0:
        print("❌ Empty training dataframe")
        return False
    required_cols = ['did', 'liked', data_dict['join_post']]
    missing_cols = [col for col in required_cols if col not in data_dict['train_df'].columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return False
    print("✅ Data integrity validated")
    return True


# ----------------------------------------
# Visualization helpers (shared)
# ----------------------------------------
import matplotlib.pyplot as plt  # type: ignore
try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None  # type: ignore
import matplotlib.patches as mpatches  # type: ignore

FIGURE_SIZE = (10, 6)
DPI = 300


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[Path] = None, best_epoch: Optional[int] = None):
    required_keys = ['train_loss', 'val_loss', 'train_auc', 'val_auc']
    if any(k not in history for k in required_keys) or len(history.get('train_loss', [])) == 0:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, history['train_auc'], 'b-', label='Train AUC', linewidth=2)
    ax2.plot(epochs, history['val_auc'], 'r-', label='Val AUC', linewidth=2)
    ax2.set_title('Training and Validation AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if best_epoch is not None:
        try:
            ax1.axvline(best_epoch, color='k', linestyle='--', alpha=0.6)
            ax2.axvline(best_epoch, color='k', linestyle='--', alpha=0.6)
        except Exception:
            pass
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.show()


def plot_model_performance(y_true: np.ndarray, y_pred_proba: np.ndarray, save_path: Optional[Path] = None):
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix  # type: ignore
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    axes[0, 0].plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    axes[0, 1].plot(recall, precision)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].grid(True, alpha=0.3)
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    else:
        axes[1, 0].imshow(cm, cmap='Blues')
        for (i, j), val in np.ndenumerate(np.array(cm)):
            axes[1, 0].text(j, i, int(val), ha='center', va='center')
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 1].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Not Liked')
    axes[1, 1].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Liked')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Prediction Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.show()


def create_user_visualization(user_tracking_results: Dict[str, Any], timestamp: str, save_dir: Path) -> None:
    if not user_tracking_results:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    # Minimal stub: leave detailed visualization to stage-specific logic if needed
    # (Kept for API compatibility)
    summary_path = save_dir / f"user_tracking_summary_{timestamp}.json"
    try:
        with open(summary_path, 'w') as f:
            json.dump(user_tracking_results, f)
    except Exception:
        pass


__all__ = [
    # Datetime
    'parse_one_ts',
    # Data IO Green Earth Ingex GCS
    'load_raw_data_ingex',
    # Data IO Digital Ocean
    'list_recent_objects_digital_ocean', 'list_all_objects_digital_ocean', 'download_parquet_files_digital_ocean', 'load_and_combine_data_digital_ocean', 'load_most_recent_raw_data_digital_ocean',
    # Detection
    'find_join_key', 'find_text_column',
    # Embeddings
    'get_embed_col_names', 'embedding_loads', 'extract_encoded_embedding_ingex', 'load_embeddings_ingex', 'compute_post_embeddings', 'compute_image_embeddings',
    # Features/columns
    'get_actual_feature_columns', 'build_user_feature_frame', 'build_candidate_posts', 'compute_post_feature_frame', 'save_bundle',
    # Relevel/topic helpers
    'discover_topics', 'compute_user_topic_mixtures', 'relevel_uniform_mixture',
    # Dataset construction
    'create_pairs_dataset',
    # Validation
    'validate_data_integrity',
    # Viz
    'plot_training_history', 'plot_model_performance', 'create_user_visualization',
]


# ----------------------------------------
# Stage 1 convenience: load most recent small raw bundle
# ----------------------------------------
def load_most_recent_raw_data_digital_ocean(max_files_per_table: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download and load a compact slice of recent posts/likes (and optional images) from Spaces.

    Selects the most recently modified up to `max_files_per_table` files for each table.
    """
    # Discover latest keys
    posts_keys, posts_info = list_all_objects_digital_ocean(SPACES_BUCKET, "bsky_firehose_posts_tmp")
    likes_keys, likes_info = list_all_objects_digital_ocean(SPACES_BUCKET, "bsky_firehose_likes_light_tmp")
    # Sort by LastModified desc using info arrays
    def _top_n(keys: List[str], info: List[dict], n: int) -> List[str]:
        if not keys or not info:
            return []
        m = {d['key']: d.get('modified') for d in info if 'key' in d}
        ordered = sorted([k for k in keys if k in m], key=lambda k: m[k], reverse=True)
        return ordered[: max(0, int(n))]

    posts_sel = _top_n(posts_keys, posts_info, max_files_per_table)
    likes_sel = _top_n(likes_keys, likes_info, max_files_per_table)

    # Download and load
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        posts_files = download_parquet_files_digital_ocean(posts_sel, SPACES_BUCKET, tmp / "posts") if posts_sel else []
        likes_files = download_parquet_files_digital_ocean(likes_sel, SPACES_BUCKET, tmp / "likes") if likes_sel else []
        posts_df, likes_df, metadata_df = load_and_combine_data_digital_ocean({
            "posts": posts_files,
            "likes": likes_files,
            # images omitted in Stage 1 bundle; keep empty
        })
    return posts_df, likes_df, metadata_df


# ----------------------------------------
# Stage 2: Featurize helpers
# ----------------------------------------
def build_candidate_posts(
    posts_df: pd.DataFrame,
    likes_df: pd.DataFrame,
    join_like: str,
    join_post: str,
    author_col: str,
    *,
    max_posts_per_author: int = 3,
    max_liked_posts_per_user: int = 100,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Select candidate posts by union of liked posts and per-author caps.

    - Always include posts that appear in likes_df[join_like].
    - Augment with up to `max_posts_per_author` posts per author (random selection).
    """
    import time
    t0 = time.time()
    
    rng = np.random.RandomState(int(rng_seed))
    join_like_str = likes_df[join_like].astype(str)
    liked_post_ids = set(join_like_str.dropna().unique().tolist())
    print(f"  Found {len(liked_post_ids)} unique liked posts")

    posts_df_local = posts_df.copy()
    posts_df_local[join_post] = posts_df_local[join_post].astype(str)

    liked_posts = posts_df_local[posts_df_local[join_post].isin(liked_post_ids)]
    print(f"  Matched {len(liked_posts)} liked posts from posts_df")
    
    extra_rows: List[pd.DataFrame] = []
    if author_col in posts_df_local.columns and max_posts_per_author > 0:
        print(f"  Sampling {max_posts_per_author} posts per author...")
        grouped = posts_df_local.groupby(author_col)
        num_authors = len(grouped)
        print(f"  Processing {num_authors} authors...")
        
        # OPTIMIZED: Use vectorized sampling instead of loop
        sampled_indices = []
        for author, g in grouped:
            if len(g) <= max_posts_per_author:
                sampled_indices.extend(g.index.tolist())
            else:
                idx = rng.choice(g.index.values, size=int(max_posts_per_author), replace=False)
                sampled_indices.extend(idx.tolist())
        
        if sampled_indices:
            extra_rows = [posts_df_local.loc[sampled_indices]]
            print(f"  Sampled {len(sampled_indices)} posts from authors")
    
    pool = [liked_posts] + extra_rows if extra_rows else [liked_posts]
    if not pool:
        return posts_df_local
    
    print(f"  Concatenating and deduplicating...")
    candidates = pd.concat(pool, ignore_index=True).drop_duplicates(subset=[join_post])
    print(f"  Built {len(candidates)} candidate posts (took {time.time()-t0:.2f}s)")
    return candidates


def compute_post_feature_frame(candidate_posts: pd.DataFrame, data_source: str, model_name: str, image_mode: str = 'auto') -> Tuple[pd.DataFrame, int]:
    """Compute embeddings for candidate posts (text always; optional image).

    image_mode: 'off' | 'on' | 'auto' (currently same as 'off' unless image_url present)
    """
    if data_source == 'digitalocean':
        text_col = find_text_column(candidate_posts)
        model_name_st = 'sentence-transformers/' + model_name.replace('_', '-')
        posts_emb_df, text_dim = compute_post_embeddings(candidate_posts, text_col, model_name_st)
        img_dim = 0
        if image_mode in ('on', 'auto') and 'image_url' in candidate_posts.columns:
            try:
                posts_emb_df, img_dim = compute_image_embeddings(posts_emb_df, 'image_url')
            except Exception:
                img_dim = 0
        return posts_emb_df, (text_dim + img_dim)
    elif data_source == 'greenearth':
        posts_emb_df, text_dim = load_embeddings_ingex(candidate_posts, model_name)
        return posts_emb_df, text_dim
    else:
        raise ValueError(f"Unsupported data_source: {data_source}")


def save_bundle(
    *,
    out_dir: Path,
    posts_emb_df: pd.DataFrame,
    likes_df: pd.DataFrame,
    join_like: str,
    join_post: str,
    text_column: str,
    author_column: str,
    data_source: str,
    embedding_model: str,
    embedding_dim: int,
    image_mode: str,
    extra_meta: Optional[Dict[str, Any]] = None,
    liked_posts_texts_path: Optional[str] = None,
) -> str:
    """Persist embedding bundle to out_dir and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = out_dir.name
    bundle_path = out_dir / f"embedding_bundle_{ts}.pkl"
    payload = {
        'posts_emb_df': posts_emb_df,
        'likes_df': likes_df,
        'join_like': join_like,
        'join_post': join_post,
        'text_column': text_column,
        'author_column': author_column,
        'data_source': data_source,
        'embedding_model': embedding_model,
        'embedding_dim': int(embedding_dim),
        'image_mode': str(image_mode),
        'meta': dict(extra_meta or {}),
    }
    if liked_posts_texts_path:
        payload['liked_posts_texts_path'] = str(liked_posts_texts_path)
    import pickle
    with open(bundle_path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    return str(bundle_path)


# ----------------------------------------
# Stage 3: Topic discovery and releveling
# ----------------------------------------
class TopicArtifacts:
    def __init__(self, topic_model: Optional[Any], pca_model: Optional[Any], global_topic_k: Optional[int]):
        self.topic_model = topic_model
        self.pca_model = pca_model
        self.global_topic_k = global_topic_k


def discover_topics(
    posts_emb_df: pd.DataFrame,
    likes_df_joinable: pd.DataFrame,
    join_like: str,
    join_post: str,
    *,
    global_topic_k: int = 20,
    random_seed: int = 42,
) -> TopicArtifacts:
    """Fit MiniBatchKMeans on post embeddings (optionally after PCA) using liked posts as samples."""
    try:
        from sklearn.decomposition import PCA  # type: ignore
        from sklearn.cluster import MiniBatchKMeans  # type: ignore
    except Exception:
        return TopicArtifacts(None, None, None)
    feat_cols = [c for c in posts_emb_df.columns if c.startswith('post_emb_') or c.startswith('image_emb_')]
    if not feat_cols:
        return TopicArtifacts(None, None, None)
    available_posts = set(posts_emb_df[join_post].astype(str).unique())
    df = likes_df_joinable.copy()
    df[join_like] = df[join_like].astype(str)
    df = df[df[join_like].isin(available_posts)]
    joined = df.merge(posts_emb_df[[join_post] + feat_cols], left_on=join_like, right_on=join_post, how='inner')
    if len(joined) == 0:
        return TopicArtifacts(None, None, None)
    X = joined[feat_cols].values.astype(np.float32, copy=False)
    pca = None
    if X.shape[1] > 256:
        pca = PCA(n_components=256, random_state=int(random_seed))
        X = pca.fit_transform(X)
    kmeans = MiniBatchKMeans(n_clusters=int(global_topic_k), random_state=int(random_seed), batch_size=min(2048, max(64, len(X))))
    kmeans.fit(X)
    return TopicArtifacts(kmeans, pca, int(global_topic_k))


def compute_user_topic_mixtures(artifacts: TopicArtifacts, posts_emb_df: pd.DataFrame, likes_df_joinable: pd.DataFrame, join_like: str, join_post: str) -> Optional[pd.DataFrame]:
    if artifacts.topic_model is None:
        return None
    feat_cols = [c for c in posts_emb_df.columns if c.startswith('post_emb_') or c.startswith('image_emb_')]
    available_posts = set(posts_emb_df[join_post].astype(str).unique())
    df = likes_df_joinable.copy()
    df[join_like] = df[join_like].astype(str)
    df = df[df[join_like].isin(available_posts)]
    joined = df.merge(posts_emb_df[[join_post] + feat_cols], left_on=join_like, right_on=join_post, how='inner')
    if len(joined) == 0:
        return None
    X = joined[feat_cols].values.astype(np.float32, copy=False)
    if artifacts.pca_model is not None:
        try:
            X = artifacts.pca_model.transform(X)
        except Exception:
            pass
    labels = artifacts.topic_model.predict(X)
    joined['_topic'] = labels
    counts = joined.groupby(['did', '_topic']).size().unstack(fill_value=0)
    # Normalize to probabilities
    mixtures = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
    mixtures.index.name = 'did'
    return mixtures


def relevel_uniform_mixture(
    *,
    users: List[str],
    user_topic_probs: pd.DataFrame,
    global_topic_k: int,
    alpha: float = 0.35,
    min_users_per_topic: int = 0,
    random_seed: int = 42,
) -> List[str]:
    """Select a subset of users whose topic mixtures are closer to uniform.

    Greedy selection to approach per-topic coverage ~ uniform with minimum users per topic constraint.
    """
    rng = np.random.RandomState(int(random_seed))
    target = np.ones((int(global_topic_k),), dtype=np.float32) / float(global_topic_k)
    kept: List[str] = []
    remaining = users.copy()
    rng.shuffle(remaining)
    # Simple heuristic: keep users with smallest KL divergence to uniform first
    import numpy as _np
    def _kl(p, q):
        p = _np.clip(p, 1e-8, 1)
        q = _np.clip(q, 1e-8, 1)
        return float((p * _np.log(p / q)).sum())
    scored = []
    for u in remaining:
        if u not in user_topic_probs.index:
            continue
        p = user_topic_probs.loc[u].values.astype(np.float32, copy=False)
        scored.append((u, _kl(p, target)))
    scored.sort(key=lambda t: t[1])
    kept = [u for (u, _s) in scored]
    if min_users_per_topic > 0:
        # Ensure minimum coverage; greedy top-up per topic
        per_topic_counts = dict((t, 0) for t in range(int(global_topic_k)))
        final: List[str] = []
        for u in kept:
            if u not in user_topic_probs.index:
                continue
            p = user_topic_probs.loc[u].values.astype(np.float32, copy=False)
            top_topic = int(np.argmax(p))
            if per_topic_counts[top_topic] < int(min_users_per_topic):
                per_topic_counts[top_topic] += 1
                final.append(u)
        if len(final) >= int(min_users_per_topic) * int(global_topic_k):
            return final
    return kept


# ----------------------------------------
# Logging utilities
# ----------------------------------------
import logging
from datetime import datetime

# Global logger instances per stage (initialized on first use)
_stage_loggers: Dict[str, logging.Logger] = {}


def get_stage_logger(stage_name: str, log_file: Optional[Path] = None) -> logging.Logger:
    """Get or create a logger for a specific stage with timestamped formatting.
    
    Args:
        stage_name: Name of the stage (e.g., 'STAGE_01_GET_DATA')
        log_file: Optional path to log file. If None, logs only to stdout.
    
    Returns:
        Configured logger instance
    """
    if stage_name in _stage_loggers:
        return _stage_loggers[stage_name]
    
    logger = logging.getLogger(f"pipeline.{stage_name}")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter with timestamp
    formatter = logging.Formatter(
        '[%(asctime)s.%(msecs)03d] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Optional file handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    _stage_loggers[stage_name] = logger
    return logger


def log_operation_start(operation_name: str, stage_name: str, logger: Optional[logging.Logger] = None) -> logging.Logger:
    """Log the start of a major operation with timestamp.
    
    Args:
        operation_name: Name of the operation being started
        stage_name: Name of the stage (e.g., 'STAGE_01_GET_DATA')
        logger: Optional logger instance. If None, will get/create one for the stage.
    
    Returns:
        Logger instance used
    """
    if logger is None:
        logger = get_stage_logger(stage_name)
    logger.info(f"Starting: {operation_name}")
    return logger


def get_device(arg_device: Optional[str]) -> str:
    import torch
    if arg_device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    else:
        return arg_device