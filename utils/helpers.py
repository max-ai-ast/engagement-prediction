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
import base64
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from datetime import datetime, timedelta, timezone
import multiprocessing as mp
import polars as pl
import subprocess
import numpy as np
import pandas as pd

# Optional heavy deps: provide stubs/fallbacks to keep imports robust
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


# Avoid HF tokenizers fork warnings/deadlocks in multiprocessing contexts
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


# ----------------------------------------
# Data loading helpers
# ----------------------------------------
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


def apply_time_filter(
    lf: pl.LazyFrame, 
    start_str: Optional[str], 
    end_str: Optional[str]
) -> pl.LazyFrame:
    """
    Apply a time filter to a polars lazyframe. 
    Note that applying the filter using strings instead of converting to datetimes allows for 
    streaming rather than loading everything into memory.
    """
    if 'record_created_at' not in lf.collect_schema().names():
        raise ValueError("Input LazyFrame does not contain 'record_created_at' column for time filtering")
    if start_str is not None:
        lf = lf.filter(pl.col("record_created_at") >= start_str)
    if end_str is not None:
        lf = lf.filter(pl.col("record_created_at") < end_str)
    return lf


def save_polars_physical_plan_image(lf: pl.LazyFrame, out_path: str):
    dot = lf.show_graph(plan_stage='physical', engine='streaming', raw_output=True)
    if dot is not None:
        Path("plan.dot").write_text(dot)
    else:
        print("\n\nNo DOT output generated!!!\n\n")
    subprocess.run(["dot", "-Tpng", "-Gdpi=220", "plan.dot", "-o", out_path], check=True) 


def load_parquet_from_prior(prior_path: Path, prefix: str) -> pl.LazyFrame:
    # Load the most recent *.parquet found in the given directory
    candidates = sorted(prior_path.glob(f"{prefix}*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No {prefix}*.parquet found under {prior_path}")
    return pl.scan_parquet(candidates[0])


# ----------------------------------------
# Embeddings helpers
# ----------------------------------------
def _get_embeddings_list_col(lf: pl.LazyFrame, embedding_model: str) -> pl.LazyFrame:
    emb_str = (
        pl.col("embeddings")
        .list.eval(
            pl.when(pl.element().struct.field("key") == embedding_model)
              .then(pl.element().struct.field("value"))
        )
        .list.drop_nulls()
        .list.get(0)
    )
    emb_vec = emb_str.map_elements(
        lambda s: _embedding_loads(s, decompress=True) if s is not None else None,
        return_dtype=pl.List(pl.Float32),
    )
    return lf.with_columns(emb_vec.alias("_emb_vec"))


def get_embed_dim(lf: pl.LazyFrame, embedding_model: str) -> int:
    lf_with_emb = _get_embeddings_list_col(lf, embedding_model)
    return (
        lf_with_emb
        .select(pl.col("_emb_vec").list.len().alias("dim"))
        .filter(pl.col("dim").is_not_null())
        .head(1)
        .collect(engine="streaming")
        .item()
    )


def expand_embeddings_polars(
    lf: pl.LazyFrame,
    embedding_model: str,
    embed_dim: int
) -> pl.LazyFrame:
    lf = _get_embeddings_list_col(lf, embedding_model)
    return (
        lf
        .with_columns(
            [pl.col("_emb_vec").list.get(i).alias(f"post_emb_{i}") for i in range(embed_dim)]
        ).drop(["embeddings", "_emb_vec"])
    )


def get_embed_col_names(dim: int) -> List[str]:
    """Generate embedding column names for given dimension."""
    return [f"post_emb_{i}" for i in range(dim)]


def _embedding_loads(s: str, decompress: Optional[bool] = None) -> list[float]:
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
def validate_dataframe_schema(
    df: pd.DataFrame | pl.DataFrame | pl.LazyFrame,
    expected_schema: Dict[str, Any],
    *,
    allow_extra_columns: bool = True,
) -> None:
    """Validate a DataFrame against an expected schema of column names and dtypes.

    Supports pandas DataFrame and polars DataFrame/LazyFrame. expected_schema maps
    column name -> dtype spec, where dtype spec can be a Python type (e.g., int,
    float, str), a pandas/numpy dtype, a dtype string, or an iterable of specs.
    """
    if not isinstance(expected_schema, dict) or not expected_schema:
        raise ValueError("expected_schema must be a non-empty dict")

    if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        schema = dict(df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema)

        polars_integer = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
        polars_float = {pl.Float32, pl.Float64}
        polars_string = {pl.String, pl.Utf8}

        def _matches_expected_dtype_polars(dtype: pl.DataType, expected: Any) -> bool:
            if isinstance(expected, (list, tuple, set)):
                return any(_matches_expected_dtype_polars(dtype, e) for e in expected)

            if isinstance(expected, pl.DataType) or type(expected).__name__ == "DataTypeClass":
                return dtype == expected

            if isinstance(expected, str):
                key = expected.strip().lower()
                if key in ("int", "int64", "integer"):
                    return dtype in polars_integer
                if key in ("float", "float64", "double"):
                    return dtype in polars_float
                if key in ("bool", "boolean"):
                    return dtype == pl.Boolean
                if key in ("string", "str", "utf8"):
                    return dtype in polars_string
                if key in ("object", "obj"):
                    return dtype == pl.Object
                if key in ("category", "categorical"):
                    return dtype == pl.Categorical
                if key in ("datetime", "datetime64", "datetime64[ns]", "datetime64[ns, tz]"):
                    return dtype == pl.Datetime
                if key in ("date",):
                    return dtype == pl.Date
                if key in ("time", "time64"):
                    return dtype == pl.Time
                if key in ("timedelta", "timedelta64", "timedelta64[ns]", "duration"):
                    return dtype == pl.Duration
                try:
                    return _matches_expected_dtype_polars(dtype, np.dtype(expected))
                except Exception:
                    return False

            if isinstance(expected, np.dtype):
                if expected.kind in ("i", "u"):
                    return dtype in polars_integer
                if expected.kind == "f":
                    return dtype in polars_float
                if expected.kind == "b":
                    return dtype == pl.Boolean
                if expected.kind == "M":
                    return dtype in (pl.Datetime, pl.Date)
                if expected.kind == "m":
                    return dtype == pl.Duration
                if expected.kind in ("U", "S", "O"):
                    return dtype in polars_string or dtype == pl.Object

            if isinstance(expected, type):
                if issubclass(expected, (bool, np.bool_)):
                    return dtype == pl.Boolean
                if issubclass(expected, (int, np.integer)):
                    return dtype in polars_integer
                if issubclass(expected, (float, np.floating)):
                    return dtype in polars_float
                if issubclass(expected, str):
                    return dtype in polars_string
                if issubclass(expected, (np.datetime64, datetime)):
                    return dtype in (pl.Datetime, pl.Date)
                if issubclass(expected, (np.timedelta64, timedelta)):
                    return dtype == pl.Duration

            return dtype == expected

        missing_cols = [col for col in expected_schema if col not in schema]
        extra_cols = [col for col in schema if col not in expected_schema] if not allow_extra_columns else []

        mismatches = []
        for col, expected in expected_schema.items():
            if col not in schema:
                continue
            if not _matches_expected_dtype_polars(schema[col], expected):
                mismatches.append((col, expected, schema[col]))

        errors = []
        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        if extra_cols:
            errors.append(f"Unexpected columns: {extra_cols}")
        if mismatches:
            formatted = ", ".join([f"{col} (expected {exp!r}, got {actual})" for col, exp, actual in mismatches])
            errors.append(f"Dtype mismatches: {formatted}")

        if errors:
            raise ValueError("Schema validation failed: " + "; ".join(errors))
        return

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame or polars DataFrame/LazyFrame, got {type(df)!r}")

    def _matches_expected_dtype_pandas(series: pd.Series, expected: Any) -> bool:
        if isinstance(expected, (list, tuple, set)):
            return any(_matches_expected_dtype_pandas(series, e) for e in expected)

        dtype = series.dtype

        if isinstance(expected, str):
            key = expected.strip().lower()
            if key in ("int", "int64", "integer"):
                return pd.api.types.is_integer_dtype(dtype)
            if key in ("float", "float64", "double"):
                return pd.api.types.is_float_dtype(dtype)
            if key in ("bool", "boolean"):
                return pd.api.types.is_bool_dtype(dtype)
            if key in ("string", "str"):
                return pd.api.types.is_string_dtype(dtype)
            if key in ("object", "obj"):
                return pd.api.types.is_object_dtype(dtype)
            if key in ("datetime", "datetime64", "datetime64[ns]", "datetime64[ns, tz]"):
                return pd.api.types.is_datetime64_any_dtype(dtype)
            if key in ("timedelta", "timedelta64", "timedelta64[ns]"):
                return pd.api.types.is_timedelta64_dtype(dtype)
            try:
                return pd.api.types.is_dtype_equal(dtype, np.dtype(expected))
            except Exception:
                return False

        if isinstance(expected, np.dtype):
            return pd.api.types.is_dtype_equal(dtype, expected)

        if isinstance(expected, type):
            if issubclass(expected, (int, np.integer)):
                return pd.api.types.is_integer_dtype(dtype)
            if issubclass(expected, (float, np.floating)):
                return pd.api.types.is_float_dtype(dtype)
            if issubclass(expected, (bool, np.bool_)):
                return pd.api.types.is_bool_dtype(dtype)
            if issubclass(expected, str):
                return pd.api.types.is_string_dtype(dtype)
            if issubclass(expected, (np.datetime64, datetime)):
                return pd.api.types.is_datetime64_any_dtype(dtype)
            if issubclass(expected, (np.timedelta64, timedelta)):
                return pd.api.types.is_timedelta64_dtype(dtype)

        try:
            return pd.api.types.is_dtype_equal(dtype, expected)
        except Exception:
            return False

    missing_cols = [col for col in expected_schema if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in expected_schema] if not allow_extra_columns else []

    mismatches = []
    for col, expected in expected_schema.items():
        if col not in df.columns:
            continue
        if not _matches_expected_dtype_pandas(df[col], expected):
            mismatches.append((col, expected, df[col].dtype))

    errors = []
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    if extra_cols:
        errors.append(f"Unexpected columns: {extra_cols}")
    if mismatches:
        formatted = ", ".join([f"{col} (expected {exp!r}, got {actual})" for col, exp, actual in mismatches])
        errors.append(f"Dtype mismatches: {formatted}")

    if errors:
        raise ValueError("Schema validation failed: " + "; ".join(errors))


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
    # Stage 1: Memory safety checks and tracking
    'get_current_memory_usage', 'log_memory_checkpoint', 'MemoryTracker',
    'estimate_parquet_memory', 'estimate_filtered_data_memory',
    'compute_memory_model_features', 'predict_memory_gb',
    'MEMORY_MODEL_COEFFICIENTS', 'MEMORY_MODEL_FEATURE_NAMES',
    'check_memory_available', 'check_data_load_safe',
    # Embeddings
    'expand_embeddings_polars', 'get_embed_col_names', 
    # Features/columns
    'get_actual_feature_columns', 'build_user_feature_frame',
    # Relevel/topic helpers
    'discover_topics', 'compute_user_topic_mixtures', 'relevel_uniform_mixture',
    # Dataset construction
    'create_pairs_dataset',
    # Validation
    'validate_dataframe_schema', 'validate_data_integrity',
    # Viz
    'plot_training_history', 'plot_model_performance', 'create_user_visualization',
]


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
        '[%(asctime)s] [%(name)s] %(message)s',
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
