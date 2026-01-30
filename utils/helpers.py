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
import time
import random
import hashlib
import tempfile
import base64
import struct
import zlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from functools import partial
import warnings
from io import BytesIO
import multiprocessing as mp
from google.cloud import storage
import re
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
            pl.col("record_created_at").str.to_datetime(time_zone="UTC").alias("record_created_at_dt")
        )
    )
    if start_dt is not None:
        lf = lf.filter(pl.col("record_created_at_dt") >= start_dt)
    if end_dt is not None:
        lf = lf.filter(pl.col("record_created_at_dt") < end_dt)
    pandas_df = lf.collect().to_pandas()

    return pandas_df


# ----------------------------------------
# Stage 1: Memory estimation and safety checks
# ----------------------------------------
try:
    import psutil  # type: ignore
except ImportError:
    psutil = None  # type: ignore

# Memory estimation constants (bytes per value after Polars/pandas expansion)
DTYPE_MEMORY_MAP = {
    'String': 50,      # Average string length estimate
    'Utf8': 50,
    'Int64': 8,
    'Int32': 4,
    'Float64': 8,
    'Float32': 4,
    'Boolean': 1,
    'Date': 8,
    'Datetime': 8,
    'List': 200,       # For embeddings column (nested structure)
}


def get_current_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage for both the process and system.
    
    Returns:
        Dict with memory stats:
        - process_rss_gb: Process resident set size in GB
        - process_vms_gb: Process virtual memory size in GB
        - system_used_gb: System used memory in GB
        - system_available_gb: System available memory in GB
        - system_total_gb: System total memory in GB
        - system_percent: System memory usage percentage
    """
    if psutil is None:
        return {'error': 'psutil not available'}
    
    process = psutil.Process()
    mem_info = process.memory_info()
    sys_mem = psutil.virtual_memory()
    
    return {
        'process_rss_gb': mem_info.rss / (1024**3),
        'process_vms_gb': mem_info.vms / (1024**3),
        'system_used_gb': sys_mem.used / (1024**3),
        'system_available_gb': sys_mem.available / (1024**3),
        'system_total_gb': sys_mem.total / (1024**3),
        'system_percent': sys_mem.percent,
    }


def log_memory_checkpoint(
    checkpoint_name: str,
    logger: Optional[Any] = None,
    *,
    include_system: bool = True,
) -> Dict[str, Any]:
    """
    Log a memory checkpoint with current usage stats.
    
    This function is designed to be called at key points in the pipeline
    to track actual memory consumption vs. estimates.
    
    Args:
        checkpoint_name: Descriptive name for this checkpoint (e.g., "after_likes_load")
        logger: Logger instance to use for output
        include_system: Whether to include system-wide memory stats
        
    Returns:
        Dict with memory stats (same as get_current_memory_usage)
    """
    stats = get_current_memory_usage()
    
    if 'error' in stats:
        if logger:
            logger.warning(f"[MEMORY] {checkpoint_name}: {stats['error']}")
        return stats
    
    if include_system:
        msg = (
            f"[MEMORY] {checkpoint_name}: "
            f"process={stats['process_rss_gb']:.3f}GB, "
            f"system={stats['system_used_gb']:.1f}/{stats['system_total_gb']:.1f}GB "
            f"({stats['system_percent']:.1f}% used), "
            f"available={stats['system_available_gb']:.1f}GB"
        )
    else:
        msg = f"[MEMORY] {checkpoint_name}: process={stats['process_rss_gb']:.3f}GB"
    
    if logger:
        logger.info(msg)
    else:
        print(msg)
    
    return stats


class MemoryTracker:
    """
    Track memory usage throughout a pipeline run for comparison with estimates.
    
    Usage:
        tracker = MemoryTracker(logger=logger)
        tracker.checkpoint("start")
        # ... do work ...
        tracker.checkpoint("after_load_likes")
        # ... more work ...
        tracker.checkpoint("end")
        tracker.summary()  # logs all checkpoints and deltas
    """
    
    def __init__(self, logger: Optional[Any] = None):
        self.logger = logger
        self.checkpoints: List[Tuple[str, float, Dict[str, Any]]] = []
        self.start_time = time.time()
    
    def checkpoint(self, name: str) -> Dict[str, Any]:
        """Record a memory checkpoint."""
        elapsed = time.time() - self.start_time
        stats = log_memory_checkpoint(name, self.logger)
        self.checkpoints.append((name, elapsed, stats))
        return stats
    
    def get_peak_process_memory_gb(self) -> float:
        """Get the peak process memory observed across all checkpoints."""
        if not self.checkpoints:
            return 0.0
        return max(
            cp[2].get('process_rss_gb', 0) 
            for cp in self.checkpoints 
            if 'process_rss_gb' in cp[2]
        )
    
    def summary(self) -> Dict[str, Any]:
        """
        Log a summary of all memory checkpoints and return the data.
        """
        if not self.checkpoints:
            if self.logger:
                self.logger.info("[MEMORY SUMMARY] No checkpoints recorded")
            return {}
        
        peak_gb = self.get_peak_process_memory_gb()
        start_gb = self.checkpoints[0][2].get('process_rss_gb', 0) if self.checkpoints else 0
        end_gb = self.checkpoints[-1][2].get('process_rss_gb', 0) if self.checkpoints else 0
        
        summary_data = {
            'n_checkpoints': len(self.checkpoints),
            'peak_process_gb': peak_gb,
            'start_process_gb': start_gb,
            'end_process_gb': end_gb,
            'growth_gb': end_gb - start_gb,
            'checkpoints': [
                {
                    'name': name,
                    'elapsed_sec': elapsed,
                    'process_gb': stats.get('process_rss_gb', 0),
                }
                for name, elapsed, stats in self.checkpoints
            ]
        }
        
        if self.logger:
            self.logger.info("=" * 60)
            self.logger.info("[MEMORY SUMMARY]")
            self.logger.info(f"  Peak process memory: {peak_gb:.3f} GB")
            self.logger.info(f"  Memory growth: {start_gb:.3f} GB -> {end_gb:.3f} GB (+{summary_data['growth_gb']:.3f} GB)")
            self.logger.info("  Checkpoints:")
            for name, elapsed, stats in self.checkpoints:
                self.logger.info(f"    {elapsed:6.1f}s  {name}: {stats.get('process_rss_gb', 0):.3f} GB")
            self.logger.info("=" * 60)
        
        return summary_data


def estimate_parquet_memory(
    paths: List[str],
    *,
    embedding_expansion_dim: int = 384,
) -> Dict[str, Any]:
    """
    Estimate memory required to load parquet files WITHOUT loading them.
    
    Uses parquet metadata (row counts, schema) to estimate in-memory size.
    
    Args:
        paths: List of parquet file paths (gs:// or local)
        embedding_expansion_dim: Number of embedding dimensions to expand (0 to skip)
    
    Returns:
        Dict with estimated_bytes, estimated_gb, total_rows, etc.
    """
    if not paths:
        return {'estimated_bytes': 0, 'estimated_gb': 0.0, 'total_rows': 0}
    
    total_rows = 0
    schema = None
    
    for path in paths:
        # Read just the metadata (no data loaded)
        try:
            lf = pl.scan_parquet(path)
            meta = lf.collect_schema()
            if schema is None:
                schema = meta
            
            # Get row count from parquet metadata
            row_count = lf.select(pl.len()).collect().item()
            total_rows += row_count
        except Exception:
            continue
    
    if schema is None or total_rows == 0:
        return {'estimated_bytes': 0, 'estimated_gb': 0.0, 'total_rows': 0}
    
    # Estimate memory per row based on schema
    bytes_per_row = 0
    has_embeddings = False
    
    for col_name, dtype in schema.items():
        dtype_str = str(dtype)
        
        if col_name == 'embeddings':
            has_embeddings = True
            # Embeddings column will be dropped after expansion
            continue
        
        # Get estimated bytes for this dtype
        matched = False
        for known_dtype, size in DTYPE_MEMORY_MAP.items():
            if known_dtype.lower() in dtype_str.lower():
                bytes_per_row += size
                matched = True
                break
        if not matched:
            bytes_per_row += 50  # Default estimate for unknown types
    
    # Add embedding expansion overhead
    if has_embeddings and embedding_expansion_dim > 0:
        # Each embedding dimension becomes a Float32 column
        bytes_per_row += embedding_expansion_dim * 4
    
    # Add polars/pandas overhead (typically 1.5-2x raw data)
    overhead_factor = 1.8
    estimated_bytes = int(total_rows * bytes_per_row * overhead_factor)
    
    return {
        'estimated_bytes': estimated_bytes,
        'estimated_gb': estimated_bytes / (1024**3),
        'total_rows': total_rows,
        'bytes_per_row': bytes_per_row,
        'n_columns': len(schema),
        'has_embeddings': has_embeddings,
    }


# ============================================================================
# MEMORY ESTIMATION MODEL
# ============================================================================
# Fitted regression model for predicting peak memory usage.
#
# Model performance: R-squared = 0.9440, Mean % error = 6.7%
#
# To re-fit after collecting new sweep data:
#   python scripts/fit_memory_model.py --input sweep_results.csv
# This will output new coefficients to paste here.
# ============================================================================

MEMORY_MODEL_COEFFICIENTS = {
    'intercept': 32.1569093112,
    'data_window_days': -0.3731594051,
    'max_liking_users_10k': 5.3712117159,
    'max_likes_per_user_100': 0.0586046975,
    'negative_posts_sample_10k': -0.0297902646,
    'log_max_liking_users': -8.7875844850,
    'sqrt_likes_initial_1e6': 4.4282688972,
    'days_x_users_10k': 0.0362670731,
    'users_x_log_users': -0.8738069443,
}

MEMORY_MODEL_FEATURE_NAMES = [
    'data_window_days',
    'max_liking_users_10k',
    'max_likes_per_user_100',
    'negative_posts_sample_10k',
    'log_max_liking_users',
    'sqrt_likes_initial_1e6',
    'days_x_users_10k',
    'users_x_log_users',
]


def compute_memory_model_features(
    data_window_days: float,
    max_liking_users: float,
    max_likes_per_user: float,
    negative_posts_sample: float,
    likes_initial: float,
) -> Dict[str, float]:
    """Compute feature values for the memory prediction model.
    
    Args:
        data_window_days: Number of days in the data window (e.g., 7, 14, 21)
        max_liking_users: Maximum number of liking users to sample
        max_likes_per_user: Cap on likes per user
        negative_posts_sample: Number of negative posts to sample
        likes_initial: Total raw likes count (from parquet metadata)
    
    Returns:
        Dict mapping feature names to values
    """
    return {
        'data_window_days': data_window_days,
        'max_liking_users_10k': max_liking_users / 10000,
        'max_likes_per_user_100': max_likes_per_user / 100,
        'negative_posts_sample_10k': negative_posts_sample / 10000,
        'log_max_liking_users': np.log10(max(max_liking_users, 1)),
        'sqrt_likes_initial_1e6': np.sqrt(likes_initial) / 1000,
        'days_x_users_10k': data_window_days * max_liking_users / 10000,
        'users_x_log_users': (max_liking_users / 10000) * np.log10(max(max_liking_users, 1)),
    }


def predict_memory_gb(
    features: Dict[str, float],
    coefficients: Optional[Dict[str, float]] = None,
) -> float:
    """Predict peak memory usage in GB using the fitted model.
    
    Args:
        features: Feature dict from compute_memory_model_features()
        coefficients: Model coefficients (defaults to MEMORY_MODEL_COEFFICIENTS)
    
    Returns:
        Estimated peak memory in GB
    """
    if coefficients is None:
        coefficients = MEMORY_MODEL_COEFFICIENTS
    
    result = coefficients['intercept']
    for name in MEMORY_MODEL_FEATURE_NAMES:
        result += coefficients[name] * features[name]
    
    # Ensure non-negative (model could predict negative for extreme edge cases)
    return max(result, 1.0)


def estimate_filtered_data_memory(
    likes_paths: List[str],
    posts_paths: List[str],
    *,
    max_liking_users: Optional[int] = None,
    max_likes_per_user: int = 100,
    min_likes_per_user: int = 2,
    negative_posts_sample: int = 100_000,
    embedding_dim: int = 384,
    logger: logging.Logger,
    data_window_days: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Estimate memory required for data loading using a fitted regression model.
    
    Uses a regression model trained on sweep_results.csv to predict peak memory
    usage based on configuration parameters (users, likes/user, window size, etc.).
    
    Args:
        likes_paths: Paths to likes parquet files
        posts_paths: Paths to posts parquet files
        max_liking_users: Cap on number of liking users (None = no cap, uses 10000 for estimation)
        max_likes_per_user: Max likes to keep per user
        min_likes_per_user: Min likes required per user (not used by model, kept for compatibility)
        negative_posts_sample: Number of negative posts to sample
        embedding_dim: Embedding dimension (not used by model, kept for compatibility)
        logger: Logger instance for output
        data_window_days: Number of days in the data window (auto-detected from file count if not provided)
        
    Returns:
        Dict with memory estimates and metadata
    """
    
    # Get raw stats from parquet metadata (no data loading)
    likes_raw = estimate_parquet_memory(likes_paths, embedding_expansion_dim=0)
    posts_raw = estimate_parquet_memory(posts_paths, embedding_expansion_dim=0)
    
    raw_likes_rows = likes_raw['total_rows']
    raw_posts_rows = posts_raw['total_rows']
    n_likes_files = len(likes_paths)
    n_posts_files = len(posts_paths)
    
    if raw_likes_rows == 0:
        return {
            'estimated_peak_gb': 0,
            'estimated_total_gb': 0,
            'error': 'No likes data found',
            'model_version': 'v1.0',
        }
    
    # Auto-detect data window days from file count if not provided
    # Each day typically has ~24 files (hourly), so divide by 24
    if data_window_days is None:
        data_window_days = max(1, n_likes_files // 24)
        if data_window_days < 1:
            data_window_days = 7  # Default fallback
    
    # Use default max_liking_users if not specified (None means no cap)
    effective_max_users = max_liking_users if max_liking_users is not None else 10000
    
    # Compute features for the model
    features = compute_memory_model_features(
        data_window_days=data_window_days,
        max_liking_users=effective_max_users,
        max_likes_per_user=max_likes_per_user,
        negative_posts_sample=negative_posts_sample,
        likes_initial=raw_likes_rows,
    )
    
    # Predict memory using fitted model
    estimated_peak_gb = predict_memory_gb(features)
    
    # Add 10% safety margin for unknown variations
    estimated_peak_gb_with_margin = estimated_peak_gb * 1.10
    
    # Build result dict (maintaining backward compatibility with key fields)
    result = {
        # Primary estimates
        'estimated_peak_gb': estimated_peak_gb_with_margin,
        'estimated_total_gb': estimated_peak_gb_with_margin,
        'estimated_peak_gb_raw': estimated_peak_gb,  # Without safety margin
        
        # Model metadata
        'model_version': 'v1.0',
        'model_r_squared': 0.9440,
        
        # Raw data stats
        'raw_likes_rows': raw_likes_rows,
        'raw_posts_rows': raw_posts_rows,
        'raw_likes_gb': likes_raw['estimated_gb'],
        'raw_posts_gb': posts_raw['estimated_gb'],
        'n_likes_files': n_likes_files,
        'n_posts_files': n_posts_files,
        
        # Model inputs (for debugging/transparency)
        'model_features': features,
        
        # Parameters used
        'params': {
            'max_liking_users': max_liking_users,
            'effective_max_users': effective_max_users,
            'max_likes_per_user': max_likes_per_user,
            'min_likes_per_user': min_likes_per_user,
            'negative_posts_sample': negative_posts_sample,
            'embedding_dim': embedding_dim,
            'data_window_days': data_window_days,
        },
    }
    
    logger.info("Memory estimation (fitted regression model):")
    logger.info(f"  Raw data: {raw_likes_rows:,} likes ({n_likes_files} files), {raw_posts_rows:,} posts ({n_posts_files} files)")
    logger.info(f"  Window: {data_window_days} days, Users: {effective_max_users:,}, Likes/user cap: {max_likes_per_user}")
    logger.info(f"  Model prediction: {estimated_peak_gb:.2f} GB (with 10% margin: {estimated_peak_gb_with_margin:.2f} GB)")
    logger.info(f"  Model R-squared: 0.9440 (mean error: 6.7%)")
    
    return result


def check_memory_available(
    estimated_bytes: int,
    *,
    max_memory_gb: Optional[float] = None,  # None = use percentage of available
    max_memory_pct: float = 0.75,  # Use at most 75% of available RAM
    logger: Optional[Any] = None,
) -> Tuple[bool, str]:
    """
    Check if estimated memory usage is safe given available system memory.
    
    Args:
        estimated_bytes: Estimated memory required
        max_memory_gb: Maximum memory in GB (None = auto based on percentage)
        max_memory_pct: Maximum percentage of available RAM to use
        logger: Optional logger for output
    
    Returns:
        Tuple of (is_safe: bool, message: str)
    """
    if psutil is None:
        return True, "psutil not available, skipping memory check"
    
    mem = psutil.virtual_memory()
    available_bytes = mem.available
    total_bytes = mem.total
    
    # Determine threshold
    if max_memory_gb is not None:
        threshold_bytes = int(max_memory_gb * (1024**3))
    else:
        threshold_bytes = int(available_bytes * max_memory_pct)
    
    is_safe = estimated_bytes <= threshold_bytes
    
    msg = (
        f"Memory check: estimated {estimated_bytes/(1024**3):.2f} GB, "
        f"threshold {threshold_bytes/(1024**3):.2f} GB, "
        f"available {available_bytes/(1024**3):.2f} GB / {total_bytes/(1024**3):.2f} GB total"
    )
    
    if not is_safe:
        msg = f"MEMORY LIMIT EXCEEDED: {msg}"
    
    if logger:
        if is_safe:
            logger.info(msg)
        else:
            logger.error(msg)
    
    return is_safe, msg


def check_data_load_safe(
    likes_paths: List[str],
    posts_paths: List[str],
    *,
    embedding_dim: int = 384,
    max_memory_gb: Optional[float] = None,
    max_memory_pct: float = 0.75,
    max_liking_users: Optional[int] = None,
    max_likes_per_user: int = 100,
    min_likes_per_user: int = 2,
    negative_posts_sample: int = 100_000,
    skip_safety_check: bool = False,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Pre-flight check before loading data. Raises MemoryError if unsafe.
    
    This function uses smart estimation that accounts for filtering parameters
    to provide more accurate memory predictions.
    
    Args:
        likes_paths: List of likes parquet paths
        posts_paths: List of posts parquet paths
        embedding_dim: Embedding dimension for memory estimation
        max_memory_gb: Maximum memory in GB (None = auto based on percentage)
        max_memory_pct: Maximum percentage of available RAM
        max_liking_users: Cap on number of liking users
        max_likes_per_user: Max likes to keep per user
        min_likes_per_user: Min likes required per user
        negative_posts_sample: Number of negative posts to sample
        skip_safety_check: If True, perform estimation but don't raise error
        logger: Optional logger
    
    Returns:
        Dict with memory estimation details
    
    Raises:
        MemoryError: If estimated memory exceeds limits (unless skip_safety_check=True)
    """
    if psutil is None:
        if logger:
            logger.warning("psutil not available, skipping memory safety check")
        return {'skipped': True, 'reason': 'psutil not available'}
    
    if logger:
        logger.info("=" * 60)
        logger.info("PRE-FLIGHT MEMORY CHECK")
        logger.info("=" * 60)
    
    # Use smart estimation that accounts for filtering
    estimation = estimate_filtered_data_memory(
        likes_paths=likes_paths,
        posts_paths=posts_paths,
        max_liking_users=max_liking_users,
        max_likes_per_user=max_likes_per_user,
        min_likes_per_user=min_likes_per_user,
        negative_posts_sample=negative_posts_sample,
        embedding_dim=embedding_dim,
        logger=logger,
    )
    
    if 'error' in estimation:
        if logger:
            logger.warning(f"Memory estimation warning: {estimation['error']}")
        return estimation
    
    # Also show raw (unfiltered) estimate for comparison
    raw_likes = estimate_parquet_memory(likes_paths, embedding_expansion_dim=0)
    raw_posts = estimate_parquet_memory(posts_paths, embedding_expansion_dim=embedding_dim)
    raw_total_gb = raw_likes['estimated_gb'] + raw_posts['estimated_gb']
    
    if logger:
        logger.info(f"  Raw data (unfiltered): ~{raw_total_gb:.2f} GB")
        logger.info(f"  Estimated peak (incremental): ~{estimation['estimated_peak_gb']:.2f} GB")
    
    # Use the peak estimate for safety check
    estimated_bytes = int(estimation['estimated_peak_gb'] * (1024**3))
    
    is_safe, msg = check_memory_available(
        estimated_bytes,
        max_memory_gb=max_memory_gb,
        max_memory_pct=max_memory_pct,
        logger=logger,
    )
    
    estimation['is_safe'] = is_safe
    estimation['memory_check_message'] = msg
    
    if logger:
        logger.info("=" * 60)
    
    if not is_safe:
        if skip_safety_check:
            if logger:
                logger.warning("Memory limit would be exceeded, but proceeding due to --skip-memory-check")
        else:
            raise MemoryError(
                f"Data load would exceed memory limits. {msg}\n"
                f"Estimated peak memory: {estimation['estimated_peak_gb']:.2f} GB\n"
                f"Options to reduce memory:\n"
                f"  - Use a shorter time window (--posts-start/--posts-end)\n"
                f"  - Reduce --max-liking-users (current: {max_liking_users})\n"
                f"  - Reduce --max-likes-per-user (current: {max_likes_per_user})\n"
                f"  - Reduce --negative-posts-sample (current: {negative_posts_sample})\n"
                f"  - Increase --max-memory-gb if you have more RAM available\n"
                f"  - Use --skip-memory-check to proceed anyway (at your own risk)"
            )
    
    return estimation


def _apply_time_filter(
    lf: pl.LazyFrame, 
    start_str: Optional[str], 
    end_str: Optional[str]
) -> pl.LazyFrame:
    if 'record_created_at' not in lf.collect_schema().names():
        raise ValueError("Input LazyFrame does not contain 'record_created_at' column for time filtering")
    if start_str is not None:
        lf = lf.filter(pl.col("record_created_at") >= start_str)
    if end_str is not None:
        lf = lf.filter(pl.col("record_created_at") < end_str)
    return lf


# ----------------------------------------
# Stage 1: Polars-based filtering for core datasets
# ----------------------------------------
def load_likes_core_polars(
    start_str: Optional[str],
    end_str: Optional[str],
    paths: List[str],
    *,
    max_liking_users: Optional[int] = None,
    max_likes_per_user: int,
    min_likes_per_user: int,
    random_seed: int,
    logger: logging.Logger,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """
    Load and filter likes data using a streaming Polars pipeline.
    
    High-level flow:
    1. Streamed pass: count likes per user
    2. Pre-filter users who don't meet min_likes_per_user
    3. Sample users from eligible pool (if cap is set)
    4. Streamed pass: keep only likes from sampled users
    5. Apply per-user random caps (NOT recency-based)
    6. Verify min-likes threshold (handles edge cases from per-user caps)
    
    This avoids materializing the full likes table in memory and returns a
    LazyFrame suitable for streaming writes (sink_parquet).
    
    Returns:
        Tuple of (likes_lf: pl.LazyFrame, stats: Dict with filtering statistics)
    """
    if not paths:
        raise ValueError(f"No likes parquet files found for time range {start_str} to {end_str}")
    
    logger.info(f"Found {len(paths)} likes parquet files")
    log_memory_checkpoint("likes_before_scan", logger)
    
    raw_lf = pl.scan_parquet(paths)
    base_lf = _apply_time_filter(raw_lf, start_str, end_str)

    # ===== PASS 1: Count likes per user (streaming) =====
    logger.info("Pass 1: Counting likes per user (streaming)...")
    user_counts_df = (
        base_lf.group_by('did')
        .agg(pl.len().alias('like_count'))
        .collect(engine="streaming")
    )
    
    n_users_initial = user_counts_df.height
    n_likes_initial = int(user_counts_df['like_count'].sum()) if n_users_initial > 0 else 0
    logger.info(f"Pass 1 complete: {n_likes_initial:,} likes from {n_users_initial:,} users")
    log_memory_checkpoint("likes_after_pass1", logger)
    
    stats = {
        'n_likes_initial': n_likes_initial,
        'n_users_initial': n_users_initial,
    }

    # ===== Pre-filter users by min_likes_per_user before sampling =====
    if min_likes_per_user > 0:
        eligible_users_df = user_counts_df.filter(pl.col('like_count') >= min_likes_per_user)
        n_users_eligible = eligible_users_df.height
        n_users_filtered = n_users_initial - n_users_eligible
        logger.info(f"Pre-filtering: {n_users_eligible:,} users meet min-likes threshold ({min_likes_per_user}), "
             f"excluded {n_users_filtered:,} users with too few likes")
        stats['n_users_eligible_for_sampling'] = n_users_eligible
        stats['n_users_excluded_min_likes'] = n_users_filtered
    else:
        eligible_users_df = user_counts_df.select('did')
        stats['n_users_eligible_for_sampling'] = n_users_initial

    # ===== Sample users if cap is set =====
    if max_liking_users is not None and eligible_users_df.height > max_liking_users:
        sampled_users_df = (
            eligible_users_df.with_columns(
                pl.col('did').hash(seed=random_seed).alias('_rand_key')
            ).sort('_rand_key').head(max_liking_users).select('did')
        )
        logger.info(
            f"Sampled {max_liking_users:,} liking users "
            f"({100*max_liking_users/eligible_users_df.height:.1f}% of eligible)"
        )
        stats['n_users_sampled'] = max_liking_users
    else:
        sampled_users_df = eligible_users_df.select('did')
        stats['n_users_sampled'] = sampled_users_df.height
    
    log_memory_checkpoint("likes_after_user_sample", logger)
    
    # ===== PASS 2: Filter likes to sampled users (lazy) =====
    logger.info("Pass 2: Filtering likes for sampled users (streaming)...")
    likes_lf = base_lf.join(sampled_users_df.lazy(), on='did', how='semi')
    
    # Compute counts per user before per-user cap
    counts_pre_cap_df = (
        likes_lf.group_by('did')
        .agg(pl.len().alias('like_count'))
        .collect(engine='streaming')
    )
    
    n_after_user_sample = int(counts_pre_cap_df['like_count'].sum()) if counts_pre_cap_df.height > 0 else 0
    pct_retained = 100.0 * n_after_user_sample / n_likes_initial if n_likes_initial > 0 else 0
    logger.info(f"Pass 2 complete: {n_after_user_sample:,} likes ({pct_retained:.1f}% retained)")
    stats['n_likes_after_user_sample'] = n_after_user_sample
    log_memory_checkpoint("likes_after_pass2", logger)
    
    # ===== Capture like count distribution BEFORE cap (for analysis/plotting) =====
    # This shows how many likes each sampled user has before we apply the per-user cap
    if counts_pre_cap_df.height > 0:
        likes_per_user_before_cap = counts_pre_cap_df['like_count'].to_list()
        stats['likes_per_user_distribution'] = likes_per_user_before_cap
        stats['likes_per_user_mean'] = float(np.mean(likes_per_user_before_cap))
        stats['likes_per_user_median'] = float(np.median(likes_per_user_before_cap))
        stats['likes_per_user_max'] = int(max(likes_per_user_before_cap))
        stats['likes_per_user_p90'] = float(np.percentile(likes_per_user_before_cap, 90))
        stats['likes_per_user_p99'] = float(np.percentile(likes_per_user_before_cap, 99))
        logger.info(f"Likes per sampled user: mean={stats['likes_per_user_mean']:.1f}, "
             f"median={stats['likes_per_user_median']:.0f}, max={stats['likes_per_user_max']}, "
             f"p90={stats['likes_per_user_p90']:.0f}, p99={stats['likes_per_user_p99']:.0f}")
    
    # ===== Apply per-user random cap (NOT recency-based) =====
    if max_likes_per_user > 0 and n_after_user_sample > 0:
        n_before_cap = n_after_user_sample
        
        # Add deterministic pseudo-random order per user, then keep top-K
        likes_lf = likes_lf.with_columns(
            pl.col('subject_uri').hash(seed=random_seed).alias('_rand_key')
        )
        likes_lf = likes_lf.with_columns(
            pl.col('_rand_key').rank('ordinal').over('did').alias('_rand_order')
        )
        likes_lf = likes_lf.filter(pl.col('_rand_order') <= max_likes_per_user)
        likes_lf = likes_lf.drop(['_rand_key', '_rand_order'])
        
        # Compute post-cap counts from pre-cap per-user counts to avoid materializing likes_lf.
        n_after_cap = int(
            counts_pre_cap_df
            .select(pl.col('like_count').clip(upper_bound=max_likes_per_user).sum())
            .item()
        )
        pct_retained = 100.0 * n_after_cap / n_before_cap if n_before_cap > 0 else 0
        logger.info(f"After per-user cap ({max_likes_per_user}): {n_after_cap:,} likes ({pct_retained:.1f}% retained)")
        stats['n_likes_after_per_user_cap'] = n_after_cap
    else:
        stats['n_likes_after_per_user_cap'] = n_after_user_sample
    
    # Select output columns
    output_cols = ['did', 'subject_uri', 'record_created_at']
    available_cols = [c for c in output_cols if c in likes_lf.collect_schema().names()]
    if available_cols:
        likes_lf = likes_lf.select(available_cols)
    
    # Convert record_created_at to datetime if it exists and is not already datetime
    schema = likes_lf.collect_schema()
    if 'record_created_at' in schema and schema['record_created_at'] != pl.Datetime:
        if schema['record_created_at'] in (pl.String, pl.Utf8):
            likes_lf = likes_lf.with_columns(
                pl.col('record_created_at').str.to_datetime(time_zone="UTC").alias('record_created_at')
            )
        else:
            likes_lf = likes_lf.with_columns(
                pl.col('record_created_at').cast(pl.Datetime).alias('record_created_at')
            )
    
    likes_df = likes_lf.collect(engine="streaming")

    stats['n_likes_final'] = len(likes_df)
    stats['n_users_final'] = likes_df['did'].n_unique() if len(likes_df) > 0 else 0

    log_memory_checkpoint("likes_final", logger)
    
    return likes_df, stats


def save_polars_physical_plan_image(lf: pl.LazyFrame, out_path: str):
    dot = lf.show_graph(plan_stage='physical', engine='streaming', raw_output=True)
    if dot is not None:
        Path("plan.dot").write_text(dot)
    else:
        print("\n\nNo DOT output generated!!!\n\n")
    subprocess.run(["dot", "-Tpng", "-Gdpi=220", "plan.dot", "-o", out_path], check=True) 


def load_posts_core_polars(
    start_str: Optional[str],
    end_str: Optional[str],
    liked_post_uris_df: pl.DataFrame,
    paths: List[str],
    *,
    negative_posts_sample: int,
    embedding_model: str,
    random_seed: int,
    logger: logging.Logger,
    out_dir: Path,
) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
    """
    Load posts data using batch processing with early embedding expansion.
    
    Key optimization: expand embeddings per-batch BEFORE accumulating.
    This reduces memory by ~98% (150KB/post raw -> 2KB/post expanded).
    
    Processing flow:
    1. Process files in batches (20 files at a time)
    2. For each batch:
       a) Reservoir sample from ALL posts (independent of like status)
       b) Collect liked posts NOT already in the random sample
       c) Expand embeddings and DROP the raw blob
    3. Accumulate only the slim expanded data
    
    Statistical independence:
    The random sample is drawn from ALL posts, not filtered by like status.
    Posts that are both liked AND randomly sampled appear once with in_random_sample=True.
    This ensures the random sample can be used for unbiased population statistics.
    
    Output columns:
    - in_random_sample: True if post was collected via reservoir sampling,
                        False if collected only because it was liked
    - To identify liked posts: join with likes_core on subject_uri = at_uri
    
    Returns:
        Tuple of (posts_df: pl.DataFrame, stats: Dict, embedding_dim: int)
    """
    if not paths:
        raise ValueError(f"No posts parquet files found for time range {start_str} to {end_str}")
    
    logger.info(f"Found {len(paths)} posts parquet files")
    log_memory_checkpoint("posts_before_scan", logger)

    posts_lf = pl.scan_parquet(paths)
    posts_lf = _apply_time_filter(posts_lf, start_str, end_str)

    # get the total number of posts and calc threshold
    n_posts_total = posts_lf.select(pl.count()).collect().item()
    logger.info(f"n_posts_total: {n_posts_total:,}")
    fraction_to_sample = negative_posts_sample / n_posts_total
    threshold_hash = int(fraction_to_sample * (2**64 - 1))

    cols_with_emb = ["at_uri", "embeddings", "record_created_at", "did", "record_text"]
    cols_no_emb = cols_with_emb.copy()
    cols_no_emb.remove("embeddings")
    
    # get posts: sampled via hash, or in liked_post_uris:
    negs_and_likes_lf = (
        posts_lf
        .select(cols_with_emb)
        .with_columns(
            pl.col("at_uri").hash(seed=random_seed).alias("_hash_key"),
        )
        .join(
            liked_post_uris_df.with_columns(pl.lit(True).alias("_is_liked")).lazy(),
            left_on="at_uri",
            right_on="subject_uri",
            how="left",
        )
        .with_columns(
            (pl.col("_hash_key") <= threshold_hash).alias("in_random_sample"),
            pl.col("_is_liked").fill_null(False).alias("is_liked"),
        )
        .filter(pl.col("in_random_sample") | pl.col("is_liked"))
        .drop(["_is_liked", "_hash_key"])
    )

    # get embedding dim
    embed_dim = get_embed_dim(posts_lf, embedding_model)
    logger.info(f"Detected embedding dimension: {embed_dim}")

    # expand embeddings into columns
    posts_core_lf = expand_embeddings_polars(negs_and_likes_lf, embedding_model, embed_dim)
    
    # Validate posts_core_lf schema
    posts_schema_with_embs = {
        'at_uri': str,
        'in_random_sample': bool,
        'did': str,
        'record_created_at': str,
        'record_text': str,
        'is_liked': bool,
    }
    for i in range(embed_dim):
        posts_schema_with_embs[f'post_emb_{i}'] = float
    validate_dataframe_schema(posts_core_lf, posts_schema_with_embs, allow_extra_columns=False)

    # write out
    logger.info(f"✓ posts_core schema validated (embed_dim={embed_dim})")

    # Save outputs as parquet
    log_operation_start('Save likes core dataset as parquet', 'STAGE_01_GET_DATA', logger)
    ts_name = out_dir.name
    posts_core_path = out_dir / f"posts_core_{ts_name}.parquet"
    
    # low row_group_size because embeddings are very large. keeps memory low
    posts_core_lf.sink_parquet(posts_core_path, compression="zstd", engine="streaming", row_group_size=128)
    log_memory_checkpoint("posts_after_sink_parquet", logger)

    # read back from the parquet file, withOUT embeddings
    posts_core_df = (
        pl
        .scan_parquet(posts_core_path)
        .select(cols_no_emb + ['is_liked', 'in_random_sample'])
        .collect(engine="streaming")
    )

    # Validate posts_core_lf schema
    posts_schema_no_embs = {
        'at_uri': str,
        'in_random_sample': bool,
        'did': str,
        'record_created_at': str,
        'record_text': str,
        'is_liked': bool,
    }
    validate_dataframe_schema(posts_core_df, posts_schema_no_embs, allow_extra_columns=False)

    # calculate metrics
    n_posts_core = posts_core_df.height
    n_liked_only = posts_core_df.filter(pl.col("is_liked") & ~pl.col("in_random_sample")).height
    n_liked_in_random = posts_core_df.filter(pl.col("is_liked") & pl.col("in_random_sample")).height
    n_random_sample = posts_core_df.filter(pl.col("in_random_sample")).height

    logger.info(f"Saved posts_core: {posts_core_path} ({n_posts_core:,} rows)")
    logger.info(f"All posts in raw data: {n_posts_total:,}")
    logger.info(f"Liked only: {n_liked_only:,}")
    logger.info(f"Liked in random sample: {n_liked_in_random:,}")
    logger.info(f"Random sample total: {n_random_sample:,}")

    # Total liked posts = those only in liked set + those also in random sample
    n_total_liked_posts = n_liked_only + n_liked_in_random
    liked_post_match_rate = 100.0 * n_total_liked_posts / liked_post_uris_df.height
    logger.info(f"Loaded {n_total_liked_posts:,} liked posts ({liked_post_match_rate:.1f}% match rate)")
    
    stats = {
        'n_posts_total': n_posts_total,
        'n_liked_posts': n_total_liked_posts,
        'n_liked_only': n_liked_only,  # Liked posts not in random sample
        'n_liked_in_random_sample': n_liked_in_random,  # Liked posts that are also in random sample
        'liked_post_match_rate': liked_post_match_rate,
        'n_random_sample': n_random_sample,
    }
    
    logger.info(f"posts_core: {n_posts_core:,} rows ({n_liked_only:,} liked-only + {n_random_sample:,} random sample)")
    logger.info(f"Embeddings already expanded during loading (dim={embed_dim})")
    
    stats['n_posts_core'] = n_posts_core
    stats['embedding_dim'] = embed_dim
    log_memory_checkpoint("posts_after_combine", logger)
    
    return posts_core_df, stats, embed_dim


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
        lambda s: embedding_loads(s, decompress=True) if s is not None else None,
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
    # Data IO Green Earth Ingex GCS
    'load_raw_data_ingex',
    # Stage 1: Memory safety checks and tracking
    'get_current_memory_usage', 'log_memory_checkpoint', 'MemoryTracker',
    'estimate_parquet_memory', 'estimate_filtered_data_memory',
    'compute_memory_model_features', 'predict_memory_gb',
    'MEMORY_MODEL_COEFFICIENTS', 'MEMORY_MODEL_FEATURE_NAMES',
    'check_memory_available', 'check_data_load_safe',
    # Stage 1: Polars-based filtering for core datasets
    'load_likes_core_polars', 'load_posts_core_polars', 'expand_embeddings_polars',
    # Embeddings
    'get_embed_col_names', 'embedding_loads', 'extract_encoded_embedding_ingex',
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
