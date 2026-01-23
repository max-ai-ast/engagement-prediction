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


def estimate_filtered_data_memory(
    likes_paths: List[str],
    posts_paths: List[str],
    *,
    max_liking_users: int = 0,
    max_likes_per_user: int = 100,
    min_likes_per_user: int = 2,
    negative_posts_sample: int = 100_000,
    embedding_dim: int = 384,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Estimate memory required using the incremental file processing approach.
    
    With incremental processing, peak memory is bounded by:
    1. Output data: filtered likes + liked posts + negative sample + expanded embeddings
    2. Working memory: one parquet file at a time during processing
    3. Temporary structures: user set during sampling, reservoir for negatives
    
    This is much more efficient than loading all raw data at once.
    
    Args:
        likes_paths: Paths to likes parquet files
        posts_paths: Paths to posts parquet files
        max_liking_users: Cap on number of liking users (0 = no cap)
        max_likes_per_user: Max likes to keep per user
        min_likes_per_user: Min likes required per user
        negative_posts_sample: Number of negative posts to sample
        embedding_dim: Embedding dimension (for posts memory)
        logger: Optional logger
        
    Returns:
        Dict with detailed memory estimates
    """
    def _log(msg: str):
        if logger:
            logger.info(msg)
    
    # Get raw stats without loading data
    likes_raw = estimate_parquet_memory(likes_paths, embedding_expansion_dim=0)
    posts_raw = estimate_parquet_memory(posts_paths, embedding_expansion_dim=0)  # No expansion yet
    
    raw_likes_rows = likes_raw['total_rows']
    raw_posts_rows = posts_raw['total_rows']
    n_likes_files = len(likes_paths)
    n_posts_files = len(posts_paths)
    
    if raw_likes_rows == 0:
        return {
            'estimated_total_gb': 0,
            'likes_estimated_gb': 0,
            'posts_estimated_gb': 0,
            'error': 'No likes data found',
        }
    
    # === Estimate filtered data sizes ===
    
    # Estimate unique users from raw data
    # Observed: ~47 likes per user on average in typical time windows
    avg_likes_per_user_observed = 47
    estimated_raw_users = max(1, raw_likes_rows // avg_likes_per_user_observed)
    
    # Calculate actual average likes per user in THIS time window
    avg_likes_per_user_in_window = raw_likes_rows / max(estimated_raw_users, 1)
    
    # Apply user cap
    if max_liking_users > 0:
        estimated_users = min(max_liking_users, estimated_raw_users)
    else:
        estimated_users = estimated_raw_users
    
    # IMPORTANT: Memory estimation should focus on INTERMEDIATE sizes during processing,
    # not final filtered output. The key insight from observations:
    #
    # Likes processing:
    # - Pass 1: Scan all files to collect user DIDs (memory: user set ~200MB)
    # - Pass 2: Collect ALL likes for sampled users before per-user cap/min-likes filter
    #   -> This is proportional to (sampled_users / total_users) × raw_likes
    #   -> These likes are stored in chunks with significant overhead (~4KB/row)
    #
    # Posts processing:
    # - Batch through files, extracting liked posts + reservoir sampling negatives
    # - Final output is relatively small (liked posts + negative sample)
    
    # Calculate INTERMEDIATE likes (before per-user cap and min-likes filter)
    # This is what actually consumes memory during Pass 2
    user_sampling_ratio = estimated_users / max(estimated_raw_users, 1)
    intermediate_likes = int(raw_likes_rows * user_sampling_ratio)
    
    # Estimate final likes after filtering (for output size)
    # Per-user cap: users with > max_likes_per_user get truncated
    # Min-likes filter: users with < min_likes_per_user get removed
    # Observed: very aggressive filtering, especially with power-law distributions
    expected_likes_per_user = min(avg_likes_per_user_in_window, max_likes_per_user)
    estimated_likes = int(estimated_users * expected_likes_per_user * 0.95)  # 5% loss to min-likes
    
    # Estimate liked posts for final output
    unique_liked_uris_estimate = int(estimated_likes * 0.5)
    liked_post_match_rate = 0.75
    estimated_liked_posts = min(int(unique_liked_uris_estimate * liked_post_match_rate), raw_posts_rows)
    
    # Total posts = liked posts + negative sample (this is final output)
    estimated_posts = estimated_liked_posts + min(negative_posts_sample, raw_posts_rows)
    
    # === Memory estimation with incremental processing ===
    
    # Bytes per row estimates (in-memory representation)
    # IMPORTANT: These are based on observed actual memory usage, not theoretical sizes
    
    # Likes during chunk accumulation have significant overhead (~4KB per row observed)
    # This includes Polars DataFrame overhead per chunk, string storage, etc.
    likes_bytes_per_row_intermediate = 4000  # During Pass 2 chunk accumulation
    likes_bytes_per_row_final = 200  # After combining into single DataFrame
    
    # Posts: use parquet metadata to estimate, with 4x expansion for in-memory
    posts_bytes_per_row_parquet = posts_raw['estimated_bytes'] / max(raw_posts_rows, 1)
    posts_bytes_per_row_raw = int(posts_bytes_per_row_parquet * 4)
    
    # Expanded posts: ~31KB per post observed (text, author, embeddings as float32s, etc.)
    posts_bytes_per_row_expanded = 32_000
    
    # Component 1: User set during Pass 1 (strings in memory)
    user_set_bytes = estimated_raw_users * 80  # ~80 bytes per DID string in set
    
    # Component 2: Working memory per batch of files
    batch_size = 20  # Files per batch
    actual_likes_batch = min(batch_size, n_likes_files)
    actual_posts_batch = min(batch_size, n_posts_files)
    
    avg_likes_per_file = raw_likes_rows / max(n_likes_files, 1)
    avg_posts_per_file = raw_posts_rows / max(n_posts_files, 1)
    
    likes_batch_rows = int(avg_likes_per_file * actual_likes_batch)
    posts_batch_rows = int(avg_posts_per_file * actual_posts_batch)
    
    likes_batch_bytes = int(likes_batch_rows * likes_bytes_per_row_intermediate * 1.5)
    posts_batch_bytes = int(posts_batch_rows * posts_bytes_per_row_raw * 1.5)
    
    # Component 3: Intermediate likes storage during Pass 2
    # This is the BIG memory consumer - we accumulate ALL likes for sampled users
    # before filtering, and each chunk has significant overhead
    likes_intermediate_bytes = int(intermediate_likes * likes_bytes_per_row_intermediate)
    
    # Component 4: Final output data (after filtering)
    likes_output_bytes = int(estimated_likes * likes_bytes_per_row_final)
    posts_output_bytes = int(estimated_posts * posts_bytes_per_row_expanded)
    
    # Component 5: Negative reservoir (100k posts with expanded embeddings)
    negative_reservoir_bytes = int(min(negative_posts_sample, raw_posts_rows) * posts_bytes_per_row_expanded)
    
    # Phase 1: Likes loading (Pass 1 + Pass 2)
    # - Pass 1: user_set_bytes (scanning for unique users)
    # - Pass 2: accumulating all likes for sampled users in chunks
    # Peak is at end of Pass 2 when all intermediate likes are in memory
    phase_1_likes = user_set_bytes + likes_intermediate_bytes
    
    # Phase 2: Posts loading with early expansion
    # After likes processing, we have ~likes_intermediate_bytes still in memory
    # (not freed until GC), plus we're accumulating:
    # - Liked posts (small: ~estimated_liked_posts × 32KB)
    # - Negative reservoir (capped at negative_posts_sample × 32KB)
    # - One raw batch being processed
    phase_2_posts = (likes_intermediate_bytes +  # Likes still in memory
                     posts_batch_bytes +          # Current batch being processed
                     negative_reservoir_bytes)    # Negative sample (expanded)
    
    # Phase 3: Final output (likes filtered, posts combined)
    # Most intermediate likes memory should be freed at this point
    phase_3_final = likes_output_bytes + posts_output_bytes
    
    peak_bytes = max(phase_1_likes, phase_2_posts, phase_3_final)
    
    # Add Python/Polars baseline overhead (~0.7 GB typical for this pipeline)
    # Use a modest 10% buffer since our estimates are now based on observed behavior
    baseline_overhead_bytes = int(0.7 * (1024**3))
    peak_bytes = int(peak_bytes * 1.1) + baseline_overhead_bytes  # 10% buffer
    
    result = {
        # Raw data stats
        'raw_likes_rows': raw_likes_rows,
        'raw_posts_rows': raw_posts_rows,
        'raw_likes_gb': likes_raw['estimated_gb'],
        'raw_posts_gb': posts_raw['estimated_gb'],
        'n_likes_files': n_likes_files,
        'n_posts_files': n_posts_files,
        # Per-row estimates (KB)
        'likes_bytes_per_row_intermediate_kb': likes_bytes_per_row_intermediate / 1024,
        'posts_bytes_per_row_expanded_kb': posts_bytes_per_row_expanded / 1024,
        # Estimated users
        'estimated_raw_users': estimated_raw_users,
        'estimated_filtered_users': estimated_users,
        'avg_likes_per_user_in_window': avg_likes_per_user_in_window,
        'user_sampling_ratio': user_sampling_ratio,
        # Estimated data sizes
        'intermediate_likes': intermediate_likes,  # Before per-user cap/min-likes filter
        'estimated_likes_rows': estimated_likes,   # After filtering (final output)
        'estimated_liked_posts': estimated_liked_posts,
        'estimated_negative_posts': min(negative_posts_sample, raw_posts_rows),
        'estimated_total_posts': estimated_posts,
        # Memory breakdown (GB)
        'mem_user_set_gb': user_set_bytes / (1024**3),
        'mem_likes_intermediate_gb': likes_intermediate_bytes / (1024**3),
        'mem_posts_batch_gb': posts_batch_bytes / (1024**3),
        'mem_negative_reservoir_gb': negative_reservoir_bytes / (1024**3),
        'mem_likes_output_gb': likes_output_bytes / (1024**3),
        'mem_posts_output_gb': posts_output_bytes / (1024**3),
        # Phase peaks
        'phase_1_likes_gb': phase_1_likes / (1024**3),
        'phase_2_posts_gb': phase_2_posts / (1024**3),
        'phase_3_final_gb': phase_3_final / (1024**3),
        # Final estimates
        'estimated_peak_gb': peak_bytes / (1024**3),
        'estimated_total_gb': peak_bytes / (1024**3),
        # Parameters used
        'params': {
            'max_liking_users': max_liking_users,
            'max_likes_per_user': max_likes_per_user,
            'min_likes_per_user': min_likes_per_user,
            'negative_posts_sample': negative_posts_sample,
            'embedding_dim': embedding_dim,
            'batch_size': batch_size,
            'actual_posts_batch': actual_posts_batch,
        },
    }
    
    _log("Memory estimation (batch processing with early embedding expansion):")
    _log(f"  Raw data: {raw_likes_rows:,} likes ({n_likes_files} files), {raw_posts_rows:,} posts ({n_posts_files} files)")
    _log(f"  User sampling: {estimated_users:,} / {estimated_raw_users:,} users ({user_sampling_ratio*100:.1f}%)")
    _log(f"  Intermediate likes (before cap/filter): {intermediate_likes:,} ({likes_bytes_per_row_intermediate/1024:.1f}KB/row)")
    _log(f"  Negative reservoir: {min(negative_posts_sample, raw_posts_rows):,} posts ({posts_bytes_per_row_expanded/1024:.1f}KB/post)")
    _log(f"  Memory phases: likes={result['phase_1_likes_gb']:.2f}GB, posts={result['phase_2_posts_gb']:.2f}GB, final={result['phase_3_final_gb']:.2f}GB")
    _log(f"  Estimated peak: {result['estimated_peak_gb']:.2f} GB")
    
    return result


def check_memory_available(
    estimated_bytes: int,
    *,
    max_memory_gb: float = 0,  # 0 = use percentage of available
    max_memory_pct: float = 0.75,  # Use at most 75% of available RAM
    logger: Optional[Any] = None,
) -> Tuple[bool, str]:
    """
    Check if estimated memory usage is safe given available system memory.
    
    Args:
        estimated_bytes: Estimated memory required
        max_memory_gb: Maximum memory in GB (0 = auto based on percentage)
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
    if max_memory_gb > 0:
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
    max_memory_gb: float = 0,
    max_memory_pct: float = 0.75,
    max_liking_users: int = 0,
    max_likes_per_user: int = 100,
    min_likes_per_user: int = 2,
    negative_posts_sample: int = 100_000,
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
        max_memory_gb: Maximum memory in GB (0 = auto based on percentage)
        max_memory_pct: Maximum percentage of available RAM
        max_liking_users: Cap on number of liking users
        max_likes_per_user: Max likes to keep per user
        min_likes_per_user: Min likes required per user
        negative_posts_sample: Number of negative posts to sample
        logger: Optional logger
    
    Returns:
        Dict with memory estimation details
    
    Raises:
        MemoryError: If estimated memory exceeds limits
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
        raise MemoryError(
            f"Data load would exceed memory limits. {msg}\n"
            f"Estimated peak memory: {estimation['estimated_peak_gb']:.2f} GB\n"
            f"Options to reduce memory:\n"
            f"  - Use a shorter time window (--posts-start/--posts-end)\n"
            f"  - Reduce --max-liking-users (current: {max_liking_users})\n"
            f"  - Reduce --max-likes-per-user (current: {max_likes_per_user})\n"
            f"  - Reduce --negative-posts-sample (current: {negative_posts_sample})\n"
            f"  - Increase --max-memory-gb if you have more RAM available"
        )
    
    return estimation


# ----------------------------------------
# Stage 1: Polars-based filtering for core datasets
# ----------------------------------------
def load_likes_core_polars(
    gcs_bucket: str,
    start_str: Optional[str],
    end_str: Optional[str],
    *,
    max_liking_users: int = 0,
    max_likes_per_user: int = 100,
    min_likes_per_user: int = 2,
    random_seed: int = 42,
    logger: Optional[Any] = None,
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """
    Load and filter likes data using memory-efficient incremental processing.
    
    Instead of loading all files at once, this processes files incrementally:
    1. First pass: scan files to collect unique user DIDs (minimal memory)
    2. Sample users if cap is set
    3. Second pass: scan files again, only keeping likes from sampled users
    4. Apply per-user caps and min-likes filters
    
    This approach keeps memory usage bounded regardless of total data size.
    
    Returns:
        Tuple of (likes_df: pl.DataFrame, stats: Dict with filtering statistics)
    """
    def _log(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(f"  {msg}")
    
    start_dt = parse_one_ts(start_str)
    end_dt = parse_one_ts(end_str)
    
    # Get file paths for likes
    paths = list_files_in_range_ingex_gcs(
        gcs_bucket=gcs_bucket,
        blob_prefix='bsky_likes',
        start=start_dt,
        end=end_dt,
    )
    
    if not paths:
        raise ValueError(f"No likes parquet files found for time range {start_str} to {end_str}")
    
    _log(f"Found {len(paths)} likes parquet files")
    log_memory_checkpoint("likes_before_scan", logger)
    
    # Batch size for processing multiple files at once
    BATCH_SIZE = 20
    
    # Helper to normalize column names and apply time filter for batch of files
    def _prepare_batch_lf(batch_paths: List[str]) -> pl.LazyFrame:
        lf = pl.scan_parquet(batch_paths)
        schema = lf.collect_schema()
        
        # Normalize column names
        col_mapping = {}
        for col in schema.names():
            if col.lower() == 'did':
                col_mapping[col] = 'did'
            elif col.lower() == 'subjecturi':
                col_mapping[col] = 'subject_uri'
            elif col.lower() == 'recordcreatedat':
                col_mapping[col] = 'record_created_at'
            elif col.lower() == 'insertedat':
                col_mapping[col] = 'inserted_at'
        
        if col_mapping:
            lf = lf.rename(col_mapping)
        
        # Apply time filter
        if 'inserted_at' in lf.collect_schema().names():
            lf = lf.with_columns(
                pl.col("inserted_at").str.to_datetime(time_zone="UTC").alias("inserted_at_dt")
            )
            if start_dt is not None:
                lf = lf.filter(pl.col("inserted_at_dt") >= start_dt)
            if end_dt is not None:
                lf = lf.filter(pl.col("inserted_at_dt") < end_dt)
        
        return lf
    
    # ===== PASS 1: Collect unique users and counts (batch processing) =====
    _log(f"Pass 1: Scanning files for unique users (batch size: {BATCH_SIZE})...")
    all_users: Set[str] = set()
    n_likes_initial = 0
    
    for batch_start in range(0, len(paths), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(paths))
        batch_paths = paths[batch_start:batch_end]
        
        lf = _prepare_batch_lf(batch_paths)
        # Only collect the 'did' column - minimal memory
        batch_users = lf.select('did').collect()
        all_users.update(batch_users['did'].to_list())
        n_likes_initial += len(batch_users)
        
        _log(f"  Scanned {batch_end}/{len(paths)} files: {n_likes_initial:,} likes, {len(all_users):,} unique users")
        log_memory_checkpoint(f"likes_pass1_batch_{batch_end}", logger)
    
    n_users_initial = len(all_users)
    _log(f"Pass 1 complete: {n_likes_initial:,} likes from {n_users_initial:,} users")
    log_memory_checkpoint("likes_after_pass1", logger)
    
    stats = {
        'n_likes_initial': n_likes_initial,
        'n_users_initial': n_users_initial,
    }
    
    # ===== Sample users if cap is set =====
    rng = np.random.RandomState(random_seed)
    
    if max_liking_users > 0 and n_users_initial > max_liking_users:
        user_list = list(all_users)
        sampled_indices = rng.choice(len(user_list), size=max_liking_users, replace=False)
        sampled_user_set = {user_list[i] for i in sampled_indices}
        _log(f"Sampled {max_liking_users:,} liking users ({100*max_liking_users/n_users_initial:.1f}% of total)")
        stats['n_users_sampled'] = max_liking_users
    else:
        sampled_user_set = all_users
        stats['n_users_sampled'] = n_users_initial
    
    # Free the full user set
    del all_users
    log_memory_checkpoint("likes_after_user_sample", logger)
    
    # ===== PASS 2: Collect likes only for sampled users (batch processing) =====
    _log(f"Pass 2: Collecting likes for sampled users (batch size: {BATCH_SIZE})...")
    likes_chunks: List[pl.DataFrame] = []
    n_likes_collected = 0
    
    for batch_start in range(0, len(paths), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(paths))
        batch_paths = paths[batch_start:batch_end]
        
        lf = _prepare_batch_lf(batch_paths)
        # Filter to sampled users before collecting
        batch_df = lf.filter(pl.col('did').is_in(sampled_user_set)).collect()
        
        if len(batch_df) > 0:
            likes_chunks.append(batch_df)
            n_likes_collected += len(batch_df)
        
        _log(f"  Processed {batch_end}/{len(paths)} files: {n_likes_collected:,} likes collected")
        log_memory_checkpoint(f"likes_pass2_batch_{batch_end}", logger)
    
    # Combine chunks
    if likes_chunks:
        likes_df = pl.concat(likes_chunks)
    else:
        # Empty result
        likes_df = pl.DataFrame({'did': [], 'subject_uri': [], 'record_created_at': []})
    
    del likes_chunks
    
    n_after_user_sample = len(likes_df)
    pct_retained = 100.0 * n_after_user_sample / n_likes_initial if n_likes_initial > 0 else 0
    _log(f"Pass 2 complete: {n_after_user_sample:,} likes ({pct_retained:.1f}% retained)")
    stats['n_likes_after_user_sample'] = n_after_user_sample
    log_memory_checkpoint("likes_after_pass2", logger)
    
    # ===== Apply per-user random cap (NOT recency-based) =====
    if max_likes_per_user > 0 and len(likes_df) > 0:
        n_before_cap = len(likes_df)
        
        # Add random ordering within each user's likes
        likes_df = likes_df.with_columns(
            pl.lit(1).cum_count().over('did').shuffle(seed=random_seed).alias('_rand_order')
        )
        likes_df = likes_df.filter(pl.col('_rand_order') <= max_likes_per_user)
        likes_df = likes_df.drop('_rand_order')
        
        n_after_cap = len(likes_df)
        pct_retained = 100.0 * n_after_cap / n_before_cap if n_before_cap > 0 else 0
        _log(f"After per-user cap ({max_likes_per_user}): {n_after_cap:,} likes ({pct_retained:.1f}% retained)")
        stats['n_likes_after_per_user_cap'] = n_after_cap
    else:
        stats['n_likes_after_per_user_cap'] = len(likes_df)
    
    # ===== Filter users with fewer than min_likes_per_user =====
    if min_likes_per_user > 0 and len(likes_df) > 0:
        n_before_min = len(likes_df)
        user_counts = likes_df.group_by('did').agg(pl.len().alias('count'))
        eligible_users = user_counts.filter(pl.col('count') >= min_likes_per_user)['did']
        likes_df = likes_df.filter(pl.col('did').is_in(eligible_users))
        
        n_after_min = len(likes_df)
        n_users_final = likes_df['did'].n_unique() if len(likes_df) > 0 else 0
        pct_retained = 100.0 * n_after_min / n_before_min if n_before_min > 0 else 0
        _log(f"After min-likes filter ({min_likes_per_user}): {n_after_min:,} likes ({pct_retained:.1f}% retained)")
        _log(f"Final: {n_users_final:,} users with {n_after_min:,} likes")
        stats['n_likes_final'] = n_after_min
        stats['n_users_final'] = n_users_final
    else:
        stats['n_likes_final'] = len(likes_df)
        stats['n_users_final'] = likes_df['did'].n_unique() if len(likes_df) > 0 else 0
    
    # Select output columns
    output_cols = ['did', 'subject_uri', 'record_created_at']
    available_cols = [c for c in output_cols if c in likes_df.columns]
    if available_cols:
        likes_df = likes_df.select(available_cols)
    
    log_memory_checkpoint("likes_final", logger)
    
    return likes_df, stats


def load_posts_core_polars(
    gcs_bucket: str,
    start_str: Optional[str],
    end_str: Optional[str],
    liked_post_uris: Set[str],
    *,
    negative_posts_sample: int = 100000,
    embedding_model: str = 'all_MiniLM_L6_v2',
    random_seed: int = 42,
    logger: Optional[Any] = None,
) -> Tuple[pl.DataFrame, Dict[str, Any], int]:
    """
    Load posts data using batch processing with early embedding expansion.
    
    Key optimization: expand embeddings per-batch BEFORE accumulating.
    This reduces memory by ~98% (150KB/post raw -> 2KB/post expanded).
    
    Processing flow:
    1. Process files in batches (20 files at a time)
    2. For each batch: filter to liked posts + sample negatives
    3. Expand embeddings and DROP the raw blob
    4. Accumulate only the slim expanded data
    
    Returns:
        Tuple of (posts_df: pl.DataFrame, stats: Dict, embedding_dim: int)
    """
    def _log(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(f"  {msg}")
    
    start_dt = parse_one_ts(start_str)
    end_dt = parse_one_ts(end_str)
    
    # Get file paths for posts
    paths = list_files_in_range_ingex_gcs(
        gcs_bucket=gcs_bucket,
        blob_prefix='bsky_posts',
        start=start_dt,
        end=end_dt,
    )
    
    if not paths:
        raise ValueError(f"No posts parquet files found for time range {start_str} to {end_str}")
    
    _log(f"Found {len(paths)} posts parquet files")
    log_memory_checkpoint("posts_before_scan", logger)
    
    # Batch size for processing
    BATCH_SIZE = 20
    
    # Helper to prepare batch with time filter
    def _prepare_batch_lf(batch_paths: List[str]) -> pl.LazyFrame:
        lf = pl.scan_parquet(batch_paths)
        
        if 'inserted_at' in lf.collect_schema().names():
            lf = lf.with_columns(
                pl.col("inserted_at").str.to_datetime(time_zone="UTC").alias("inserted_at_dt")
            )
            if start_dt is not None:
                lf = lf.filter(pl.col("inserted_at_dt") >= start_dt)
            if end_dt is not None:
                lf = lf.filter(pl.col("inserted_at_dt") < end_dt)
        
        return lf
    
    # Initialize random state
    rng = np.random.RandomState(random_seed)
    
    # Accumulators - these store EXPANDED data (small memory footprint)
    liked_posts_expanded: List[pl.DataFrame] = []
    negative_reservoir_expanded: List[pl.DataFrame] = []
    n_non_liked_seen = 0
    n_posts_total = 0
    n_liked_posts = 0
    embed_dim = 0  # Will be set on first successful expansion
    
    _log(f"Processing {len(paths)} files in batches of {BATCH_SIZE} with early embedding expansion...")
    
    for batch_start in range(0, len(paths), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(paths))
        batch_paths = paths[batch_start:batch_end]
        
        lf = _prepare_batch_lf(batch_paths)
        batch_df = lf.collect()
        n_batch = len(batch_df)
        n_posts_total += n_batch
        
        if n_batch == 0:
            continue
        
        # Check for at_uri column
        if 'at_uri' not in batch_df.columns:
            _log(f"  Warning: Batch {batch_end} missing 'at_uri' column, skipping")
            continue
        
        # === Extract liked posts from this batch ===
        batch_liked = batch_df.filter(pl.col('at_uri').is_in(liked_post_uris))
        
        if len(batch_liked) > 0:
            # EARLY EXPANSION: Expand embeddings and drop raw blob
            batch_liked_expanded, batch_embed_dim = _expand_embeddings_chunk(
                batch_liked, embedding_model, logger=None  # Quiet for per-batch
            )
            if batch_embed_dim > 0:
                embed_dim = batch_embed_dim
            batch_liked_expanded = batch_liked_expanded.with_columns(pl.lit(True).alias('is_liked'))
            liked_posts_expanded.append(batch_liked_expanded)
            n_liked_posts += len(batch_liked_expanded)
        
        # === Reservoir sampling for negative posts ===
        if negative_posts_sample > 0:
            batch_non_liked = batch_df.filter(~pl.col('at_uri').is_in(liked_post_uris))
            n_batch_non_liked = len(batch_non_liked)
            
            if n_batch_non_liked > 0:
                current_reservoir_size = sum(len(r) for r in negative_reservoir_expanded)
                
                if current_reservoir_size < negative_posts_sample:
                    # Reservoir not full - add what we need
                    space_remaining = negative_posts_sample - current_reservoir_size
                    if n_batch_non_liked <= space_remaining:
                        sample_to_add = batch_non_liked
                    else:
                        sample_to_add = batch_non_liked.sample(n=space_remaining, seed=rng.randint(0, 2**31))
                    
                    # EARLY EXPANSION for negatives too
                    sample_expanded, _ = _expand_embeddings_chunk(
                        sample_to_add, embedding_model, logger=None
                    )
                    sample_expanded = sample_expanded.with_columns(pl.lit(False).alias('is_liked'))
                    negative_reservoir_expanded.append(sample_expanded)
                else:
                    # Reservoir full - probabilistic replacement (simplified for batches)
                    # For batch processing, we use a simpler approximation:
                    # Replace a fraction of the reservoir proportional to batch size
                    n_non_liked_seen += n_batch_non_liked
                    replace_prob = min(1.0, n_batch_non_liked / n_non_liked_seen)
                    
                    if rng.random() < replace_prob * 0.3:  # ~30% chance per qualifying batch
                        # Replace one chunk with a sample from this batch
                        if negative_reservoir_expanded:
                            replace_idx = rng.randint(0, len(negative_reservoir_expanded))
                            sample_size = min(n_batch_non_liked, len(negative_reservoir_expanded[replace_idx]))
                            replacement = batch_non_liked.sample(n=sample_size, seed=rng.randint(0, 2**31))
                            replacement_expanded, _ = _expand_embeddings_chunk(
                                replacement, embedding_model, logger=None
                            )
                            replacement_expanded = replacement_expanded.with_columns(pl.lit(False).alias('is_liked'))
                            negative_reservoir_expanded[replace_idx] = replacement_expanded
        
        # Free batch data
        del batch_df
        
        n_neg_current = sum(len(r) for r in negative_reservoir_expanded)
        _log(f"  Processed {batch_end}/{len(paths)} files: {n_posts_total:,} posts, {n_liked_posts:,} liked (expanded), {n_neg_current:,} negative reservoir")
        log_memory_checkpoint(f"posts_batch_{batch_end}", logger)
    
    _log(f"Extracted {len(liked_post_uris):,} unique liked post IDs")
    
    stats = {
        'n_posts_total': n_posts_total,
        'n_liked_posts': n_liked_posts,
        'liked_post_match_rate': 100.0 * n_liked_posts / len(liked_post_uris) if liked_post_uris else 0,
    }
    
    _log(f"Loaded {n_liked_posts:,} liked posts ({stats['liked_post_match_rate']:.1f}% match rate)")
    
    # Combine liked posts (already expanded)
    if liked_posts_expanded:
        liked_posts_df = pl.concat(liked_posts_expanded)
    else:
        liked_posts_df = None
    
    del liked_posts_expanded
    
    # Combine negative reservoir (already expanded)
    if negative_reservoir_expanded:
        neg_sample_df = pl.concat(negative_reservoir_expanded)
        # Trim to exact sample size if we collected more
        if len(neg_sample_df) > negative_posts_sample:
            neg_sample_df = neg_sample_df.sample(n=negative_posts_sample, seed=random_seed)
        stats['n_negative_sample'] = len(neg_sample_df)
        _log(f"Negative sample: {len(neg_sample_df):,} posts from reservoir")
    else:
        neg_sample_df = None
        stats['n_negative_sample'] = 0
    
    del negative_reservoir_expanded
    
    # Combine liked and negative sample posts
    if liked_posts_df is not None and neg_sample_df is not None:
        posts_combined = pl.concat([liked_posts_df, neg_sample_df])
    elif liked_posts_df is not None:
        posts_combined = liked_posts_df
    elif neg_sample_df is not None:
        posts_combined = neg_sample_df
    else:
        posts_combined = pl.DataFrame()
    
    n_combined = len(posts_combined)
    _log(f"posts_core: {n_combined:,} rows ({n_liked_posts:,} liked + {stats['n_negative_sample']:,} negative)")
    _log(f"Embeddings already expanded during loading (dim={embed_dim})")
    stats['n_posts_core'] = n_combined
    stats['embedding_dim'] = embed_dim
    log_memory_checkpoint("posts_after_combine", logger)
    
    return posts_combined, stats, embed_dim


def _expand_embeddings_chunk(
    posts_df: pl.DataFrame,
    embedding_model: str,
    logger: Optional[Any] = None,
) -> Tuple[pl.DataFrame, int]:
    """
    Expand embeddings for a chunk of posts and drop the raw blob.
    
    This is an internal helper for early embedding expansion during batch loading.
    Returns (expanded_df without embeddings column, embedding_dim).
    """
    if 'embeddings' not in posts_df.columns or len(posts_df) == 0:
        return posts_df, 0
    
    # Convert to pandas for embedding extraction
    pdf = posts_df.to_pandas()
    
    # Find embedding dimension from first valid example
    embed_dim = None
    for emb_list in pdf['embeddings']:
        if emb_list is None:
            continue
        emb_str = extract_encoded_embedding_ingex(emb_list, embedding_model)
        if emb_str is not None:
            try:
                sample_emb = embedding_loads(emb_str, decompress=True)
                embed_dim = len(sample_emb)
                break
            except Exception:
                continue
    
    if embed_dim is None:
        # No valid embeddings - just drop the column
        if 'embeddings' in posts_df.columns:
            posts_df = posts_df.drop('embeddings')
        return posts_df, 0
    
    # Extract embeddings for all rows
    n_rows = len(pdf)
    emb_array = np.zeros((n_rows, embed_dim), dtype=np.float32)
    
    for i, emb_list in enumerate(pdf['embeddings']):
        if emb_list is None:
            continue
        emb_str = extract_encoded_embedding_ingex(emb_list, embedding_model)
        if emb_str is not None:
            try:
                emb_array[i] = embedding_loads(emb_str, decompress=True)
            except Exception:
                continue
    
    # Create embedding column names
    emb_col_names = [f'post_emb_{i}' for i in range(embed_dim)]
    
    # Add embedding columns to dataframe
    emb_df = pd.DataFrame(emb_array, columns=emb_col_names)
    pdf = pd.concat([pdf.reset_index(drop=True), emb_df], axis=1)
    
    # Drop the original embeddings column (the large raw blob)
    pdf = pdf.drop(columns=['embeddings'])
    
    # Convert back to polars
    result_df = pl.from_pandas(pdf)
    
    return result_df, embed_dim


def expand_embeddings_polars(
    posts_df: pl.DataFrame,
    embedding_model: str = 'all_MiniLM_L6_v2',
    logger: Optional[Any] = None,
) -> Tuple[pl.DataFrame, int]:
    """
    Expand the 'embeddings' column into separate post_emb_* columns.
    
    The embeddings column contains a list of dicts like:
    [{'key': 'all_MiniLM_L6_v2', 'value': '<base85-encoded>'}, ...]
    
    Returns:
        Tuple of (posts_df with expanded embeddings, embedding_dim)
    """
    def _log(msg: str):
        if logger:
            logger.info(msg)
        else:
            print(f"  {msg}")
    
    if 'embeddings' not in posts_df.columns:
        _log("No 'embeddings' column found, skipping expansion")
        return posts_df, 0
    
    # Convert to pandas for embedding extraction (complex nested structure)
    _log("Expanding embeddings column...")
    log_memory_checkpoint("embeddings_before_expand", logger)
    pdf = posts_df.to_pandas()
    
    # Find embedding dimension from first valid example
    embed_dim = None
    for emb_list in pdf['embeddings']:
        if emb_list is None:
            continue
        emb_str = extract_encoded_embedding_ingex(emb_list, embedding_model)
        if emb_str is not None:
            try:
                sample_emb = embedding_loads(emb_str, decompress=True)
                embed_dim = len(sample_emb)
                break
            except Exception:
                continue
    
    if embed_dim is None:
        _log(f"No valid embeddings found for model {embedding_model}")
        return posts_df, 0
    
    _log(f"Embedding dimension: {embed_dim}")
    
    # Extract embeddings for all rows
    n_rows = len(pdf)
    emb_array = np.zeros((n_rows, embed_dim), dtype=np.float32)
    n_valid = 0
    
    for i, emb_list in enumerate(pdf['embeddings']):
        if emb_list is None:
            continue
        emb_str = extract_encoded_embedding_ingex(emb_list, embedding_model)
        if emb_str is not None:
            try:
                emb_array[i] = embedding_loads(emb_str, decompress=True)
                n_valid += 1
            except Exception:
                continue
    
    _log(f"Expanded {n_valid:,}/{n_rows:,} embeddings ({100*n_valid/n_rows:.1f}%)")
    
    # Create embedding column names
    emb_col_names = [f'post_emb_{i}' for i in range(embed_dim)]
    
    # Add embedding columns to dataframe
    emb_df = pd.DataFrame(emb_array, columns=emb_col_names)
    pdf = pd.concat([pdf.reset_index(drop=True), emb_df], axis=1)
    
    # Drop the original embeddings column (too large for parquet)
    pdf = pdf.drop(columns=['embeddings'])
    
    # Convert back to polars
    result_df = pl.from_pandas(pdf)
    log_memory_checkpoint("embeddings_after_expand", logger)
    
    return result_df, embed_dim


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
                if issubclass(expected, (int, np.integer)):
                    return dtype in polars_integer
                if issubclass(expected, (float, np.floating)):
                    return dtype in polars_float
                if issubclass(expected, (bool, np.bool_)):
                    return dtype == pl.Boolean
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
    'check_memory_available', 'check_data_load_safe',
    # Stage 1: Polars-based filtering for core datasets
    'load_likes_core_polars', 'load_posts_core_polars', 'expand_embeddings_polars',
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
    'validate_dataframe_schema', 'validate_data_integrity',
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
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Select candidate posts by union of liked posts and per-author caps.

    - Caps likes per user to max_liked_posts_per_user (random sampling across time).
    - Always include posts that appear in the (capped) likes_df[join_like].
    - Augment with up to `max_posts_per_author` posts per author (random selection).
    
    Returns:
        Tuple of (candidate_posts_df, capped_likes_df)
    """
    import time
    t0 = time.time()
    
    rng = np.random.RandomState(int(rng_seed))
    
    # Cap likes per user with RANDOM sampling across time.
    # NOTE: In deployment/inference, we would want recency-based capping (most recent N likes)
    # to reflect current user preferences. However, during training we use random sampling
    # across a consistent time period so the model can learn recency effects from the data
    # rather than having them baked in by the sampling strategy.
    likes_df_capped = likes_df.copy()
    n_likes_before = len(likes_df_capped)
    
    if max_liked_posts_per_user > 0 and 'did' in likes_df_capped.columns:
        def _cap_user_likes(g: pd.DataFrame) -> pd.DataFrame:
            if len(g) <= max_liked_posts_per_user:
                return g
            return g.sample(n=max_liked_posts_per_user, random_state=rng)
        
        likes_df_capped = likes_df_capped.groupby('did', group_keys=False).apply(_cap_user_likes)
        n_likes_after = len(likes_df_capped)
        print(f"  Capped likes per user: {n_likes_before} -> {n_likes_after} (max {max_liked_posts_per_user}/user)")
    else:
        n_likes_after = n_likes_before
    
    join_like_str = likes_df_capped[join_like].astype(str)
    liked_post_ids = set(join_like_str.dropna().unique().tolist())
    print(f"  Found {len(liked_post_ids)} unique liked posts (after per-user cap)")

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
    return candidates, likes_df_capped


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
