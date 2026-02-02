#!/usr/bin/env python3

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timezone
import polars as pl
import numpy as np
import logging

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