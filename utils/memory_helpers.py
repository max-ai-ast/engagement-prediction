#!/usr/bin/env python3
"""
Memory Helpers Module
=====================

This module provides utilities for:
1. Memory tracking and monitoring during pipeline execution
2. Memory estimation using a fitted regression model
3. Running memory profiling sweeps, exporting results, and fitting models

The memory model artifacts (configs, results, weights) are stored in:
    utils/memory_helper_artifacts/

To update the memory model:
    1. Generate a sweep config: generate_sweep_config()
    2. Run the sweep: run_memory_sweep()
    3. Export results: export_sweep_results_from_clearml()
    4. Fit new model: fit_memory_model()
    5. Save weights: save_model_weights()
"""

from __future__ import annotations

import csv
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import logging

import numpy as np
import polars as pl

try:
    import psutil  # type: ignore
except ImportError:
    psutil = None  # type: ignore

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None  # type: ignore

# ============================================================================
# PATHS AND CONSTANTS
# ============================================================================

ARTIFACTS_DIR = Path(__file__).parent / "memory_helper_artifacts"
DEFAULT_MODEL_WEIGHTS_FILE = ARTIFACTS_DIR / "model_weights_latest.json"

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
    
    def checkpoint(self, name: str, *, quiet: bool = False) -> Dict[str, Any]:
        """Record a memory checkpoint.
        
        Args:
            name: Descriptive name for this checkpoint
            quiet: If True, store data but skip logging the inline message.
                   Useful when the summary will show all checkpoints anyway.
        """
        elapsed = time.time() - self.start_time
        if quiet:
            stats = get_current_memory_usage()
        else:
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
        Return a summary of all memory checkpoints.
        
        Note: This method only returns data; it does not log anything.
        The caller (e.g., _log_data_attrition_report) is responsible for
        formatting and logging the summary as needed.
        """
        if not self.checkpoints:
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
    
    # Batch scan all files at once - Polars handles multiple files natively
    # This is much faster than iterating per-file (671 files: 147s -> ~2s)
    try:
        lf = pl.scan_parquet(paths)
        schema = lf.collect_schema()
        total_rows = lf.select(pl.len()).collect().item()
    except Exception:
        return {'estimated_bytes': 0, 'estimated_gb': 0.0, 'total_rows': 0}
    
    if total_rows == 0:
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
# Model weights are loaded from memory_helper_artifacts/model_weights_latest.json
#
# To update the memory model:
#   1. Run a sweep: config = generate_sweep_config(...); run_memory_sweep(config)
#   2. Export results: export_sweep_results_from_clearml(tag="...")
#   3. Fit model: weights = fit_memory_model("sweep_results_xxx.csv")
#   4. Save: save_model_weights(weights, "model_weights_xxx.json")
#
# The constants below are DEPRECATED and kept only as fallback.
# ============================================================================

# DEPRECATED: These are kept as fallback; prefer loading from JSON
MEMORY_MODEL_COEFFICIENTS = {
    'intercept': -74.5927239155,
    'data_window_days': -0.0708091837,
    'max_liking_users_10k': 12.0706132179,
    'max_likes_per_user_100': 1.5620756976,
    'negative_posts_sample_10k': 0.2271053690,
    'log_max_liking_users': 15.4635658117,
    'sqrt_likes_initial_1e6': 2.8735780908,
    'days_x_users_10k': -0.0101803924,
    'users_x_log_users': -1.9725666267,
}

# DEPRECATED: Feature names are stored in the JSON weights file
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


def load_model_weights(
    model_file: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load memory model weights from a JSON file.
    
    Args:
        model_file: Path to the JSON weights file. Defaults to 
            memory_helper_artifacts/model_weights_latest.json
    
    Returns:
        Dict containing:
        - 'coefficients': Dict mapping feature names to coefficients
        - 'feature_names': List of feature names in order
        - 'version': Version string (e.g., "260202")
        - 'r_squared': Model R-squared value
        - 'fitted_at': ISO timestamp when model was fitted
    
    Raises:
        FileNotFoundError: If the weights file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    if model_file is None:
        model_file = DEFAULT_MODEL_WEIGHTS_FILE
    
    model_file = Path(model_file)
    if not model_file.is_absolute():
        model_file = ARTIFACTS_DIR / model_file
    
    with open(model_file, 'r') as f:
        data = json.load(f)
    
    return data


def save_model_weights(
    weights: Dict[str, Any],
    output_file: Optional[Union[str, Path]] = None,
    *,
    set_as_latest: bool = True,
) -> Path:
    """Save memory model weights to a JSON file.
    
    Args:
        weights: Dict containing 'coefficients', 'feature_names', 'version', 
            'r_squared', 'fitted_at', and optionally other metadata
        output_file: Filename or path for the output JSON. If just a filename,
            it will be saved in memory_helper_artifacts/. If None, uses
            model_weights_{version}.json
        set_as_latest: If True, also copy to model_weights_latest.json
    
    Returns:
        Path to the saved weights file
    
    Example:
        weights = fit_memory_model(df)
        save_model_weights(weights, "model_weights_260205.json")
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    if output_file is None:
        version = weights.get('version', datetime.now().strftime('%y%m%d'))
        output_file = f"model_weights_{version}.json"
    
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = ARTIFACTS_DIR / output_path
    
    with open(output_path, 'w') as f:
        json.dump(weights, f, indent=2)
    
    if set_as_latest:
        latest_path = ARTIFACTS_DIR / "model_weights_latest.json"
        with open(latest_path, 'w') as f:
            json.dump(weights, f, indent=2)
    
    return output_path


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
    model_file: Optional[Path] = None,
) -> float:
    """Predict peak memory usage in GB using the fitted model.
    
    Args:
        features: Feature dict from compute_memory_model_features()
        coefficients: Model coefficients dict with 'intercept' and feature names as keys.
            If None, loads from model_file or the default latest weights file.
        model_file: Path to model weights JSON file. Used only if coefficients is None.
            Defaults to memory_helper_artifacts/model_weights_latest.json
    
    Returns:
        Estimated peak memory in GB
    
    Example:
        features = compute_memory_model_features(
            data_window_days=14,
            max_liking_users=50000,
            max_likes_per_user=100,
            negative_posts_sample=50000,
            likes_initial=10_000_000,
        )
        estimated_gb = predict_memory_gb(features)
    """
    if coefficients is None:
        # Load from file
        weights = load_model_weights(model_file or DEFAULT_MODEL_WEIGHTS_FILE)
        coefficients = weights['coefficients']
        feature_names = weights.get('feature_names', MEMORY_MODEL_FEATURE_NAMES)
    else:
        # Use provided coefficients; determine feature names from keys
        feature_names = [k for k in coefficients.keys() if k != 'intercept']
    
    result = coefficients['intercept']
    for name in feature_names:
        if name in features and name in coefficients:
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
    # Use batch scanning for both - much faster than per-file iteration
    likes_raw = estimate_parquet_memory(likes_paths, embedding_expansion_dim=0)
    posts_raw = estimate_parquet_memory(posts_paths, embedding_expansion_dim=0)
    # Also compute posts with embedding expansion for "raw unfiltered" display
    posts_raw_with_embeddings = estimate_parquet_memory(posts_paths, embedding_expansion_dim=embedding_dim)
    
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
        
        # Raw data stats
        'raw_likes_rows': raw_likes_rows,
        'raw_posts_rows': raw_posts_rows,
        'raw_likes_gb': likes_raw['estimated_gb'],
        'raw_posts_gb': posts_raw['estimated_gb'],
        'raw_posts_gb_with_embeddings': posts_raw_with_embeddings['estimated_gb'],
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
    
    # Use raw estimates from estimate_filtered_data_memory (no redundant calls)
    raw_total_gb = estimation['raw_likes_gb'] + estimation['raw_posts_gb_with_embeddings']
    
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


# ============================================================================
# SWEEP CONFIGURATION
# ============================================================================

def generate_sweep_config(
    name: str,
    tag: str,
    *,
    days_options: List[int] = [7, 14, 21],
    users_options: List[int] = [10000, 50000, 100000],
    likes_per_user_options: List[int] = [100, 500],
    negative_sample_options: List[int] = [10000, 50000, 100000],
    posts_start: str = "2026-01-01",
    gcs_bucket: str = "greenearth-471522-ingex-extract-stage",
    min_likes_per_user: int = 2,
    embedding_model: str = "all_MiniLM_L6_v2",
    project: str = "Engagement Prediction",
) -> Dict[str, Any]:
    """Generate a sweep configuration programmatically.
    
    Creates a configuration dict with experiments ordered by expected load
    (fastest to slowest) based on: days × users × likes_per_user.
    
    Args:
        name: Sweep name (e.g., "data_sweep_260205")
        tag: ClearML tag for filtering (e.g., "data-sweep-26-02-05")
        days_options: List of data window days to test
        users_options: List of max_liking_users values to test
        likes_per_user_options: List of max_likes_per_user values to test
        negative_sample_options: List of negative_posts_sample values to test
        posts_start: Start date for data window (ISO format)
        gcs_bucket: GCS bucket name for data
        min_likes_per_user: Minimum likes per user filter
        embedding_model: SentenceTransformer model name
        project: ClearML project name
    
    Returns:
        Dict containing sweep config ready for run_memory_sweep() or save_sweep_config()
    
    Example:
        config = generate_sweep_config(
            name="data_sweep_260205",
            tag="data-sweep-26-02-05",
            days_options=[7, 14],
            users_options=[10000, 50000],
        )
        save_sweep_config(config, "sweep_config_260205.yml")
    """
    # Generate all experiment combinations
    experiments_raw = []
    for days in days_options:
        for users in users_options:
            for likes in likes_per_user_options:
                for neg in negative_sample_options:
                    load_factor = days * users * likes
                    experiments_raw.append({
                        'days': days,
                        'users': users,
                        'likes': likes,
                        'neg': neg,
                        'load': load_factor,
                    })
    
    # Sort by load factor (fastest first)
    experiments_raw.sort(key=lambda x: (x['load'], x['days'], x['users'], x['likes'], x['neg']))
    
    # Build experiment list with names
    experiments = []
    for idx, exp in enumerate(experiments_raw, 1):
        exp_name = f"{idx:02d}_{exp['days']}d_{exp['users']//1000}ku_{exp['likes']}l_{exp['neg']//1000}kn"
        experiments.append({
            'name': exp_name,
            'params': {
                'data_window_days': exp['days'],
                'max_liking_users': exp['users'],
                'max_likes_per_user': exp['likes'],
                'negative_posts_sample': exp['neg'],
            }
        })
    
    config = {
        'sweep': {
            'name': name,
            'description': f"Memory profiling sweep generated on {datetime.now().isoformat()}",
            'project': project,
            'tags': [tag],
        },
        'fixed': {
            'gcs_bucket': gcs_bucket,
            'posts_start': posts_start,
            'likes_start': posts_start,
            'min_likes_per_user': min_likes_per_user,
            'cap_random_seed': 42,
            'embedding_model': embedding_model,
            'memory_check': 'ignore',
            'stop_after': 'get_data',
        },
        'experiments': experiments,
        'execution': {
            'mode': 'sequential',
            'delay_between_runs': 10,
            'continue_on_failure': True,
            'output_base': 'outputs/sweeps',
        },
    }
    
    return config


def save_sweep_config(config: Dict[str, Any], output_file: Union[str, Path]) -> Path:
    """Save a sweep configuration to a YAML file.
    
    Args:
        config: Sweep configuration dict from generate_sweep_config()
        output_file: Filename or path. If just a filename, saved in artifacts dir.
    
    Returns:
        Path to the saved config file
    """
    if yaml is None:
        raise ImportError("PyYAML is required to save sweep configs. Install with: pip install pyyaml")
    
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    output_path = Path(output_file)
    if not output_path.is_absolute():
        output_path = ARTIFACTS_DIR / output_path
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    
    return output_path


def load_sweep_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a sweep configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML config file. If not absolute and file doesn't
            exist, will look in artifacts directory.
    
    Returns:
        Sweep configuration dict
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is missing required sections
    """
    if yaml is None:
        raise ImportError("PyYAML is required to load sweep configs. Install with: pip install pyyaml")
    
    path = Path(config_path)
    if not path.exists() and not path.is_absolute():
        path = ARTIFACTS_DIR / path
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path) as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    if 'sweep' not in config:
        raise ValueError("Config missing required section: sweep")
    if 'fixed' not in config:
        raise ValueError("Config missing required section: fixed")
    if 'sweep_params' not in config and 'experiments' not in config:
        raise ValueError("Config must have either 'sweep_params' (grid search) or 'experiments' (explicit list)")
    
    return config


def _generate_experiment_grid(config: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Generate list of experiments from a sweep config.
    
    Supports two modes:
    1. Grid search: 'sweep_params' defines parameter grid (all combinations)
    2. Explicit list: 'experiments' defines ordered list of specific experiments
    
    Args:
        config: Sweep configuration dict
    
    Returns:
        List of (experiment_name, params_dict) tuples
    """
    fixed_params = config.get('fixed', {})
    
    # Mode 2: Explicit experiment list (preserves order, allows custom names)
    if 'experiments' in config:
        experiments = []
        for exp_def in config['experiments']:
            exp_name = exp_def.get('name', f"exp_{len(experiments)+1:03d}")
            exp_params = dict(fixed_params)
            exp_params.update(exp_def.get('params', {}))
            
            # Handle data_window_days -> posts_end, likes_end conversion
            if 'data_window_days' in exp_params:
                days = exp_params.pop('data_window_days')
                start_date_str = fixed_params.get('posts_start', '2026-01-01')
                start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
                end_date = start_date + timedelta(days=days)
                end_date_str = end_date.strftime('%Y-%m-%d')
                exp_params['posts_end'] = end_date_str
                exp_params['likes_end'] = end_date_str
            
            experiments.append((exp_name, exp_params))
        return experiments
    
    # Mode 1: Grid search over sweep_params
    sweep_params = config.get('sweep_params', {}).copy()
    data_window_days = sweep_params.pop('data_window_days', None)
    
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[name] for name in param_names]
    
    experiments = []
    index = 1
    
    if data_window_days:
        start_date_str = fixed_params.get('posts_start', '2026-01-01')
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        
        for days in data_window_days:
            end_date = start_date + timedelta(days=days)
            end_date_str = end_date.strftime('%Y-%m-%d')
            
            if param_names:
                for values in itertools.product(*param_values):
                    exp_params = dict(fixed_params)
                    for name, value in zip(param_names, values):
                        exp_params[name] = value
                    exp_params['posts_end'] = end_date_str
                    exp_params['likes_end'] = end_date_str
                    exp_name = _generate_experiment_name(exp_params, index)
                    experiments.append((exp_name, exp_params))
                    index += 1
            else:
                exp_params = dict(fixed_params)
                exp_params['posts_end'] = end_date_str
                exp_params['likes_end'] = end_date_str
                exp_name = _generate_experiment_name(exp_params, index)
                experiments.append((exp_name, exp_params))
                index += 1
    else:
        for values in itertools.product(*param_values):
            exp_params = dict(fixed_params)
            for name, value in zip(param_names, values):
                exp_params[name] = value
            exp_name = _generate_experiment_name(exp_params, index)
            experiments.append((exp_name, exp_params))
            index += 1
    
    return experiments


def _generate_experiment_name(params: Dict[str, Any], index: int) -> str:
    """Generate a descriptive experiment name from parameters."""
    posts_end = params.get('posts_end', 'unknown')
    max_users = params.get('max_liking_users', 0)
    max_likes = params.get('max_likes_per_user', 0)
    neg_sample = params.get('negative_posts_sample', 0)
    min_likes = params.get('min_likes_per_user', 2)
    
    posts_start = params.get('posts_start', '2026-01-01')
    try:
        start = datetime.strptime(posts_start, '%Y-%m-%d')
        end = datetime.strptime(posts_end, '%Y-%m-%d')
        days = (end - start).days
    except (ValueError, TypeError):
        days = '?'
    
    min_likes_suffix = f"_min{min_likes}" if min_likes != 2 else ""
    
    return f"sweep_{index:03d}_days{days}_users{max_users//1000}k_likes{max_likes}_neg{neg_sample//1000}k{min_likes_suffix}"


# ============================================================================
# SWEEP EXECUTION
# ============================================================================

def _build_cli_args(params: Dict[str, Any], experiment_name: str, tags: List[str]) -> List[str]:
    """Build CLI arguments for a single experiment."""
    args = ['python', 'cli.py']
    
    # Add experiment tracking
    args.extend(['--experiment-task', experiment_name])
    for tag in tags:
        args.extend(['--experiment-tags', tag])
    
    # Add all parameters
    for key, value in params.items():
        if value is None:
            continue
        
        flag = f"--{key.replace('_', '-')}"
        
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, list):
            args.append(flag)
            args.extend([str(v) for v in value])
        else:
            args.extend([flag, str(value)])
    
    args.append('--foreground')
    return args


def _run_single_experiment(
    args: List[str],
    experiment_name: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a single experiment and return results."""
    result = {
        'name': experiment_name,
        'args': args,
        'success': False,
        'return_code': None,
        'duration_seconds': None,
        'error': None,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"{'='*60}")
        if dry_run:
            print(f"[DRY RUN] Would execute:")
        print(f"  {' '.join(args)}")
    
    if dry_run:
        result['success'] = True
        result['dry_run'] = True
        return result
    
    start_time = time.time()
    try:
        proc = subprocess.run(
            args,
            capture_output=False,
            text=True,
        )
        result['return_code'] = proc.returncode
        result['success'] = (proc.returncode == 0)
    except Exception as e:
        result['error'] = str(e)
    
    result['duration_seconds'] = time.time() - start_time
    
    if verbose:
        status = "SUCCESS" if result['success'] else "FAILED"
        duration = result.get('duration_seconds', 0)
        print(f"\n[{status}] {experiment_name} completed in {duration:.1f}s")
    
    return result


def _load_sweep_progress(progress_file: Path) -> Dict[str, Any]:
    """Load sweep progress from file."""
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {'completed': [], 'failed': []}


def _save_sweep_progress(progress_file: Path, progress: Dict[str, Any]) -> None:
    """Save sweep progress to file."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def run_memory_sweep(
    config: Dict[str, Any],
    *,
    dry_run: bool = False,
    resume: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run a memory profiling sweep.
    
    Executes each experiment in the sweep configuration sequentially via cli.py,
    tracking progress and allowing resume on failure.
    
    Args:
        config: Sweep configuration from generate_sweep_config() or load_sweep_config()
        dry_run: If True, print commands without executing
        resume: If True, skip already-completed experiments
        verbose: If True, print progress information
    
    Returns:
        Dict with sweep results:
        - 'sweep_name': Name of the sweep
        - 'total_experiments': Total number of experiments
        - 'completed': Number completed successfully
        - 'failed': Number that failed
        - 'results': List of individual experiment results
        - 'progress_file': Path to progress tracking file
    
    Example:
        config = generate_sweep_config("sweep_260205", "data-sweep-26-02-05")
        results = run_memory_sweep(config, dry_run=True)  # Preview
        results = run_memory_sweep(config)  # Actually run
    """
    if yaml is None:
        raise ImportError("PyYAML is required to run sweeps. Install with: pip install pyyaml")
    
    sweep_name = config['sweep'].get('name', 'data_sweep')
    sweep_tags = config['sweep'].get('tags', [])
    execution = config.get('execution', {})
    
    delay = execution.get('delay_between_runs', 10)
    continue_on_failure = execution.get('continue_on_failure', True)
    output_base = Path(execution.get('output_base', 'outputs/sweeps'))
    
    experiments = _generate_experiment_grid(config)
    
    print(f"\n{'#'*60}")
    print(f"# Memory Sweep: {sweep_name}")
    print(f"# Total experiments: {len(experiments)}")
    print(f"# Tags: {sweep_tags}")
    print(f"{'#'*60}")
    
    sweep_dir = output_base / f"{sweep_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    progress_file = sweep_dir / 'progress.json'
    
    if resume and progress_file.exists():
        progress = _load_sweep_progress(progress_file)
        completed_names = set(progress['completed'])
        print(f"\nResuming sweep: {len(completed_names)} experiments already completed")
    else:
        progress = {'completed': [], 'failed': [], 'skipped': []}
        completed_names = set()
        sweep_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to sweep directory
    config_copy = sweep_dir / 'sweep_config.yml'
    if not config_copy.exists():
        with open(config_copy, 'w') as f:
            yaml.dump(config, f)
    
    results = []
    
    for i, (exp_name, exp_params) in enumerate(experiments):
        if exp_name in completed_names:
            print(f"\n[SKIP] {exp_name} (already completed)")
            continue
        
        cli_args = _build_cli_args(exp_params, exp_name, sweep_tags)
        result = _run_single_experiment(cli_args, exp_name, dry_run=dry_run, verbose=verbose)
        results.append(result)
        
        if result['success']:
            progress['completed'].append(exp_name)
        else:
            progress['failed'].append(exp_name)
            if not continue_on_failure and not dry_run:
                print(f"\n[ABORT] Stopping sweep due to failure")
                break
        
        if not dry_run:
            _save_sweep_progress(progress_file, progress)
        
        if i < len(experiments) - 1 and not dry_run and delay > 0:
            print(f"\nWaiting {delay}s before next experiment...")
            time.sleep(delay)
    
    print(f"\n{'#'*60}")
    print(f"# Sweep Complete")
    print(f"# Successful: {len(progress['completed'])}")
    print(f"# Failed: {len(progress['failed'])}")
    print(f"# Progress file: {progress_file}")
    print(f"{'#'*60}")
    
    return {
        'sweep_name': sweep_name,
        'total_experiments': len(experiments),
        'completed': len(progress['completed']),
        'failed': len(progress['failed']),
        'results': results,
        'progress_file': str(progress_file),
    }


# ============================================================================
# CLEARML EXPORT
# ============================================================================

# CSV field names for sweep results (in order)
SWEEP_RESULTS_FIELDNAMES = [
    'task_id', 'task_name', 'status',
    'data_window_days', 'posts_start', 'posts_end', 'likes_start', 'likes_end',
    'max_liking_users', 'max_likes_per_user', 'negative_posts_sample', 'min_likes_per_user',
    'memory_peak_gb', 'memory_estimated_peak_gb', 'memory_start_gb',
    'memory_end_gb', 'memory_growth_gb', 'memory_estimate_accuracy_pct',
    'output_likes', 'output_posts', 'embedding_dim',
    'likes_initial', 'likes_final', 'users_initial', 'users_final',
    'users_eligible', 'users_sampled', 'likes_after_user_sample',
    'likes_after_cap', 'likes_final_pre_join', 'users_final_pre_join',
    'retention_users_pct', 'retention_likes_pct',
    'likes_per_user_mean', 'likes_per_user_median', 'likes_per_user_max',
    'likes_per_user_p90', 'likes_per_user_p99',
    'posts_total', 'posts_liked', 'posts_random_sample', 'posts_match_rate',
]


def _extract_clearml_parameter(params: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Extract parameter value from ClearML's path-based structure."""
    prefixes = [
        'General/run/data/',
        'General/overrides/',
        'General/run/featurize/',
        'General/run/meta/',
        'General/',
    ]
    
    for prefix in prefixes:
        path = f'{prefix}{key}'
        if path in params:
            return params[path]
    
    if key in params:
        return params[key]
    
    for param_key in params.keys():
        if param_key.endswith(f'/{key}'):
            return params[param_key]
    
    return default


def _extract_clearml_metric(metrics: Dict[str, Any], metric_name: str, alternatives: Optional[List[str]] = None) -> Optional[float]:
    """Extract a metric value from ClearML metrics dict."""
    summary_level1 = metrics.get('Summary', {})
    if not isinstance(summary_level1, dict):
        return None
    
    summary_metrics = summary_level1.get('Summary', {})
    if not isinstance(summary_metrics, dict):
        summary_metrics = summary_level1
    
    names_to_try = [metric_name]
    if alternatives:
        names_to_try.extend(alternatives)
    
    for name in names_to_try:
        if name in summary_metrics:
            metric_data = summary_metrics[name]
            if isinstance(metric_data, dict):
                value = metric_data.get('last', metric_data.get('value', None))
                if value is not None:
                    return float(value)
            elif metric_data is not None:
                return float(metric_data)
    
    return None


def _derive_data_window_days(params: Dict[str, Any]) -> Optional[int]:
    """Derive data_window_days from posts_end and posts_start if not directly available."""
    days = _extract_clearml_parameter(params, 'data_window_days')
    if days is not None:
        return int(days)
    
    posts_start = _extract_clearml_parameter(params, 'posts_start')
    posts_end = _extract_clearml_parameter(params, 'posts_end')
    
    if posts_start and posts_end:
        try:
            start = datetime.strptime(posts_start, '%Y-%m-%d')
            end = datetime.strptime(posts_end, '%Y-%m-%d')
            return (end - start).days
        except (ValueError, TypeError):
            pass
    
    return None


def _extract_clearml_task_data(task: Any) -> Optional[Dict[str, Any]]:
    """Extract all relevant data from a ClearML task."""
    try:
        task_id = task.id
        task_name = task.name
        status = task.status
        
        params = task.get_parameters()
        
        metrics_dict = {}
        try:
            reported_scalars = task.get_reported_scalars()
            if reported_scalars:
                for title, series_dict in reported_scalars.items():
                    if isinstance(series_dict, dict):
                        for series, data in series_dict.items():
                            if isinstance(data, dict):
                                value = data.get('last') or data.get('value')
                            else:
                                value = data
                            
                            if value is not None:
                                if title and title != 'Summary':
                                    key = f"{title} - {series}" if series else title
                                else:
                                    key = series if series else title
                                
                                if 'Summary' not in metrics_dict:
                                    metrics_dict['Summary'] = {}
                                metrics_dict['Summary'][key] = {'last': value}
        except Exception:
            pass
        
        try:
            last_scalars = task.get_last_scalar_metrics()
            if last_scalars and isinstance(last_scalars, dict):
                for key, value in last_scalars.items():
                    if 'Summary' not in metrics_dict:
                        metrics_dict['Summary'] = {}
                    if isinstance(value, dict):
                        metrics_dict['Summary'][key] = value
                    else:
                        metrics_dict['Summary'][key] = {'last': value}
        except Exception:
            pass
        
        posts_start = _extract_clearml_parameter(params, 'posts_start')
        posts_end = _extract_clearml_parameter(params, 'posts_end')
        likes_start = _extract_clearml_parameter(params, 'likes_start')
        likes_end = _extract_clearml_parameter(params, 'likes_end')
        
        name_match = re.match(r'(\d+)_(\d+)d_(\d+)ku_(\d+)l_(\d+)kn', task_name)
        
        data_window_days = _derive_data_window_days(params)
        if data_window_days is None and name_match:
            data_window_days = int(name_match.group(2))
        
        max_liking_users = _extract_clearml_parameter(params, 'max_liking_users')
        if max_liking_users is None and name_match:
            max_liking_users = int(name_match.group(3)) * 1000
        max_liking_users = max_liking_users or 0
        
        max_likes_per_user = (
            _extract_clearml_parameter(params, 'max_likes_per_user') or
            _extract_clearml_parameter(params, 'max_liked_posts_per_user')
        )
        if max_likes_per_user is None and name_match:
            max_likes_per_user = int(name_match.group(4))
        max_likes_per_user = max_likes_per_user or 0
        
        negative_posts_sample = (
            _extract_clearml_parameter(params, 'negative_posts_sample') or
            _extract_clearml_parameter(params, 'negative_sample_size')
        )
        if negative_posts_sample is None and name_match:
            negative_posts_sample = int(name_match.group(5)) * 1000
        negative_posts_sample = negative_posts_sample or 0
        
        min_likes_per_user = _extract_clearml_parameter(params, 'min_likes_per_user', 2)
        
        return {
            'task_id': task_id,
            'task_name': task_name,
            'status': status,
            'data_window_days': data_window_days,
            'posts_start': posts_start,
            'posts_end': posts_end,
            'likes_start': likes_start,
            'likes_end': likes_end,
            'max_liking_users': max_liking_users,
            'max_likes_per_user': max_likes_per_user,
            'negative_posts_sample': negative_posts_sample,
            'min_likes_per_user': min_likes_per_user,
            'memory_peak_gb': _extract_clearml_metric(metrics_dict, 'Memory - Peak GB'),
            'memory_estimated_peak_gb': _extract_clearml_metric(metrics_dict, 'Memory - Estimated Peak GB'),
            'memory_start_gb': _extract_clearml_metric(metrics_dict, 'Memory - Start GB'),
            'memory_end_gb': _extract_clearml_metric(metrics_dict, 'Memory - End GB'),
            'memory_growth_gb': _extract_clearml_metric(metrics_dict, 'Memory - Growth GB'),
            'memory_estimate_accuracy_pct': _extract_clearml_metric(metrics_dict, 'Memory - Estimate Accuracy %'),
            'output_likes': _extract_clearml_metric(metrics_dict, 'Output - Likes (final)'),
            'output_posts': _extract_clearml_metric(metrics_dict, 'Output - Posts (final)'),
            'embedding_dim': _extract_clearml_metric(metrics_dict, 'Output - Embedding Dim'),
            'likes_initial': _extract_clearml_metric(metrics_dict, 'Likes - 1 Initial Likes'),
            'likes_final': _extract_clearml_metric(metrics_dict, 'Likes - 7 Final Likes (post-join)'),
            'users_initial': _extract_clearml_metric(metrics_dict, 'Likes - 1 Initial Users'),
            'users_final': _extract_clearml_metric(metrics_dict, 'Likes - 7 Final Users (post-join)'),
            'users_eligible': _extract_clearml_metric(metrics_dict, 'Likes - 2 Eligible Users (min-likes)'),
            'users_sampled': _extract_clearml_metric(metrics_dict, 'Likes - 3 Sampled Users'),
            'likes_after_user_sample': _extract_clearml_metric(metrics_dict, 'Likes - 4 Likes After User Sample'),
            'likes_after_cap': _extract_clearml_metric(metrics_dict, 'Likes - 5 Likes After Per-User Cap'),
            'likes_final_pre_join': _extract_clearml_metric(metrics_dict, 'Likes - 6 Final Likes (pre-join)'),
            'users_final_pre_join': _extract_clearml_metric(metrics_dict, 'Likes - 6 Final Users (pre-join)'),
            'retention_users_pct': _extract_clearml_metric(metrics_dict, 'Retention - Users %'),
            'retention_likes_pct': _extract_clearml_metric(metrics_dict, 'Retention - Likes %'),
            'likes_per_user_mean': _extract_clearml_metric(metrics_dict, 'Distribution - Likes/User Mean'),
            'likes_per_user_median': _extract_clearml_metric(metrics_dict, 'Distribution - Likes/User Median'),
            'likes_per_user_max': _extract_clearml_metric(metrics_dict, 'Distribution - Likes/User Max'),
            'likes_per_user_p90': _extract_clearml_metric(metrics_dict, 'Distribution - Likes/User P90'),
            'likes_per_user_p99': _extract_clearml_metric(metrics_dict, 'Distribution - Likes/User P99'),
            'posts_total': _extract_clearml_metric(metrics_dict, 'Posts - 1 Total (time-filtered)'),
            'posts_liked': _extract_clearml_metric(metrics_dict, 'Posts - 2 Liked Posts Found'),
            'posts_random_sample': _extract_clearml_metric(metrics_dict, 'Posts - 3 Random Sample'),
            'posts_match_rate': _extract_clearml_metric(metrics_dict, 'Posts - Match Rate %'),
        }
    except Exception as e:
        print(f"  Warning: Failed to extract data from task {task.name} ({task.id}): {e}")
        return None


def export_sweep_results_from_clearml(
    tag: str,
    *,
    project: str = "Engagement Prediction",
    output_file: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """Export sweep results from ClearML to CSV.
    
    Queries ClearML for completed tasks matching the specified tag, extracts
    metrics and parameters, and saves to CSV.
    
    Args:
        tag: ClearML tag to filter tasks (e.g., "data-sweep-26-02-02")
        project: ClearML project name
        output_file: Output CSV filename. If just a filename, saved in artifacts dir.
            If None, uses sweep_results_{tag}.csv
        dry_run: If True, preview results without saving
        verbose: If True, print progress for each task
    
    Returns:
        List of dicts with extracted task data
    
    Example:
        data = export_sweep_results_from_clearml(
            tag="data-sweep-26-02-05",
            output_file="sweep_results_260205.csv"
        )
    """
    try:
        from clearml import Task
    except ImportError:
        raise ImportError("clearml is required for export. Install with: pip install clearml")
    
    tags = [tag]
    print(f"Querying ClearML for tasks with tag '{tag}' in project '{project}'...")
    
    task_ids = Task.query_tasks(
        project_name=project,
        tags=tags,
        task_filter={'status': ['completed']}
    )
    
    unified_pattern = re.compile(r'^\d{1,2}_\d+d_\d+ku_\d+l_\d+kn$')
    
    print(f"Found {len(task_ids)} completed task IDs, loading and filtering...")
    tasks = []
    skipped = 0
    for task_id in task_ids:
        try:
            task = Task.get_task(task_id=task_id)
            if unified_pattern.match(task.name):
                tasks.append(task)
            else:
                skipped += 1
        except Exception as e:
            print(f"  Warning: Failed to load task {task_id}: {e}")
    
    print(f"Successfully loaded {len(tasks)} tasks")
    if skipped > 0:
        print(f"  (Skipped {skipped} tasks with non-matching names)")
    
    if not tasks:
        print("No tasks found.")
        return []
    
    print(f"\nExtracting data from {len(tasks)} tasks...")
    all_data = []
    for i, task in enumerate(tasks, 1):
        if verbose:
            print(f"  [{i}/{len(tasks)}] Processing {task.name}...")
        data = _extract_clearml_task_data(task)
        if data:
            all_data.append(data)
    
    if not all_data:
        print("No data extracted.")
        return []
    
    if dry_run:
        print(f"\n[DRY RUN] Would write {len(all_data)} records")
        print("\nPreview:")
        for d in all_data:
            print(f"  - {d['task_name']}: peak={d['memory_peak_gb']}GB, "
                  f"days={d['data_window_days']}, users={d['max_liking_users']}")
    else:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        
        if output_file is None:
            safe_tag = tag.replace('-', '_').replace(' ', '_')
            output_file = f"sweep_results_{safe_tag}.csv"
        
        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = ARTIFACTS_DIR / output_path
        
        print(f"\nWriting {len(all_data)} records to {output_path}...")
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=SWEEP_RESULTS_FIELDNAMES)
            writer.writeheader()
            writer.writerows(all_data)
        
        print(f"Successfully exported {len(all_data)} experiments")
    
    memory_peak_count = sum(1 for d in all_data if d['memory_peak_gb'] is not None)
    print(f"\nSummary:")
    print(f"  - Tasks with memory_peak_gb: {memory_peak_count}/{len(all_data)}")
    
    return all_data


# ============================================================================
# MODEL FITTING
# ============================================================================

def _load_sweep_data_from_csv(csv_path: Path) -> Tuple[List[Dict], List[str]]:
    """Load sweep results from CSV file."""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []
    return rows, columns


def _filter_valid_runs(rows: List[Dict], min_memory_gb: float = 15.0) -> List[Dict]:
    """Filter to valid completed runs with positive memory measurements."""
    valid = []
    for row in rows:
        if row.get('status') != 'completed':
            continue
        
        try:
            memory_peak = float(row.get('memory_peak_gb', 0))
            if memory_peak <= 0:
                continue
        except (ValueError, TypeError):
            continue
        
        try:
            if memory_peak < min_memory_gb:
                continue
        except (ValueError, TypeError):
            continue
        
        valid.append(row)
    
    return valid


def _compute_model_features_from_row(row: Dict) -> Dict[str, float]:
    """Compute feature values from a sweep result row.
    
    This mirrors compute_memory_model_features() but works from CSV row data.
    """
    data_window_days = float(row.get('data_window_days', 7))
    max_liking_users = float(row.get('max_liking_users', 10000))
    max_likes_per_user = float(row.get('max_likes_per_user', 100))
    negative_posts_sample = float(row.get('negative_posts_sample', 10000))
    likes_initial = float(row.get('likes_initial', 0))
    
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


def _fit_linear_regression(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit linear regression using normal equations.
    
    Returns coefficients (intercept first) and R-squared.
    """
    n_samples = X.shape[0]
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    
    XtX = X_with_intercept.T @ X_with_intercept
    Xty = X_with_intercept.T @ y
    
    coefficients = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    
    y_pred = X_with_intercept @ coefficients
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return coefficients, r_squared


def _cross_validate_model(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> Tuple[float, float]:
    """Perform k-fold cross-validation and return mean/std R-squared."""
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_size = n_samples // n_folds
    r2_scores = []
    
    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_samples
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        coeffs, _ = _fit_linear_regression(X_train, y_train)
        
        X_test_with_intercept = np.column_stack([np.ones(len(X_test)), X_test])
        y_pred = X_test_with_intercept @ coeffs
        
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r2_scores.append(r2)
    
    return np.mean(r2_scores), np.std(r2_scores)


def _compute_prediction_errors(X: np.ndarray, y: np.ndarray, coeffs: np.ndarray) -> Dict[str, float]:
    """Compute various error metrics."""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    y_pred = X_with_intercept @ coeffs
    
    abs_errors = np.abs(y - y_pred)
    pct_errors = np.abs(y - y_pred) / y * 100
    
    return {
        'mae_gb': float(np.mean(abs_errors)),
        'max_abs_error_gb': float(np.max(abs_errors)),
        'mean_pct_error': float(np.mean(pct_errors)),
        'median_pct_error': float(np.median(pct_errors)),
        'max_pct_error': float(np.max(pct_errors)),
        'p90_pct_error': float(np.percentile(pct_errors, 90)),
    }


def fit_memory_model(
    data: Union[str, Path, List[Dict]],
    *,
    min_samples: int = 30,
    min_memory_gb: float = 15.0,
    version: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Fit a memory prediction model from sweep results.
    
    Args:
        data: Either path to CSV file, or list of dicts from export_sweep_results_from_clearml()
        min_samples: Minimum number of valid samples required
        min_memory_gb: Minimum memory_peak_gb to consider a valid run
        version: Version string for the model (default: YYMMDD)
        verbose: If True, print progress and results
    
    Returns:
        Dict with model weights ready for save_model_weights():
        - 'version': Version string
        - 'fitted_at': ISO timestamp
        - 'r_squared': R-squared on training data
        - 'cv_r_squared': Cross-validation R-squared
        - 'mean_pct_error': Mean percentage error
        - 'coefficients': Dict mapping feature names to coefficients
        - 'feature_names': List of feature names in order
        - 'n_samples': Number of samples used for fitting
    
    Example:
        # From CSV file
        weights = fit_memory_model("sweep_results_260205.csv")
        
        # From export results
        data = export_sweep_results_from_clearml(tag="data-sweep-26-02-05")
        weights = fit_memory_model(data)
        save_model_weights(weights, "model_weights_260205.json")
    """
    # Load data if path provided
    if isinstance(data, (str, Path)):
        csv_path = Path(data)
        if not csv_path.is_absolute():
            csv_path = ARTIFACTS_DIR / csv_path
        
        if verbose:
            print(f"Loading data from {csv_path}...")
        rows, _ = _load_sweep_data_from_csv(csv_path)
        if verbose:
            print(f"  Loaded {len(rows)} rows")
    else:
        rows = data
        if verbose:
            print(f"Processing {len(rows)} records...")
    
    # Filter to valid runs
    valid_rows = _filter_valid_runs(rows, min_memory_gb=min_memory_gb)
    if verbose:
        print(f"  {len(valid_rows)} valid runs after filtering")
    
    if len(valid_rows) < min_samples:
        raise ValueError(f"Need at least {min_samples} valid samples, got {len(valid_rows)}")
    
    # Compute features
    if verbose:
        print("\nComputing features...")
    feature_dicts = [_compute_model_features_from_row(row) for row in valid_rows]
    feature_names = list(feature_dicts[0].keys())
    
    X = np.array([[fd[name] for name in feature_names] for fd in feature_dicts])
    y = np.array([float(row['memory_peak_gb']) for row in valid_rows])
    
    if verbose:
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Target range: {y.min():.1f} - {y.max():.1f} GB")
    
    # Fit model
    if verbose:
        print("\nFitting linear regression...")
    coefficients, r_squared = _fit_linear_regression(X, y)
    if verbose:
        print(f"  R-squared (train): {r_squared:.4f}")
    
    # Cross-validation
    if verbose:
        print("\nCross-validation (5-fold)...")
    cv_mean, cv_std = _cross_validate_model(X, y)
    if verbose:
        print(f"  R-squared (CV): {cv_mean:.4f} +/- {cv_std:.4f}")
    
    # Compute error metrics
    errors = _compute_prediction_errors(X, y, coefficients)
    if verbose:
        print("\nPrediction errors on training data:")
        print(f"  Mean absolute error: {errors['mae_gb']:.2f} GB")
        print(f"  Mean % error: {errors['mean_pct_error']:.1f}%")
        print(f"  90th percentile % error: {errors['p90_pct_error']:.1f}%")
    
    # Build coefficients dict
    coeffs_dict = {'intercept': float(coefficients[0])}
    for i, name in enumerate(feature_names):
        coeffs_dict[name] = float(coefficients[i + 1])
    
    if verbose:
        print("\nModel coefficients:")
        for name, value in coeffs_dict.items():
            print(f"  {name}: {value:.6f}")
    
    # Build result
    if version is None:
        version = datetime.now().strftime('%y%m%d')
    
    return {
        'version': version,
        'fitted_at': datetime.now(timezone.utc).isoformat(),
        'r_squared': float(r_squared),
        'cv_r_squared': float(cv_mean),
        'cv_r_squared_std': float(cv_std),
        'mean_pct_error': errors['mean_pct_error'],
        'mae_gb': errors['mae_gb'],
        'coefficients': coeffs_dict,
        'feature_names': feature_names,
        'n_samples': len(valid_rows),
    }