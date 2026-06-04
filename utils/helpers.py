#!/usr/bin/env python3

"""
Consolidated Helpers for Engagement Prediction Pipeline

This module centralizes the shared helper functions used across pipeline stages.
Only truly cross-stage utilities live here. Stage-specific helpers should live
inside their respective stage scripts (e.g., utils/03_train/stage_train_mlp.py).
"""

from __future__ import annotations

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any, TYPE_CHECKING
from datetime import datetime, timedelta, timezone
import multiprocessing as mp
import subprocess

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np  # type: ignore
    import pandas as pd  # type: ignore
    import polars as pl  # type: ignore
    from utils.pipeline.core import Context


# Avoid HF tokenizers fork warnings/deadlocks in multiprocessing contexts
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


TIMESTAMP_COL_NAME = "record_created_at"

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


def parse_one_ts_strict(raw_ts: str) -> datetime:
    for fmt in KNOWN_TS_FORMATS:
        try:
            dt = datetime.strptime(raw_ts, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    raise ValueError(f"Unrecognized datetime format: {raw_ts!r}")


def parse_one_ts(raw_ts: Optional[str]) -> Optional[datetime]:
    """Parse a single timestamp string into a timezone-aware datetime (UTC)."""
    if raw_ts is None:
        return None
    return parse_one_ts_strict(raw_ts)


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
    import polars as pl
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
    import polars as pl
    candidates = sorted(prior_path.glob(f"{prefix}*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No {prefix}*.parquet found under {prior_path}")
    return pl.scan_parquet(candidates[0])


def find_author_idx_artifact_path(context: Context) -> Optional[Path]:
    get_data_dir = context.get_active_stage_inputs().get("01_get_data")
    if get_data_dir is None:
        get_data_dir = context.get_artifact_dir("get_data")
    if get_data_dir is None:
        return None
    candidates = sorted(
        Path(get_data_dir).glob("author_idx_*.parquet"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


# ----------------------------------------
# Feature column helpers
# ----------------------------------------
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
    import numpy as np
    import pandas as pd
    try:
        from tqdm import tqdm  # type: ignore
    except Exception:  # pragma: no cover
        def tqdm(iterable=None, *args, **kwargs):
            return iterable if iterable is not None else range(kwargs.get('total', 0) or 0)
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
    column name -> dtype spec, where dtype spec can be None for a presence-only
    check, a Python type (e.g., int, float, str), a pandas/numpy dtype, a dtype
    string, or an iterable of specs.
    """
    import numpy as np
    import pandas as pd
    import polars as pl
    if not isinstance(expected_schema, dict) or not expected_schema:
        raise ValueError("expected_schema must be a non-empty dict")

    if isinstance(df, (pl.DataFrame, pl.LazyFrame)):
        schema = dict(df.collect_schema() if isinstance(df, pl.LazyFrame) else df.schema)

        polars_integer = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}
        polars_float = {pl.Float32, pl.Float64}
        polars_string = {pl.String, pl.Utf8}

        def _matches_expected_dtype_polars(dtype: pl.DataType, expected: Any) -> bool:
            if expected is None:
                return True
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
        if expected is None:
            return True
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

FIGURE_SIZE = (10, 6)
DPI = 300


def _configure_matplotlib_backend():
    """Configure matplotlib to use non-interactive Agg backend.
    
    This should be called before any matplotlib.pyplot imports to avoid
    display issues in headless environments. Checks if matplotlib has already
    been imported and only sets the backend if it hasn't, avoiding warnings
    about changing backends after initialization.
    """
    import sys
    if 'matplotlib' not in sys.modules:
        import matplotlib
        matplotlib.use("Agg")


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[Path] = None, best_epoch: Optional[int] = None):
    _configure_matplotlib_backend()
    import matplotlib.pyplot as plt  # type: ignore
    if any(k not in history for k in ['train_loss', 'val_loss']) or len(history.get('train_loss', [])) == 0:
        return
    train_metric_key = next((k for k in history if k.startswith('train_') and k != 'train_loss'), None)
    metric_key = train_metric_key.replace('train_', '', 1) if train_metric_key is not None else None
    val_metric_key = f'val_{metric_key}' if metric_key is not None else None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if train_metric_key is not None and val_metric_key in history:
        ax2.plot(epochs, history[train_metric_key], 'b-', label=f'Train {metric_key}', linewidth=2)
        ax2.plot(epochs, history[val_metric_key], 'r-', label=f'Val {metric_key}', linewidth=2)
    ax2.set_title('Training and Validation Metric')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel(metric_key or 'Metric')
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
    plt.close(fig)


def plot_model_performance(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[Path] = None,
    title_suffix: str = "",
):
    import numpy as np
    _configure_matplotlib_backend()
    import matplotlib.pyplot as plt  # type: ignore
    try:
        import seaborn as sns  # type: ignore
    except Exception:  # pragma: no cover
        sns = None  # type: ignore
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, confusion_matrix  # type: ignore
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    axes[0, 0].plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title(f'ROC Curve {title_suffix}'.strip())
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    axes[0, 1].plot(recall, precision)
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title(f'Precision-Recall Curve {title_suffix}'.strip())
    axes[0, 1].grid(True, alpha=0.3)
    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)
    if sns is not None:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    else:
        axes[1, 0].imshow(cm, cmap='Blues')
        for (i, j), val in np.ndenumerate(np.array(cm)):
            axes[1, 0].text(j, i, int(val), ha='center', va='center')
    axes[1, 0].set_title(f'Confusion Matrix {title_suffix}'.strip())
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    axes[1, 1].hist(y_pred_proba[y_true == 0], bins=50, alpha=0.7, label='Not Liked')
    axes[1, 1].hist(y_pred_proba[y_true == 1], bins=50, alpha=0.7, label='Liked')
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Prediction Distribution {title_suffix}'.strip())
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    plt.close()


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


# ----------------------------------------
# Logging utilities
# ----------------------------------------
import logging

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
    logger.info("=" * 60)
    logger.info(f"Starting: {operation_name}")
    return logger


def log_prior_stage_inputs(
    context: "Context",
    logger: logging.Logger,
    *,
    header: str = "Resolved prior inputs used",
) -> None:
    """Log the concrete prior stage artifact directories used by the active stage."""
    prior_inputs = context.get_active_stage_inputs()
    if not prior_inputs:
        logger.info(f"{header}: none")
        return

    logger.info(f"{header}:")
    for folder, path in sorted(prior_inputs.items()):
        logger.info(f"  {folder}: {path}")


def get_device(arg_device: Optional[str]) -> str:
    import torch
    if arg_device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    else:
        return arg_device


# ----------------------------------------
# PyTorch utilities
# ----------------------------------------

def clear_cuda_memory():
    """Aggressively clear CUDA cache and run garbage collection.
    
    Useful for freeing GPU memory between model creation, particularly when
    experimenting with different model sizes or batch sizes.
    """
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.
    
    Args:
        seed: Random seed value
        
    Note:
        This ensures deterministic behavior for model initialization, data
        shuffling, and stochastic operations like dropout. However, some
        CUDA operations may still have non-deterministic behavior.
    """
    import random as _random
    import numpy as np
    import torch
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
