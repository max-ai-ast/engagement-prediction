#!/usr/bin/env python3

"""
Consolidated Helpers for Engagement Prediction Pipeline

This module centralizes the shared helper functions used across pipeline stages.
Only truly cross-stage utilities live here. Stage-specific helpers should live
inside their respective stage scripts (e.g., utils/04_train/stage_train_mlp.py).
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
    column name -> dtype spec, where dtype spec can be a Python type (e.g., int,
    float, str), a pandas/numpy dtype, a dtype string, or an iterable of specs.
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
        # PCA components must be <= min(n_samples, n_features)
        max_components = min(256, X.shape[0], X.shape[1])
        if max_components > global_topic_k:  # Only use PCA if it's useful
            pca = PCA(n_components=max_components, random_state=int(random_seed))
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
