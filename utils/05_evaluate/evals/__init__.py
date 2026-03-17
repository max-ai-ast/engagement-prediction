#!/usr/bin/env python3

"""
Evaluation Modules Framework for Engagement Prediction Pipeline

This module provides:
- EvalContext: Standardized data structure passed to all evaluation modules
- EvalModule: Abstract base class for evaluation modules
- discover_modules(): Auto-discovery function to find all eval modules
- run_all_modules(): Orchestrator to run all discovered modules

Each evaluation module should:
1. Subclass EvalModule
2. Implement the `run(ctx: EvalContext) -> Dict[str, Any]` method
3. Save artifacts to ctx.output_dir / self.name
4. Return a summary dict with key metrics

Example module structure:
    class MyEvalModule(EvalModule):
        name = "my_evaluation"
        description = "Computes some metrics"
        
        def run(self, ctx: EvalContext) -> Dict[str, Any]:
            out_dir = ctx.output_dir / self.name
            out_dir.mkdir(parents=True, exist_ok=True)
            # ... compute metrics, save plots ...
            return {"metric1": value1, "metric2": value2}
"""

from __future__ import annotations

import importlib
import json
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd


@dataclass
class EvalContext:
    """
    Standardized evaluation context passed to all evaluation modules.
    
    Attributes:
        predictions_df: DataFrame with columns [did, post_id, y_true, y_pred_proba]
                       Contains holdout predictions from training stage.
        user_metadata_df: DataFrame with columns [did, num_embedding_likes, num_total_likes]
                         Contains per-user metadata computed from the bundle.
        output_dir: Base output directory for all evaluation artifacts.
        timestamp: Timestamp string for artifact naming.
        config: Optional configuration dict for evaluation parameters.
    """
    predictions_df: pd.DataFrame
    user_metadata_df: pd.DataFrame
    output_dir: Path
    timestamp: str
    config: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def num_holdout_users(self) -> int:
        """Get count of holdout users."""
        return int(self.predictions_df['did'].nunique())
    
    @property
    def num_predictions(self) -> int:
        """Get total number of predictions."""
        return len(self.predictions_df)


class EvalModule(ABC):
    """
    Abstract base class for evaluation modules.
    
    Each evaluation module computes specific metrics on holdout predictions
    and saves artifacts (plots, CSVs, JSON summaries) to its own subdirectory.
    
    Subclasses must define:
        - name: str - unique identifier for the module (used for output subdirectory)
        - description: str - human-readable description of what the module computes
        - run(ctx: EvalContext) -> Dict[str, Any] - main evaluation logic
    """
    
    name: str = "base_module"
    description: str = "Base evaluation module"
    
    @abstractmethod
    def run(self, ctx: EvalContext) -> Dict[str, Any]:
        """
        Run the evaluation module.
        
        Args:
            ctx: EvalContext containing predictions, user metadata, and output paths.
        
        Returns:
            Dict with summary metrics and artifact paths.
        """
        pass
    
    def get_output_dir(self, ctx: EvalContext) -> Path:
        """Get the output directory for this module's artifacts."""
        out_dir = ctx.output_dir / self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    
    def save_json(self, data: Dict[str, Any], path: Path) -> None:
        """Save a dict as JSON with proper serialization."""
        def _serialize(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, pd.Series):
                return obj.tolist()
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            return str(obj)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=_serialize)


def discover_modules() -> List[Type[EvalModule]]:
    """
    Auto-discover all evaluation modules in the evals package.
    
    Scans for Python files in this directory (excluding __init__.py),
    imports them, and collects all EvalModule subclasses.
    
    Returns:
        List of EvalModule subclasses found in the package.
    """
    modules: List[Type[EvalModule]] = []
    package_dir = Path(__file__).parent
    
    for finder, module_name, is_pkg in pkgutil.iter_modules([str(package_dir)]):
        if module_name.startswith('_'):
            continue
        
        try:
            module = importlib.import_module(f".{module_name}", package=__name__)
            
            # Find all EvalModule subclasses in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, EvalModule) and 
                    attr is not EvalModule):
                    modules.append(attr)
        except Exception as e:
            print(f"Warning: Failed to import evaluation module '{module_name}': {e}")
    
    return modules


def run_all_modules(
    ctx: EvalContext,
    modules: Optional[List[Type[EvalModule]]] = None,
    skip_modules: Optional[List[str]] = None,
    only_modules: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run all evaluation modules and collect results.
    
    Args:
        ctx: EvalContext to pass to each module.
        modules: Optional list of module classes to run. If None, auto-discovers modules.
        skip_modules: Optional list of module names to skip.
        only_modules: Optional list of module names to run exclusively.
    
    Returns:
        Dict mapping module name -> module results dict.
    """
    if modules is None:
        modules = discover_modules()
    
    skip_set = set(skip_modules or [])
    only_set = set(only_modules) if only_modules else None
    results: Dict[str, Dict[str, Any]] = {}
    
    for module_cls in modules:
        module = module_cls()
        
        if only_set is not None and module.name not in only_set:
            print(f"  Skipping module: {module.name}")
            continue
        if module.name in skip_set:
            print(f"  Skipping module: {module.name}")
            continue
        
        print(f"  Running module: {module.name} - {module.description}")
        try:
            result = module.run(ctx)
            results[module.name] = {
                'status': 'success',
                'description': module.description,
                **result,
            }
            print(f"    Completed: {module.name}")
        except Exception as e:
            import traceback
            results[module.name] = {
                'status': 'error',
                'description': module.description,
                'error': str(e),
                'traceback': traceback.format_exc(),
            }
            print(f"    Error in {module.name}: {e}")
    
    return results


# Utility functions for common metric computations

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> Dict[str, float]:
    """Compute precision, recall, and F1 from binary labels and predicted probabilities.

    These three metrics are robust to noisy negatives, unlike accuracy and
    AUC-ROC which treat the negative class as ground truth.

    Returns:
        Dict with keys ``precision``, ``recall``, ``f1``.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    y_pred_binary = (y_pred_proba > 0.5).astype(int)
    return {
        'precision': float(precision_score(y_true, y_pred_binary, zero_division=0)),
        'recall':    float(recall_score(y_true, y_pred_binary, zero_division=0)),
        'f1':        float(f1_score(y_true, y_pred_binary, zero_division=0)),
    }


def compute_per_user_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-user classification metrics from predictions.

    Args:
        predictions_df: DataFrame with columns [did, post_id, y_true, y_pred_proba]

    Returns:
        DataFrame with columns [did, num_samples, num_positive, num_negative,
                               precision, recall, f1, accuracy, auc_roc]
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    rows = []
    for user_id, group in predictions_df.groupby('did'):
        y_true = group['y_true'].values
        y_pred_proba = group['y_pred_proba'].values
        y_pred_binary = (y_pred_proba > 0.5).astype(int)

        num_samples = len(y_true)
        num_positive = int(y_true.sum())

        row = {
            'did': str(user_id),
            'num_samples': num_samples,
            'num_positive': num_positive,
            'num_negative': num_samples - num_positive,
            'accuracy': float(accuracy_score(y_true, y_pred_binary)) if num_samples > 0 else float('nan'),
            **compute_classification_metrics(y_true, y_pred_proba),
            'auc_roc': (
                float(roc_auc_score(y_true, y_pred_proba))
                if len(set(y_true)) > 1
                else float('nan')
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def compute_gini_coefficient(values: np.ndarray) -> float:
    """
    Compute the Gini coefficient for a distribution of values.
    
    The Gini coefficient measures inequality in a distribution:
    - 0 = perfect equality (everyone has the same value)
    - 1 = maximum inequality (one person has everything)
    
    Args:
        values: Array of non-negative values.
    
    Returns:
        Gini coefficient (float between 0 and 1).
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    
    # If there are no valid values, Gini is undefined
    if len(values) == 0:
        return float('nan')
    
    # If all (non-NaN) values sum to zero, this is perfect equality
    if values.sum() == 0:
        return 0.0
    
    # Sort values
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    # Compute Gini using the formula: G = (2 * sum(i * x_i) - (n+1) * sum(x_i)) / (n * sum(x_i))
    cumsum = np.cumsum(sorted_values)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
    
    return float(gini)


def compute_lorenz_curve(values: np.ndarray) -> tuple:
    """
    Compute the Lorenz curve for a distribution of values.
    
    The Lorenz curve shows the cumulative share of the total held by
    the bottom x% of the population.
    
    Args:
        values: Array of non-negative values.
    
    Returns:
        Tuple of (cumulative_population_share, cumulative_value_share)
        Both arrays range from 0 to 1.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    
    if len(values) == 0:
        return np.array([0, 1]), np.array([0, 1])
    
    # Sort values
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    # Cumulative sum
    cumsum = np.cumsum(sorted_values)
    total = cumsum[-1] if cumsum[-1] > 0 else 1
    
    # Cumulative shares (prepend 0 for origin point)
    cumulative_population = np.concatenate([[0], np.arange(1, n + 1) / n])
    cumulative_value = np.concatenate([[0], cumsum / total])
    
    return cumulative_population, cumulative_value


__all__ = [
    'EvalContext',
    'EvalModule',
    'discover_modules',
    'run_all_modules',
    'compute_classification_metrics',
    'compute_per_user_metrics',
    'compute_gini_coefficient',
    'compute_lorenz_curve',
]
