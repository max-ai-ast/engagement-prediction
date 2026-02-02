#!/usr/bin/env python3
"""
Fit a regression model to predict peak memory usage from configuration parameters.

This script reads sweep_results.csv, engineers features, fits a linear regression
model, and outputs the coefficients for use in utils/helpers.py.

Usage:
    python scripts/fit_memory_model.py
    python scripts/fit_memory_model.py --input sweep_results.csv --min-samples 50
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_sweep_data(csv_path: Path) -> Tuple[List[Dict], List[str]]:
    """Load sweep results from CSV file."""
    import csv
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        columns = reader.fieldnames or []
    
    return rows, columns


def filter_valid_runs(rows: List[Dict]) -> List[Dict]:
    """Filter to valid completed runs with positive memory measurements."""
    valid = []
    for row in rows:
        # Must be completed
        if row.get('status') != 'completed':
            continue
        
        # Must have valid memory peak
        try:
            memory_peak = float(row.get('memory_peak_gb', 0))
            if memory_peak <= 0:
                continue
        except (ValueError, TypeError):
            continue
        
        # Must have produced output (not a failed/empty run)
        try:
            output_likes = float(row.get('output_likes', 0))
            # Allow runs with 0 output_likes but valid memory - they still consumed memory
            # But filter out runs that look like early failures (very low memory)
            if memory_peak < 15:  # Less than 15 GB suggests early failure
                continue
        except (ValueError, TypeError):
            continue
        
        valid.append(row)
    
    return valid


def compute_features(row: Dict) -> Dict[str, float]:
    """Compute feature values from a sweep result row.
    
    Features are designed to capture the key drivers of memory consumption:
    - data_window_days: More days = more raw data
    - max_liking_users: More users = more likes to process
    - negative_posts_sample: More negatives = more post embeddings
    - likes_initial: Raw data size from parquet metadata
    
    We use log/sqrt transforms for features with large ranges to capture
    sub-linear scaling relationships.
    """
    # Extract raw values
    data_window_days = float(row.get('data_window_days', 7))
    max_liking_users = float(row.get('max_liking_users', 10000))
    max_likes_per_user = float(row.get('max_likes_per_user', 100))
    negative_posts_sample = float(row.get('negative_posts_sample', 10000))
    likes_initial = float(row.get('likes_initial', 0))
    
    # Compute engineered features
    features = {
        # Base features (scaled for numerical stability)
        'data_window_days': data_window_days,
        'max_liking_users_10k': max_liking_users / 10000,
        'max_likes_per_user_100': max_likes_per_user / 100,
        'negative_posts_sample_10k': negative_posts_sample / 10000,
        
        # Log transform for users (captures sub-linear scaling)
        'log_max_liking_users': np.log10(max(max_liking_users, 1)),
        
        # Square root of raw likes (sub-linear scaling with data size)
        'sqrt_likes_initial_1e6': np.sqrt(likes_initial) / 1000,
        
        # Interaction: window * users (more days with more users = multiplicative effect)
        'days_x_users_10k': data_window_days * max_liking_users / 10000,
        
        # Interaction: users * log(users) for super-linear user scaling
        'users_x_log_users': (max_liking_users / 10000) * np.log10(max(max_liking_users, 1)),
    }
    
    return features


def fit_linear_regression(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit linear regression using normal equations.
    
    Returns coefficients (including intercept as first element) and R-squared.
    """
    # Add intercept column
    n_samples = X.shape[0]
    X_with_intercept = np.column_stack([np.ones(n_samples), X])
    
    # Solve normal equations: (X'X)^-1 X'y
    XtX = X_with_intercept.T @ X_with_intercept
    Xty = X_with_intercept.T @ y
    
    # Use pseudo-inverse for numerical stability
    coefficients = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    
    # Compute R-squared
    y_pred = X_with_intercept @ coefficients
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return coefficients, r_squared


def cross_validate(X: np.ndarray, y: np.ndarray, n_folds: int = 5) -> Tuple[float, float]:
    """Perform k-fold cross-validation and return mean/std R-squared."""
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    fold_size = n_samples // n_folds
    r2_scores = []
    
    for fold in range(n_folds):
        # Split into train/test
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n_samples
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit on train
        coeffs, _ = fit_linear_regression(X_train, y_train)
        
        # Evaluate on test
        X_test_with_intercept = np.column_stack([np.ones(len(X_test)), X_test])
        y_pred = X_test_with_intercept @ coeffs
        
        ss_res = np.sum((y_test - y_pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r2_scores.append(r2)
    
    return np.mean(r2_scores), np.std(r2_scores)


def compute_prediction_errors(X: np.ndarray, y: np.ndarray, coeffs: np.ndarray) -> Dict[str, float]:
    """Compute various error metrics."""
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    y_pred = X_with_intercept @ coeffs
    
    # Absolute errors
    abs_errors = np.abs(y - y_pred)
    
    # Percentage errors
    pct_errors = np.abs(y - y_pred) / y * 100
    
    return {
        'mae_gb': np.mean(abs_errors),
        'max_abs_error_gb': np.max(abs_errors),
        'mean_pct_error': np.mean(pct_errors),
        'median_pct_error': np.median(pct_errors),
        'max_pct_error': np.max(pct_errors),
        'p90_pct_error': np.percentile(pct_errors, 90),
    }


def generate_code_snippet(feature_names: List[str], coefficients: np.ndarray, r_squared: float) -> str:
    """Generate Python code snippet for embedding in helpers.py."""
    lines = [
        "# Memory estimation model coefficients",
        "# Fitted on sweep_results.csv using scripts/fit_memory_model.py",
        f"# R-squared: {r_squared:.4f}",
        "#",
        "# Usage:",
        "#   features = compute_memory_model_features(...)",
        "#   estimated_gb = predict_memory_gb(features)",
        "",
        "MEMORY_MODEL_COEFFICIENTS = {",
        f"    'intercept': {coefficients[0]:.10f},",
    ]
    
    for i, name in enumerate(feature_names):
        lines.append(f"    '{name}': {coefficients[i+1]:.10f},")
    
    lines.append("}")
    lines.append("")
    lines.append("")
    lines.append("MEMORY_MODEL_FEATURE_NAMES = [")
    for name in feature_names:
        lines.append(f"    '{name}',")
    lines.append("]")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Fit memory estimation model")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=PROJECT_ROOT / "sweep_results.csv",
        help="Path to sweep_results.csv"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=30,
        help="Minimum number of valid samples required"
    )
    parser.add_argument(
        "--output-code",
        type=Path,
        default=None,
        help="Output file for generated code snippet (default: stdout)"
    )
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    rows, columns = load_sweep_data(args.input)
    print(f"  Loaded {len(rows)} rows")
    
    # Filter to valid runs
    valid_rows = filter_valid_runs(rows)
    print(f"  {len(valid_rows)} valid runs after filtering")
    
    if len(valid_rows) < args.min_samples:
        print(f"Error: Need at least {args.min_samples} valid samples, got {len(valid_rows)}")
        sys.exit(1)
    
    # Compute features for all rows
    print("\nComputing features...")
    feature_dicts = [compute_features(row) for row in valid_rows]
    feature_names = list(feature_dicts[0].keys())
    
    # Build feature matrix and target vector
    X = np.array([[fd[name] for name in feature_names] for fd in feature_dicts])
    y = np.array([float(row['memory_peak_gb']) for row in valid_rows])
    
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Target range: {y.min():.1f} - {y.max():.1f} GB")
    print(f"  Features: {feature_names}")
    
    # Fit model
    print("\nFitting linear regression...")
    coefficients, r_squared = fit_linear_regression(X, y)
    print(f"  R-squared (train): {r_squared:.4f}")
    
    # Cross-validation
    print("\nCross-validation (5-fold)...")
    cv_mean, cv_std = cross_validate(X, y)
    print(f"  R-squared (CV): {cv_mean:.4f} +/- {cv_std:.4f}")
    
    # Compute error metrics
    print("\nPrediction errors on training data:")
    errors = compute_prediction_errors(X, y, coefficients)
    print(f"  Mean absolute error: {errors['mae_gb']:.2f} GB")
    print(f"  Max absolute error: {errors['max_abs_error_gb']:.2f} GB")
    print(f"  Mean % error: {errors['mean_pct_error']:.1f}%")
    print(f"  Median % error: {errors['median_pct_error']:.1f}%")
    print(f"  90th percentile % error: {errors['p90_pct_error']:.1f}%")
    print(f"  Max % error: {errors['max_pct_error']:.1f}%")
    
    # Print coefficients
    print("\nModel coefficients:")
    print(f"  intercept: {coefficients[0]:.6f}")
    for i, name in enumerate(feature_names):
        print(f"  {name}: {coefficients[i+1]:.6f}")
    
    # Compare with current estimator
    print("\nComparison with current estimator:")
    current_estimates = [float(row.get('memory_estimated_peak_gb', 0)) for row in valid_rows]
    current_pct_errors = [abs(est - act) / act * 100 for est, act in zip(current_estimates, y) if est > 0]
    if current_pct_errors:
        print(f"  Current estimator mean % error: {np.mean(current_pct_errors):.1f}%")
        print(f"  New model mean % error: {errors['mean_pct_error']:.1f}%")
        improvement = np.mean(current_pct_errors) / errors['mean_pct_error']
        print(f"  Improvement factor: {improvement:.1f}x")
    
    # Generate code snippet
    print("\n" + "=" * 60)
    print("Code snippet for utils/helpers.py:")
    print("=" * 60)
    code_snippet = generate_code_snippet(feature_names, coefficients, r_squared)
    print(code_snippet)
    
    if args.output_code:
        args.output_code.write_text(code_snippet)
        print(f"\nCode snippet written to {args.output_code}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
