#!/usr/bin/env python3
"""
Export sweep results from ClearML for memory estimation model building.

This script queries ClearML's API to extract all metrics and parameters
from the data sweep experiments, saving them to a CSV file ready for
modeling.

Usage:
    python scripts/export_sweep_results.py --output sweep_results.csv
    python scripts/export_sweep_results.py --output sweep_results.csv --project "Engagement Prediction"
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from clearml import Task
except ImportError:
    print("Error: clearml not installed. Install with: conda install -c conda-forge clearml")
    sys.exit(1)


def extract_parameter_value(params: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Extract parameter value from ClearML's path-based structure.
    
    ClearML stores parameters with paths like:
    - General/run/data/posts_start
    - General/overrides/max_likes_per_user
    - General/run/featurize/min_likes_per_user
    """
    # Try various path prefixes
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
    
    # Also check top level
    if key in params:
        return params[key]
    
    # Try searching all keys for a match
    for param_key in params.keys():
        if param_key.endswith(f'/{key}'):
            return params[param_key]
    
    return default


def extract_metric_value(metrics: Dict[str, Any], metric_name: str, alternatives: Optional[List[str]] = None) -> Optional[float]:
    """Extract a metric value from ClearML metrics dict.
    
    ClearML stores single-value metrics with double nesting:
    metrics['Summary']['Summary']['Memory - Peak GB'] = {'last': value, 'min': value, 'max': value}
    """
    # Metrics are double-nested under 'Summary' -> 'Summary'
    summary_level1 = metrics.get('Summary', {})
    if not isinstance(summary_level1, dict):
        return None
    
    # The actual metrics are in the inner 'Summary' key
    summary_metrics = summary_level1.get('Summary', {})
    if not isinstance(summary_metrics, dict):
        # Try using summary_level1 directly as fallback
        summary_metrics = summary_level1
    
    # Build list of names to try
    names_to_try = [metric_name]
    if alternatives:
        names_to_try.extend(alternatives)
    
    # Try each name
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


def derive_data_window_days(params: Dict[str, Any]) -> Optional[int]:
    """Derive data_window_days from posts_end and posts_start if not directly available."""
    # Try to get from params
    days = extract_parameter_value(params, 'data_window_days')
    if days is not None:
        return int(days)
    
    # Derive from dates
    posts_start = extract_parameter_value(params, 'posts_start')
    posts_end = extract_parameter_value(params, 'posts_end')
    likes_start = extract_parameter_value(params, 'likes_start')
    likes_end = extract_parameter_value(params, 'likes_end')
    
    if posts_start and posts_end:
        from datetime import datetime
        try:
            start = datetime.strptime(posts_start, '%Y-%m-%d')
            end = datetime.strptime(posts_end, '%Y-%m-%d')
            return (end - start).days
        except (ValueError, TypeError):
            pass
    
    return None


def query_sweep_tasks(
    project_name: str = "Engagement Prediction",
    tags: Optional[List[str]] = None,
    task_name_pattern: Optional[str] = None,
) -> List[Task]:
    """Query ClearML for tasks matching the unified sweep criteria.
    
    Only includes tasks from data_sweep_unified.yml, which have names like:
    "01_7d_10ku_100l_10kn", "02_7d_10ku_100l_50kn", etc.
    """
    if tags is None:
        tags = ["data-sweep", "memory-analysis", "attrition-analysis"]
    
    print(f"Querying ClearML for unified sweep tasks in project '{project_name}'...")
    print(f"  Tags: {tags}")
    print(f"  Filter: Task names matching unified sweep pattern (NN_*d_*ku_*l_*kn)")
    
    # Query tasks - returns task IDs (strings)
    task_ids = Task.query_tasks(
        project_name=project_name,
        tags=tags,
        task_filter={
            'status': ['completed'],  # Only completed tasks
        }
    )
    
    # Convert task IDs to Task objects and filter to unified sweep naming pattern
    # Unified sweep tasks have names like: "01_7d_10ku_100l_10kn", "02_7d_10ku_100l_50kn"
    # Pattern: starts with 1-2 digits, underscore, then config params
    import re
    unified_pattern = re.compile(r'^\d{1,2}_\d+d_\d+ku_\d+l_\d+kn$')
    
    print(f"Found {len(task_ids)} completed task IDs, loading and filtering Task objects...")
    tasks = []
    skipped_old_format = 0
    for task_id in task_ids:
        try:
            task = Task.get_task(task_id=task_id)
            # Only include tasks matching unified sweep naming pattern
            if unified_pattern.match(task.name):
                tasks.append(task)
            else:
                skipped_old_format += 1
        except Exception as e:
            print(f"  Warning: Failed to load task {task_id}: {e}")
            continue
    
    print(f"Successfully loaded {len(tasks)} unified sweep tasks")
    if skipped_old_format > 0:
        print(f"  (Skipped {skipped_old_format} tasks with old naming format)")
    return tasks


def extract_task_data(task: Task) -> Optional[Dict[str, Any]]:
    """Extract all relevant data from a ClearML task."""
    try:
        # Get task info
        task_id = task.id
        task_name = task.name
        status = task.status
        
        # Get parameters
        params = task.get_parameters()
        
        # Get metrics - unified sweep uses log_single_value which stores under 'Summary'
        metrics_dict = {}
        
        # Method 1: get_reported_scalars (primary method for single values)
        # For log_single_value(), ClearML stores as {title: {series: {last: value}}}
        # For unified sweep, title is typically empty or 'Summary', series is the metric name
        try:
            reported_scalars = task.get_reported_scalars()
            if reported_scalars:
                for title, series_dict in reported_scalars.items():
                    if isinstance(series_dict, dict):
                        for series, data in series_dict.items():
                            # Get the value
                            if isinstance(data, dict):
                                value = data.get('last') or data.get('value')
                            else:
                                value = data
                            
                            if value is not None:
                                # Store under 'Summary' with the metric name as key
                                # The metric name is the series (for single values) or title-series combo
                                if title and title != 'Summary':
                                    key = f"{title} - {series}" if series else title
                                else:
                                    key = series if series else title
                                
                                if 'Summary' not in metrics_dict:
                                    metrics_dict['Summary'] = {}
                                metrics_dict['Summary'][key] = {'last': value}
        except Exception as e:
            pass
        
        # Method 2: get_last_scalar_metrics (fallback, may have different structure)
        try:
            last_scalars = task.get_last_scalar_metrics()
            if last_scalars:
                # This might return a flat dict or nested structure
                if isinstance(last_scalars, dict):
                    for key, value in last_scalars.items():
                        if 'Summary' not in metrics_dict:
                            metrics_dict['Summary'] = {}
                        if isinstance(value, dict):
                            metrics_dict['Summary'][key] = value
                        else:
                            metrics_dict['Summary'][key] = {'last': value}
        except Exception as e:
            pass
        
        
        # Extract key parameters from ClearML's path-based structure
        posts_start = extract_parameter_value(params, 'posts_start')
        posts_end = extract_parameter_value(params, 'posts_end')
        likes_start = extract_parameter_value(params, 'likes_start')
        likes_end = extract_parameter_value(params, 'likes_end')
        
        # Parse task name as fallback for parameters
        # Pattern: NN_Xd_Yku_Zl_Wkn (e.g., 01_7d_10ku_100l_10kn)
        import re
        name_match = re.match(r'(\d+)_(\d+)d_(\d+)ku_(\d+)l_(\d+)kn', task_name)
        
        # Derive data_window_days from dates or task name
        data_window_days = derive_data_window_days(params)
        if data_window_days is None and name_match:
            data_window_days = int(name_match.group(2))
        
        # max_liking_users
        max_liking_users = extract_parameter_value(params, 'max_liking_users')
        if max_liking_users is None and name_match:
            max_liking_users = int(name_match.group(3)) * 1000  # "10ku" = 10000
        max_liking_users = max_liking_users or 0
        
        # max_likes_per_user (CLI uses max_likes_per_user, ClearML stores as max_liked_posts_per_user)
        max_likes_per_user = (
            extract_parameter_value(params, 'max_likes_per_user') or
            extract_parameter_value(params, 'max_liked_posts_per_user')
        )
        if max_likes_per_user is None and name_match:
            max_likes_per_user = int(name_match.group(4))  # "100l" = 100
        max_likes_per_user = max_likes_per_user or 0
        
        # negative_posts_sample
        negative_posts_sample = (
            extract_parameter_value(params, 'negative_posts_sample') or
            extract_parameter_value(params, 'negative_sample_size')
        )
        if negative_posts_sample is None and name_match:
            negative_posts_sample = int(name_match.group(5)) * 1000  # "10kn" = 10000
        negative_posts_sample = negative_posts_sample or 0
        
        min_likes_per_user = extract_parameter_value(params, 'min_likes_per_user', 2)
        
        # Extract memory metrics (unified sweep uses readable names)
        memory_peak_gb = extract_metric_value(metrics_dict, 'Memory - Peak GB')
        memory_estimated_peak_gb = extract_metric_value(metrics_dict, 'Memory - Estimated Peak GB')
        memory_start_gb = extract_metric_value(metrics_dict, 'Memory - Start GB')
        memory_end_gb = extract_metric_value(metrics_dict, 'Memory - End GB')
        memory_growth_gb = extract_metric_value(metrics_dict, 'Memory - Growth GB')
        memory_estimate_accuracy_pct = extract_metric_value(metrics_dict, 'Memory - Estimate Accuracy %')
        
        # Extract output sizes
        output_likes = extract_metric_value(metrics_dict, 'Output - Likes (final)')
        output_posts = extract_metric_value(metrics_dict, 'Output - Posts (final)')
        embedding_dim = extract_metric_value(metrics_dict, 'Output - Embedding Dim')
        
        # Extract attrition metrics (unified sweep uses readable names)
        likes_initial = extract_metric_value(metrics_dict, 'Likes - 1 Initial Likes')
        likes_final = extract_metric_value(metrics_dict, 'Likes - 7 Final Likes (post-join)')
        users_initial = extract_metric_value(metrics_dict, 'Likes - 1 Initial Users')
        users_final = extract_metric_value(metrics_dict, 'Likes - 7 Final Users (post-join)')
        retention_users_pct = extract_metric_value(metrics_dict, 'Retention - Users %')
        retention_likes_pct = extract_metric_value(metrics_dict, 'Retention - Likes %')
        
        # Extract intermediate pipeline metrics
        users_eligible = extract_metric_value(metrics_dict, 'Likes - 2 Eligible Users (min-likes)')
        users_sampled = extract_metric_value(metrics_dict, 'Likes - 3 Sampled Users')
        likes_after_user_sample = extract_metric_value(metrics_dict, 'Likes - 4 Likes After User Sample')
        likes_after_cap = extract_metric_value(metrics_dict, 'Likes - 5 Likes After Per-User Cap')
        likes_final_pre_join = extract_metric_value(metrics_dict, 'Likes - 6 Final Likes (pre-join)')
        users_final_pre_join = extract_metric_value(metrics_dict, 'Likes - 6 Final Users (pre-join)')
        
        # Extract distribution stats
        likes_per_user_mean = extract_metric_value(metrics_dict, 'Distribution - Likes/User Mean')
        likes_per_user_median = extract_metric_value(metrics_dict, 'Distribution - Likes/User Median')
        likes_per_user_max = extract_metric_value(metrics_dict, 'Distribution - Likes/User Max')
        likes_per_user_p90 = extract_metric_value(metrics_dict, 'Distribution - Likes/User P90')
        likes_per_user_p99 = extract_metric_value(metrics_dict, 'Distribution - Likes/User P99')
        
        # Extract posts metrics
        posts_total = extract_metric_value(metrics_dict, 'Posts - 1 Total (time-filtered)')
        posts_liked = extract_metric_value(metrics_dict, 'Posts - 2 Liked Posts Found')
        posts_random_sample = extract_metric_value(metrics_dict, 'Posts - 3 Random Sample')
        posts_match_rate = extract_metric_value(metrics_dict, 'Posts - Match Rate %')
        
        return {
            # Task identification
            'task_id': task_id,
            'task_name': task_name,
            'status': status,
            
            # Hyperparameters (inputs for model)
            'data_window_days': data_window_days,
            'posts_start': posts_start,
            'posts_end': posts_end,
            'likes_start': likes_start,
            'likes_end': likes_end,
            'max_liking_users': max_liking_users,
            'max_likes_per_user': max_likes_per_user,
            'negative_posts_sample': negative_posts_sample,
            'min_likes_per_user': min_likes_per_user,
            
            # Memory metrics (target for prediction)
            'memory_peak_gb': memory_peak_gb,
            'memory_estimated_peak_gb': memory_estimated_peak_gb,
            'memory_start_gb': memory_start_gb,
            'memory_end_gb': memory_end_gb,
            'memory_growth_gb': memory_growth_gb,
            'memory_estimate_accuracy_pct': memory_estimate_accuracy_pct,
            
            # Output sizes (features for model)
            'output_likes': output_likes,
            'output_posts': output_posts,
            'embedding_dim': embedding_dim,
            
            # Attrition metrics (features)
            'likes_initial': likes_initial,
            'likes_final': likes_final,
            'users_initial': users_initial,
            'users_final': users_final,
            'users_eligible': users_eligible,
            'users_sampled': users_sampled,
            'likes_after_user_sample': likes_after_user_sample,
            'likes_after_cap': likes_after_cap,
            'likes_final_pre_join': likes_final_pre_join,
            'users_final_pre_join': users_final_pre_join,
            'retention_users_pct': retention_users_pct,
            'retention_likes_pct': retention_likes_pct,
            
            # Distribution stats (features)
            'likes_per_user_mean': likes_per_user_mean,
            'likes_per_user_median': likes_per_user_median,
            'likes_per_user_max': likes_per_user_max,
            'likes_per_user_p90': likes_per_user_p90,
            'likes_per_user_p99': likes_per_user_p99,
            
            # Posts metrics (features)
            'posts_total': posts_total,
            'posts_liked': posts_liked,
            'posts_random_sample': posts_random_sample,
            'posts_match_rate': posts_match_rate,
        }
    except Exception as e:
        print(f"  Warning: Failed to extract data from task {task.name} ({task.id}): {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Export sweep results from ClearML for memory estimation modeling"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='sweep_results.csv',
        help='Output CSV file path (default: sweep_results.csv)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='Engagement Prediction',
        help='ClearML project name (default: "Engagement Prediction")'
    )
    parser.add_argument(
        '--tags',
        nargs='+',
        default=['data-sweep', 'memory-analysis', 'attrition-analysis'],
        help='Tags to filter tasks (default: data-sweep memory-analysis attrition-analysis)'
    )
    parser.add_argument(
        '--task-pattern',
        type=str,
        default=None,
        help='Optional regex pattern to filter task names (default: unified sweep pattern)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed progress'
    )
    
    args = parser.parse_args()
    
    # Query tasks (unified sweep only by default)
    tasks = query_sweep_tasks(
        project_name=args.project,
        tags=args.tags,
        task_name_pattern=args.task_pattern,  # If None, uses unified sweep pattern
    )
    
    if not tasks:
        print("No tasks found. Exiting.")
        return 1
    
    # Extract data from each task
    print(f"\nExtracting data from {len(tasks)} tasks...")
    all_data = []
    for i, task in enumerate(tasks, 1):
        if args.verbose:
            print(f"  [{i}/{len(tasks)}] Processing {task.name}...")
        data = extract_task_data(task)
        if data:
            all_data.append(data)
    
    if not all_data:
        print("No data extracted. Exiting.")
        return 1
    
    # Write to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nWriting {len(all_data)} records to {output_path}...")
    
    # Define column order (parameters first, then target, then features)
    fieldnames = [
        # Identification
        'task_id', 'task_name', 'status',
        # Hyperparameters (inputs)
        'data_window_days', 'posts_start', 'posts_end', 'likes_start', 'likes_end',
        'max_liking_users', 'max_likes_per_user', 'negative_posts_sample', 'min_likes_per_user',
        # Memory metrics (target)
        'memory_peak_gb', 'memory_estimated_peak_gb', 'memory_start_gb',
        'memory_end_gb', 'memory_growth_gb', 'memory_estimate_accuracy_pct',
        # Output sizes (features)
        'output_likes', 'output_posts', 'embedding_dim',
            # Attrition (features)
            'likes_initial', 'likes_final', 'users_initial', 'users_final',
            'users_eligible', 'users_sampled', 'likes_after_user_sample',
            'likes_after_cap', 'likes_final_pre_join', 'users_final_pre_join',
            'retention_users_pct', 'retention_likes_pct',
        # Distribution (features)
        'likes_per_user_mean', 'likes_per_user_median', 'likes_per_user_max',
        'likes_per_user_p90', 'likes_per_user_p99',
        # Posts (features)
        'posts_total', 'posts_liked', 'posts_random_sample', 'posts_match_rate',
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_data)
    
    print(f"✓ Successfully exported {len(all_data)} unified sweep experiments to {output_path}")
    print(f"\nSummary:")
    memory_peak_count = sum(1 for d in all_data if d['memory_peak_gb'] is not None)
    all_params_count = sum(1 for d in all_data if all(d.get(k) is not None for k in ['data_window_days', 'max_liking_users', 'max_likes_per_user']))
    print(f"  - Tasks with memory_peak_gb: {memory_peak_count}/{len(all_data)}")
    print(f"  - Tasks with all params: {all_params_count}/{len(all_data)}")
    
    if memory_peak_count < len(all_data):
        missing = len(all_data) - memory_peak_count
        print(f"\n  WARNING: {missing} tasks missing memory_peak_gb")
        print(f"  These may be from older runs or may have failed before logging metrics")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
