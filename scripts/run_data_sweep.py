#!/usr/bin/env python3
"""
Data Sweep Runner
=================

Orchestrates a hyperparameter sweep over data filtering parameters to measure
their impact on N (sample sizes) and memory consumption.

Usage:
    python scripts/run_data_sweep.py --config configs/data_sweep.yml
    python scripts/run_data_sweep.py --config configs/data_sweep.yml --dry-run
    python scripts/run_data_sweep.py --config configs/data_sweep.yml --resume

The sweep:
1. Reads parameter grid from the config file
2. Generates all combinations
3. Runs each experiment sequentially via cli.py run-all
4. Each experiment is tracked as a separate ClearML task
5. Results can be compared in the ClearML dashboard

Monitoring in ClearML:
- Go to your ClearML dashboard (e.g., https://app.clear.ml)
- Navigate to Projects > "Engagement Prediction"
- Filter by tags: "data-sweep"
- Use "Compare" to view metrics across experiments
- Key metrics to compare:
  - get_data/n_users_final
  - get_data/n_likes_core
  - get_data/memory_peak_gb
  - get_data/memory_growth_gb
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_sweep_config(config_path: str) -> Dict[str, Any]:
    """Load and validate sweep configuration."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path) as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    if 'sweep' not in config:
        raise ValueError("Config missing required section: sweep")
    if 'fixed' not in config:
        raise ValueError("Config missing required section: fixed")
    
    # Must have either sweep_params (grid search) or experiments (explicit list)
    if 'sweep_params' not in config and 'experiments' not in config:
        raise ValueError("Config must have either 'sweep_params' (grid search) or 'experiments' (explicit list)")
    
    return config


def generate_experiment_grid(config: Dict[str, Any]) -> List[tuple]:
    """
    Generate experiments from the sweep config.
    
    Supports two modes:
    1. Grid search: 'sweep_params' defines parameter grid (all combinations)
    2. Explicit list: 'experiments' defines ordered list of specific experiments
    
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
    
    # Handle special case: data_window_days -> posts_end, likes_end (ALIGNED)
    data_window_days = sweep_params.pop('data_window_days', None)
    
    # Generate all combinations of independent parameters
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
                    exp_name = generate_experiment_name(exp_params, index)
                    experiments.append((exp_name, exp_params))
                    index += 1
            else:
                exp_params = dict(fixed_params)
                exp_params['posts_end'] = end_date_str
                exp_params['likes_end'] = end_date_str
                exp_name = generate_experiment_name(exp_params, index)
                experiments.append((exp_name, exp_params))
                index += 1
    else:
        for values in itertools.product(*param_values):
            exp_params = dict(fixed_params)
            for name, value in zip(param_names, values):
                exp_params[name] = value
            exp_name = generate_experiment_name(exp_params, index)
            experiments.append((exp_name, exp_params))
            index += 1
    
    return experiments


def generate_experiment_name(params: Dict[str, Any], index: int) -> str:
    """Generate a descriptive experiment name from parameters."""
    # Extract key parameters for the name
    posts_end = params.get('posts_end', 'unknown')
    max_users = params.get('max_liking_users', 0)
    max_likes = params.get('max_likes_per_user', 0)
    neg_sample = params.get('negative_posts_sample', 0)
    min_likes = params.get('min_likes_per_user', 2)
    
    # Calculate days from start
    posts_start = params.get('posts_start', '2026-01-01')
    try:
        start = datetime.strptime(posts_start, '%Y-%m-%d')
        end = datetime.strptime(posts_end, '%Y-%m-%d')
        days = (end - start).days
    except (ValueError, TypeError):
        days = '?'
    
    # Include min_likes in name if not default (2)
    min_likes_suffix = f"_min{min_likes}" if min_likes != 2 else ""
    
    return f"sweep_{index:03d}_days{days}_users{max_users//1000}k_likes{max_likes}_neg{neg_sample//1000}k{min_likes_suffix}"


def build_cli_args(params: Dict[str, Any], experiment_name: str, tags: List[str]) -> List[str]:
    """Build CLI arguments for a single experiment."""
    args = ['python', 'cli.py', 'run-all']
    
    # Add experiment tracking
    args.extend(['--experiment-task', experiment_name])
    for tag in tags:
        args.extend(['--experiment-tags', tag])
    
    # Add all parameters
    for key, value in params.items():
        if value is None:
            continue
        
        # Convert key to CLI flag format
        flag = f"--{key.replace('_', '-')}"
        
        if isinstance(value, bool):
            if value:
                args.append(flag)
        elif isinstance(value, list):
            args.append(flag)
            args.extend([str(v) for v in value])
        else:
            args.extend([flag, str(value)])
    
    # Always run in foreground for sequential execution
    args.append('--foreground')
    
    return args


def run_experiment(
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
            capture_output=False,  # Let output go to terminal
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


def load_progress(progress_file: Path) -> Dict[str, Any]:
    """Load sweep progress from file."""
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {'completed': [], 'failed': []}


def save_progress(progress_file: Path, progress: Dict[str, Any]) -> None:
    """Save sweep progress to file."""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with open(progress_file, 'w') as f:
        json.dump(progress, f, indent=2)


def run_sweep(
    config: Dict[str, Any],
    dry_run: bool = False,
    resume: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the full parameter sweep."""
    sweep_name = config['sweep'].get('name', 'data_sweep')
    sweep_tags = config['sweep'].get('tags', [])
    execution = config.get('execution', {})
    
    delay = execution.get('delay_between_runs', 10)
    continue_on_failure = execution.get('continue_on_failure', True)
    output_base = Path(execution.get('output_base', 'outputs/sweeps'))
    
    # Generate experiment list (returns list of (name, params) tuples)
    experiments = generate_experiment_grid(config)
    
    print(f"\n{'#'*60}")
    print(f"# Data Sweep: {sweep_name}")
    print(f"# Total experiments: {len(experiments)}")
    print(f"# Tags: {sweep_tags}")
    print(f"{'#'*60}")
    
    # Progress tracking
    sweep_dir = output_base / f"{sweep_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    progress_file = sweep_dir / 'progress.json'
    
    if resume and progress_file.exists():
        progress = load_progress(progress_file)
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
        # Skip if already completed
        if exp_name in completed_names:
            print(f"\n[SKIP] {exp_name} (already completed)")
            continue
        
        # Build and run
        cli_args = build_cli_args(exp_params, exp_name, sweep_tags)
        result = run_experiment(cli_args, exp_name, dry_run=dry_run, verbose=verbose)
        results.append(result)
        
        # Update progress
        if result['success']:
            progress['completed'].append(exp_name)
        else:
            progress['failed'].append(exp_name)
            if not continue_on_failure and not dry_run:
                print(f"\n[ABORT] Stopping sweep due to failure (continue_on_failure=false)")
                break
        
        # Save progress
        if not dry_run:
            save_progress(progress_file, progress)
        
        # Delay between runs (skip after last)
        if i < len(experiments) - 1 and not dry_run and delay > 0:
            print(f"\nWaiting {delay}s before next experiment...")
            time.sleep(delay)
    
    # Summary
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


def main():
    parser = argparse.ArgumentParser(
        description="Run a data filtering hyperparameter sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what will run (no actual execution)
  python scripts/run_data_sweep.py --config configs/data_sweep.yml --dry-run

  # Run the full sweep
  python scripts/run_data_sweep.py --config configs/data_sweep.yml

  # Resume an interrupted sweep
  python scripts/run_data_sweep.py --config configs/data_sweep.yml --resume

Monitoring in ClearML:
  1. Go to your ClearML dashboard
  2. Navigate to Projects > "Engagement Prediction"
  3. Filter by tags: "data-sweep"
  4. Select multiple experiments and click "Compare"
  5. View metrics like:
     - get_data/n_users_final
     - get_data/n_likes_core  
     - get_data/memory_peak_gb
        """,
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to sweep configuration YAML file',
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Print what would run without executing',
    )
    parser.add_argument(
        '--resume', '-r',
        action='store_true',
        help='Resume an interrupted sweep (skip completed experiments)',
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Reduce output verbosity',
    )
    
    args = parser.parse_args()
    
    try:
        config = load_sweep_config(args.config)
        result = run_sweep(
            config,
            dry_run=args.dry_run,
            resume=args.resume,
            verbose=not args.quiet,
        )
        return 0 if result['failed'] == 0 else 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
