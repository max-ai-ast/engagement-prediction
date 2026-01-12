#!/usr/bin/env python3

"""
Unified CLI for Engagement Prediction Pipeline
=============================================

Subcommands:
- preprocess: Run data preprocessing and save processed dataset
- train: Train model using preprocessed data
- evaluate: Evaluate trained model on held-out users
- train-eval: Train using an embedding bundle + user splits, then run full-feed evaluation

Usage examples:
    python cli.py preprocess --days 5 --min-likes 5 --prediction-posts-per-user 2
    python cli.py train --load-processed auto --epochs 150
    python cli.py evaluate --model-path auto --data-path auto --create-plots
    python cli.py train-eval \
        --embedding-bundle /srv/vox/engagement_prediction/wills_tinkering_folder/outputs/<ts>_run_.../precompute/embedding_bundle_<ts>.pkl \
        --user-splits /srv/vox/engagement_prediction/wills_tinkering_folder/outputs/<ts>_run_.../relevel/user_splits.json
"""

import argparse
import sys
import subprocess
from pathlib import Path
from typing import Optional
import os
import json
import time
import importlib.util

# Avoid heavy imports at module import time; import lazily inside handlers

TINKERING_DIR = Path(__file__).parent
OUTPUTS_DIR = TINKERING_DIR / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
PROCESSED_DATA_DIR = TINKERING_DIR / "processed_data"
RESULTS_DIR = OUTPUTS_DIR / "holdout_evaluation_results"

# Mirror defaults to avoid importing training module for help text
DEFAULT_HIDDEN_DIMS = [64, 32, 16]
DEFAULT_DROPOUT_RATE = 0.5
DEFAULT_WEIGHT_DECAY = 0.1
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_EPOCHS = 300
DEFAULT_PATIENCE = 10
DEFAULT_DEVICE = "cuda" if False else "cpu"  # will be overridden by training module at runtime

DEFAULT_GCS_BUCKET = 'greenearth-471522-ingex-extract-stage'

def cmd_run_all(args) -> int:
    """Run the 4-stage pipeline.

    Creates a run directory up front and backgrounds itself with nohup unless --foreground.
    """
    outputs_dir = OUTPUTS_DIR
    outputs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Create run_dir deterministically up front
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        d_part = f"d{int(args.max_files_per_table)}"
        cap_part = f"mppa{int(args.max_posts_per_author)}"
        suffix = f"{d_part}_{cap_part}"
        if args.run_name:
            rn = str(args.run_name).strip().replace(' ', '_')
            if rn:
                suffix = f"{suffix}_{rn}"
        run_dir = outputs_dir / f"{timestamp}_run_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Choose log path inside run_dir
    initial_log = Path(args._initial_log) if args._initial_log else (run_dir / "run-all.log")
    try:
        initial_log.parent.mkdir(parents=True, exist_ok=True)
        with open(initial_log, 'a') as f:
            f.write(f"run-all started at {timestamp}\n")
    except Exception:
        pass

    if not args.foreground:
        # Background via nohup by re-invoking run-all with --foreground and pinned --output-dir
        import shlex
        cli_args = []
        for k, v in vars(args).items():
            if k in ("command", "foreground", "_initial_log", "output_dir", "func"):
                continue
            if v is None or v is False:
                continue
            opt = f"--{k.replace('_','-')}"
            if isinstance(v, bool):
                cli_args.append(opt)
            elif isinstance(v, list):
                cli_args.extend([opt] + [str(x) for x in v])
            else:
                cli_args.extend([opt, str(v)])
        cli_args.extend(["--foreground", "--_initial-log", str(initial_log), "--output-dir", str(run_dir.resolve())])

        py = shlex.quote(sys.executable)
        script = shlex.quote(str(Path(__file__).resolve()))
        args_str = ' '.join(shlex.quote(a) for a in (["run-all"] + cli_args))
        redir = shlex.quote(str(initial_log))
        cmd = f"nohup {py} {script} {args_str} > {redir} 2>&1 & echo $!"
        print(f"▶️  Backgrounding run-all with nohup. Log: {initial_log}")
        import subprocess as sp
        proc = sp.run(["bash", "-lc", cmd], stdout=sp.PIPE, stderr=sp.PIPE, text=True)
        if proc.returncode == 0:
            pid_str = (proc.stdout or "").strip().splitlines()[-1] if (proc.stdout or "").strip() else None
            pid_file = run_dir / "run-all.pid"
            if pid_str and pid_str.isdigit():
                try:
                    with open(pid_file, "w") as f:
                        f.write(pid_str + "\n")
                except Exception:
                    pass
                print(f"✅ run-all started in background (PID {pid_str}). Kill with: kill {pid_str}\n📝 PID file: {pid_file}")
            else:
                print("✅ run-all started in background")
            return 0
        print("❌ Failed to start run-all in background")
        return proc.returncode or 1

    # Foreground execution: call the internal exec directly
    # Ensure args.output_dir is set so subsequent stages use this run_dir
    if not args.output_dir:
        setattr(args, 'output_dir', str(run_dir.resolve()))
    return cmd__run_all_exec(args)


def cmd__run_all_exec(args) -> int:
    """Execute the 6-stage modular pipeline in the foreground sequentially."""
    # Build Context and invoke stages via registry
    from utils.pipeline.core import Context
    from utils.pipeline import registry as reg

    run_dir = Path(args.output_dir).resolve()
    # In sequential execution, always allow stages to resolve latest artifacts from prior stages
    ctx = Context(run_dir=run_dir, use_latest=True)

    # Helper: map stage keys to enumerated folder names
    # Override relevel stage key if --relevel-method is specified
    relevel_method = getattr(args, 'relevel_method', None)
    relevel_key = 'relevel'  # default
    if relevel_method == 'gini':
        relevel_key = 'relevel_gini'
    elif relevel_method == 'simple':
        relevel_key = 'relevel_simple'
    elif relevel_method == 'uniform':
        relevel_key = 'relevel'
    
    # Override train stage key if --model-type is specified
    model_type = getattr(args, 'model_type', 'mlp')
    train_key = 'train'  # default MLP
    if model_type == 'two-tower':
        train_key = 'train_two_tower'
    
    stage_order = ['get_data', 'featurize', relevel_key, 'split', train_key, 'evaluate']
    stage_folder = {}
    for key in stage_order:
        _mp, _folder = reg.get_stage_spec(key)
        stage_folder[key] = _folder

    # Respect selective reruns
    start_from = getattr(args, 'start_from', None)
    stop_after = getattr(args, 'stop_after', None)
    # Map 'relevel' to the actual relevel key if specified
    if start_from == 'relevel':
        start_from = relevel_key
    if stop_after == 'relevel':
        stop_after = relevel_key
    # Map 'train' to the actual train key (train or train_two_tower)
    if start_from == 'train':
        start_from = train_key
    if stop_after == 'train':
        stop_after = train_key
    start_idx = stage_order.index(start_from) if start_from in stage_order else 0
    stop_idx = stage_order.index(stop_after) if stop_after in stage_order else (len(stage_order) - 1)

    # Pin prior outputs if provided
    def _pin_prior(arg_name: str, stage_key: str):
        path_str = getattr(args, arg_name, None)
        if path_str:
            p = Path(path_str)
            if p.exists():
                ctx.prior_outputs[stage_folder[stage_key]] = p

    _pin_prior('prior_get_data', 'get_data')
    _pin_prior('prior_featurize', 'featurize')
    _pin_prior('prior_relevel', relevel_key)  # Use the selected relevel key
    _pin_prior('prior_split', 'split')
    _pin_prior('prior_train', 'train')

    # Optional interactive chooser (foreground only)
    def _maybe_choose_prior(stage_key: str):
        if not getattr(args, 'pick_prior', False):
            return
        folder = stage_folder[stage_key]
        base = (run_dir / folder)
        if not base.exists():
            return
        subdirs = [p for p in base.iterdir() if p.is_dir()]
        if len(subdirs) <= 1:
            return
        # Prompt only in foreground mode
        if not bool(getattr(args, 'foreground', False)):
            return
        subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"\nPick prior output for stage '{stage_key}' under {base}:")
        for i, p in enumerate(subdirs):
            print(f"  [{i}] {p.name}")
        try:
            choice = input("Enter index (blank for latest): ").strip()
            if choice:
                idx = int(choice)
                if 0 <= idx < len(subdirs):
                    ctx.prior_outputs[folder] = subdirs[idx]
        except Exception:
            pass

    # Execute selected subset
    for idx, key in enumerate(stage_order):
        if idx < start_idx or idx > stop_idx:
            continue
        # Before running, offer prior selection for this stage's dependency (if any)
        if key != 'get_data':
            prev_key = stage_order[idx - 1]
            if stage_folder[prev_key] not in ctx.prior_outputs:
                _maybe_choose_prior(prev_key)
        label_map = {
            'get_data': "Stage 1: Get data…",
            'featurize': "Stage 2: Featurize…",
            'relevel': "Stage 3: Relevel (uniform mixture)…",
            'relevel_gini': "Stage 3: Relevel (Gini-optimized)…",
            'relevel_simple': "Stage 3: Relevel (simple)…",
            'split': "Stage 4: Split users…",
            'train': "Stage 5: Train model (MLP)…",
            'train_two_tower': "Stage 5: Train model (Two-Tower)…",
            'evaluate': "Stage 6: Evaluate model…",
        }
        label = label_map.get(key, f"Stage {idx+1}: {key}…")
        print(f"\n[{idx+1}/6] ▶️  {label}")
        reg.run_stage(key, ctx, args)

    print("\n✅ run-all completed successfully")
    return 0
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Engagement Prediction Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run-all (modular 6-stage end-to-end)
    p_all = subparsers.add_parser("run-all", help="Run all 6 stages end-to-end. Defaults to background with nohup.")
    # Stage 1 options
    p_all.add_argument("--data-source", type=str, choices=["greenearth", "digitalocean"], default="greenearth")
    p_all.add_argument("--gcs-bucket", type=str, default=DEFAULT_GCS_BUCKET, help="GCS bucket name for ingex data")
    p_all.add_argument("--posts-start", type=str, default=None, help="ISO date string for ingex GCS posts start (inclusive)")
    p_all.add_argument("--posts-end", type=str, default=None, help="ISO date string for ingex GCS posts end (exclusive)")
    p_all.add_argument("--likes-start", type=str, default=None, help="ISO date string for ingex GCS likes start (inclusive)")
    p_all.add_argument("--likes-end", type=str, default=None, help="ISO date string for ingex GCS likes end (exclusive)")
    p_all.add_argument("--max-files-per-table", type=int, default=5)
    p_all.add_argument("--image-mode", type=str, choices=["auto", "off", "on"], default="auto")
    p_all.add_argument("--max-posts-per-author", type=int, default=3)
    p_all.add_argument("--max-liked-posts-per-user", type=int, default=100)
    p_all.add_argument("--cap-random-seed", type=int, default=42)
    p_all.add_argument("--output-dir", type=str, default=None, help="Optional explicit run directory root")
    p_all.add_argument("--run-name", type=str, default=None, help="Optional suffix for Stage 1 run dir name")
    p_all.add_argument("--debug", action="store_true", help="Enable verbose debug logging for Stage 1")
    # Stage 2 options
    p_all.add_argument("--global-topic-k", type=int, default=20)
    p_all.add_argument("--relevel-method", type=str, choices=["uniform", "gini", "simple"], default="uniform",
                      help="Which relevel script to use: 'uniform' (default), 'gini' (Gini-optimized), or 'simple'")
    p_all.add_argument("--relevel-strategy", type=str, choices=["none", "uniform_mixture_balanced"], default="uniform_mixture_balanced")
    p_all.add_argument("--relevel-alpha", type=float, default=0.35)
    p_all.add_argument("--relevel-min-users-per-topic", type=int, default=0)
    p_all.add_argument("--min-likes-per-user", type=int, default=10)
    p_all.add_argument("--val-ratio", type=float, default=0.2)
    p_all.add_argument("--holdout-ratio", type=float, default=0.2)
    p_all.add_argument("--random-seed", type=int, default=42)
    p_all.add_argument("--embedding-model", type=str, choices=["all_MiniLM_L6_v2", "all_MiniLM_L12_v2"], default='all_MiniLM_L6_v2', help="The SentenceTransformers model to use to look up precomputed embeddings or generate them.")
    # Stage 5 (train) model selection
    p_all.add_argument("--model-type", type=str, choices=["mlp", "two-tower"], default="mlp",
                      help="Model architecture: 'mlp' (default MLP) or 'two-tower' (two-tower with attention)")
    # Two-tower specific options
    p_all.add_argument("--shared-dim", type=int, default=128, help="Two-tower shared embedding dimension")
    p_all.add_argument("--user-hidden-dim", type=int, default=256, help="Two-tower user encoder hidden dimension")
    p_all.add_argument("--post-hidden-dim", type=int, default=256, help="Two-tower post encoder hidden dimension")
    p_all.add_argument("--num-attention-heads", type=int, default=4, help="Two-tower attention heads")
    p_all.add_argument("--num-attention-layers", type=int, default=2, help="Two-tower attention layers")
    p_all.add_argument("--max-history-len", type=int, default=20, help="Two-tower max user history length")
    # Stage 5 options (shared)
    p_all.add_argument("--epochs", type=int, default=300)
    p_all.add_argument("--batch-size", type=int, default=256)
    p_all.add_argument("--learning-rate", type=float, default=0.001)
    p_all.add_argument("--weight-decay", type=float, default=0.1)
    p_all.add_argument("--hidden-dims", type=int, nargs="+", default=None)
    p_all.add_argument("--dropout-rate", type=float, default=0.5)
    p_all.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p_all.add_argument("--patience", type=int, default=50, help="Early stopping patience")
    p_all.add_argument("--no-plots", action="store_true")
    p_all.add_argument("--no-save-model", action="store_true")
    # Stage 4 options (subset)
    p_all.add_argument("--eval-batch-size", type=int, default=8192)
    p_all.add_argument("--eval-max-users", type=int, default=0, help="0 = all eligible")
    # Selection behavior
    p_all.add_argument("--use-latest", action="store_true", help="(Deprecated) Always enabled during sequential run-all")
    # Selective reruns and prior pinning
    p_all.add_argument("--start-from", type=str, choices=["get_data","featurize","relevel","split","train","evaluate"], default=None,
                      help="Begin execution at this stage")
    p_all.add_argument("--stop-after", type=str, choices=["get_data","featurize","relevel","split","train","evaluate"], default=None,
                      help="Stop after this stage completes")
    p_all.add_argument("--prior-get-data", type=str, default=None, help="Path to a specific 01_get_data/<ts> directory")
    p_all.add_argument("--prior-featurize", type=str, default=None, help="Path to a specific 02_featurize/<ts> directory")
    p_all.add_argument("--prior-relevel", type=str, default=None, help="Path to a specific 03_relevel/<ts> directory")
    p_all.add_argument("--prior-split", type=str, default=None, help="Path to a specific 04_split/<ts> directory")
    p_all.add_argument("--prior-train", type=str, default=None, help="Path to a specific 05_train/<ts> directory")
    p_all.add_argument("--pick-prior", action="store_true", help="If multiple prior outputs exist, prompt to pick (foreground only)")
    # Execution behavior
    p_all.add_argument("--foreground", action="store_true", help="Run in foreground (default: background with nohup)")
    p_all.add_argument("--_initial-log", type=str, default=None, help=argparse.SUPPRESS)
    p_all.set_defaults(func=cmd_run_all)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main()) 