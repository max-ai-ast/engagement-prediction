#!/usr/bin/env python3

"""
Unified CLI for Engagement Prediction Pipeline
=============================================

Subcommands:
- run-all: Run the 6-stage pipeline end-to-end (ingest → featurize → relevel → split → train → evaluate)

Usage examples:
    python cli.py run-all --epochs 150 --embedding-model all_MiniLM_L12_v2
    python cli.py run-all --config configs/pipeline.yml --foreground
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time
import copy

# Avoid heavy imports at module import time; import lazily inside handlers

TINKERING_DIR = Path(__file__).parent
OUTPUTS_DIR = TINKERING_DIR / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
PROCESSED_DATA_DIR = TINKERING_DIR / "processed_data"
RESULTS_DIR = OUTPUTS_DIR / "holdout_evaluation_results"

# Central default map for all run-all parameters
DEFAULTS: Dict[str, Any] = {
    # Stage 1
    "data_source": "greenearth",
    "gcs_bucket": 'greenearth-471522-ingex-extract-stage',
    "posts_start": None,
    "posts_end": None,
    "likes_start": None,
    "likes_end": None,
    "max_files_per_table": 5,
    "image_mode": "auto",
    "max_posts_per_author": 3,
    "max_liked_posts_per_user": 100,
    "cap_random_seed": 42,
    "output_dir": None,
    "run_name": None,
    "debug": False,
    # Stage 2/3/4
    "global_topic_k": 20,
    "relevel_method": "uniform",
    "relevel_strategy": "uniform_mixture_balanced",
    "relevel_alpha": 0.35,
    "relevel_min_users_per_topic": 0,
    "min_likes_per_user": 10,
    "val_ratio": 0.2,
    "holdout_ratio": 0.2,
    "random_seed": 42,
    "embedding_model": "all_MiniLM_L6_v2",
    # Stage 5 (train)
    "model_type": "mlp",
    "shared_dim": 128,
    "user_hidden_dim": 256,
    "post_hidden_dim": 256,
    "num_attention_heads": 4,
    "num_attention_layers": 2,
    "max_history_len": 20,
    "epochs": 300,
    "batch_size": 256,
    "learning_rate": 0.001,
    "weight_decay": 0.1,
    "hidden_dims": [64, 32, 16],
    "dropout_rate": 0.5,
    "device": "cpu",
    "patience": 50,
    "no_plots": False,
    "no_save_model": False,
    # Stage 6 (eval)
    "eval_batch_size": 8192,
    "eval_max_users": 0,
    # Selection/prior behavior
    "use_latest": False,
    "start_from": None,
    "stop_after": None,
    "prior_get_data": None,
    "prior_featurize": None,
    "prior_relevel": None,
    "prior_split": None,
    "prior_train": None,
    "pick_prior": False,
    # Execution behavior
    "foreground": False,
    "_initial_log": None,
}


def _help_with_default(text: Optional[str], key: str) -> Optional[str]:
    """Append default value text without duplicating default assignments."""
    default_val = DEFAULTS.get(key, None)
    if text is None:
        text = ""
    if default_val is None:
        return text
    return f"{text} (default: {default_val})"


def _load_config_file(path_str: str) -> Dict[str, Any]:
    """Load a YAML (or JSON) config file mapping CLI args to values."""
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None  # type: ignore
    if yaml is not None:
        data = yaml.safe_load(path.read_text())
    else:
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise RuntimeError("PyYAML is not installed and the config file is not valid JSON") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping of argument names to values")
    # Normalize kebab-case to snake_case to match argparse dest names
    return {k.replace("-", "_"): v for k, v in data.items()}


def _merge_args_with_config(raw_args: argparse.Namespace) -> argparse.Namespace:
    """Apply defaults, then config file values, then CLI overrides."""
    args_dict = vars(raw_args).copy()
    command = args_dict.get("command")
    func = args_dict.get("func")
    config_path = args_dict.pop("config", None)
    config_data: Dict[str, Any] = {}
    if config_path:
        config_data = _load_config_file(config_path)

    merged: Dict[str, Any] = copy.deepcopy(DEFAULTS)
    if config_data:
        unknown_keys = set(config_data.keys()) - set(DEFAULTS.keys())
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown_keys))}")
        merged.update(config_data)
    merged.update({k: v for k, v in args_dict.items() if k not in ("command", "func")})
    final_ns = argparse.Namespace(**merged)
    # Preserve argparse-injected metadata
    setattr(final_ns, "command", command)
    setattr(final_ns, "func", func)
    return final_ns

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
    parser = argparse.ArgumentParser(
        description="Engagement Prediction Pipeline CLI",
        argument_default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="YAML/JSON config file with run-all parameters (CLI flags override config)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run-all (modular 6-stage end-to-end)
    p_all = subparsers.add_parser("run-all", help="Run all 6 stages end-to-end. Defaults to background with nohup.")
    # Stage 1 options
    p_all.add_argument("--data-source", type=str, choices=["greenearth", "digitalocean"], default=argparse.SUPPRESS,
                      help=_help_with_default("Source for raw input data - posts and likes", "data_source"))
    p_all.add_argument("--gcs-bucket", type=str, default=argparse.SUPPRESS,
                      help=_help_with_default("GCS bucket name for ingex data", "gcs_bucket"))
    p_all.add_argument("--posts-start", type=str, default=argparse.SUPPRESS,
                      help=_help_with_default("ISO date string for ingex GCS posts start (inclusive)", "posts_start"))
    p_all.add_argument("--posts-end", type=str, default=argparse.SUPPRESS,
                      help=_help_with_default("ISO date string for ingex GCS posts end (exclusive)", "posts_end"))
    p_all.add_argument("--likes-start", type=str, default=argparse.SUPPRESS,
                      help=_help_with_default("ISO date string for ingex GCS likes start (inclusive)", "likes_start"))
    p_all.add_argument("--likes-end", type=str, default=argparse.SUPPRESS,
                      help=_help_with_default("ISO date string for ingex GCS likes end (exclusive)", "likes_end"))
    p_all.add_argument("--max-files-per-table", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Maximum files to read per ingex table", "max_files_per_table"))
    p_all.add_argument("--image-mode", type=str, choices=["auto", "off", "on"], default=argparse.SUPPRESS,
                      help=_help_with_default("Control image handling during data pull", "image_mode"))
    p_all.add_argument("--max-posts-per-author", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Cap on posts per author during ingestion", "max_posts_per_author"))
    p_all.add_argument("--max-liked-posts-per-user", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Cap on liked posts per user during ingestion", "max_liked_posts_per_user"))
    p_all.add_argument("--cap-random-seed", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Random seed for ingestion capping", "cap_random_seed"))
    p_all.add_argument("--output-dir", type=str, default=argparse.SUPPRESS,
                      help=_help_with_default("Optional explicit run directory root", "output_dir"))
    p_all.add_argument("--run-name", type=str, default=argparse.SUPPRESS,
                      help=_help_with_default("Optional suffix for Stage 1 run dir name", "run_name"))
    p_all.add_argument("--debug", action="store_true", default=argparse.SUPPRESS,
                      help=_help_with_default("Enable verbose debug logging for Stage 1", "debug"))
    # Stage 2 options
    p_all.add_argument("--global-topic-k", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Number of global topics", "global_topic_k"))
    p_all.add_argument("--relevel-method", type=str, choices=["uniform", "gini", "simple"], default=argparse.SUPPRESS,
                      help=_help_with_default("Which relevel script to use: uniform, gini, or simple", "relevel_method"))
    p_all.add_argument("--relevel-strategy", type=str, choices=["none", "uniform_mixture_balanced"], default=argparse.SUPPRESS,
                      help=_help_with_default("Relevel weighting strategy", "relevel_strategy"))
    p_all.add_argument("--relevel-alpha", type=float, default=argparse.SUPPRESS,
                      help=_help_with_default("Alpha parameter for relevel weighting", "relevel_alpha"))
    p_all.add_argument("--relevel-min-users-per-topic", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Minimum users per topic when releveling", "relevel_min_users_per_topic"))
    p_all.add_argument("--min-likes-per-user", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Minimum likes per user for inclusion", "min_likes_per_user"))
    p_all.add_argument("--val-ratio", type=float, default=argparse.SUPPRESS,
                      help=_help_with_default("Validation ratio", "val_ratio"))
    p_all.add_argument("--holdout-ratio", type=float, default=argparse.SUPPRESS,
                      help=_help_with_default("Holdout ratio", "holdout_ratio"))
    p_all.add_argument("--random-seed", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Random seed for splitting", "random_seed"))
    p_all.add_argument("--embedding-model", type=str, choices=["all_MiniLM_L6_v2", "all_MiniLM_L12_v2"], default=argparse.SUPPRESS,
                      help=_help_with_default("SentenceTransformers model for embeddings", "embedding_model"))
    # Stage 5 (train) model selection
    p_all.add_argument("--model-type", type=str, choices=["mlp", "two-tower"], default=argparse.SUPPRESS,
                      help=_help_with_default("Model architecture: mlp or two-tower", "model_type"))
    # Two-tower specific options
    p_all.add_argument("--shared-dim", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Two-tower shared embedding dimension", "shared_dim"))
    p_all.add_argument("--user-hidden-dim", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Two-tower user encoder hidden dimension", "user_hidden_dim"))
    p_all.add_argument("--post-hidden-dim", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Two-tower post encoder hidden dimension", "post_hidden_dim"))
    p_all.add_argument("--num-attention-heads", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Two-tower attention heads", "num_attention_heads"))
    p_all.add_argument("--num-attention-layers", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Two-tower attention layers", "num_attention_layers"))
    p_all.add_argument("--max-history-len", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Two-tower max user history length", "max_history_len"))
    # Stage 5 options (shared)
    p_all.add_argument("--epochs", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Training epochs", "epochs"))
    p_all.add_argument("--batch-size", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Training batch size", "batch_size"))
    p_all.add_argument("--learning-rate", type=float, default=argparse.SUPPRESS,
                      help=_help_with_default("Learning rate", "learning_rate"))
    p_all.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS,
                      help=_help_with_default("Weight decay", "weight_decay"))
    p_all.add_argument("--hidden-dims", type=int, nargs="+", default=argparse.SUPPRESS,
                      help=_help_with_default("Hidden layer sizes", "hidden_dims"))
    p_all.add_argument("--dropout-rate", type=float, default=argparse.SUPPRESS,
                      help=_help_with_default("Dropout rate", "dropout_rate"))
    p_all.add_argument("--device", type=str, default=argparse.SUPPRESS,
                      help=_help_with_default("Device for training", "device"))
    p_all.add_argument("--patience", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Early stopping patience", "patience"))
    p_all.add_argument("--no-plots", action="store_true", default=argparse.SUPPRESS,
                      help=_help_with_default("Disable training plots", "no_plots"))
    p_all.add_argument("--no-save-model", action="store_true", default=argparse.SUPPRESS,
                      help=_help_with_default("Skip saving model checkpoints", "no_save_model"))
    # Stage 4 options (subset)
    p_all.add_argument("--eval-batch-size", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("Batch size for evaluation", "eval_batch_size"))
    p_all.add_argument("--eval-max-users", type=int, default=argparse.SUPPRESS,
                      help=_help_with_default("0 = all eligible users for evaluation", "eval_max_users"))
    # Selection behavior
    p_all.add_argument("--use-latest", action="store_true", default=argparse.SUPPRESS,
                      help=_help_with_default("(Deprecated) Always enabled during sequential run-all", "use_latest"))
    # Selective reruns and prior pinning
    p_all.add_argument("--start-from", type=str, choices=["get_data","featurize","relevel","split","train","evaluate"], default=argparse.SUPPRESS,
                      help=_help_with_default("Begin execution at this stage", "start_from"))
    p_all.add_argument("--stop-after", type=str, choices=["get_data","featurize","relevel","split","train","evaluate"], default=argparse.SUPPRESS,
                      help=_help_with_default("Stop after this stage completes", "stop_after"))
    p_all.add_argument("--prior-get-data", type=str, default=argparse.SUPPRESS,
                      help=_help_with_default("Path to a specific 01_get_data/<ts> directory", "prior_get_data"))
    p_all.add_argument("--prior-featurize", type=str, default=argparse.SUPPRESS,
                      help=_help_with_default("Path to a specific 02_featurize/<ts> directory", "prior_featurize"))
    p_all.add_argument("--prior-relevel", type=str, default=argparse.SUPPRESS,
                      help=_help_with_default("Path to a specific 03_relevel/<ts> directory", "prior_relevel"))
    p_all.add_argument("--prior-split", type=str, default=argparse.SUPPRESS,
                      help=_help_with_default("Path to a specific 04_split/<ts> directory", "prior_split"))
    p_all.add_argument("--prior-train", type=str, default=argparse.SUPPRESS,
                      help=_help_with_default("Path to a specific 05_train/<ts> directory", "prior_train"))
    p_all.add_argument("--pick-prior", action="store_true", default=argparse.SUPPRESS,
                      help=_help_with_default("If multiple prior outputs exist, prompt to pick (foreground only)", "pick_prior"))
    # Execution behavior
    p_all.add_argument("--foreground", action="store_true", default=argparse.SUPPRESS,
                      help=_help_with_default("Run in foreground (default: background with nohup)", "foreground"))
    p_all.add_argument("--_initial-log", type=str, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    p_all.set_defaults(func=cmd_run_all)

    return parser


def main() -> int:
    parser = build_parser()
    raw_args = parser.parse_args()
    merged_args = _merge_args_with_config(raw_args)
    return merged_args.func(merged_args)


if __name__ == "__main__":
    sys.exit(main()) 
