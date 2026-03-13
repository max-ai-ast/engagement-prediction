#!/usr/bin/env python3

"""
Unified CLI for Engagement Prediction Pipeline
=============================================

Runs the 5-stage pipeline end-to-end (get_data → target_posts → user_history → train → evaluate).

Note: The historical `run-all` subcommand is now optional (kept for backwards compatibility).

Usage examples:
    python cli.py --user-encoder summarized --epochs 150 --embedding-model all_MiniLM_L12_v2
    python cli.py --user-encoder full_transformer --model-type two-tower --config config.yml
"""

import argparse
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
import copy

from utils.experiment_tracking import build_experiment_tracker
from utils.pipeline.core import (
    Context,
    generate_run_timestamp,
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_RUNS_DIR,
    new_pipeline_run_dir,
    ensure_pipeline_run_dir,
    update_latest_symlink,
)


# Avoid heavy imports at module import time; import lazily inside handlers

CLI_FILE_DIR = Path(__file__).parent

# Central default map for all run-all parameters
DEFAULTS: Dict[str, Any] = {
    # Stage 1: Data filtering
    "gcs_bucket": 'greenearth-471522-ingex-extract-stage',
    "posts_start": None,
    "posts_end": None,
    "likes_start": None,
    "likes_end": None,
    "max_liking_users": None,  # None = no limit; sample this many unique liking users
    "max_likes_per_user": 100,  # Stage 1: random cap on likes per user (NOT recency-based)
    "min_likes_per_user": 2,  # Stage 1: minimum likes for user inclusion
    "negative_posts_sample": 100000,  # Stage 1: random posts for negative cases
    "cap_random_seed": 42,
    "max_memory_gb": None,  # Stage 1: max memory in GB (None = auto based on percentage)
    "max_memory_pct": 0.75,  # Stage 1: max percentage of available RAM to use
    "memory_check": "full",  # Stage 1: memory check mode (full/ignore/skip)
    "output_dir": None,
    "debug": False,
    "random_seed": 42,
    "embedding_model": "all_MiniLM_L6_v2",
    "skip_embeddings": False,
    # Stage 2 Target posts and Split
    "max_prior_likes": None,  # Stage 3: cap on prior likes per target for user history (None = no cap)
    "history_buffer_hours": None,  # Stage 3: buffer in hours between seen_at and prior-like cutoff (None = no buffer)
    "neg_sample_bucket": "1h",
    "train_start": None,
    "val_start": None,
    "holdout_user_fraction": 0.2,
    "holdout_user_seed": 42,
    "holdout_start": None,
    "holdout_end": None,
    # Stage 4 (train) - Model architecture
    "user_summarization": "mean",  # MLP user-history summarization: mean, ema, linear_recency
    "ema_alpha": 0.1,  # EMA smoothing factor (only used when user_summarization=ema)
    "user_encoder": "summarized",  # User encoder type: must be explicitly specified and compatible with model_type
    "model_type": "mlp",
    "shared_dim": 128,
    "user_hidden_dim": 256,
    "user_output_dim": 128,  # Output dim for MLPModel's user encoder in full_transformer mode; separate from shared_dim used in TwoTower
    "use_post_encoder": True,  # True means using a transformation on the post embedding (e.g. single layer neural net). False uses the post embedding directly.
    "post_hidden_dim": 256,
    "num_attention_heads": 4,
    "num_attention_layers": 2,
    "max_history_len": 20,
    "attention_dropout": 0.1,  # Dropout rate for attention-based user encoders
    "epochs": 300,
    "batch_size": 256,
    "learning_rate": 0.001,
    "weight_decay_mlp": 0.1,
    "weight_decay_two_tower": 0.01,
    "hidden_dims": [64, 32, 16],
    "dropout_rate_mlp": 0.5,
    "dropout_rate_two_tower": 0.1,
    "prediction_posts_per_user": 1,
    "device": None,
    "patience": 50,
    "run_tag": None,  # Optional tag appended to training output directory name
    "no_plots": False,
    "no_save_model": False,
    "disable_progress": False,  # Disable progress bars during training
    # Stage 4 (train) - DataLoader settings
    "num_dataloader_workers": 4,
    "dataloader_pin_memory": True,
    "dataloader_persistent_workers": True,
    "dataloader_prefetch_factor": 2,
    # Stage 4 (train) - Learning rate scheduler
    "lr_scheduler_factor": 0.5,
    "lr_scheduler_patience": 5,
    # Stage 4 (train) - Training optimization
    "gradient_clip_max_norm": 1.0,
    # Stage 5 (eval)
    "eval_batch_size": 8192,
    "eval_holdout_type": "unseen_users",
    "skip_modules": None,  # Comma-separated eval module names to skip (None = run all)
    # Selection/prior behavior
    "use_latest": False,
    "start_from": None,
    "stop_after": None,
    "pick_prior": False,
    # Prior pins (optional): may be a stage_run_id (dir name under artifacts/<stage>/)
    # or a path (absolute, or relative to --output-dir).
    "prior_01_get_data": None,
    "prior_02_target_posts": None,
    "prior_03_user_history": None,
    "prior_04_train": None,
    # Execution behavior
    # Default is foreground execution (recommended for ClearML remote execution).
    "background": False,
    "_initial_log": None,
    # Experiment tracking
    "experiment_tracker": "clearml",
    "experiment_project": "Engagement Prediction",
    "experiment_task": None,
    "experiment_tags": None,
    # ClearML / model registry
    # If set, used as ClearML Task output URI (e.g. gs://...); if None, ClearML uses its default output.
    "model_output_uri": 'gs://greenearth-471522-engagement-prediction-test',
}


def _help_with_default(text: Optional[str], key: str) -> Optional[str]:
    """Append default value text without duplicating default assignments."""
    default_val = DEFAULTS.get(key, None)
    if text is None:
        text = ""
    if default_val is None:
        return text
    return f"{text} (default: {default_val})"


def _arg_key_from_flag(flag: str) -> str:
    """Convert a CLI flag (e.g., --posts-start) to the DEFAULTS key."""
    return flag.lstrip("-").replace("-", "_")


def _add_arg_with_default(parser: argparse.ArgumentParser, flag: str, *, key: Optional[str] = None,
                          help_text: Optional[str] = None, **kwargs: Any) -> None:
    """Add an argument with standardized default-aware help text."""
    if help_text is not None:
        effective_key = key or _arg_key_from_flag(flag)
        kwargs["help"] = _help_with_default(help_text or "", effective_key)
    parser.add_argument(flag, **kwargs)


def _extract_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for key, default in DEFAULTS.items():
        if hasattr(args, key):
            value = getattr(args, key)
            if value != default:
                overrides[key] = value
    return overrides


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


def _build_effective_config_for_background_run(
    args: argparse.Namespace, *, output_root: Path, initial_log: Path
) -> Dict[str, Any]:
    """Materialize an effective config to re-invoke run-all in the background.

    We prefer passing a config file rather than reconstructing CLI flags from
    argparse dest names, since some args intentionally use a different flag name
    (e.g. `use_post_encoder` is controlled via `--post-encoder/--no-post-encoder`).
    """
    cfg: Dict[str, Any] = {k: getattr(args, k) for k in DEFAULTS.keys()}
    cfg["output_dir"] = str(Path(output_root).resolve())
    cfg["_initial_log"] = str(initial_log)
    # Prevent recursive backgrounding: the child process should run in the foreground.
    cfg["background"] = False
    return cfg


def _generate_run_name(args: argparse.Namespace) -> str:
    stages_str = "all"
    if args.start_from is not None or args.stop_after is not None:
        if args.start_from == args.stop_after:
            stages_str = args.start_from
        else:
            if args.start_from is None:
                stages_str = "start_to_"
            else:
                stages_str = f"{args.start_from}_to_"
            if args.stop_after is None:
                stages_str += "end"
            else:
                stages_str += args.stop_after

    stages_str += f"_{args.model_type}"
    return stages_str


def _resolve_run_dir(args: argparse.Namespace, run_timestamp: str) -> Path:
    """Resolve the output root directory as an absolute path.

    ClearML remote execution may run with a different working directory than local runs.
    If `--output-dir` is provided as a relative path, interpret it relative to the repo root
    (this file's directory) to keep behavior stable across environments.
    """
    output_dir = args.output_dir
    if output_dir:
        p = Path(str(output_dir)).expanduser()
        if not p.is_absolute():
            p = (CLI_FILE_DIR / p)
        return p.resolve()
    return (CLI_FILE_DIR / "outputs").resolve()


def _resolve_pipeline_run_dir(args: argparse.Namespace, *, output_root: Path, run_timestamp: str) -> Path:
    runs_dir = (Path(output_root) / "runs").resolve()
    pinned = (os.environ.get("ENGAGEMENT_PIPELINE_RUN_ID") or "").strip()
    if pinned:
        return ensure_pipeline_run_dir(runs_dir, pipeline_run_id=pinned).resolve()
    run_name = _generate_run_name(args)
    base_name = f"{run_timestamp}_{run_name}"
    return new_pipeline_run_dir(runs_dir, base_name=base_name).resolve()


def _resolve_prior_spec(
    spec: Optional[str],
    *,
    output_root: Path,
    artifacts_dir: Path,
    stage_folder: str,
) -> Optional[Path]:
    """Resolve a prior pin to a concrete artifact directory path.

    `spec` may be:
      - an absolute path
      - a path relative to output_root
      - a stage_run_id (directory name under artifacts/<stage_folder>/)
    """
    if spec is None:
        return None
    s = str(spec).strip()
    if not s:
        return None

    p = Path(s).expanduser()
    candidate = p if p.is_absolute() else (Path(output_root) / p)
    if candidate.exists():
        return candidate.resolve()

    by_id = (Path(artifacts_dir) / stage_folder / s)
    if by_id.exists():
        return by_id.resolve()

    raise FileNotFoundError(
        f"Could not resolve prior spec for '{stage_folder}': {spec!r}. "
        f"Expected an existing path (absolute or relative to {Path(output_root).resolve()}) "
        f"or a stage_run_id under {Path(artifacts_dir).resolve() / stage_folder}."
    )


def cmd_run_all(args: argparse.Namespace) -> int:
    """Run the 5-stage pipeline.

    Creates a run directory up front and backgrounds itself with nohup if --background.
    """
    # Store the single timestamp in Context; for background runs we pass it via env.
    run_timestamp = (os.environ.get("ENGAGEMENT_RUN_TIMESTAMP") or "").strip() or generate_run_timestamp()

    if bool(args.background):
        output_root = _resolve_run_dir(args, run_timestamp=run_timestamp)
        output_root.mkdir(parents=True, exist_ok=True)
        run_dir = _resolve_pipeline_run_dir(args, output_root=output_root, run_timestamp=run_timestamp)
        update_latest_symlink(output_root / "runs", run_dir)

        # Choose log path inside run_dir
        if args._initial_log:
            initial_log = Path(args._initial_log)
        else:
            initial_log = (run_dir / "run-all.log")
        try:
            initial_log.parent.mkdir(parents=True, exist_ok=True)
            with open(initial_log, 'a') as f:
                f.write(f"run-all started at {run_timestamp}\n")
        except Exception:
            pass

        # Background via nohup by re-invoking run-all in the foreground (background disabled)
        # with a pinned --output-dir.
        import shlex
        resolved_config = _build_effective_config_for_background_run(
            args, output_root=output_root, initial_log=initial_log
        )
        resolved_config_path = run_dir / "run-all.resolved-config.json"
        resolved_config_path.write_text(json.dumps(resolved_config, indent=2, sort_keys=True) + "\n")
        cli_args = ["--config", str(resolved_config_path)]

        py = shlex.quote(sys.executable)
        script = shlex.quote(str(Path(__file__).resolve()))
        args_str = ' '.join(shlex.quote(a) for a in cli_args)
        redir = shlex.quote(str(initial_log))
        env_prefix = (
            f"ENGAGEMENT_RUN_TIMESTAMP={shlex.quote(run_timestamp)} "
            f"ENGAGEMENT_PIPELINE_RUN_ID={shlex.quote(run_dir.name)}"
        )
        cmd = f"{env_prefix} nohup {py} {script} {args_str} > {redir} 2>&1 & echo $!"
        print(f"▶️  Backgrounding run-all with nohup. Log: {initial_log}")
        import subprocess as sp
        proc = sp.run(["bash", "-lc", cmd], stdout=sp.PIPE, stderr=sp.PIPE, text=True)
        if proc.returncode == 0:
            pid_str = (proc.stdout or "").strip().splitlines()[-1] if (proc.stdout or "").strip() else None
            pid_file = (run_dir / "run-all.pid")
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

    # Foreground execution: initialize experiment tracker and run
    # Only initialize ClearML here (not before backgrounding) to avoid creating
    # a task in the parent process that gets "aborted" when the parent exits.
    #
    # Pre-import torch so it's fully cached in sys.modules before ClearML patches
    # builtins.__import__. ClearML's patched importer breaks torch's internal
    # circular import chain (torch.jit._async -> torch.utils.set_module).
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    tracker = build_experiment_tracker(
        args.experiment_tracker,
        project_name=args.experiment_project,
        task_name=args.experiment_task or _generate_run_name(args),
        tags=args.experiment_tags,
        model_output_uri=args.model_output_uri,
    )
    # ClearML remote execution can override parameters on the server/UI.
    # Connect args and rehydrate a Namespace so downstream code sees the updated values.
    args = tracker.connect_args(args, "Args")

    output_root = _resolve_run_dir(args, run_timestamp=run_timestamp)
    output_root.mkdir(parents=True, exist_ok=True)

    # Resolve pipeline run dir after ClearML connects args, since output_dir might have been overridden.
    run_dir = _resolve_pipeline_run_dir(args, output_root=output_root, run_timestamp=run_timestamp)
    update_latest_symlink(output_root / "runs", run_dir)

    # Ensure args.output_dir is set (this is the output root).
    setattr(args, 'output_dir', str(output_root))
    setattr(args, "_argv", sys.argv[:])

    # Pipeline run scaffolding
    artifacts_dir = (output_root / "artifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name in ("tmp", "metrics", "plots", "logs"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)

    run_resolved_config_path = run_dir / "run-all.resolved-config.json"
    args_dict = {k: v for k, v in vars(args).items() if k != "func" and not callable(v)}
    run_resolved_config_path.write_text(json.dumps(args_dict, indent=2, sort_keys=True) + "\n")

    lineage_path = run_dir / "lineage.yaml"
    if not lineage_path.exists():
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(CLI_FILE_DIR),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            git_sha = (proc.stdout or "").strip() if proc.returncode == 0 else None
        except Exception:
            git_sha = None
        lineage_path.write_text(json.dumps({
            "pipeline_run_id": run_dir.name,
            "created_at": datetime.now().isoformat(),
            "git_sha": git_sha,
            "argv": sys.argv[:],
            "stages": {},
        }, indent=2, sort_keys=True) + "\n")

    # In sequential execution, always allow stages to resolve latest artifacts from prior stages
    ctx = Context(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        runs_dir=(output_root / "runs").resolve(),
        pipeline_run_id=run_dir.name,
        run_timestamp=run_timestamp,
        use_latest=True,
        tracker=tracker,
    )
    return cmd__run_all_exec(args, ctx)


def cmd__run_all_exec(args: argparse.Namespace, ctx: Context) -> int:
    """Execute the modular pipeline stages in the foreground sequentially."""
    # Build Context and invoke stages via registry
    from utils.pipeline import registry as reg

    run_dir = Path(ctx.run_dir).resolve()
    artifacts_dir = Path(ctx.artifacts_dir).resolve()
    output_root = Path(args.output_dir).resolve()

    # Apply non-interactive prior pins (paths or stage_run_ids).
    prior_01_get_data = _resolve_prior_spec(
        getattr(args, "prior_01_get_data", None),
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="01_get_data",
    )
    prior_02_target_posts = _resolve_prior_spec(
        getattr(args, "prior_02_target_posts", None),
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="02_target_posts",
    )
    prior_03_user_history = _resolve_prior_spec(
        getattr(args, "prior_03_user_history", None),
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="03_user_history",
    )
    prior_04_train = _resolve_prior_spec(
        getattr(args, "prior_04_train", None),
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="04_train",
    )
    if prior_01_get_data is not None:
        ctx.prior_outputs["01_get_data"] = prior_01_get_data
    if prior_02_target_posts is not None:
        ctx.prior_outputs["02_target_posts"] = prior_02_target_posts
    if prior_03_user_history is not None:
        ctx.prior_outputs["03_user_history"] = prior_03_user_history
    if prior_04_train is not None:
        ctx.prior_outputs["04_train"] = prior_04_train
    
    # Override train stage key if --model-type is specified
    # Do not default to MLP if model name is not recognized - raise error instead
    model_type = args.model_type
    train_key = 'train_mlp'  # default MLP
    if model_type == 'mlp':
        train_key = 'train_mlp'
    elif model_type == 'two-tower':
        train_key = 'train_two_tower'
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # --- Validation for --user-encoder ---
    user_encoder = args.user_encoder
    valid_encoders = {
        "mlp": ("summarized", "full_transformer"),
        "two-tower": ("summarized", "full_transformer", "cross_attention"),
    }
    allowed = valid_encoders.get(model_type, ())
    if user_encoder not in allowed:
        raise ValueError(
            f"--user-encoder '{user_encoder}' is not valid for --model-type '{model_type}'. "
            + f"Allowed values: {allowed}"
        )
    
    stage_order = ['get_data', 'target_posts', 'user_history', train_key, 'evaluate']
    stage_folder = {}
    for key in stage_order:
        _mp, _folder = reg.get_stage_spec(key)
        stage_folder[key] = _folder

    # Respect selective reruns (map the generic "train" alias to the concrete train stage key)
    start_from = args.start_from
    if start_from == 'train':
        start_from = train_key
    stop_after = args.stop_after
    if stop_after == 'train':
        stop_after = train_key
    if start_from and start_from not in stage_order:
        raise ValueError(f"Unrecognized start_from: {start_from}. Please choose from: {stage_order}")
    if stop_after and stop_after not in stage_order:
        raise ValueError(f"Unrecognized stop_after: {stop_after}. Please choose from: {stage_order}")
    start_idx = stage_order.index(start_from) if start_from in stage_order else 0
    stop_idx = stage_order.index(stop_after) if stop_after in stage_order else (len(stage_order) - 1)

    # Optional interactive chooser (foreground only)
    def _maybe_choose_prior(stage_key: str):
        if not args.pick_prior:
            return
        folder = stage_folder[stage_key]
        base = (artifacts_dir / folder)
        if not base.exists():
            return
        subdirs = [p for p in base.iterdir() if p.is_dir()]
        if len(subdirs) <= 1:
            return
        # Prompt only in foreground mode
        if bool(args.background):
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
    try:
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
                'target_posts': "Stage 2: Generate target posts…",
                'user_history': "Stage 3: Generate user history…",
                'train_mlp': "Stage 4: Train model (MLP)…",
                'train_two_tower': "Stage 4: Train model (Two-Tower)…",
                'evaluate': "Stage 5: Evaluate model…",
            }
            label = label_map.get(key, f"Stage {idx+1}: {key}…")
            print(f"\n[{idx+1}/5] ▶️  {label}")
            reg.run_stage(key, ctx, args)
    finally:
        ctx.tracker.close()

    print("\n✅ run-all completed successfully")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Engagement Prediction Pipeline CLI",
        argument_default=argparse.SUPPRESS,
    )
    # Backwards compatible vestige: `run-all` used to be a subcommand; now it's implicit.
    parser.add_argument(
        "command",
        nargs="?",
        default="run-all",
        choices=["run-all"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--config",
        type=str,
        help="YAML/JSON config file with run-all parameters (CLI flags override config)",
    )
    # run-all (modular 5-stage end-to-end)
    p_all = parser
    # Stage 1 options
    _add_arg_with_default(p_all, "--gcs-bucket", type=str, default=argparse.SUPPRESS,
                          help_text="GCS bucket name for ingex data")
    _add_arg_with_default(p_all, "--posts-start", type=str, default=argparse.SUPPRESS,
                          help_text="ISO date string for ingex GCS posts start (inclusive)")
    _add_arg_with_default(p_all, "--posts-end", type=str, default=argparse.SUPPRESS,
                          help_text="ISO date string for ingex GCS posts end (exclusive)")
    _add_arg_with_default(p_all, "--likes-start", type=str, default=argparse.SUPPRESS,
                          help_text="ISO date string for ingex GCS likes start (inclusive)")
    _add_arg_with_default(p_all, "--likes-end", type=str, default=argparse.SUPPRESS,
                          help_text="ISO date string for ingex GCS likes end (exclusive)")
    _add_arg_with_default(p_all, "--max-liking-users", type=int, default=argparse.SUPPRESS,
                          help_text="Cap on total liking users to sample (None = no limit)")
    _add_arg_with_default(p_all, "--max-likes-per-user", type=int, default=argparse.SUPPRESS,
                          help_text="Random cap on likes per user in Stage 1 (NOT recency-based)")
    _add_arg_with_default(p_all, "--negative-posts-sample", type=int, default=argparse.SUPPRESS,
                          help_text="Number of random posts to sample for negative cases in Stage 1")
    _add_arg_with_default(p_all, "--cap-random-seed", type=int, default=argparse.SUPPRESS,
                          help_text="Random seed for ingestion capping")
    _add_arg_with_default(p_all, "--max-memory-gb", type=float, default=argparse.SUPPRESS,
                          help_text="Maximum memory to use in GB (None = auto based on available RAM)")
    _add_arg_with_default(p_all, "--max-memory-pct", type=float, default=argparse.SUPPRESS,
                          help_text="Maximum percentage of available RAM to use (default: 0.75)")
    _add_arg_with_default(p_all, "--memory-check", type=str, choices=["full", "ignore", "skip"],
                          default=argparse.SUPPRESS,
                          help_text="Memory check mode: full (enforce limits), ignore (log only), skip (no estimation)")
    _add_arg_with_default(p_all, "--output-dir", type=str, default=argparse.SUPPRESS,
                          help_text="Optional explicit run directory root")
    _add_arg_with_default(p_all, "--debug", action="store_true", default=argparse.SUPPRESS,
                          help_text="Enable verbose debug logging for Stage 1")
    _add_arg_with_default(p_all, "--random-seed", type=int, default=argparse.SUPPRESS,
                          help_text="Random seed for splitting")
    _add_arg_with_default(p_all, "--embedding-model", type=str, choices=["all_MiniLM_L6_v2", "all_MiniLM_L12_v2"],
                          default=argparse.SUPPRESS, help_text="SentenceTransformers model for embeddings")
    _add_arg_with_default(p_all, "--skip-embeddings", action="store_true", default=argparse.SUPPRESS,
                          help_text="Skip embedding validation/memmap write in Stage 1 (faster iteration; later stages that need embeddings will fail)")
    # Stage 2/3 options
    _add_arg_with_default(p_all, "--max-prior-likes", type=int, default=argparse.SUPPRESS,
                          help_text="Cap on prior likes per target in Stage 3 user history (None = no cap, keeps all prior likes)")
    _add_arg_with_default(p_all, "--history-buffer-hours", type=float, default=argparse.SUPPRESS,
                          help_text="Buffer in hours subtracted from seen_at when determining prior likes for user history (None = no buffer)")
    _add_arg_with_default(p_all, "--neg-sample-bucket", type=str, default=argparse.SUPPRESS,
                          help_text="Duration (e.g. 1h) of time buckets for picking negative samples near positive (liked) posts")
    _add_arg_with_default(p_all, "--train-start", type=str, default=argparse.SUPPRESS,
                          help_text="ISO date string for start of training dataset window")
    _add_arg_with_default(p_all, "--val-start", type=str, default=argparse.SUPPRESS,
                          help_text="ISO date string for start of validation dataset window. Must be >= train-start")
    _add_arg_with_default(p_all, "--holdout-user-fraction", type=float, default=argparse.SUPPRESS,
                          help_text="Fraction of users to hold out for evaluation (0-1). Users are assigned deterministically via hashing.")
    _add_arg_with_default(p_all, "--holdout-user-seed", type=int, default=argparse.SUPPRESS,
                          help_text="Seed for deterministic holdout user assignment (combined with user ID in hash)")
    _add_arg_with_default(p_all, "--holdout-start", type=str, default=argparse.SUPPRESS,
                          help_text="ISO date string for start of seen-users holdout window. Non-holdout users' rows at/after this date become holdout_seen_users. Must be after val-start.")
    _add_arg_with_default(p_all, "--holdout-end", type=str, default=argparse.SUPPRESS,
                          help_text="ISO date string for end of holdout window. Applies to both holdout_seen_users and holdout_unseen_users. Rows at/after this date get split=None. Default: no upper bound.")
    _add_arg_with_default(p_all, "--global-topic-k", type=int, default=argparse.SUPPRESS,
                          help_text="Number of global topics")
    _add_arg_with_default(p_all, "--min-likes-per-user", type=int, default=argparse.SUPPRESS,
                          help_text="Minimum likes per user for inclusion (used in Stage 1 filtering and later stages)")
    # Stage 4 (train) user summarization + model selection
    _add_arg_with_default(p_all, "--user-summarization", type=str, choices=["mean", "ema", "linear_recency"],
                          default=argparse.SUPPRESS,
                          help_text="User-history summarization strategy for MLP (mean, ema, linear_recency)")
    _add_arg_with_default(p_all, "--ema-alpha", type=float, default=argparse.SUPPRESS,
                          help_text="EMA smoothing factor (0,1]. Higher = more weight on recent likes. Only used when --user-summarization=ema")
    _add_arg_with_default(p_all, "--user-encoder", type=str, choices=["summarized", "full_transformer", "cross_attention", "attention"],
                          default=argparse.SUPPRESS, help_text="User encoder type (must match model-type: summarized for mlp; full_transformer/cross_attention for two-tower). "
                          "Note: 'attention' is a deprecated alias for 'full_transformer'.")
    _add_arg_with_default(p_all, "--model-type", type=str, choices=["mlp", "two-tower"],
                          default=argparse.SUPPRESS, help_text="Model architecture: mlp or two-tower")
    # Two-tower specific options
    _add_arg_with_default(p_all, "--shared-dim", type=int, default=argparse.SUPPRESS,
                          help_text="Two-tower shared embedding dimension")
    _add_arg_with_default(p_all, "--user-hidden-dim", type=int, default=argparse.SUPPRESS,
                          help_text="User encoder hidden dimension")
    _add_arg_with_default(p_all, "--user-output-dim", type=int, default=argparse.SUPPRESS,
                          help_text="User encoder output dimension")
    _add_arg_with_default(p_all, "--post-hidden-dim", type=int, default=argparse.SUPPRESS,
                          help_text="Two-tower post encoder hidden dimension")
    _add_arg_with_default(p_all, "--post-encoder", key="use_post_encoder", dest="use_post_encoder",
                          action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS,
                          help_text="Enable or disable a neural post-tower encoder. Use --post-encoder to enable, --no-post-encoder to disable")
    _add_arg_with_default(p_all, "--num-attention-heads", type=int, default=argparse.SUPPRESS,
                          help_text="Two-tower attention heads")
    _add_arg_with_default(p_all, "--num-attention-layers", type=int, default=argparse.SUPPRESS,
                          help_text="Two-tower attention layers")
    _add_arg_with_default(p_all, "--max-history-len", type=int, default=argparse.SUPPRESS,
                          help_text="Max user history length")
    _add_arg_with_default(p_all, "--attention-dropout", type=float, default=argparse.SUPPRESS,
                          help_text="Dropout rate for attention-based user encoders")
    # Stage 5 options (shared)
    _add_arg_with_default(p_all, "--epochs", type=int, default=argparse.SUPPRESS,
                          help_text="Training epochs")
    _add_arg_with_default(p_all, "--batch-size", type=int, default=argparse.SUPPRESS,
                          help_text="Training batch size")
    _add_arg_with_default(p_all, "--learning-rate", type=float, default=argparse.SUPPRESS,
                          help_text="Learning rate")
    _add_arg_with_default(p_all, "--weight-decay-mlp", type=float, default=argparse.SUPPRESS,
                          help_text="Weight decay for MLP model")
    _add_arg_with_default(p_all, "--weight-decay-two-tower", type=float, default=argparse.SUPPRESS,
                          help_text="Weight decay for two tower model")
    _add_arg_with_default(p_all, "--hidden-dims", type=int, nargs="+", default=argparse.SUPPRESS,
                          help_text="Hidden layer sizes")
    _add_arg_with_default(p_all, "--dropout-rate-mlp", type=float, default=argparse.SUPPRESS,
                          help_text="Dropout rate for MLP model")
    _add_arg_with_default(p_all, "--dropout-rate-two-tower", type=float, default=argparse.SUPPRESS,
                          help_text="Dropout rate for two tower model")
    _add_arg_with_default(p_all, "--prediction-posts-per-user", type=float, default=argparse.SUPPRESS,
                          help_text="Prediction posts per user")
    _add_arg_with_default(p_all, "--device", type=str, choices=["cpu", "cuda"], default=argparse.SUPPRESS,
                          help_text="Device for training")
    _add_arg_with_default(p_all, "--patience", type=int, default=argparse.SUPPRESS,
                          help_text="Early stopping patience")
    _add_arg_with_default(p_all, "--run-tag", type=str, default=argparse.SUPPRESS,
                          help_text="Tag appended to training output directory name (e.g. mlp_summarized_mean)")
    _add_arg_with_default(p_all, "--no-plots", action="store_true", default=argparse.SUPPRESS,
                          help_text="Disable training plots")
    _add_arg_with_default(p_all, "--no-save-model", action="store_true", default=argparse.SUPPRESS,
                          help_text="Skip saving model checkpoints")
    _add_arg_with_default(p_all, "--disable-progress", action="store_true", default=argparse.SUPPRESS,
                          help_text="Disable progress bars during training")
    # Stage 4 (train) - DataLoader settings
    _add_arg_with_default(p_all, "--num-dataloader-workers", type=int, default=argparse.SUPPRESS,
                          help_text="Number of DataLoader worker processes")
    _add_arg_with_default(p_all, "--dataloader-pin-memory", action="store_true", default=argparse.SUPPRESS,
                          help_text="Enable DataLoader pin_memory for faster GPU transfer")
    _add_arg_with_default(p_all, "--dataloader-persistent-workers", action="store_true", default=argparse.SUPPRESS,
                          help_text="Keep DataLoader workers alive between epochs")
    _add_arg_with_default(p_all, "--dataloader-prefetch-factor", type=int, default=argparse.SUPPRESS,
                          help_text="Number of batches to prefetch per DataLoader worker")
    # Stage 4 (train) - Learning rate scheduler
    _add_arg_with_default(p_all, "--lr-scheduler-factor", type=float, default=argparse.SUPPRESS,
                          help_text="Factor by which to reduce learning rate")
    _add_arg_with_default(p_all, "--lr-scheduler-patience", type=int, default=argparse.SUPPRESS,
                          help_text="Number of epochs with no improvement before reducing LR")
    # Stage 4 (train) - Training optimization
    _add_arg_with_default(p_all, "--gradient-clip-max-norm", type=float, default=argparse.SUPPRESS,
                          help_text="Maximum gradient norm for clipping (two-tower only)")
    # Stage 5 options (subset)
    _add_arg_with_default(p_all, "--eval-batch-size", type=int, default=argparse.SUPPRESS,
                          help_text="Batch size for evaluation")
    _add_arg_with_default(p_all, "--eval-holdout-type", type=str, default=argparse.SUPPRESS,
                          choices=["unseen_users", "seen_users"],
                          help_text="Which holdout set to use for evaluation: unseen_users (user-split) or seen_users (temporal after val period)")
    _add_arg_with_default(p_all, "--skip-modules", type=str, default=argparse.SUPPRESS,
                          help_text="Comma-separated list of evaluation module names to skip (e.g. cold_start_curves,performance_inequality)")
    # Selection behavior
    _add_arg_with_default(p_all, "--use-latest", action="store_true", default=argparse.SUPPRESS,
                          help_text="(Deprecated) Always enabled during sequential run-all")
    # Selective reruns and prior pinning
    _add_arg_with_default(p_all, "--start-from", type=str,
                          choices=["get_data", "target_posts", "user_history", "train", "train_mlp", "train_two_tower", "evaluate"],
                          default=argparse.SUPPRESS, help_text="Begin execution at this stage")
    _add_arg_with_default(p_all, "--stop-after", type=str,
                          choices=["get_data", "target_posts", "user_history", "train", "train_mlp", "train_two_tower", "evaluate"],
                          default=argparse.SUPPRESS, help_text="Stop after this stage completes")
    _add_arg_with_default(p_all, "--pick-prior", action="store_true", default=argparse.SUPPRESS,
                          help_text="If multiple prior outputs exist, prompt to pick (foreground only)")
    _add_arg_with_default(p_all, "--prior-01-get-data", type=str, default=argparse.SUPPRESS,
                          help_text="Pin prior Stage 1 (01_get_data) artifact dir by stage_run_id or path")
    _add_arg_with_default(p_all, "--prior-02-target-posts", type=str, default=argparse.SUPPRESS,
                          help_text="Pin prior Stage 2 (02_target_posts) artifact dir by stage_run_id or path")
    _add_arg_with_default(p_all, "--prior-03-user-history", type=str, default=argparse.SUPPRESS,
                          help_text="Pin prior Stage 3 (03_user_history) artifact dir by stage_run_id or path")
    _add_arg_with_default(p_all, "--prior-04-train", type=str, default=argparse.SUPPRESS,
                          help_text="Pin prior Stage 4 (04_train) artifact dir by stage_run_id or path (used by eval)")
    # Execution behavior
    _add_arg_with_default(p_all, "--background", action="store_true", default=argparse.SUPPRESS,
                          help_text="Run in background with nohup (default: foreground)")
    p_all.add_argument("--_initial-log", type=str, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    # Experiment tracking
    _add_arg_with_default(p_all, "--experiment-tracker", type=str, choices=["none", "clearml"], default=argparse.SUPPRESS,
                          help_text="Type of experiment tracker to use")
    _add_arg_with_default(p_all, "--experiment-project", type=str, default=argparse.SUPPRESS,
                          help_text="Experiment tracking project name")
    _add_arg_with_default(p_all, "--experiment-task", type=str, default=argparse.SUPPRESS,
                          help_text="Experiment tracking task name")
    _add_arg_with_default(p_all, "--experiment-tags", type=str, nargs="*", default=argparse.SUPPRESS,
                          help_text="Optional tags for the experiment tracker")
    _add_arg_with_default(p_all, "--model-output-uri", type=str, default=argparse.SUPPRESS,
                          help_text="Model/task output URI for ClearML (e.g. gs://bucket/path)")
    p_all.set_defaults(func=cmd_run_all)

    return parser


def main() -> int:
    parser = build_parser()
    raw_args = parser.parse_args()
    merged_args = _merge_args_with_config(raw_args)
    return merged_args.func(merged_args)


if __name__ == "__main__":
    sys.exit(main()) 
