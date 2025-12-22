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
def _load_run_training_pipeline():
    """Dynamically load run_training_pipeline from utils/05_train/stage_train.py (numeric folder)."""
    stage_path = Path(__file__).parent / "utils" / "05_train" / "stage_train.py"
    spec = importlib.util.spec_from_file_location("utils_stage_train", str(stage_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load training stage module at {stage_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return getattr(mod, "run_training_pipeline")


def _load_run_two_tower_pipeline():
    """Dynamically load run_two_tower_pipeline from utils/05_train/stage_train_two_tower.py."""
    stage_path = Path(__file__).parent / "utils" / "05_train" / "stage_train_two_tower.py"
    spec = importlib.util.spec_from_file_location("utils_stage_train_two_tower", str(stage_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load two-tower training stage module at {stage_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return getattr(mod, "run_two_tower_pipeline")


def _load_stage_evaluate():
    """Dynamically load utils/06_evaluate/stage_evaluate.py module."""
    stage_path = Path(__file__).parent / "utils" / "06_evaluate" / "stage_evaluate.py"
    spec = importlib.util.spec_from_file_location("utils_stage_evaluate", str(stage_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load evaluate stage module at {stage_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def _auto_latest_processed() -> Optional[str]:
    """Return latest processed data file path or None if not found."""
    if not PROCESSED_DATA_DIR.exists():
        return None
    candidates = list(PROCESSED_DATA_DIR.glob("processed_data_*.pkl"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def _auto_latest_model() -> Optional[str]:
    """Return latest model checkpoint path or None if not found."""
    if not CHECKPOINT_DIR.exists():
        return None
    patterns = ["final_engagement_model_*.pth", "engagement_model_*.pth", "*.pth"]
    candidates = []
    for pat in patterns:
        candidates.extend(CHECKPOINT_DIR.glob(pat))
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    return str(latest)


def cmd_preprocess(args) -> int:
    # Lazy import from src
    # Deprecated: preprocess now handled by stage scripts; keep stub to avoid src import
    class DataPreprocessor:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
        def run(self, *args, **kwargs):
            print("Preprocess via stage scripts under utils/, not src.preprocess. (stub)")

    pre = DataPreprocessor(
        days=args.days,
        min_likes_per_user=args.min_likes,
        max_likes_per_liker=args.max_likes_per_liker,
        val_ratio=args.test_ratio,
        holdout_ratio=args.holdout_ratio,
        random_seed=args.random_seed,
        max_samples=args.max_samples,
        drop_unliked_posts=args.drop_unliked,
        limit_images=args.limit_images,
        prediction_posts_per_user=args.prediction_posts_per_user,
        require_liked_negatives=args.require_liked_negatives,
        verbose_logging=args.verbose,
        # New topic discovery + releveling
        global_topic_k=getattr(args, 'global_topic_k', 20),
        relevel_strategy=getattr(args, 'relevel_strategy', None),
        relevel_alpha=getattr(args, 'relevel_alpha', 0.35),
        relevel_min_users_per_topic=getattr(args, 'relevel_min_users_per_topic', 0),
        # User features
        user_features=getattr(args, 'user_features', 'topic_mixture'),
        user_k=getattr(args, 'user_k', 3),
        min_cluster_size=getattr(args, 'min_cluster_size', 3),
        max_embedding_posts_per_user=getattr(args, 'max_embedding_posts_per_user', 20),
    )
    out_path = pre.run_complete_preprocessing()
    print(f"\n✅ Preprocessing complete: {out_path}")
    print(f"Next: python cli.py train --load-processed {out_path}")
    return 0


def cmd_train(args) -> int:
    processed_path = args.load_processed
    if processed_path in (None, "auto"):
        processed_path = _auto_latest_processed()
        if not processed_path:
            print("❌ No processed data found. Run 'preprocess' first or provide --load-processed path.")
            return 1
        print(f"📂 Using latest processed data: {Path(processed_path).name}")

    # Use stage-based training pipeline (dynamic import due to numeric folder name)
    run_training_pipeline = _load_run_training_pipeline()

    results = run_training_pipeline(
        days=args.days,
        min_likes_per_user=args.min_likes_per_user,
        test_ratio=args.test_ratio,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate,
        device=args.device,
        random_seed=args.random_seed,
        save_model=not args.no_save_model,
        generate_plots=not args.no_plots,
        max_samples=args.max_samples,
        drop_unliked_posts=args.drop_unliked_posts,
        limit_images=args.limit_images,
        load_processed=processed_path,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        disable_progress=getattr(args, 'disable_progress', False),
        tqdm_mininterval=getattr(args, 'tqdm_mininterval', None),
        tqdm_miniters=getattr(args, 'tqdm_miniters', None),
    )

    if results.get("model_path"):
        print(f"\n✅ Training complete. Model: {results['model_path']}")
        print("Next: python cli.py evaluate --model-path auto --data-path auto")
    else:
        print("\n✅ Training complete. Use 'evaluate' to evaluate on holdout users.")
    return 0


def cmd_evaluate(args) -> int:
    model_path = args.model_path
    if model_path in (None, "auto"):
        model_path = _auto_latest_model()
        if not model_path:
            print("❌ No model found in checkpoints. Train a model first or provide --model-path.")
            return 1
        print(f"🤖 Using latest model: {Path(model_path).name}")
    else:
        print(f"🤖 Using model: {Path(model_path).name}")

    data_path = args.data_path
    if data_path in (None, "auto"):
        data_path = _auto_latest_processed()
        if not data_path:
            print("❌ No processed data found. Run 'preprocess' first or provide --data-path.")
            return 1
        print(f"📊 Using latest data: {Path(data_path).name}")
    else:
        print(f"📊 Using data: {Path(data_path).name}")

    # Route to stage 6 evaluator run function using a minimal context (dynamic import)
    _stage_eval = _load_stage_evaluate()
    class _Ctx:
        def __init__(self, run_dir: str):
            self.run_dir = run_dir
            self.use_latest = True
            self.prior_outputs = {}
    # Resolve run_dir heuristically from provided paths
    rd = Path(args.output_dir or RESULTS_DIR)
    ctx = _Ctx(str(rd))
    eval_args = args  # pass through
    _stage_eval.run(ctx, eval_args)
    print("\n✅ Evaluation completed.")
    return 0


def cmd_train_eval(args) -> int:
    """Train using an embedding bundle + splits (auto-discovered from run-dir when provided), then run full-feed evaluation."""
    # Determine model type
    model_type = getattr(args, 'model_type', 'mlp')
    
    # Resolve run_dir (preferred) and auto-select bundle/splits if not explicitly provided
    run_dir: Optional[Path] = Path(args.run_dir).resolve() if args.run_dir else None

    def _infer_run_dir_from_path(p: Optional[str]) -> Optional[Path]:
        if not p:
            return None
        try:
            path_obj = Path(p).resolve()
            # .../precompute/embedding_bundle_*.pkl → run_dir
            if path_obj.parent.name == 'precompute':
                return path_obj.parent.parent
            # .../relevel(_and_split_custom)/user_splits.json → run_dir
            if path_obj.parent.name in ('relevel', 'relevel_and_split_custom'):
                return path_obj.parent.parent
            # .../train/(<ts>/)checkpoints/*.pth → run_dir
            if path_obj.parent.name == 'checkpoints' and path_obj.parent.parent.name == 'train':
                # support optional timestamp level
                maybe_ts = path_obj.parent.parent
                if maybe_ts.parent.name == 'train':
                    return maybe_ts.parent.parent
                return path_obj.parent.parent.parent
        except Exception:
            return None
        return None

    if run_dir is None:
        run_dir = _infer_run_dir_from_path(args.embedding_bundle) or _infer_run_dir_from_path(args.user_splits)

    # Resolve embedding bundle
    if args.embedding_bundle:
        bundle_path = Path(args.embedding_bundle).resolve()
    else:
        if not run_dir:
            print("❌ Provide --run-dir or --embedding-bundle so I can locate the embedding bundle")
            return 2
        precompute_dir = run_dir / 'precompute'
        candidates = sorted(precompute_dir.glob('embedding_bundle_*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print(f"❌ No embedding_bundle_*.pkl found under {precompute_dir}")
            return 2
        bundle_path = candidates[0].resolve()
        print(f"📦 Auto-selected embedding bundle: {bundle_path}")

    # Resolve user_splits
    if args.user_splits:
        splits_path = Path(args.user_splits).resolve()
    else:
        if not run_dir:
            print("❌ Provide --run-dir or --user-splits so I can locate user_splits.json")
            return 2
        custom_splits = run_dir / 'relevel_and_split_custom' / 'user_splits.json'
        default_splits = run_dir / 'relevel' / 'user_splits.json'
        if custom_splits.exists():
            splits_path = custom_splits.resolve()
            print(f"🧩 Auto-selected custom user splits: {splits_path}")
        elif default_splits.exists():
            splits_path = default_splits.resolve()
            print(f"🧩 Auto-selected default user splits: {splits_path}")
        else:
            print(f"❌ Could not find user_splits.json under {run_dir}/relevel_and_split_custom or {run_dir}/relevel")
            return 2

    # Ensure run_dir set for downstream saves
    if run_dir is None:
        try:
            # Fallback to outputs root
            run_dir = OUTPUTS_DIR
        except Exception:
            run_dir = OUTPUTS_DIR

    # Train based on model type
    if model_type == 'two-tower':
        print("🏗️  Using Two-Tower model architecture")
        run_two_tower_pipeline = _load_run_two_tower_pipeline()
        results = run_two_tower_pipeline(
            embedding_bundle=str(bundle_path),
            user_splits=str(splits_path),
            shared_dim=getattr(args, 'shared_dim', 128),
            user_hidden_dim=getattr(args, 'user_hidden_dim', 256),
            post_hidden_dim=getattr(args, 'post_hidden_dim', 256),
            num_attention_heads=getattr(args, 'num_attention_heads', 4),
            num_attention_layers=getattr(args, 'num_attention_layers', 2),
            max_history_len=getattr(args, 'max_history_len', 20),
            dropout_rate=args.dropout_rate,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
            random_seed=args.random_seed,
            output_dir=run_dir,
            disable_progress=getattr(args, 'disable_progress', False),
        )
    else:
        print("🏗️  Using MLP model architecture")
        run_training_pipeline = _load_run_training_pipeline()
        results = run_training_pipeline(
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            hidden_dims=args.hidden_dims,
            dropout_rate=args.dropout_rate,
            device=args.device,
            random_seed=args.random_seed,
            save_model=not args.no_save_model,
            generate_plots=not args.no_plots,
            embedding_bundle=str(bundle_path),
            user_splits=str(splits_path),
            # Schema is set internally by training (multi_centroid by default)
            topic_model_path=args.topic_model_path,
            topic_pca_path=args.topic_pca_path,
            output_dir=run_dir,
            disable_progress=getattr(args, 'disable_progress', False),
            tqdm_mininterval=getattr(args, 'tqdm_mininterval', None),
            tqdm_miniters=getattr(args, 'tqdm_miniters', None),
        )

    # Report training results
    if model_type == 'two-tower':
        training_results = results.get("results", {})
        print(f"\n✅ Two-Tower training complete.")
        if 'model_path' in training_results:
            print(f"   Model: {training_results['model_path']}")
    else:
        trained_model_path = results.get("model_path")
        if trained_model_path:
            print(f"\n✅ Training complete. Model: {trained_model_path}")
        else:
            print("\n✅ Training complete.")

    # Directly invoke stage 6 run (pairs mode)
    _stage_eval = _load_stage_evaluate()
    class _Ctx:
        def __init__(self, run_dir: str):
            self.run_dir = run_dir
            self.use_latest = True
            self.prior_outputs = {}
    ctx = _Ctx(str(run_dir.resolve()))
    _stage_eval.run(ctx, args)
    print("\n✅ Train → Evaluate completed")
    return 0


def _build_python_cmd(script_path: Path, args_list: list[str]) -> list[str]:
    return [sys.executable, str(script_path)] + args_list


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

    # preprocess
    p_pre = subparsers.add_parser("preprocess", help="Run data preprocessing")
    p_pre.add_argument("--days", type=int, default=5)
    p_pre.add_argument("--min-likes", type=int, default=4)
    p_pre.add_argument("--max-likes-per-liker", type=int, default=None)
    p_pre.add_argument("--test-ratio", type=float, default=0.2)
    p_pre.add_argument("--holdout-ratio", type=float, default=0.2)
    p_pre.add_argument("--random-seed", type=int, default=42)
    p_pre.add_argument("--max-samples", type=int, default=None)
    p_pre.add_argument("--drop-unliked", action="store_true")
    p_pre.add_argument("--limit-images", action="store_true")
    p_pre.add_argument("--prediction-posts-per-user", type=int, default=1)
    p_pre.add_argument("--require-liked-negatives", action="store_true", 
                       help="Sample negative examples only from posts liked by someone else in dataset")
    p_pre.add_argument("--verbose", action="store_true")
    # Topic discovery + releveling options
    p_pre.add_argument("--global-topic-k", type=int, default=20)
    p_pre.add_argument("--relevel-strategy", type=str, default=None,
                       help="Set to 'uniform_mixture_balanced' to enable user-level topic releveling")
    p_pre.add_argument("--relevel-alpha", type=float, default=0.35)
    p_pre.add_argument("--relevel-min-users-per-topic", type=int, default=0)
    # User feature options
    p_pre.add_argument("--user-features", type=str, default='topic_mixture',
                      choices=['topic_mixture','multi_centroid','mean'],
                      help="User feature representation for training data")
    p_pre.add_argument("--user-k", type=int, default=3, help="K for multi-centroid user features")
    p_pre.add_argument("--min-cluster-size", type=int, default=3, help="Min posts per cluster for per-user KMeans")
    p_pre.add_argument("--max-embedding-posts-per-user", type=int, default=20,
                      help="Cap embedding posts per user used to compute user features")
    p_pre.set_defaults(func=cmd_preprocess)

    # train
    p_train = subparsers.add_parser("train", help="Train model using preprocessed data")
    p_train.add_argument("--load-processed", type=str, default="auto", help="Path to processed data .pkl or 'auto'")
    p_train.add_argument("--days", type=int, default=5)
    p_train.add_argument("--min-likes-per-user", type=int, default=4)
    p_train.add_argument("--test-ratio", type=float, default=0.2)
    p_train.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_train.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    p_train.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p_train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p_train.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p_train.add_argument("--hidden-dims", type=int, nargs="+", default=None)
    p_train.add_argument("--dropout-rate", type=float, default=DEFAULT_DROPOUT_RATE)
    p_train.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p_train.add_argument("--random-seed", type=int, default=42)
    p_train.add_argument("--no-save-model", action="store_true")
    p_train.add_argument("--no-plots", action="store_true")
    p_train.add_argument("--max-samples", type=int, default=None)
    p_train.add_argument("--drop-unliked-posts", action="store_true")
    p_train.add_argument("--limit-images", action="store_true")
    p_train.add_argument("--output-dir", type=str, default=None, help="Optional run directory; checkpoints/plots/logs will be created under this path")
    # Progress controls
    p_train.add_argument("--disable-progress", action="store_true", help="Disable tqdm progress bars during training")
    p_train.add_argument("--tqdm-mininterval", type=float, default=None, help="Min seconds between tqdm refreshes during training")
    p_train.add_argument("--tqdm-miniters", type=int, default=None, help="Min iterations between tqdm refreshes during training")
    p_train.set_defaults(func=cmd_train)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate trained model on holdout users")
    p_eval.add_argument("--model-path", type=str, default="auto", help="Path to .pth or 'auto'")
    p_eval.add_argument("--data-path", type=str, default="auto", help="Path to processed .pkl or 'auto'")
    p_eval.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    p_eval.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p_eval.add_argument("--random-seed", type=int, default=42)
    p_eval.add_argument("--batch-size", type=int, default=256)
    p_eval.add_argument("--create-plots", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    # train-eval
    p_te = subparsers.add_parser("train-eval", help="Train, then run full-feed evaluation using bundle + splits")
    # IO resolution: allow --run-dir to auto-discover, or explicit paths
    p_te.add_argument("--run-dir", type=str, default=None, help="Run directory; auto-discovers bundle and splits when provided")
    p_te.add_argument("--embedding-bundle", type=str, default=None, help="Optional path to embedding_bundle_*.pkl (auto from --run-dir if omitted)")
    p_te.add_argument("--user-splits", type=str, default=None, help="Optional path to user_splits.json (auto from --run-dir if omitted)")
    # Model type selection
    p_te.add_argument("--model-type", type=str, choices=["mlp", "two-tower"], default="mlp",
                      help="Model architecture: 'mlp' (default) or 'two-tower'")
    # Two-tower specific options
    p_te.add_argument("--shared-dim", type=int, default=128, help="Two-tower shared embedding dimension")
    p_te.add_argument("--user-hidden-dim", type=int, default=256, help="Two-tower user encoder hidden dimension")
    p_te.add_argument("--post-hidden-dim", type=int, default=256, help="Two-tower post encoder hidden dimension")
    p_te.add_argument("--num-attention-heads", type=int, default=4, help="Two-tower attention heads")
    p_te.add_argument("--num-attention-layers", type=int, default=2, help="Two-tower attention layers")
    p_te.add_argument("--max-history-len", type=int, default=20, help="Two-tower max user history length")
    # Topic-model options are ignored by training (default multi_centroid); kept for eval compatibility if needed
    p_te.add_argument("--topic-model-path", type=str, default=None)
    p_te.add_argument("--topic-pca-path", type=str, default=None)
    # Training knobs (subset of train)
    p_te.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p_te.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    p_te.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    p_te.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p_te.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p_te.add_argument("--hidden-dims", type=int, nargs="+", default=None)
    p_te.add_argument("--dropout-rate", type=float, default=DEFAULT_DROPOUT_RATE)
    p_te.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    p_te.add_argument("--random-seed", type=int, default=42)
    p_te.add_argument("--no-save-model", action="store_true")
    p_te.add_argument("--no-plots", action="store_true")
    # Progress / logging controls for training
    p_te.add_argument("--disable-progress", action="store_true", help="Disable tqdm progress bars during training")
    p_te.add_argument("--tqdm-mininterval", type=float, default=None, help="Min seconds between tqdm refreshes during training")
    p_te.add_argument("--tqdm-miniters", type=int, default=None, help="Min iterations between tqdm refreshes during training")
    # Evaluation knobs
    p_te.add_argument("--max-users", type=int, default=0, help="0 = all eligible holdout users")
    p_te.add_argument("--eval-batch-size", type=int, default=None, help="Override eval scoring batch size (defaults to script)")
    p_te.set_defaults(func=cmd_train_eval)

    # run-all (modular 6-stage end-to-end)
    p_all = subparsers.add_parser("run-all", help="Run all 6 stages end-to-end. Defaults to background with nohup.")
    # Stage 1 options
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