"""Compare checkpoint-backed rankers on shared bucketed candidate sets."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from utils.pipeline.dependencies import (
    resolve_stage_dependencies_for_run,
    validate_explicit_prior_pin_consistency,
)
from utils.pipeline.core import (
    Context,
    generate_run_timestamp,
    new_stage_artifact_dir,
)


DEFAULT_COMPARE_SPLITS = ["val", "val_unseen_users", "holdout_unseen_users", "holdout_seen_users"]
DEFAULT_COMPARE_METRICS_TOP_KS = [30]
DEFAULT_COMPARE_BATCH_SIZE = 256
DEFAULT_COMPARE_RANDOM_SEED = 42
DEFAULT_COMPARE_NUM_DATALOADER_WORKERS = 4
DEFAULT_COMPARE_DATALOADER_PIN_MEMORY = True
DEFAULT_COMPARE_DATALOADER_PERSISTENT_WORKERS = True
DEFAULT_COMPARE_DATALOADER_PREFETCH_FACTOR = 2
DEFAULT_COMPARE_DISABLE_PROGRESS = False
DEFAULT_COMPARE_BST_CANDIDATE_CHUNK_SIZE = 1024
VALID_COMPARE_MODEL_TYPES = {"two-tower", "bst-ranker"}


def _parse_compare_model_spec(raw: str) -> Dict[str, str]:
    parts = str(raw).split(":", 2)
    if len(parts) != 3:
        raise ValueError("Model spec must have format name:type:path")
    name, model_type, checkpoint_path = (part.strip() for part in parts)
    if not name:
        raise ValueError("Model spec name must not be empty")
    if model_type not in VALID_COMPARE_MODEL_TYPES:
        valid = ", ".join(sorted(VALID_COMPARE_MODEL_TYPES))
        raise ValueError(f"Model spec type must be one of: {valid}")
    if not checkpoint_path:
        raise ValueError("Model spec checkpoint path must not be empty")
    return {
        "name": name,
        "model_type": model_type,
        "checkpoint_path": checkpoint_path,
    }


def _resolve_compare_checkpoint_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Compare-rankers checkpoint not found: {path}")
    return path


def _require_compare_model_config(model_spec: Dict[str, str], config: Dict[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(
            f"compare-rankers model '{model_spec['name']}' config is missing required key "
            f"{key!r}: {model_spec['checkpoint_path']}"
        )
    return config[key]


def _validate_compare_author_config(model_spec: Dict[str, str], config: Dict[str, Any]) -> None:
    if _require_compare_model_config(model_spec, config, "use_author_embedding_table") is not True:
        raise ValueError(
            f"compare-rankers assumes author embeddings for every model, but model "
            f"'{model_spec['name']}' does not have use_author_embedding_table=True"
        )
    _require_compare_model_config(model_spec, config, "author_embedding_dim")
    _require_compare_model_config(model_spec, config, "author_table_num_rows")
    _require_compare_model_config(model_spec, config, "author_unknown_dropout_rate")


def _validate_compare_bst_config(model_spec: Dict[str, str], config: Dict[str, Any]) -> None:
    if model_spec["model_type"] != "bst-ranker":
        return
    _require_compare_model_config(model_spec, config, "content_projection_dim")
    _require_compare_model_config(model_spec, config, "author_projection_dim")


def _resolve_compare_max_history_len(
    args: argparse.Namespace,
    *,
    model_specs: List[Dict[str, str]],
    model_configs: Dict[str, Dict[str, Any]],
) -> int:
    if hasattr(args, "max_history_len"):
        value = int(getattr(args, "max_history_len"))
        if value <= 0:
            raise ValueError("compare-rankers --max-history-len must be positive")
        return value

    values_by_model: Dict[str, int] = {}
    for model_spec in model_specs:
        value = _require_compare_model_config(
            model_spec,
            model_configs[model_spec["name"]],
            "max_history_len",
        )
        values_by_model[model_spec["name"]] = int(value)

    distinct_values = sorted(set(values_by_model.values()))
    if len(distinct_values) > 1:
        details = ", ".join(f"{name}={value}" for name, value in sorted(values_by_model.items()))
        raise ValueError(
            "compare-rankers requires matching max_history_len across compared models "
            f"unless --max-history-len is passed explicitly: {details}"
        )
    max_history_len = distinct_values[0]
    if max_history_len <= 0:
        raise ValueError(f"compare-rankers max_history_len must be positive, got {max_history_len}")
    return max_history_len


def _metric_value_for_csv(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _write_compare_metrics_csv(
    path: Path,
    *,
    model_specs: List[Dict[str, str]],
    metrics_by_model: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    specs_by_name = {spec["name"]: spec for spec in model_specs}
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model_name", "model_type", "checkpoint_path", "split", "metric", "value"],
        )
        writer.writeheader()
        for model_name, split_metrics in metrics_by_model.items():
            spec = specs_by_name[model_name]
            for split_name, metrics in split_metrics.items():
                for metric_name, metric_value in sorted(metrics.items()):
                    writer.writerow({
                        "model_name": model_name,
                        "model_type": spec["model_type"],
                        "checkpoint_path": spec["checkpoint_path"],
                        "split": split_name,
                        "metric": metric_name,
                        "value": _metric_value_for_csv(metric_value),
                    })


def _make_compare_adapter(
    model_spec: Dict[str, str],
    *,
    bst_candidate_chunk_size: int,
    config_overrides: Optional[Dict[str, Any]],
):
    from utils.ranking_adapters import BstPthAdapter, TwoTowerPthAdapter

    if model_spec["model_type"] == "two-tower":
        if config_overrides is None:
            return TwoTowerPthAdapter(model_spec["checkpoint_path"])
        return TwoTowerPthAdapter(model_spec["checkpoint_path"], config_overrides=config_overrides)
    if model_spec["model_type"] == "bst-ranker":
        if config_overrides is None:
            return BstPthAdapter(
                model_spec["checkpoint_path"],
                candidate_chunk_size=bst_candidate_chunk_size,
            )
        return BstPthAdapter(
            model_spec["checkpoint_path"],
            candidate_chunk_size=bst_candidate_chunk_size,
            config_overrides=config_overrides,
        )
    raise ValueError(f"Unsupported model type: {model_spec['model_type']}")


def cmd_compare_rankers(
    args: argparse.Namespace,
    *,
    resolve_run_dir: Callable[[argparse.Namespace, str], Path],
    resolve_prior_spec: Callable[..., Optional[Path]],
) -> int:
    """Compare saved ranker checkpoints on shared bucketed candidate sets."""
    if getattr(args, "config", None):
        raise SystemExit("--config is not supported for compare-rankers")
    raw_model_specs = list(getattr(args, "model", []) or [])
    if not raw_model_specs:
        raise SystemExit("compare-rankers requires at least one --model name:type:path")

    run_timestamp = generate_run_timestamp()
    if not hasattr(args, "output_dir"):
        setattr(args, "output_dir", None)
    output_root = resolve_run_dir(args, run_timestamp)
    artifacts_dir = (output_root / "artifacts").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_dir = new_stage_artifact_dir(artifacts_dir, "compare_rankers", tag="rankers")

    from utils.helpers import get_stage_logger, log_operation_start

    logger = get_stage_logger(f"COMPARE_RANKERS", log_file=out_dir / "stage.log")
    log_operation_start("Compare rankers", "COMPARE_RANKERS", logger)
    t0 = time.time()

    from utils.ranking_adapters import load_checkpoint_config

    parsed_specs: List[Dict[str, str]] = []
    model_configs: Dict[str, Dict[str, Any]] = {}
    seen_names = set()
    for raw in raw_model_specs:
        spec = _parse_compare_model_spec(raw)
        if spec["name"] in seen_names:
            raise ValueError(f"Duplicate compare-rankers model name: {spec['name']}")
        seen_names.add(spec["name"])
        spec["checkpoint_path"] = str(_resolve_compare_checkpoint_path(spec["checkpoint_path"]))
        config = load_checkpoint_config(spec["checkpoint_path"])
        _validate_compare_author_config(spec, config)
        _validate_compare_bst_config(spec, config)
        model_configs[spec["name"]] = config
        parsed_specs.append(spec)

    requested_splits = list(getattr(args, "splits", DEFAULT_COMPARE_SPLITS))
    metrics_top_ks = list(getattr(args, "metrics_top_ks", DEFAULT_COMPARE_METRICS_TOP_KS))
    batch_size = int(getattr(args, "batch_size", DEFAULT_COMPARE_BATCH_SIZE))
    random_seed = int(getattr(args, "random_seed", DEFAULT_COMPARE_RANDOM_SEED))
    num_workers = int(getattr(args, "num_dataloader_workers", DEFAULT_COMPARE_NUM_DATALOADER_WORKERS))
    pin_memory = bool(getattr(args, "dataloader_pin_memory", DEFAULT_COMPARE_DATALOADER_PIN_MEMORY))
    persistent_workers = bool(getattr(args, "dataloader_persistent_workers", DEFAULT_COMPARE_DATALOADER_PERSISTENT_WORKERS))
    prefetch_factor = int(getattr(args, "dataloader_prefetch_factor", DEFAULT_COMPARE_DATALOADER_PREFETCH_FACTOR))
    bst_candidate_chunk_size = int(getattr(args, "bst_candidate_chunk_size", DEFAULT_COMPARE_BST_CANDIDATE_CHUNK_SIZE))
    disable_progress = bool(getattr(args, "disable_progress", DEFAULT_COMPARE_DISABLE_PROGRESS))
    eval_max_history_len = _resolve_compare_max_history_len(
        args,
        model_specs=parsed_specs,
        model_configs=model_configs,
    )
    use_popularity_feature_for_compare = any(
        spec["model_type"] == "bst-ranker"
        and bool(model_configs[spec["name"]].get("bst_use_popularity_feature", False))
        for spec in parsed_specs
    )

    print(f"Batch Size: {batch_size}")
    print(f"Candidate Chunk Size: {bst_candidate_chunk_size}")
    print(f"Num workers: {num_workers}")
    print(f"Pin memory: {pin_memory}")
    print(f"Prefetch Factor: {prefetch_factor}")
    print(f"Persistent Workers: {persistent_workers}")

    import torch
    from torch.utils.data import DataLoader
    from utils.dataloaders import (
        BucketedBatchSampler,
        BucketedEngagementDataset,
        load_bucketed_training_data,
    )
    from utils.matrix_ranking import evaluate_matrix_scorer

    device_arg = getattr(args, "device", None)
    device = str(device_arg) if device_arg else ("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = out_dir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = Context(
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        runs_dir=(output_root / "runs").resolve(),
        pipeline_run_id=out_dir.name,
        run_timestamp=run_timestamp,
        use_latest=True,
    )

    prior_01_get_data = resolve_prior_spec(
        getattr(args, "prior_01_get_data", None),
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="01_get_data",
    )
    prior_02_user_history = resolve_prior_spec(
        getattr(args, "prior_02_user_history", None),
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder="02_user_history",
    )
    if prior_01_get_data is not None:
        ctx.prior_outputs["01_get_data"] = prior_01_get_data
    if prior_02_user_history is not None:
        ctx.prior_outputs["02_user_history"] = prior_02_user_history
    validate_explicit_prior_pin_consistency(ctx)
    resolved_priors = resolve_stage_dependencies_for_run(
        ctx=ctx,
        consumer_stage_folder="03_train",
    )
    ctx.prior_outputs.update(resolved_priors)

    embeddings_mmap, likes_core_df, posts_core_df, history_df, _author_idx_mapping_df, embed_dim = load_bucketed_training_data(
        ctx,
        logger=logger,
        require_target_hour_history_popularity=use_popularity_feature_for_compare,
    )

    worker_kw: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        worker_kw.update(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    split_loaders: Dict[str, DataLoader] = {}
    split_row_counts: Dict[str, int] = {}
    skipped_splits: List[str] = []
    for split_name in requested_splits:
        dataset = BucketedEngagementDataset(
            embeddings_mmap=embeddings_mmap,
            likes_core_df=likes_core_df,
            posts_core_df=posts_core_df,
            history_df=history_df,
            split=split_name,
            max_history_len=eval_max_history_len,
            embed_dim=embed_dim,
            use_author_embedding_table=True,
            use_popularity_feature=use_popularity_feature_for_compare,
            logger=logger,
        )
        split_row_counts[split_name] = len(dataset)
        if len(dataset) == 0:
            skipped_splits.append(split_name)
            logger.warning(f"No rows for split '{split_name}', skipping.")
            continue
        split_loaders[split_name] = DataLoader(
            dataset,
            batch_sampler=BucketedBatchSampler(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                seed=random_seed,
            ),
            collate_fn=dataset.collate_batch,
            **worker_kw,
        )

    metrics_by_model: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for model_spec in parsed_specs:
        model_name = model_spec["name"]
        logger.info(
            f"Evaluating model '{model_name}' ({model_spec['model_type']}): {model_spec['checkpoint_path']}"
        )
        adapter = _make_compare_adapter(
            model_spec,
            bst_candidate_chunk_size=bst_candidate_chunk_size,
            config_overrides=None,
        )
        metrics_by_model[model_name] = {}
        for split_name, loader in split_loaders.items():
            result = evaluate_matrix_scorer(
                adapter,
                loader,
                device,
                metrics_top_ks,
                collect_ranking_rows=False,
                progress_desc=f"{model_name} {split_name}",
                disable_progress=disable_progress,
            )
            metrics = result["metrics"]
            metrics_by_model[model_name][split_name] = metrics
            logger.info(f"Metrics for {model_name}/{split_name}:\n{json.dumps(metrics, indent=2)}")
        del adapter
        if device == "cuda":
            torch.cuda.empty_cache()

    summary = {
        "timestamp": run_timestamp,
        "runtime_seconds": time.time() - t0,
        "device": device,
        "splits": requested_splits,
        "skipped_splits": skipped_splits,
        "split_row_counts": split_row_counts,
        "metrics_top_ks": metrics_top_ks,
        "batch_size": batch_size,
        "bst_candidate_chunk_size": bst_candidate_chunk_size,
        "max_history_len": eval_max_history_len,
        "model_specs": parsed_specs,
        "metrics": metrics_by_model,
        "prior_inputs": {k: str(v) for k, v in ctx.get_active_stage_inputs().items()},
    }

    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str, sort_keys=True) + "\n")
    (out_dir / "model_specs.json").write_text(json.dumps(parsed_specs, indent=2, sort_keys=True) + "\n")
    _write_compare_metrics_csv(
        out_dir / "metrics.csv",
        model_specs=parsed_specs,
        metrics_by_model=metrics_by_model,
    )

    info_lines = [
        "stage: compare_rankers",
        f"timestamp: {run_timestamp}",
        f"runtime_seconds: {time.time() - t0:.2f}",
        f"device: {device}",
        f"models: {', '.join(spec['name'] for spec in parsed_specs)}",
        f"splits: {', '.join(requested_splits)}",
        f"skipped_splits: {', '.join(skipped_splits) if skipped_splits else 'none'}",
        f"metrics_top_ks: {', '.join(str(k) for k in metrics_top_ks)}",
        f"batch_size: {batch_size}",
        f"max_history_len: {eval_max_history_len}",
    ]
    for folder, path in sorted(ctx.get_active_stage_inputs().items()):
        info_lines.append(f"prior_input_{folder}: {path}")
    (out_dir / "stage_info.txt").write_text("\n".join(info_lines) + "\n")

    logger.info(f"Compare-rankers complete. Output: {out_dir}")
    print(f"✅ compare-rankers completed successfully: {out_dir}")
    return 0
