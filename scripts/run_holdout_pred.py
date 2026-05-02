#!/usr/bin/env python3
"""Post-hoc holdout-prediction generator for trained cells.

This is the companion to ``--skip-holdout-pred``: when a sweep cell trains
without materializing the holdout dataset (to keep RSS bounded at sweep
scale, see ``260428_like_biases/jobs/0006_sweep02_memory_prep.md``), this
script regenerates ``predictions/holdout_<unseen_users,seen_users>.parquet``
from the cell's saved checkpoint.

It rebuilds the same Dataset that the train stage would have built, runs the
saved model in inference mode, and writes parquets with the same schema as
the train-stage versions: ``did, post_id, y_true, y_pred_proba``.

Usage::

    scripts/run_holdout_pred.py <train_cell_dir> [--holdout-type unseen_users|seen_users|both]
                                                  [--batch-size N]
                                                  [--device cpu|cuda]
                                                  [--num-workers N]

Where ``<train_cell_dir>`` is a 04_train output directory containing
``training_config.json`` and ``checkpoints/<engagement_model_*|two_tower_*>.pth``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset

# We use the synthetic_feed loader path so model reconstruction stays in
# one place across the bias diagnostic and this helper.
import importlib

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.dataloaders import (  # noqa: E402  (sys.path adjusted above)
    SequenceEngagementDataset,
    SummarizedEngagementDataset,
    get_summarizer,
    load_training_data,
)
from utils.pipeline.core import Context  # noqa: E402

_synthetic_feed = importlib.import_module("utils.05_evaluate.evals.synthetic_feed")


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("run_holdout_pred")
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] [%(name)s] %(message)s"))
        logger.addHandler(h)
    logger.setLevel(logging.INFO)
    return logger


def _resolve_run_dir(train_cell_dir: Path) -> Path:
    # train_cell_dir = <run_dir>/04_train/<ts>_<run_tag>/
    return train_cell_dir.parent.parent


def _load_training_config(train_cell_dir: Path) -> Dict[str, Any]:
    cfg_path = train_cell_dir / "training_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"training_config.json not found at {cfg_path}")
    with open(cfg_path) as f:
        return json.load(f)


def _build_dataset(
    split_name: str,
    embeddings_mmap: np.ndarray,
    target_posts_df: pl.DataFrame,
    history_df: pl.DataFrame,
    cfg: Dict[str, Any],
    embed_dim: int,
    logger: logging.Logger,
) -> Dataset:
    """Reconstruct the dataset the train stage would have built for this split."""
    user_encoder = cfg.get("user_encoder") or cfg.get("user_encoder_type") or "summarized"
    if user_encoder == "summarized":
        summarizer_name = cfg.get("user_summarization") or "mean"
        ema_alpha = float(cfg.get("ema_alpha") or 0.1)
        summarizer = get_summarizer(summarizer_name, ema_alpha=ema_alpha)
        return SummarizedEngagementDataset(
            embeddings_mmap, target_posts_df, history_df, split=split_name,
            summarizer=summarizer, embed_dim=embed_dim, logger=logger,
        )
    max_history_len = int(cfg.get("max_history_len") or 100)
    return SequenceEngagementDataset(
        embeddings_mmap, target_posts_df, history_df, split=split_name,
        max_history_len=max_history_len, embed_dim=embed_dim, logger=logger,
    )


def _score_dataset(
    model: torch.nn.Module,
    dataset: Dataset,
    model_type: str,
    embed_dim: int,
    batch_size: int,
    num_workers: int,
    device: str,
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """Inference loop matching the train stages' holdout block.

    Returns (user_ids, post_ids, y_true, y_pred_proba) as plain numpy arrays.
    """
    loader_kw: Dict[str, Any] = dict(
        batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=False,
    )
    if num_workers > 0:
        loader_kw["persistent_workers"] = False
    loader = DataLoader(dataset, **loader_kw)

    model = model.to(device)
    model.eval()

    user_ids: List[str] = []
    post_ids: List[str] = []
    ys_parts: List[torch.Tensor] = []
    ps_parts: List[torch.Tensor] = []

    with torch.inference_mode():
        for batch in loader:
            if model_type == "two_tower":
                _, scores = model.compute_loss_and_preds(batch, device, embed_dim)
                preds = torch.sigmoid(scores).cpu()
            else:
                _, preds = model.compute_loss_and_preds(batch, device)
                preds = preds.cpu()
                if preds.ndim == 0:
                    preds = preds.unsqueeze(0)
            labels = batch["label"]
            if labels.ndim == 0:
                labels = labels.unsqueeze(0)
            ps_parts.append(preds)
            ys_parts.append(labels)
            uid = batch["user_id"]
            pid = batch["post_id"]
            if isinstance(uid, str):
                user_ids.append(uid)
                post_ids.append(pid)
            else:
                user_ids.extend(list(uid))
                post_ids.extend(list(pid))

    y_true = torch.cat(ys_parts).numpy()
    y_pred = torch.cat(ps_parts).numpy()
    return user_ids, post_ids, y_true, y_pred


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("train_cell_dir", type=Path, help="04_train cell directory containing training_config.json and checkpoints/")
    ap.add_argument("--holdout-type", choices=["unseen_users", "seen_users", "both"], default="both",
                    help="Which holdout split to predict (default: both)")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers (default 0 = single process; safer for sequential phase)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None,
                    help="Override device (default: cuda if available)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-generate parquet even if it already exists")
    args = ap.parse_args()

    logger = _setup_logger()
    train_cell_dir: Path = args.train_cell_dir.resolve()
    if not train_cell_dir.exists():
        logger.error(f"Train cell dir does not exist: {train_cell_dir}")
        return 2

    cfg = _load_training_config(train_cell_dir)
    model_type = cfg.get("model_type", "mlp")
    logger.info(f"Train cell: {train_cell_dir}")
    logger.info(f"Model type: {model_type}; user_encoder={cfg.get('user_encoder') or cfg.get('user_encoder_type')}")

    predictions_dir = train_cell_dir / "predictions"
    predictions_dir.mkdir(exist_ok=True)

    holdout_types = ["unseen_users", "seen_users"] if args.holdout_type == "both" else [args.holdout_type]

    pending: List[str] = []
    for ht in holdout_types:
        out_path = predictions_dir / f"holdout_{ht}.parquet"
        if out_path.exists() and not args.overwrite:
            logger.info(f"Skipping {ht}: already exists at {out_path}")
        else:
            pending.append(ht)
    if not pending:
        logger.info("Nothing to do.")
        return 0

    # Resolve run_dir + load upstream data
    run_dir = _resolve_run_dir(train_cell_dir)
    context = Context(run_dir=run_dir, use_latest=True)
    logger.info(f"Loading upstream data from run_dir={run_dir}")
    embeddings_mmap, target_posts_df, history_df, embed_dim = load_training_data(
        run_dir=run_dir, context=context, logger=logger,
    )

    # Load model via the bias-diagnostic loader (one place for arch-aware code).
    # Mirrors _find_checkpoint's selection: prefer the rich-metadata
    # checkpoint (e.g. ``engagement_model_<timestamp>.pth``) over the
    # bare ``_best.pth`` / ``_best_weights.pth`` files which lack
    # constructor hyperparameters.
    ckpt_glob = _synthetic_feed._MODEL_TYPE_TO_CKPT_GLOB.get(model_type)
    if ckpt_glob is None:
        logger.error(f"Unknown model_type: {model_type}")
        return 2
    ckpt_dir = train_cell_dir / "checkpoints"
    candidates = sorted(
        ckpt_dir.glob(ckpt_glob),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        logger.error(f"No checkpoints matching '{ckpt_glob}' in {ckpt_dir}")
        return 2
    full_ckpts = [
        c for c in candidates
        if "_best" not in c.stem and "_weights" not in c.stem
    ]
    ckpt_path = full_ckpts[0] if full_ckpts else candidates[0]
    logger.info(f"Checkpoint: {ckpt_path.name}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    model, _ckpt = _synthetic_feed._load_model(ckpt_path, device)

    overall_t0 = time.time()
    written: List[str] = []
    for ht in pending:
        split_name = f"holdout_{ht}"
        t0 = time.time()
        try:
            dataset = _build_dataset(
                split_name=split_name,
                embeddings_mmap=embeddings_mmap,
                target_posts_df=target_posts_df,
                history_df=history_df,
                cfg=cfg,
                embed_dim=embed_dim,
                logger=logger,
            )
        except Exception as e:
            logger.warning(f"Failed to build dataset for {split_name}: {e}")
            continue
        if len(dataset) == 0:
            logger.info(f"Dataset for {split_name} is empty; skipping.")
            continue
        logger.info(f"Scoring {len(dataset)} rows for {split_name}...")
        uids, pids, y_true, y_pred = _score_dataset(
            model=model,
            dataset=dataset,
            model_type=model_type,
            embed_dim=embed_dim,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        out_path = predictions_dir / f"{split_name}.parquet"
        pl.DataFrame({
            "did": uids,
            "post_id": pids,
            "y_true": y_true,
            "y_pred_proba": y_pred,
        }).write_parquet(out_path)
        written.append(str(out_path))
        logger.info(f"Wrote {out_path} ({len(y_true)} rows) in {time.time() - t0:.1f}s")

    logger.info(f"All done in {time.time() - overall_t0:.1f}s. Wrote: {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
