#!/usr/bin/env python3

"""
One-off recovery script: backfill the missing ``inferences_core_*.parquet``
artifact for a Stage 1 run that crashed during inference loading.

Used to recover ``outputs/20260429_143355`` after a JSON-decode bug killed
Stage 1 right after the heavy work (likes_core / posts_core / embeddings)
had already landed on disk.  The bug itself is fixed in
``utils/01_get_data/stage_get_data.py:_load_inferences_core_polars``; this
script just lets us avoid re-doing the ~5 hour GCS download + 2.5 hour
embedding write.

Usage (from the repo root):

    python3 scripts/recover_inferences.py \\
        --run-dir outputs/20260429_143355 \\
        --posts-start 2026-04-01 --posts-end 2026-04-28 \\
        [--gcs-bucket greenearth-471522-ingex-extract-stage]

The script writes:
- ``<run-dir>/01_get_data/<ts>/inferences_core_<ts>.parquet``
- ``<run-dir>/01_get_data/<ts>/stage_info.txt`` (overwriting any existing one)
- ``<run-dir>/01_get_data/<ts>/summary.json``  (overwriting any existing one)

so the run becomes a fully valid 01_get_data output that downstream
stages (and the cap-arch sweep harness) will pick up via
``select_prior_output``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import logging
from importlib.util import module_from_spec, spec_from_file_location

import polars as pl

from utils.helpers import parse_one_ts
from utils.pipeline.core import list_stage_outputs


def _load_stage_get_data():
    """Load utils/01_get_data/stage_get_data.py as a module.

    The directory name ``01_get_data`` starts with a digit, so it cannot
    be imported with ``from utils.01_get_data...`` syntax.  We follow the
    same pattern as ``utils.pipeline.core.load_run_callable``.
    """
    module_path = REPO_ROOT / "utils" / "01_get_data" / "stage_get_data.py"
    spec = spec_from_file_location("stage_get_data", str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module spec for {module_path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_stage = _load_stage_get_data()
_list_files_with_timestamps_ingex_gcs = _stage._list_files_with_timestamps_ingex_gcs
_load_inferences_core_polars = _stage._load_inferences_core_polars


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, type=Path,
                   help="Pipeline run directory containing 01_get_data/<ts>/")
    p.add_argument("--posts-start", required=True,
                   help="ISO date string for start of posts window (e.g. 2026-04-01)")
    p.add_argument("--posts-end", required=True,
                   help="ISO date string for end of posts window (e.g. 2026-04-28)")
    p.add_argument("--gcs-bucket", default="greenearth-471522-ingex-extract-stage")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("recover_inferences")

    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = (REPO_ROOT / run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    stage_outs = list_stage_outputs(run_dir, "01_get_data")
    if not stage_outs:
        raise SystemExit(f"No 01_get_data subdirs found in {run_dir}")
    out_dir: Path = stage_outs[0]
    ts = out_dir.name
    logger.info(f"Recovering Stage 1 inferences for run {run_dir.name} (stage_run_id={ts})")

    posts_core_candidates = sorted(out_dir.glob("posts_core_*.parquet"))
    if not posts_core_candidates:
        raise SystemExit(f"No posts_core_*.parquet found in {out_dir}")
    posts_core_path = posts_core_candidates[0]
    likes_core_candidates = sorted(out_dir.glob("likes_core_*.parquet"))
    if not likes_core_candidates:
        raise SystemExit(f"No likes_core_*.parquet found in {out_dir}")
    likes_core_path = likes_core_candidates[0]
    embeddings_candidates = sorted(out_dir.glob("embeddings_*.npy"))
    embeddings_path = embeddings_candidates[0] if embeddings_candidates else None

    logger.info(f"posts_core: {posts_core_path.name}")
    logger.info(f"likes_core: {likes_core_path.name}")
    logger.info(f"embeddings: {embeddings_path.name if embeddings_path else 'NONE'}")

    posts_start_dt = parse_one_ts(args.posts_start)
    posts_end_dt = parse_one_ts(args.posts_end)

    inferences_paths, _ = _list_files_with_timestamps_ingex_gcs(
        gcs_bucket=args.gcs_bucket,
        blob_prefix="bsky_inferences",
        start=posts_start_dt,
        end=posts_end_dt,
    )
    logger.info(f"Found {len(inferences_paths):,} bsky_inferences files in window")

    if not inferences_paths:
        raise SystemExit("No bsky_inferences files in the requested window")

    t0 = time.time()
    logger.info("Loading posts_core (at_uri only)...")
    posts_core_df = pl.read_parquet(posts_core_path, columns=["at_uri"])
    n_posts = len(posts_core_df)
    logger.info(f"Loaded {n_posts:,} post URIs")

    logger.info("Scanning + decoding inference parquets (uses fixed _load_inferences_core_polars)...")
    inferences_df, inference_stats = _load_inferences_core_polars(
        posts_core_df=posts_core_df,
        inferences_paths=inferences_paths,
        logger=logger,
    )
    n_inferences = len(inferences_df)
    logger.info(f"Decoded {n_inferences:,} inference rows ({inference_stats.get('coverage_pct', 0):.1f}% coverage)")

    inferences_core_path = out_dir / f"inferences_core_{ts}.parquet"
    inferences_df.write_parquet(inferences_core_path, compression="zstd")
    logger.info(f"Wrote {inferences_core_path}")

    n_likes = pl.scan_parquet(likes_core_path).select(pl.len()).collect().item()
    embed_dim = 0
    if embeddings_path is not None:
        match = re.search(r"embeddings_(.+)\.npy", embeddings_path.name)
        if match:
            try:
                import numpy as np
                arr = np.load(embeddings_path, mmap_mode="r")
                embed_dim = int(arr.shape[1])
            except Exception as e:  # pragma: no cover - best-effort
                logger.warning(f"Could not infer embedding dim from {embeddings_path}: {e}")

    info_lines = [
        "stage: get_data",
        "runtime_seconds: NA (recovered)",
        "settings: recovered_via=scripts/recover_inferences.py",
        f"inputs: GCS bucket={args.gcs_bucket}",
        f"N_likes_core: {n_likes}",
        f"N_posts_core: {n_posts}",
        f"embedding_dim: {embed_dim}",
        f"embeddings_file: {embeddings_path.name if embeddings_path is not None else 'NONE'}",
        f"N_inferences_core: {n_inferences}",
        f"inferences_file: {inferences_core_path.name}",
    ]
    (out_dir / "stage_info.txt").write_text("\n".join(info_lines) + "\n")
    logger.info(f"Wrote {out_dir / 'stage_info.txt'}")

    summary = {
        "recovered": True,
        "recovery_script": "scripts/recover_inferences.py",
        "gcs_bucket": args.gcs_bucket,
        "posts_start": args.posts_start,
        "posts_end": args.posts_end,
        "outputs": {
            "likes_core_rows": n_likes,
            "posts_core_rows": n_posts,
            "embedding_dim": embed_dim,
            "embeddings_file": embeddings_path.name if embeddings_path is not None else None,
            "inferences_core_rows": n_inferences,
            "inferences_file": inferences_core_path.name,
        },
        "inference_stats": inference_stats,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {out_dir / 'summary.json'}")

    elapsed = time.time() - t0
    logger.info(f"Recovery complete in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
