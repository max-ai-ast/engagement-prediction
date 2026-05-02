"""
run_sweep_eval.py — run Stage 5 eval in parallel for all cells in a sweep.

Usage:
    python3 scripts/run_sweep_eval.py <sweep_root> [--max-workers N]

Finds every 04_train/<ts>_<run_tag>/ cell under sweep_root, derives the
cap-level output dir and run tag, and runs:
    cli.py --output-dir <cap_dir> --start-from evaluate --stop-after evaluate
           --run-tag <run_tag>
in parallel. Skips cells that already have a bias_by_trait_*.parquet artifact.
"""

import argparse
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [run_sweep_eval] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def find_cells(sweep_root: Path):
    """Return list of (cap_dir, run_tag, cell_dir) for every trained cell."""
    cells = []
    for cfg in sorted(sweep_root.rglob("training_config.json")):
        cell_dir = cfg.parent
        # cell_dir = <sweep_root>/<cap_label>/04_train/<ts>_<run_tag>/
        cap_dir = cell_dir.parent.parent
        # run_tag = directory name minus leading timestamp (YYYYMMDD_HHMMSS_)
        parts = cell_dir.name.split("_", 2)
        run_tag = parts[2] if len(parts) == 3 else cell_dir.name
        cells.append((cap_dir, run_tag, cell_dir))
    return cells


def already_evaluated(cell_dir: Path) -> bool:
    bias_parquets = list(
        (cell_dir / "evals").rglob("bias_by_trait_*.parquet")
        if (cell_dir / "evals").exists()
        else []
    )
    return len(bias_parquets) > 0


def run_eval(cap_dir: Path, run_tag: str, cell_dir: Path, log_dir: Path) -> bool:
    label = f"{cap_dir.name}/{run_tag}"

    if already_evaluated(cell_dir):
        log.info(f"[{label}] already evaluated — skipping")
        return True

    log_file = log_dir / f"eval_{cap_dir.name}_{run_tag}.log"
    cmd = [
        "python3", "cli.py",
        "--output-dir", str(cap_dir),
        "--start-from", "evaluate",
        "--stop-after", "evaluate",
        "--run-tag", run_tag,
    ]
    log.info(f"[{label}] Starting eval")
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        log.info(f"[{label}] ✓ eval complete")
        return True
    else:
        log.error(f"[{label}] ✗ eval failed (exit={result.returncode}) — see {log_file}")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sweep_root", help="Path to sweep root dir")
    p.add_argument("--max-workers", type=int, default=1,
                   help="Max parallel eval workers (default 4)")
    args = p.parse_args()

    sweep_root = Path(args.sweep_root).resolve()
    if not sweep_root.is_dir():
        sys.exit(f"sweep_root not found: {sweep_root}")

    log_dir = sweep_root / "sweep_logs"
    log_dir.mkdir(exist_ok=True)

    cells = find_cells(sweep_root)
    log.info(f"Found {len(cells)} cells in {sweep_root.name}")

    passed = failed = skipped = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {
            pool.submit(run_eval, cap_dir, run_tag, cell_dir, log_dir): (cap_dir, run_tag)
            for cap_dir, run_tag, cell_dir in cells
        }
        for fut in as_completed(futures):
            ok = fut.result()
            if ok:
                passed += 1
            else:
                failed += 1

    log.info(f"Eval complete: {passed} passed, {failed} failed out of {len(cells)}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
