import os
from pathlib import Path

from utils.pipeline.core import select_prior_output


def test_select_prior_output_prefers_latest_alt_stage_dir(tmp_path):
    run_dir = Path(tmp_path)
    older = run_dir / "03_relevel" / "20240101_000000"
    newer = run_dir / "03_relevel" / "20240102_000000"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    chosen = select_prior_output(run_dir, "relevel")

    assert chosen == newer


def test_select_prior_output_honors_explicit_prior_path(tmp_path):
    run_dir = Path(tmp_path)
    explicit = run_dir / "custom_prior"
    other = run_dir / "05_train" / "20240101_000000"
    explicit.mkdir(parents=True)
    other.mkdir(parents=True)

    chosen = select_prior_output(run_dir, "train", prior_path=explicit)

    assert chosen == explicit
