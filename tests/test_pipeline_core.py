import os
from pathlib import Path

from utils.pipeline.core import select_prior_output


def test_select_prior_output_prefers_latest(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    older = artifacts_dir / "03_user_history" / "20240101_000000"
    newer = artifacts_dir / "03_user_history" / "20240102_000000"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    chosen = select_prior_output(artifacts_dir=artifacts_dir, stage_folder="03_user_history")

    assert chosen == newer


def test_select_prior_output_honors_explicit_prior_path(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    explicit = Path(tmp_path) / "custom_prior"
    other = artifacts_dir / "04_train" / "20240101_000000"
    explicit.mkdir(parents=True)
    other.mkdir(parents=True)

    chosen = select_prior_output(artifacts_dir=artifacts_dir, stage_folder="04_train", prior_path=explicit)

    assert chosen == explicit
