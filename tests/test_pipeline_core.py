import os
import json
import argparse
from pathlib import Path

from utils.pipeline.core import Context, select_prior_output, list_stage_outputs


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


def test_list_stage_outputs_sorts_by_timestamp_then_mtime(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    stage_folder = "02_target_posts"
    base = artifacts_dir / stage_folder
    base.mkdir(parents=True, exist_ok=True)

    # Newest timestamp should win even if its mtime is older.
    older_ts_newer_mtime = base / "20240101_000000_abcd1234"
    newer_ts_older_mtime = base / "20240102_000000_zzzz9999"
    older_ts_newer_mtime.mkdir(parents=True, exist_ok=True)
    newer_ts_older_mtime.mkdir(parents=True, exist_ok=True)
    os.utime(older_ts_newer_mtime, (200, 200))
    os.utime(newer_ts_older_mtime, (100, 100))

    # Same timestamp: tie-break by mtime.
    same_ts_older_mtime = base / "20240103_000000_tag_11111111"
    same_ts_newer_mtime = base / "20240103_000000_tag_22222222"
    same_ts_older_mtime.mkdir(parents=True, exist_ok=True)
    same_ts_newer_mtime.mkdir(parents=True, exist_ok=True)
    os.utime(same_ts_older_mtime, (10, 10))
    os.utime(same_ts_newer_mtime, (20, 20))

    outs = list_stage_outputs(artifacts_dir=artifacts_dir, stage_folder=stage_folder)
    assert outs[0] == same_ts_newer_mtime
    assert outs[1] == same_ts_older_mtime
    assert outs[2] == newer_ts_older_mtime
    assert outs[3] == older_ts_newer_mtime


def test_stage_metadata_json_includes_nulls(tmp_path):
    run_dir = Path(tmp_path) / "runs" / "run1"
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", pipeline_run_id="run1")
    ctx.begin_stage("target_posts", "02_target_posts")
    out_dir = ctx.new_stage_dir(tag="test")

    args = argparse.Namespace(foo=None, bar="baz")
    ctx.finalize_stage(stage_key="target_posts", stage_folder="02_target_posts", output_dir=out_dir, args=args, argv=None)

    manifest_path = out_dir / "manifest.json"
    resolved_config_path = out_dir / "resolved_config.json"
    assert manifest_path.exists()
    assert resolved_config_path.exists()

    manifest = json.loads(manifest_path.read_text())
    assert "argv" in manifest
    assert manifest["argv"] is None

    resolved = json.loads(resolved_config_path.read_text())
    assert "foo" in resolved
    assert resolved["foo"] is None


def test_new_stage_dir_rejects_mismatched_stage_folder_when_active(tmp_path):
    run_dir = Path(tmp_path) / "runs" / "run1"
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=True)

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", pipeline_run_id="run1")
    ctx.begin_stage("target_posts", "02_target_posts")

    try:
        ctx.new_stage_dir("03_user_history")
        assert False, "Expected ValueError for mismatched stage folder"
    except ValueError as e:
        assert "mismatch" in str(e).lower()
