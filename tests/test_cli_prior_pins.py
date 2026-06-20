import json
from pathlib import Path

import pytest

import cli
from utils.pipeline.core import Context
from utils.pipeline.dependencies import (
    get_stage_folder_to_keys,
    get_stage_input_folders,
    resolve_stage_dependencies_for_run,
    validate_explicit_prior_pin_consistency,
)


def _make_stage_output(
    artifacts_dir: Path,
    stage_folder: str,
    stage_run_id: str,
    *,
    inputs=None,
) -> Path:
    out_dir = artifacts_dir / stage_folder / stage_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage_folder": stage_folder,
        "stage_run_id": stage_run_id,
        "inputs": inputs or {},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest) + "\n")
    return out_dir


def test_resolve_prior_spec_resolves_stage_run_id(tmp_path):
    output_root = Path(tmp_path) / "out"
    artifacts_dir = output_root / "artifacts"
    stage_folder = "02_user_history"
    stage_run_id = "20260102_000000_abcd1234"
    target = artifacts_dir / stage_folder / stage_run_id
    target.mkdir(parents=True)

    resolved = cli._resolve_prior_spec(
        stage_run_id,
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder=stage_folder,
    )

    assert resolved == target.resolve()


def test_resolve_prior_spec_resolves_relative_path_against_output_root(tmp_path):
    output_root = Path(tmp_path) / "out"
    artifacts_dir = output_root / "artifacts"
    stage_folder = "02_user_history"
    p = output_root / "some" / "custom_prior"
    p.mkdir(parents=True)

    resolved = cli._resolve_prior_spec(
        "some/custom_prior",
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        stage_folder=stage_folder,
    )

    assert resolved == p.resolve()


def test_resolve_prior_spec_raises_if_missing(tmp_path):
    output_root = Path(tmp_path) / "out"
    artifacts_dir = output_root / "artifacts"
    with pytest.raises(FileNotFoundError):
        cli._resolve_prior_spec(
            "does_not_exist",
            output_root=output_root,
            artifacts_dir=artifacts_dir,
            stage_folder="01_get_data",
        )


def test_get_stage_folder_to_keys_is_derived_from_registry():
    assert get_stage_folder_to_keys() == {
        "01_get_data": ("get_data",),
        "02_user_history": ("user_history",),
        "03_train": ("train_mlp", "train_two_tower", "train_bst_ranker", "train_din_ranker"),
        "04_evaluate": ("evaluate",),
    }


def test_get_stage_input_folders_is_derived_from_stage_order():
    assert get_stage_input_folders() == {
        "01_get_data": [],
        "02_user_history": ["01_get_data"],
        "03_train": ["01_get_data", "02_user_history"],
        "04_evaluate": ["01_get_data", "02_user_history", "03_train"],
    }


def test_resolve_stage_dependencies_for_train_follows_latest_downstream_lineage(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    get_data_old = _make_stage_output(artifacts_dir, "01_get_data", "20260101_000000_oldget")
    get_data_new = _make_stage_output(artifacts_dir, "01_get_data", "20260105_000000_newget")
    _make_stage_output(
        artifacts_dir,
        "02_user_history",
        "20260103_000000_oldhistory",
        inputs={"01_get_data": str(get_data_old)},
    )
    user_history_new = _make_stage_output(
        artifacts_dir,
        "02_user_history",
        "20260106_000000_newhistory",
        inputs={"01_get_data": str(get_data_new)},
    )

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", use_latest=True)

    resolved = resolve_stage_dependencies_for_run(
        ctx=ctx,
        consumer_stage_folder="03_train",
    )

    assert resolved == {
        "01_get_data": get_data_new.resolve(),
        "02_user_history": user_history_new.resolve(),
    }


def test_resolve_stage_dependencies_raises_on_misaligned_explicit_pins(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    get_data_old = _make_stage_output(artifacts_dir, "01_get_data", "20260101_000000_oldget")
    get_data_new = _make_stage_output(artifacts_dir, "01_get_data", "20260104_000000_newget")
    _make_stage_output(
        artifacts_dir,
        "02_user_history",
        "20260102_000000_oldhistory",
        inputs={"01_get_data": str(get_data_old)},
    )
    user_history_new = _make_stage_output(
        artifacts_dir,
        "02_user_history",
        "20260106_000000_newhistory",
        inputs={"01_get_data": str(get_data_new)},
    )

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", use_latest=True)
    ctx.prior_outputs["01_get_data"] = get_data_old
    ctx.prior_outputs["02_user_history"] = user_history_new

    with pytest.raises(ValueError, match="Misaligned inputs for stage '03_train'"):
        resolve_stage_dependencies_for_run(
            ctx=ctx,
            consumer_stage_folder="03_train",
        )


def test_resolve_stage_dependencies_for_evaluate_infers_inputs_from_train_manifest(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    get_data_old = _make_stage_output(artifacts_dir, "01_get_data", "20260101_000000_oldget")
    _make_stage_output(artifacts_dir, "01_get_data", "20260109_000000_newget")
    user_history_old = _make_stage_output(
        artifacts_dir,
        "02_user_history",
        "20260103_000000_oldhistory",
        inputs={"01_get_data": str(get_data_old)},
    )
    train_old = _make_stage_output(
        artifacts_dir,
        "03_train",
        "20260104_000000_oldtrain",
        inputs={
            "01_get_data": str(get_data_old),
            "02_user_history": str(user_history_old),
        },
    )

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", use_latest=True)

    resolved = resolve_stage_dependencies_for_run(
        ctx=ctx,
        consumer_stage_folder="04_evaluate",
    )

    assert resolved == {
        "01_get_data": get_data_old.resolve(),
        "02_user_history": user_history_old.resolve(),
        "03_train": train_old.resolve(),
    }


def test_validate_explicit_prior_pin_consistency_raises_on_misaligned_stage1_history_pins(tmp_path):
    artifacts_dir = Path(tmp_path) / "artifacts"
    run_dir = Path(tmp_path) / "runs" / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    get_data_old = _make_stage_output(artifacts_dir, "01_get_data", "20260101_000000_oldget")
    get_data_new = _make_stage_output(artifacts_dir, "01_get_data", "20260104_000000_newget")
    user_history_new = _make_stage_output(
        artifacts_dir,
        "02_user_history",
        "20260105_000000_newhistory",
        inputs={"01_get_data": str(get_data_new)},
    )

    ctx = Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=Path(tmp_path) / "runs", use_latest=True)
    ctx.prior_outputs["01_get_data"] = get_data_old
    ctx.prior_outputs["02_user_history"] = user_history_new

    with pytest.raises(ValueError, match="Explicit prior pins are inconsistent"):
        validate_explicit_prior_pin_consistency(ctx)
