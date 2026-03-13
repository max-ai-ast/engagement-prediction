from pathlib import Path

import pytest

import cli


def test_resolve_prior_spec_resolves_stage_run_id(tmp_path):
    output_root = Path(tmp_path) / "out"
    artifacts_dir = output_root / "artifacts"
    stage_folder = "02_target_posts"
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
    stage_folder = "03_user_history"
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

