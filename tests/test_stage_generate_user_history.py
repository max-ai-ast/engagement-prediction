from __future__ import annotations

from datetime import datetime
from importlib import util
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal


def _load_stage_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "utils" / "02_featurize" / "stage_generate_user_history.py"
    spec = util.spec_from_file_location("stage_generate_user_history", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load stage_generate_user_history module")
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _collect_sorted(lf: pl.LazyFrame) -> pl.DataFrame:
    return lf.collect().sort(["did", "record_created_at_bucket", "subject_uri"])


def test_generate_user_history_expands_buckets_and_uniques():
    stage = _load_stage_module()
    likes_core_lf = pl.DataFrame(
        {
            "did": ["u1", "u1", "u2"],
            "record_created_at": [
                datetime(2024, 1, 1, 0, 5),
                datetime(2024, 1, 1, 0, 45),
                datetime(2024, 1, 1, 10, 10),
            ],
            "subject_uri": ["s1", "s1", "s2"],
        }
    ).lazy()

    out = _collect_sorted(
        stage._generate_user_history_from_likes(
            likes_core_lf,
            bucket_duration="hourly",
            num_buckets_lookback=2,
            max_likes_per_bucket=None,
            random_seed=0,
        )
    )

    expected = pl.DataFrame(
        {
            "did": ["u1", "u1", "u2", "u2"],
            "record_created_at_bucket": [
                datetime(2024, 1, 1, 1, 0),
                datetime(2024, 1, 1, 2, 0),
                datetime(2024, 1, 1, 11, 0),
                datetime(2024, 1, 1, 12, 0),
            ],
            "subject_uri": ["s1", "s1", "s2", "s2"],
        }
    ).sort(["did", "record_created_at_bucket", "subject_uri"])

    assert_frame_equal(out, expected)


def test_generate_user_history_sampling_caps_to_max():
    stage = _load_stage_module()
    likes_core_lf = pl.DataFrame(
        {
            "did": ["u1", "u1", "u1"],
            "record_created_at": [
                datetime(2024, 1, 1, 0, 1),
                datetime(2024, 1, 1, 0, 2),
                datetime(2024, 1, 1, 0, 3),
            ],
            "subject_uri": ["s1", "s2", "s3"],
        }
    ).lazy()

    out = _collect_sorted(
        stage._generate_user_history_from_likes(
            likes_core_lf,
            bucket_duration="hourly",
            num_buckets_lookback=1,
            max_likes_per_bucket=2,
            random_seed=1,
        )
    )

    assert out.height == 2
    assert set(out["subject_uri"]) <= {"s1", "s2", "s3"}


def test_generate_user_history_sampling_handles_small_groups():
    stage = _load_stage_module()
    likes_core_lf = pl.DataFrame(
        {
            "did": ["u1"],
            "record_created_at": [datetime(2024, 1, 1, 0, 1)],
            "subject_uri": ["s1"],
        }
    ).lazy()

    out = _collect_sorted(
        stage._generate_user_history_from_likes(
            likes_core_lf,
            bucket_duration="hourly",
            num_buckets_lookback=1,
            max_likes_per_bucket=5,
            random_seed=7,
        )
    )

    assert out.height == 1
    assert out["subject_uri"][0] == "s1"


def test_generate_user_history_zero_lookback_returns_null_bucket():
    stage = _load_stage_module()
    likes_core_lf = pl.DataFrame(
        {
            "did": ["u1"],
            "record_created_at": [datetime(2024, 1, 1, 0, 1)],
            "subject_uri": ["s1"],
        }
    ).lazy()

    out = _collect_sorted(
        stage._generate_user_history_from_likes(
            likes_core_lf,
            bucket_duration="hourly",
            num_buckets_lookback=0,
            max_likes_per_bucket=None,
            random_seed=None,
        )
    )

    assert out.height == 1
    assert out["record_created_at_bucket"][0] is None
