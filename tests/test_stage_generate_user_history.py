"""
Tests for stage_generate_user_history.py (user-hour history directory).
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture(scope="session")
def stage_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "utils" / "02_user_history" / "stage_generate_user_history.py"
    spec = importlib.util.spec_from_file_location("stage_generate_user_history", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage_generate_user_history"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def build_history(stage_module):
    return stage_module._build_user_history_directory


def _make_test_logger() -> logging.Logger:
    logger = logging.getLogger("test_user_history")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    return logger


def _hour(ts: datetime) -> datetime:
    return ts.replace(minute=0, second=0, microsecond=0)


def _make_likes(
    dids: list[str],
    timestamps: list[datetime],
    subject_uris: list[str],
    emb_idxs: list[int],
    author_idxs: list[int | None] | None = None,
    like_hour_buckets: list[datetime] | None = None,
) -> pl.LazyFrame:
    data = {
        "did": dids,
        "record_created_at": timestamps,
        "like_hour_bucket": like_hour_buckets or [_hour(ts) for ts in timestamps],
        "subject_uri": subject_uris,
        "emb_idx": emb_idxs,
    }
    if author_idxs is not None:
        data["author_idx"] = author_idxs
    return pl.DataFrame(data).lazy()


def _history_by_bucket(df: pl.DataFrame) -> dict[datetime, list[int]]:
    return {
        row["like_hour_bucket"]: list(row["prior_emb_indices"])
        for row in df.iter_rows(named=True)
    }


def _history_ages_by_bucket(df: pl.DataFrame) -> dict[datetime, list[float]]:
    return {
        row["like_hour_bucket"]: list(row["prior_like_age_hours_at_bucket_start"])
        for row in df.iter_rows(named=True)
    }


def test_user_hour_history_preserves_empty_first_bucket(build_history):
    logger = _make_test_logger()
    likes_lf = _make_likes(
        ["u1", "u1", "u1"],
        [
            datetime(2024, 1, 1, 10, 15),
            datetime(2024, 1, 1, 11, 20),
            datetime(2024, 1, 1, 12, 5),
        ],
        ["p1", "p2", "p3"],
        [100, 200, 300],
    )

    result = build_history(
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect().sort("like_hour_bucket")

    assert result.height == 3
    histories = _history_by_bucket(result)
    assert histories[datetime(2024, 1, 1, 10)] == []
    assert histories[datetime(2024, 1, 1, 11)] == [100]
    assert histories[datetime(2024, 1, 1, 12)] == [200, 100]
    age_histories = _history_ages_by_bucket(result)
    assert age_histories[datetime(2024, 1, 1, 10)] == []
    assert age_histories[datetime(2024, 1, 1, 11)] == pytest.approx([0.75])
    assert age_histories[datetime(2024, 1, 1, 12)] == pytest.approx([2.0 / 3.0, 1.75])
    assert result.filter(pl.col("like_hour_bucket") == datetime(2024, 1, 1, 10))["raw_prior_count"][0] == 0


def test_user_hour_history_recency_ordering_and_capping(build_history):
    logger = _make_test_logger()
    likes_lf = _make_likes(
        ["u1", "u1", "u1", "u1"],
        [
            datetime(2024, 1, 5, 0, 0),
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 10, 0, 0),
            datetime(2024, 1, 7, 0, 0),
        ],
        ["p1", "p2", "p3", "p4"],
        [10, 20, 30, 40],
    )

    result = build_history(
        likes_lf=likes_lf,
        max_prior_likes=2,
        logger=logger,
    ).collect()

    row = result.filter(pl.col("like_hour_bucket") == datetime(2024, 1, 10))
    assert row["prior_emb_indices"][0].to_list() == [40, 10]
    assert row["prior_like_age_hours_at_bucket_start"][0].to_list() == pytest.approx([72.0, 120.0])
    assert row["raw_prior_count"][0] == 3


def test_user_hour_history_excludes_same_hour_likes(build_history):
    logger = _make_test_logger()
    likes_lf = _make_likes(
        ["u1", "u1", "u1"],
        [
            datetime(2024, 1, 1, 10, 5),
            datetime(2024, 1, 1, 11, 10),
            datetime(2024, 1, 1, 11, 50),
        ],
        ["p1", "p2", "p3"],
        [1, 2, 3],
    )

    result = build_history(
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    row = result.filter(pl.col("like_hour_bucket") == datetime(2024, 1, 1, 11))
    assert row["prior_emb_indices"][0].to_list() == [1]
    assert row["prior_like_age_hours_at_bucket_start"][0].to_list() == pytest.approx([55.0 / 60.0])
    assert row["raw_prior_count"][0] == 1


def test_user_hour_history_multiple_users(build_history):
    logger = _make_test_logger()
    likes_lf = _make_likes(
        ["u1", "u1", "u2", "u2"],
        [
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 2, 0, 0),
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 3, 0, 0),
        ],
        ["a1", "a2", "b1", "b2"],
        [1, 2, 11, 12],
    )

    result = build_history(
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    histories = {
        (row["did"], row["like_hour_bucket"]): list(row["prior_emb_indices"])
        for row in result.iter_rows(named=True)
    }
    assert histories[("u1", datetime(2024, 1, 2))] == [1]
    assert histories[("u2", datetime(2024, 1, 3))] == [11]


def test_user_hour_history_output_schema(build_history):
    logger = _make_test_logger()
    likes_lf = _make_likes(
        ["u1"],
        [datetime(2024, 1, 1, 0, 0)],
        ["p1"],
        [100],
    )

    result = build_history(
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    assert result.columns == [
        "did",
        "like_hour_bucket",
        "prior_emb_indices",
        "raw_prior_count",
        "prior_like_age_hours_at_bucket_start",
    ]
    assert result.schema["prior_emb_indices"] == pl.List(pl.UInt32)
    assert result.schema["prior_like_age_hours_at_bucket_start"] == pl.List(pl.Float32)


def test_user_hour_author_indices_preserve_order_and_unknowns(build_history):
    logger = _make_test_logger()
    likes_lf = _make_likes(
        ["u1", "u1", "u1", "u1"],
        [
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            datetime(2024, 1, 1, 12, 0),
            datetime(2024, 1, 1, 13, 0),
        ],
        ["p1", "p2", "p3", "p4"],
        [100, 200, 300, 400],
        author_idxs=[2, None, 4, 9],
    )

    result = build_history(
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    row = result.filter(pl.col("like_hour_bucket") == datetime(2024, 1, 1, 13))
    assert row["prior_emb_indices"][0].to_list() == [300, 200, 100]
    assert row["prior_like_age_hours_at_bucket_start"][0].to_list() == pytest.approx([1.0, 2.0, 3.0])
    assert row["prior_author_indices"][0].to_list() == [4, None, 2]


def test_user_hour_without_author_idx_omits_author_history(build_history):
    logger = _make_test_logger()
    likes_lf = _make_likes(
        ["u1"],
        [datetime(2024, 1, 1, 10, 0)],
        ["p1"],
        [100],
    )

    result = build_history(
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    assert "prior_author_indices" not in result.columns
