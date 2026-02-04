import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture(scope="session")
def stage_target_posts_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "utils/02_target_posts/stage_target_posts.py"
    spec = importlib.util.spec_from_file_location("stage_target_posts", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage_target_posts"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _dt(y, m, d, h=0, mi=0):
    return datetime(y, m, d, h, mi, tzinfo=timezone.utc)


def test_get_liked_target_posts_selects_expected_columns(stage_target_posts_module):
    likes_df = pl.DataFrame(
        [
            {
                "did": "user_a",
                "record_created_at": _dt(2024, 1, 1),
                "subject_uri": "post:1",
                "emb_idx": 5,
                "extra_col": "ignored",
            }
        ]
    )

    out = stage_target_posts_module._get_liked_target_posts(likes_df.lazy()).collect()

    assert out.columns == [
        "did",
        "record_created_at",
        "subject_uri",
        "emb_idx",
        "was_liked",
    ]
    assert out["was_liked"].to_list() == [True]


def test_negative_target_posts_deterministic_and_bucketed(stage_target_posts_module):
    args = argparse.Namespace(random_seed=7, neg_sample_bucket="1d")

    posts_df = pl.DataFrame(
        [
            {"at_uri": "post:1", "record_created_at": _dt(2024, 1, 1, 1), "emb_idx": 1},
            {"at_uri": "post:2", "record_created_at": _dt(2024, 1, 1, 2), "emb_idx": 2},
            {"at_uri": "post:3", "record_created_at": _dt(2024, 1, 2, 1), "emb_idx": 3},
            {"at_uri": "post:4", "record_created_at": _dt(2024, 1, 2, 2), "emb_idx": 4},
        ]
    )
    likes_df = pl.DataFrame(
        [
            {"did": "user_a", "subject_uri": "post:1", "record_created_at": _dt(2024, 1, 1, 12), "emb_idx": 10},
            {"did": "user_b", "subject_uri": "post:4", "record_created_at": _dt(2024, 1, 2, 12), "emb_idx": 20},
        ]
    )

    liked_lf = stage_target_posts_module._get_liked_target_posts(likes_df.lazy())
    first = stage_target_posts_module._get_negative_target_posts(args, posts_df.lazy(), liked_lf).collect()
    second = stage_target_posts_module._get_negative_target_posts(args, posts_df.lazy(), liked_lf).collect()

    assert first.height == likes_df.height
    assert first["subject_uri"].to_list() == second["subject_uri"].to_list()

    expected_bucket = {
        "user_a": datetime(2024, 1, 1, tzinfo=timezone.utc).date(),
        "user_b": datetime(2024, 1, 2, tzinfo=timezone.utc).date(),
    }
    for did, ts in zip(first["did"].to_list(), first["record_created_at"].to_list()):
        assert ts.date() == expected_bucket[did]


def test_negative_target_posts_requires_bucket(stage_target_posts_module):
    args = argparse.Namespace(random_seed=0, neg_sample_bucket=None)
    posts_df = pl.DataFrame(
        [{"at_uri": "post:1", "record_created_at": _dt(2024, 1, 1), "emb_idx": 1}]
    )
    likes_df = pl.DataFrame(
        [{"did": "user_a", "subject_uri": "post:1", "record_created_at": _dt(2024, 1, 1), "emb_idx": 1}]
    )

    liked_lf = stage_target_posts_module._get_liked_target_posts(likes_df.lazy())
    with pytest.raises(ValueError, match="bucket size"):
        stage_target_posts_module._get_negative_target_posts(args, posts_df.lazy(), liked_lf).collect()


def test_get_target_posts_combines_pos_and_neg(stage_target_posts_module):
    args = argparse.Namespace(random_seed=42, neg_sample_bucket="1d")
    posts_df = pl.DataFrame(
        [
            {"at_uri": "post:1", "record_created_at": _dt(2024, 1, 1), "emb_idx": 1},
            {"at_uri": "post:2", "record_created_at": _dt(2024, 1, 1, 1), "emb_idx": 2},
        ]
    )
    likes_df = pl.DataFrame(
        [
            {"did": "user_a", "subject_uri": "post:1", "record_created_at": _dt(2024, 1, 1, 2), "emb_idx": 5},
            {"did": "user_b", "subject_uri": "post:2", "record_created_at": _dt(2024, 1, 1, 3), "emb_idx": 6},
        ]
    )

    out = stage_target_posts_module._get_target_posts(args, posts_df.lazy(), likes_df.lazy()).collect()

    assert out.height == 4
    assert out.filter(pl.col("was_liked") == True).height == 2
    assert out.filter(pl.col("was_liked") == False).height == 2


def test_get_train_start_fallbacks(stage_target_posts_module):
    args = argparse.Namespace(train_start="2024-01-02", posts_start="2024-01-03", likes_start="2024-01-04")
    assert stage_target_posts_module._get_train_start(args) == "2024-01-02"

    args = argparse.Namespace(train_start=None, posts_start="2024-01-03", likes_start="2024-01-04")
    assert stage_target_posts_module._get_train_start(args) == "2024-01-03"

    args = argparse.Namespace(train_start=None, posts_start=None, likes_start="2024-01-04")
    assert stage_target_posts_module._get_train_start(args) == "2024-01-04"


def test_get_train_start_missing_raises(stage_target_posts_module):
    args = argparse.Namespace(train_start=None, posts_start=None, likes_start=None)
    with pytest.raises(ValueError, match="train window start"):
        stage_target_posts_module._get_train_start(args)


def test_apply_temporal_splits_with_holdout(stage_target_posts_module):
    args = argparse.Namespace(
        train_start="2024-01-02T00:00:00",
        posts_start=None,
        likes_start=None,
        val_start="2024-01-04T00:00:00",
        holdout_start="2024-01-06T00:00:00",
    )
    df = pl.DataFrame(
        {
            "record_created_at": [
                _dt(2024, 1, 1),
                _dt(2024, 1, 2),
                _dt(2024, 1, 4),
                _dt(2024, 1, 6),
                _dt(2024, 1, 7),
            ],
            "did": ["u1"] * 5,
            "subject_uri": ["p1", "p2", "p3", "p4", "p5"],
            "emb_idx": [0, 1, 2, 3, 4],
            "was_liked": [True] * 5,
        }
    )

    out = stage_target_posts_module._apply_temporal_splits(args, df.lazy()).collect()
    assert out["split"].to_list() == [None, "train", "val", "holdout", "holdout"]


def test_apply_temporal_splits_validation_checks(stage_target_posts_module):
    args = argparse.Namespace(
        train_start="2024-01-03T00:00:00",
        posts_start=None,
        likes_start=None,
        val_start="2024-01-02T00:00:00",
        holdout_start=None,
    )
    df = pl.DataFrame(
        {
            "record_created_at": [_dt(2024, 1, 2)],
            "did": ["u1"],
            "subject_uri": ["p1"],
            "emb_idx": [0],
            "was_liked": [True],
        }
    )

    with pytest.raises(ValueError, match="Train start date"):
        stage_target_posts_module._apply_temporal_splits(args, df.lazy()).collect()

    args = argparse.Namespace(
        train_start="2024-01-01T00:00:00",
        posts_start=None,
        likes_start=None,
        val_start="2024-01-02T00:00:00",
        holdout_start="2024-01-01T12:00:00",
    )
    with pytest.raises(ValueError, match="holdout start"):
        stage_target_posts_module._apply_temporal_splits(args, df.lazy()).collect()

    args = argparse.Namespace(
        train_start="2024-01-01T00:00:00",
        posts_start=None,
        likes_start=None,
        val_start=None,
        holdout_start=None,
    )
    with pytest.raises(ValueError, match="Validation window start"):
        stage_target_posts_module._apply_temporal_splits(args, df.lazy()).collect()
