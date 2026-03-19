import argparse
import logging
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


@pytest.fixture
def dummy_logger():
    logger = logging.getLogger("test_stage_target_posts")
    logger.setLevel(logging.INFO)
    return logger


@pytest.fixture
def dummy_context(tmp_path):
    from utils.pipeline.core import Context

    return Context(run_dir=tmp_path)


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
    posts_df = pl.DataFrame(
        [
            {
                "at_uri": "post:1",
                "record_created_at": _dt(2023, 12, 31, 23),
                "did": "author_a",
                "emb_idx": 99,
                "record_text": "hello",
            }
        ]
    )

    out = stage_target_posts_module._get_liked_target_posts(
        likes_df.lazy(), posts_df.lazy()
    ).collect()

    assert out.columns == [
        "target_did",
        "seen_at",
        "like_uri",
        "like_emb_idx",
        "like_posted_at",
        "like_author_did",
    ]
    assert out["target_did"].to_list() == ["user_a"]
    assert out["seen_at"].to_list() == [_dt(2024, 1, 1)]
    assert out["like_uri"].to_list() == ["post:1"]
    assert out["like_emb_idx"].to_list() == [5]
    assert out["like_posted_at"].to_list() == [_dt(2023, 12, 31, 23)]
    assert out["like_author_did"].to_list() == ["author_a"]


def test_negative_target_posts_deterministic_and_bucketed(
    stage_target_posts_module, dummy_logger, dummy_context
):
    args = argparse.Namespace(random_seed=7, neg_sample_bucket="1d")

    posts_df = pl.DataFrame(
        [
            {"at_uri": "post:1", "record_created_at": _dt(2024, 1, 1, 1), "emb_idx": 1, "did": "author_1"},
            {"at_uri": "post:2", "record_created_at": _dt(2024, 1, 1, 2), "emb_idx": 2, "did": "author_2"},
            {"at_uri": "post:3", "record_created_at": _dt(2024, 1, 2, 1), "emb_idx": 3, "did": "author_3"},
            {"at_uri": "post:4", "record_created_at": _dt(2024, 1, 2, 2), "emb_idx": 4, "did": "author_4"},
        ]
    )
    likes_df = pl.DataFrame(
        [
            {"did": "user_a", "subject_uri": "post:1", "record_created_at": _dt(2024, 1, 1, 12), "emb_idx": 10},
            {"did": "user_b", "subject_uri": "post:4", "record_created_at": _dt(2024, 1, 2, 12), "emb_idx": 20},
        ]
    )

    liked_lf = stage_target_posts_module._get_liked_target_posts(
        likes_df.lazy(), posts_df.lazy()
    )
    first = stage_target_posts_module._get_negative_target_posts(
        args, posts_df.lazy(), liked_lf, dummy_logger, dummy_context
    ).collect()
    second = stage_target_posts_module._get_negative_target_posts(
        args, posts_df.lazy(), liked_lf, dummy_logger, dummy_context
    ).collect()

    assert first.height == likes_df.height
    first_sorted = first.sort(["target_did", "like_uri"])
    second_sorted = second.sort(["target_did", "like_uri"])
    assert first_sorted["neg_uri"].to_list() == second_sorted["neg_uri"].to_list()

    joined = (
        first
        .join(
            posts_df.select(
                [
                    pl.col("at_uri").alias("like_uri"),
                    pl.col("record_created_at").alias("like_posted_at"),
                ]
            ),
            on="like_uri",
            how="left",
        )
        .join(
            posts_df.select(
                [
                    pl.col("at_uri").alias("neg_uri"),
                    pl.col("record_created_at").alias("neg_posted_at"),
                ]
            ),
            on="neg_uri",
            how="left",
        )
    )
    for like_ts, neg_ts in zip(
        joined["like_posted_at"].to_list(), joined["neg_posted_at"].to_list()
    ):
        assert like_ts.date() == neg_ts.date()


def test_negative_target_posts_excludes_liked_by_target(
    stage_target_posts_module, dummy_logger, dummy_context
):
    args = argparse.Namespace(random_seed=11, neg_sample_bucket="1d")

    posts_df = pl.DataFrame(
        [
            {"at_uri": "post:1", "record_created_at": _dt(2024, 1, 1, 1), "emb_idx": 1, "did": "author_1"},
            {"at_uri": "post:2", "record_created_at": _dt(2024, 1, 1, 2), "emb_idx": 2, "did": "author_2"},
            {"at_uri": "post:3", "record_created_at": _dt(2024, 1, 1, 3), "emb_idx": 3, "did": "author_3"},
        ]
    )
    likes_df = pl.DataFrame(
        [
            {"did": "user_a", "subject_uri": "post:1", "record_created_at": _dt(2024, 1, 1, 10), "emb_idx": 10},
            {"did": "user_a", "subject_uri": "post:2", "record_created_at": _dt(2024, 1, 1, 11), "emb_idx": 11},
            {"did": "user_b", "subject_uri": "post:3", "record_created_at": _dt(2024, 1, 1, 12), "emb_idx": 12},
        ]
    )

    liked_lf = stage_target_posts_module._get_liked_target_posts(
        likes_df.lazy(), posts_df.lazy()
    )
    out = stage_target_posts_module._get_negative_target_posts(
        args, posts_df.lazy(), liked_lf, dummy_logger, dummy_context
    ).collect()

    liked_pairs = set(zip(likes_df["did"].to_list(), likes_df["subject_uri"].to_list()))
    for did, neg_uri in zip(out["target_did"].to_list(), out["neg_uri"].to_list()):
        assert (did, neg_uri) not in liked_pairs

    user_a_negs = out.filter(pl.col("target_did") == "user_a")["neg_uri"].to_list()
    assert len(user_a_negs) == 2
    assert set(user_a_negs) == {"post:3"}


def test_negative_target_posts_requires_bucket(
    stage_target_posts_module, dummy_logger, dummy_context
):
    args = argparse.Namespace(random_seed=0, neg_sample_bucket=None)
    posts_df = pl.DataFrame(
        [{"at_uri": "post:1", "record_created_at": _dt(2024, 1, 1), "emb_idx": 1, "did": "author_1"}]
    )
    likes_df = pl.DataFrame(
        [{"did": "user_a", "subject_uri": "post:1", "record_created_at": _dt(2024, 1, 1), "emb_idx": 1}]
    )

    liked_lf = stage_target_posts_module._get_liked_target_posts(
        likes_df.lazy(), posts_df.lazy()
    )
    with pytest.raises(ValueError, match="bucket size"):
        stage_target_posts_module._get_negative_target_posts(
            args, posts_df.lazy(), liked_lf, dummy_logger, dummy_context
        ).collect()


def test_resolve_negative_bucket_index_skips_liked_positions(stage_target_posts_module):
    assert stage_target_posts_module._resolve_negative_bucket_index(
        {"liked_idx_list": [1, 4], "neg_rank": 0, "bucket_size": 6}
    ) == 0
    assert stage_target_posts_module._resolve_negative_bucket_index(
        {"liked_idx_list": [1, 4], "neg_rank": 1, "bucket_size": 6}
    ) == 2
    assert stage_target_posts_module._resolve_negative_bucket_index(
        {"liked_idx_list": [1, 4], "neg_rank": 2, "bucket_size": 6}
    ) == 3
    assert stage_target_posts_module._resolve_negative_bucket_index(
        {"liked_idx_list": [1, 4], "neg_rank": 3, "bucket_size": 6}
    ) == 5


def test_get_target_posts_emits_negative_pairs(
    stage_target_posts_module, dummy_logger, dummy_context
):
    args = argparse.Namespace(random_seed=42, neg_sample_bucket="1d")
    posts_df = pl.DataFrame(
        [
            {"at_uri": "post:1", "record_created_at": _dt(2024, 1, 1), "emb_idx": 1, "did": "author_1"},
            {"at_uri": "post:2", "record_created_at": _dt(2024, 1, 1, 1), "emb_idx": 2, "did": "author_2"},
        ]
    )
    likes_df = pl.DataFrame(
        [
            {"did": "user_a", "subject_uri": "post:1", "record_created_at": _dt(2024, 1, 1, 2), "emb_idx": 5},
            {"did": "user_b", "subject_uri": "post:2", "record_created_at": _dt(2024, 1, 1, 3), "emb_idx": 6},
        ]
    )

    out = stage_target_posts_module._get_target_posts(
        args, posts_df.lazy(), likes_df.lazy(), dummy_logger, dummy_context
    ).collect()

    assert out.height == likes_df.height
    assert out.columns == [
        "target_did",
        "seen_at",
        "like_uri",
        "like_emb_idx",
        "like_author_did",
        "neg_uri",
        "neg_emb_idx",
        "neg_author_did",
    ]
    assert out["neg_uri"].null_count() == 0
    assert out["neg_emb_idx"].null_count() == 0
    assert out["neg_author_did"].null_count() == 0


def test_holdout_assignment_stable_when_adding_users(stage_target_posts_module):
    seed = 123
    fraction = 0.5
    dids = ["user_u1", "user_u2", "user_u3"]
    dids_with_extra = dids + ["some_new_user"]

    holdout_before = stage_target_posts_module._select_holdout_users(dids, seed, fraction)
    holdout_after = stage_target_posts_module._select_holdout_users(
        dids_with_extra, seed, fraction
    )

    assert {did: (did in holdout_before) for did in dids} == {
        did: (did in holdout_after) for did in dids
    }


def test_run_regression_snapshot_small_dataset(stage_target_posts_module, tmp_path):
    from utils.pipeline.core import Context

    # --- create a minimal prior 01_get_data output ---
    prior_dir = tmp_path / "01_get_data" / "20240101_000000"
    prior_dir.mkdir(parents=True, exist_ok=True)

    posts_df = pl.DataFrame(
        [
            {
                "did": "author_1",
                "at_uri": "post:1",
                "record_created_at": _dt(2024, 1, 1, 1),
                "emb_idx": 1,
                "record_text": "t1",
                "is_liked": False,
                "in_random_sample": True,
            },
            {
                "did": "author_2",
                "at_uri": "post:2",
                "record_created_at": _dt(2024, 1, 1, 2),
                "emb_idx": 2,
                "record_text": "t2",
                "is_liked": False,
                "in_random_sample": True,
            },
            {
                "did": "author_3",
                "at_uri": "post:3",
                "record_created_at": _dt(2024, 1, 1, 3),
                "emb_idx": 3,
                "record_text": "t3",
                "is_liked": False,
                "in_random_sample": True,
            },
            {
                "did": "author_4",
                "at_uri": "post:4",
                "record_created_at": _dt(2024, 1, 2, 1),
                "emb_idx": 4,
                "record_text": "t4",
                "is_liked": False,
                "in_random_sample": True,
            },
            {
                "did": "author_5",
                "at_uri": "post:5",
                "record_created_at": _dt(2024, 1, 2, 2),
                "emb_idx": 5,
                "record_text": "t5",
                "is_liked": False,
                "in_random_sample": True,
            },
            {
                "did": "author_6",
                "at_uri": "post:6",
                "record_created_at": _dt(2024, 1, 2, 3),
                "emb_idx": 6,
                "record_text": "t6",
                "is_liked": False,
                "in_random_sample": True,
            },
        ]
    )
    likes_df = pl.DataFrame(
        [
            {
                "did": "user_u1",
                "subject_uri": "post:1",
                "record_created_at": _dt(2024, 1, 2, 10),
                "emb_idx": 101,
            },
            {
                "did": "user_u1",
                "subject_uri": "post:2",
                "record_created_at": _dt(2024, 1, 4, 10),
                "emb_idx": 102,
            },
            {
                "did": "user_u2",
                "subject_uri": "post:4",
                "record_created_at": _dt(2024, 1, 6, 10),
                "emb_idx": 201,
            },
            {
                "did": "user_u3",
                "subject_uri": "post:5",
                "record_created_at": _dt(2024, 1, 2, 11),
                "emb_idx": 301,
            },
            {
                "did": "user_u3",
                "subject_uri": "post:6",
                "record_created_at": _dt(2024, 1, 4, 11),
                "emb_idx": 302,
            },
            {
                "did": "user_u3",
                "subject_uri": "post:3",
                "record_created_at": _dt(2024, 1, 6, 11),
                "emb_idx": 303,
            },
        ]
    )

    posts_df.write_parquet(prior_dir / "posts_core_20240101_000000.parquet")
    likes_df.write_parquet(prior_dir / "likes_core_20240101_000000.parquet")

    context = Context(
        run_dir=tmp_path,
        run_timestamp="20240108_000000",
        prior_outputs={"01_get_data": prior_dir},
    )

    args = argparse.Namespace(
        random_seed=7,
        neg_sample_bucket="1d",
        train_start="2024-01-01T00:00:00+0000",
        val_start="2024-01-03T00:00:00+0000",
        holdout_start="2024-01-05T00:00:00+0000",
        holdout_end="2024-01-07T00:00:00+0000",
        holdout_user_fraction=0.5,
        holdout_user_seed=123,
        posts_start=None,
        likes_start=None,
    )

    res = stage_target_posts_module.run(context, args)
    out_path = Path(res["artifacts"]["user_summary_path"])
    out_df = pl.read_parquet(out_path).sort(["target_did", "like_uri"])

    expected = pl.DataFrame(
        [
            {
                "target_did": "user_u1",
                "seen_at": _dt(2024, 1, 2, 10),
                "like_uri": "post:1",
                "like_emb_idx": 101,
                "like_author_did": "author_1",
                "neg_uri": "post:3",
                "neg_emb_idx": 3,
                "neg_author_did": "author_3",
                "split": "holdout_unseen_users",
            },
            {
                "target_did": "user_u1",
                "seen_at": _dt(2024, 1, 4, 10),
                "like_uri": "post:2",
                "like_emb_idx": 102,
                "like_author_did": "author_2",
                "neg_uri": "post:3",
                "neg_emb_idx": 3,
                "neg_author_did": "author_3",
                "split": "holdout_unseen_users",
            },
            {
                "target_did": "user_u2",
                "seen_at": _dt(2024, 1, 6, 10),
                "like_uri": "post:4",
                "like_emb_idx": 201,
                "like_author_did": "author_4",
                "neg_uri": "post:6",
                "neg_emb_idx": 6,
                "neg_author_did": "author_6",
                "split": "holdout_unseen_users",
            },
            {
                "target_did": "user_u3",
                "seen_at": _dt(2024, 1, 6, 11),
                "like_uri": "post:3",
                "like_emb_idx": 303,
                "like_author_did": "author_3",
                "neg_uri": "post:1",
                "neg_emb_idx": 1,
                "neg_author_did": "author_1",
                "split": "holdout_seen_users",
            },
            {
                "target_did": "user_u3",
                "seen_at": _dt(2024, 1, 2, 11),
                "like_uri": "post:5",
                "like_emb_idx": 301,
                "like_author_did": "author_5",
                "neg_uri": "post:4",
                "neg_emb_idx": 4,
                "neg_author_did": "author_4",
                "split": "train",
            },
            {
                "target_did": "user_u3",
                "seen_at": _dt(2024, 1, 4, 11),
                "like_uri": "post:6",
                "like_emb_idx": 302,
                "like_author_did": "author_6",
                "neg_uri": "post:4",
                "neg_emb_idx": 4,
                "neg_author_did": "author_4",
                "split": "val",
            },
        ]
    ).sort(["target_did", "like_uri"])

    assert out_df.columns == expected.columns
    assert out_df.to_dicts() == expected.to_dicts()


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


def _make_split_args(**overrides):
    """Build a Namespace with sensible defaults for _apply_splits tests."""
    defaults = dict(
        train_start="2024-01-01T00:00:00",
        posts_start=None,
        likes_start=None,
        val_start="2024-01-15T00:00:00",
        holdout_user_fraction=0.5,
        holdout_user_seed=42,
        holdout_start=None,
        holdout_end=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _multi_user_df():
    """Four users, each with rows spanning the full time range."""
    rows = []
    for uid in ["u1", "u2", "u3", "u4"]:
        for i, ts in enumerate([
            _dt(2024, 1, 2),   # train window
            _dt(2024, 1, 10),  # train window
            _dt(2024, 1, 16),  # val window
            _dt(2024, 1, 20),  # val window
        ]):
            rows.append({
                "seen_at": ts,
                "target_did": uid,
                "like_uri": f"{uid}_p{i}",
                "like_emb_idx": i,
                "like_author_did": "a1",
                "neg_uri": f"{uid}_n{i}",
                "neg_emb_idx": 10 + i,
                "neg_author_did": "a2",
            })
    return pl.DataFrame(rows)


def test_apply_splits_user_based_holdout(stage_target_posts_module, dummy_logger):
    """Holdout users get split='holdout_unseen_users' on ALL their rows; others get train/val."""
    df = _multi_user_df()
    args = _make_split_args()
    out = stage_target_posts_module._apply_splits(args, df.lazy(), dummy_logger).collect()

    holdout_users = set(
        out.filter(pl.col("split") == "holdout_unseen_users")["target_did"].unique().to_list()
    )
    train_users = set(
        out.filter(pl.col("split") == "train")["target_did"].unique().to_list()
    )
    val_users = set(
        out.filter(pl.col("split") == "val")["target_did"].unique().to_list()
    )

    assert len(holdout_users) > 0, "Expected at least one holdout user"
    assert len(train_users | val_users) > 0, "Expected at least one train/val user"
    assert holdout_users.isdisjoint(train_users), "Holdout users must not appear in train"
    assert holdout_users.isdisjoint(val_users), "Holdout users must not appear in val"

    for uid in holdout_users:
        user_splits = out.filter(pl.col("target_did") == uid)["split"].to_list()
        assert all(s == "holdout_unseen_users" for s in user_splits), (
            f"All rows for holdout user {uid} should be 'holdout_unseen_users', got {user_splits}"
        )


def test_apply_splits_temporal_trainval(stage_target_posts_module, dummy_logger):
    """Non-holdout users' rows are split temporally by val_start."""
    df = _multi_user_df()
    args = _make_split_args()
    out = stage_target_posts_module._apply_splits(args, df.lazy(), dummy_logger).collect()

    holdout_users = set(
        out.filter(pl.col("split") == "holdout_unseen_users")["target_did"].unique().to_list()
    )
    trainval = out.filter(~pl.col("target_did").is_in(holdout_users))
    val_start = _dt(2024, 1, 15)

    for row in trainval.iter_rows(named=True):
        if row["seen_at"] < val_start:
            assert row["split"] == "train", f"Row before val_start should be train: {row}"
        else:
            assert row["split"] == "val", f"Row at/after val_start should be val: {row}"


def test_apply_splits_no_leakage(stage_target_posts_module, dummy_logger):
    """No user ID appears in both holdout_unseen_users and train/val."""
    df = _multi_user_df()
    args = _make_split_args()
    out = stage_target_posts_module._apply_splits(args, df.lazy(), dummy_logger).collect()

    holdout_dids = set(
        out.filter(pl.col("split") == "holdout_unseen_users")["target_did"].unique().to_list()
    )
    trainval_dids = set(
        out.filter(pl.col("split").is_in(["train", "val"]))["target_did"].unique().to_list()
    )
    assert holdout_dids.isdisjoint(trainval_dids)


def test_apply_splits_reproducibility(stage_target_posts_module, dummy_logger):
    """Same seed + fraction produces the same holdout assignment."""
    df = _multi_user_df()
    args = _make_split_args()
    out1 = stage_target_posts_module._apply_splits(args, df.lazy(), dummy_logger).collect()
    out2 = stage_target_posts_module._apply_splits(args, df.lazy(), dummy_logger).collect()
    assert out1["split"].to_list() == out2["split"].to_list()

    args_diff = _make_split_args(holdout_user_seed=999)
    out3 = stage_target_posts_module._apply_splits(args_diff, df.lazy(), dummy_logger).collect()
    h1 = set(out1.filter(pl.col("split") == "holdout_unseen_users")["target_did"].unique().to_list())
    h3 = set(out3.filter(pl.col("split") == "holdout_unseen_users")["target_did"].unique().to_list())
    assert h1 != h3, "Different seeds should (almost certainly) produce different assignments"


def test_apply_splits_holdout_end(stage_target_posts_module, dummy_logger):
    """When holdout_end is set, holdout rows after that date get split=None."""
    df = _multi_user_df()
    args = _make_split_args(holdout_end="2024-01-16T00:00:00")
    out = stage_target_posts_module._apply_splits(args, df.lazy(), dummy_logger).collect()

    holdout_users = set(
        out.filter(pl.col("split") == "holdout_unseen_users")["target_did"].unique().to_list()
    )
    assert len(holdout_users) > 0

    for uid in holdout_users:
        user_rows = out.filter(pl.col("target_did") == uid)
        for row in user_rows.iter_rows(named=True):
            if row["seen_at"] < _dt(2024, 1, 16):
                assert row["split"] == "holdout_unseen_users"
            else:
                assert row["split"] is None, (
                    f"Holdout user {uid} row at {row['seen_at']} should be None "
                    f"(past holdout_end), got {row['split']}"
                )


def test_apply_splits_validation_checks(stage_target_posts_module, dummy_logger):
    """Validation of argument constraints."""
    lf = _multi_user_df().lazy()

    with pytest.raises(ValueError, match="Train start date"):
        stage_target_posts_module._apply_splits(
            _make_split_args(train_start="2024-01-20T00:00:00", val_start="2024-01-15T00:00:00"),
            lf, dummy_logger,
        )

    with pytest.raises(ValueError, match="Validation window start"):
        stage_target_posts_module._apply_splits(
            _make_split_args(val_start=None),
            lf, dummy_logger,
        )

    with pytest.raises(ValueError, match="holdout_user_fraction must be in"):
        stage_target_posts_module._apply_splits(
            _make_split_args(holdout_user_fraction=0.0),
            lf, dummy_logger,
        )

    with pytest.raises(ValueError, match="holdout_user_fraction must be in"):
        stage_target_posts_module._apply_splits(
            _make_split_args(holdout_user_fraction=1.0),
            lf, dummy_logger,
        )

    with pytest.raises(ValueError, match="holdout_start must be after val_start"):
        stage_target_posts_module._apply_splits(
            _make_split_args(holdout_start="2024-01-10T00:00:00"),
            lf, dummy_logger,
        )


def test_apply_splits_rows_before_train_start_are_none(stage_target_posts_module, dummy_logger):
    """Rows before train_start for non-holdout users get split=None."""
    df = pl.DataFrame({
        "seen_at": [_dt(2023, 12, 25), _dt(2024, 1, 5)],
        "target_did": ["only_user", "only_user"],
        "like_uri": ["p1", "p2"],
        "like_emb_idx": [0, 1],
        "like_author_did": ["a1", "a1"],
        "neg_uri": ["n1", "n2"],
        "neg_emb_idx": [10, 11],
        "neg_author_did": ["a2", "a2"],
    })
    args = _make_split_args(holdout_user_fraction=0.01, holdout_user_seed=9999)
    out = stage_target_posts_module._apply_splits(args, df.lazy(), dummy_logger).collect()
    splits = out["split"].to_list()
    if splits[0] != "holdout_unseen_users":
        assert splits[0] is None, "Row before train_start should be None for non-holdout user"


def test_apply_splits_seen_users_holdout(stage_target_posts_module, dummy_logger):
    """When holdout_start is set, non-holdout users' rows after it become holdout_seen_users."""
    df = _multi_user_df()
    # holdout_start at Jan 18 means rows at Jan 20 for non-holdout users → holdout_seen_users
    args = _make_split_args(holdout_start="2024-01-18T00:00:00")
    out = stage_target_posts_module._apply_splits(args, df.lazy(), dummy_logger).collect()

    unseen_users = set(
        out.filter(pl.col("split") == "holdout_unseen_users")["target_did"].unique().to_list()
    )
    seen_holdout_rows = out.filter(pl.col("split") == "holdout_seen_users")
    assert seen_holdout_rows.height > 0, "Expected some holdout_seen_users rows"

    # holdout_seen_users rows must belong to non-holdout (train/val) users
    seen_holdout_users = set(seen_holdout_rows["target_did"].unique().to_list())
    assert seen_holdout_users.isdisjoint(unseen_users), (
        "holdout_seen_users must not include unseen holdout users"
    )

    # All seen holdout rows must be at/after holdout_start
    holdout_start_ts = _dt(2024, 1, 18)
    for row in seen_holdout_rows.iter_rows(named=True):
        assert row["seen_at"] >= holdout_start_ts, (
            f"holdout_seen_users row at {row['seen_at']} should be >= holdout_start"
        )

    # Val rows for non-holdout users should be < holdout_start
    val_rows = out.filter(pl.col("split") == "val")
    for row in val_rows.iter_rows(named=True):
        assert row["seen_at"] < holdout_start_ts, (
            f"Val row at {row['seen_at']} should be < holdout_start when holdout_start is set"
        )


def test_apply_splits_seen_users_holdout_with_holdout_end(stage_target_posts_module, dummy_logger):
    """holdout_end bounds both seen and unseen holdout sets."""
    df = _multi_user_df()
    args = _make_split_args(
        holdout_start="2024-01-18T00:00:00",
        holdout_end="2024-01-19T00:00:00",
    )
    out = stage_target_posts_module._apply_splits(args, df.lazy(), dummy_logger).collect()

    # Rows at Jan 20 should be None for all users (past holdout_end)
    jan20_rows = out.filter(pl.col("seen_at") == _dt(2024, 1, 20))
    for row in jan20_rows.iter_rows(named=True):
        assert row["split"] is None, (
            f"Row at {row['seen_at']} for {row['target_did']} should be None (past holdout_end), "
            f"got {row['split']}"
        )


def test_apply_splits_no_seen_holdout_without_holdout_start(stage_target_posts_module, dummy_logger):
    """Without holdout_start, no holdout_seen_users rows are produced."""
    df = _multi_user_df()
    args = _make_split_args()  # holdout_start=None by default
    out = stage_target_posts_module._apply_splits(args, df.lazy(), dummy_logger).collect()

    seen_holdout = out.filter(pl.col("split") == "holdout_seen_users")
    assert seen_holdout.height == 0, "No holdout_seen_users expected when holdout_start is None"
