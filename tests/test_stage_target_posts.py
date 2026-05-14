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

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = tmp_path / "artifacts"
    runs_dir = tmp_path / "runs"
    return Context(run_dir=run_dir, artifacts_dir=artifacts_dir, runs_dir=runs_dir, pipeline_run_id="test")


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


def _apply_splits_for_test(
    stage_target_posts_module,
    args: argparse.Namespace,
    target_posts_df: pl.DataFrame,
    logger: logging.Logger,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    target_posts_lf, author_idx_lf = stage_target_posts_module._apply_splits(
        args,
        target_posts_df.lazy(),
        logger,
    )
    return target_posts_lf.collect(), author_idx_lf.collect()


UNSEEN_USER_SPLITS = ["train_unseen_users", "val_unseen_users", "holdout_unseen_users"]


def _unseen_users(out: pl.DataFrame) -> set[str]:
    return set(
        out.filter(pl.col("split").is_in(UNSEEN_USER_SPLITS))["target_did"].unique().to_list()
    )


def test_apply_splits_user_based_holdout(stage_target_posts_module, dummy_logger):
    """Holdout users get unseen split labels by time window; others get train/val."""
    df = _multi_user_df()
    args = _make_split_args()
    out, _ = _apply_splits_for_test(stage_target_posts_module, args, df, dummy_logger)

    holdout_users = _unseen_users(out)
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
        user_rows = out.filter(pl.col("target_did") == uid)
        for row in user_rows.iter_rows(named=True):
            if row["seen_at"] < _dt(2024, 1, 15):
                assert row["split"] == "train_unseen_users"
            else:
                assert row["split"] == "val_unseen_users"


def test_apply_splits_temporal_trainval(stage_target_posts_module, dummy_logger):
    """Non-holdout users' rows are split temporally by val_start."""
    df = _multi_user_df()
    args = _make_split_args()
    out, _ = _apply_splits_for_test(stage_target_posts_module, args, df, dummy_logger)

    holdout_users = _unseen_users(out)
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
    out, _ = _apply_splits_for_test(stage_target_posts_module, args, df, dummy_logger)

    holdout_dids = _unseen_users(out)
    trainval_dids = set(
        out.filter(pl.col("split").is_in(["train", "val"]))["target_did"].unique().to_list()
    )
    assert holdout_dids.isdisjoint(trainval_dids)


def test_apply_splits_reproducibility(stage_target_posts_module, dummy_logger):
    """Same seed + fraction produces the same holdout assignment."""
    df = _multi_user_df()
    args = _make_split_args()
    out1, _ = _apply_splits_for_test(stage_target_posts_module, args, df, dummy_logger)
    out2, _ = _apply_splits_for_test(stage_target_posts_module, args, df, dummy_logger)
    assert out1["split"].to_list() == out2["split"].to_list()

    args_diff = _make_split_args(holdout_user_seed=999)
    out3, _ = _apply_splits_for_test(stage_target_posts_module, args_diff, df, dummy_logger)
    h1 = _unseen_users(out1)
    h3 = _unseen_users(out3)
    assert h1 != h3, "Different seeds should (almost certainly) produce different assignments"


def test_apply_splits_holdout_end(stage_target_posts_module, dummy_logger):
    """When holdout_end is set, holdout rows after that date get split=None."""
    df = _multi_user_df()
    args = _make_split_args(holdout_end="2024-01-16T00:00:00")
    out, _ = _apply_splits_for_test(stage_target_posts_module, args, df, dummy_logger)

    holdout_users = _unseen_users(out)
    assert len(holdout_users) > 0

    for uid in holdout_users:
        user_rows = out.filter(pl.col("target_did") == uid)
        for row in user_rows.iter_rows(named=True):
            if row["seen_at"] < _dt(2024, 1, 15):
                assert row["split"] == "train_unseen_users"
            elif row["seen_at"] < _dt(2024, 1, 16):
                assert row["split"] == "val_unseen_users"
            else:
                assert row["split"] is None, (
                    f"Holdout user {uid} row at {row['seen_at']} should be None "
                    f"(past holdout_end), got {row['split']}"
                )


def test_apply_splits_validation_checks(stage_target_posts_module, dummy_logger):
    """Validation of argument constraints."""
    df = _multi_user_df()

    with pytest.raises(ValueError, match="Train start date"):
        _apply_splits_for_test(
            stage_target_posts_module,
            _make_split_args(train_start="2024-01-20T00:00:00", val_start="2024-01-15T00:00:00"),
            df,
            dummy_logger,
        )

    with pytest.raises(ValueError, match="Validation window start"):
        _apply_splits_for_test(
            stage_target_posts_module,
            _make_split_args(val_start=None),
            df,
            dummy_logger,
        )

    with pytest.raises(ValueError, match="holdout_user_fraction must be in"):
        _apply_splits_for_test(
            stage_target_posts_module,
            _make_split_args(holdout_user_fraction=0.0),
            df,
            dummy_logger,
        )

    with pytest.raises(ValueError, match="holdout_user_fraction must be in"):
        _apply_splits_for_test(
            stage_target_posts_module,
            _make_split_args(holdout_user_fraction=1.0),
            df,
            dummy_logger,
        )

    with pytest.raises(ValueError, match="holdout_start must be after val_start"):
        _apply_splits_for_test(
            stage_target_posts_module,
            _make_split_args(holdout_start="2024-01-10T00:00:00"),
            df,
            dummy_logger,
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
    out, _ = _apply_splits_for_test(stage_target_posts_module, args, df, dummy_logger)
    splits = out["split"].to_list()
    if splits[0] != "holdout_unseen_users":
        assert splits[0] is None, "Row before train_start should be None for non-holdout user"


def test_apply_splits_seen_users_holdout(stage_target_posts_module, dummy_logger):
    """When holdout_start is set, non-holdout users' rows after it become holdout_seen_users."""
    df = _multi_user_df()
    # holdout_start at Jan 18 means rows at Jan 20 for non-holdout users → holdout_seen_users
    args = _make_split_args(holdout_start="2024-01-18T00:00:00")
    out, _ = _apply_splits_for_test(stage_target_posts_module, args, df, dummy_logger)

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
    out, _ = _apply_splits_for_test(stage_target_posts_module, args, df, dummy_logger)

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
    out, _ = _apply_splits_for_test(stage_target_posts_module, args, df, dummy_logger)

    seen_holdout = out.filter(pl.col("split") == "holdout_seen_users")
    assert seen_holdout.height == 0, "No holdout_seen_users expected when holdout_start is None"


def test_apply_splits_builds_train_only_author_idx_mapping(
    stage_target_posts_module, dummy_logger
):
    """Author mapping should be derived from train target rows only."""
    df = pl.DataFrame([
        {
            "seen_at": _dt(2024, 1, 2),
            "target_did": "target_user",
            "like_uri": "train_like_1",
            "like_emb_idx": 101,
            "like_author_did": "author_like_train",
            "neg_uri": "train_neg_1",
            "neg_emb_idx": 201,
            "neg_author_did": "author_neg_train",
        },
        {
            "seen_at": _dt(2024, 1, 3),
            "target_did": "target_user",
            "like_uri": "train_like_2",
            "like_emb_idx": 102,
            "like_author_did": "author_like_train",
            "neg_uri": "train_neg_2",
            "neg_emb_idx": 201,
            "neg_author_did": "author_neg_train",
        },
        {
            "seen_at": _dt(2024, 1, 16),
            "target_did": "target_user",
            "like_uri": "val_like_1",
            "like_emb_idx": 301,
            "like_author_did": "author_val_only",
            "neg_uri": "val_neg_1",
            "neg_emb_idx": 101,
            "neg_author_did": "author_like_train",
        },
    ])
    args = _make_split_args(holdout_user_fraction=0.01, holdout_user_seed=42)

    target_posts_df, author_idx_df = _apply_splits_for_test(
        stage_target_posts_module, args, df, dummy_logger
    )

    assert set(author_idx_df["author_did"].to_list()) == {
        "author_like_train",
        "author_neg_train",
    }
    assert "author_val_only" not in author_idx_df["author_did"].to_list()

    counts_by_author = {
        row["author_did"]: row["author_train_count"]
        for row in author_idx_df.select(["author_did", "author_train_count"]).unique().iter_rows(named=True)
    }
    assert counts_by_author == {
        "author_like_train": 2,
        "author_neg_train": 2,
    }

    like_author_indices = (
        author_idx_df
        .filter(pl.col("author_did") == "author_like_train")
        .select("author_idx")
        .unique()
    )
    assert like_author_indices.height == 1
    assert author_idx_df.schema["author_idx"] == pl.UInt32

    val_row = target_posts_df.filter(pl.col("like_uri") == "val_like_1").row(0, named=True)
    assert val_row["split"] == "val"
    assert val_row["like_author_idx"] is None
    assert val_row["neg_author_idx"] is not None

    train_rows = target_posts_df.filter(pl.col("split") == "train")
    assert train_rows["like_author_idx"].null_count() == 0
    assert train_rows["neg_author_idx"].null_count() == 0


def _reference_negative_target_posts_old_impl(
    *,
    args: argparse.Namespace,
    posts_df: pl.DataFrame,
    liked_target_posts_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Reference implementation that mirrors the pre-mem-reduction approach:
    construct `unliked_idx_list` via set-difference, then index into it by `neg_rank`.
    """
    bucket = args.neg_sample_bucket
    if bucket is None:
        raise ValueError("No bucket size specified for negative samples!")

    posts_lf = (
        posts_df.lazy()
        .select(
            [
                pl.col("at_uri").alias("neg_uri"),
                pl.col("record_created_at").alias("neg_posted_at"),
                pl.col("emb_idx").alias("neg_emb_idx"),
                pl.col("did").alias("neg_author_did"),
            ]
        )
        .with_columns(pl.col("neg_posted_at").dt.truncate(bucket).alias("bucket"))
        .sort(["bucket", "neg_posted_at", "neg_uri"])
        .with_columns(
            (pl.col("neg_uri").cum_count().over("bucket") - 1)
            .cast(pl.Int64)
            .alias("idx_in_bucket")
        )
    )

    bucket_sizes_lf = posts_lf.group_by("bucket").len().rename({"len": "bucket_size"})

    likes_with_bucket_lf = (
        liked_target_posts_df.lazy()
        .with_columns(pl.col("like_posted_at").dt.truncate(bucket).alias("bucket"))
    )

    liked_idx_by_user_bucket_lf = (
        liked_target_posts_df.lazy()
        .join(
            posts_lf.select(["neg_uri", "bucket", "idx_in_bucket"]),
            left_on="like_uri",
            right_on="neg_uri",
            how="inner",
        )
        .group_by(["target_did", "bucket"])
        .agg(pl.col("idx_in_bucket").unique().sort().alias("liked_idx_list"))
        .with_columns(pl.col("liked_idx_list").list.len().alias("liked_count"))
    )

    user_bucket_candidates_lf = (
        likes_with_bucket_lf.join(bucket_sizes_lf, on="bucket", how="inner")
        .select(["target_did", "bucket", "bucket_size"])
        .unique()
        .join(liked_idx_by_user_bucket_lf, on=["target_did", "bucket"], how="left")
        .with_columns(
            pl.col("liked_idx_list")
            .fill_null(pl.lit([], dtype=pl.List(pl.Int64)))
            .alias("liked_idx_list")
        )
        .with_columns(pl.int_ranges(0, pl.col("bucket_size")).alias("all_idx"))
        .with_columns(
            pl.col("all_idx")
            .list.set_difference(pl.col("liked_idx_list"))
            .list.sort()
            .alias("unliked_idx_list")
        )
        .with_columns(pl.col("unliked_idx_list").list.len().alias("unliked_count"))
        .select(["target_did", "bucket", "unliked_idx_list", "unliked_count"])
    )

    likes_lf = (
        likes_with_bucket_lf.join(bucket_sizes_lf, on="bucket", how="inner")
        .with_columns(
            pl.struct([pl.col("target_did"), pl.col("like_uri")])
            .hash(seed=args.random_seed)
            .cast(pl.UInt64)
            .alias("seed")
        )
        .join(user_bucket_candidates_lf, on=["target_did", "bucket"], how="left")
        .with_columns(
            pl.when(pl.col("unliked_count") > 0)
            .then((pl.col("seed") % pl.col("unliked_count").cast(pl.UInt64)).cast(pl.Int64))
            .otherwise(None)
            .alias("neg_rank")
        )
        .with_columns(
            pl.when(pl.col("neg_rank").is_not_null())
            .then(pl.col("unliked_idx_list").list.get(pl.col("neg_rank")))
            .otherwise(None)
            .alias("neg_idx")
        )
    )

    likes_lf = likes_lf.filter(pl.col("neg_idx").is_not_null())

    return (
        likes_lf.join(
            posts_lf.select(
                ["bucket", "idx_in_bucket", "neg_uri", "neg_emb_idx", "neg_author_did"]
            ),
            left_on=["bucket", "neg_idx"],
            right_on=["bucket", "idx_in_bucket"],
            how="left",
        )
        .select(
            [
                "target_did",
                "seen_at",
                "like_uri",
                "like_emb_idx",
                "like_author_did",
                "neg_uri",
                "neg_emb_idx",
                "neg_author_did",
            ]
        )
        .collect()
    )


def test_negative_sampling_matches_pre_mem_reduction_reference(
    stage_target_posts_module, dummy_logger, dummy_context
):
    args = argparse.Namespace(random_seed=123, neg_sample_bucket="1d")

    posts_df = pl.DataFrame(
        [
            # Jan 1 bucket (5 posts)
            {"at_uri": "p:a", "record_created_at": _dt(2024, 1, 1, 1), "emb_idx": 1, "did": "auth_1"},
            {"at_uri": "p:b", "record_created_at": _dt(2024, 1, 1, 2), "emb_idx": 2, "did": "auth_2"},
            {"at_uri": "p:c", "record_created_at": _dt(2024, 1, 1, 3), "emb_idx": 3, "did": "auth_3"},
            {"at_uri": "p:d", "record_created_at": _dt(2024, 1, 1, 4), "emb_idx": 4, "did": "auth_4"},
            {"at_uri": "p:e", "record_created_at": _dt(2024, 1, 1, 5), "emb_idx": 5, "did": "auth_5"},
            # Jan 2 bucket (3 posts)
            {"at_uri": "p:f", "record_created_at": _dt(2024, 1, 2, 1), "emb_idx": 6, "did": "auth_6"},
            {"at_uri": "p:g", "record_created_at": _dt(2024, 1, 2, 2), "emb_idx": 7, "did": "auth_7"},
            {"at_uri": "p:h", "record_created_at": _dt(2024, 1, 2, 3), "emb_idx": 8, "did": "auth_8"},
        ]
    )

    likes_df = pl.DataFrame(
        [
            # user_1 likes 2/5 in Jan 1 bucket
            {"did": "user_1", "subject_uri": "p:b", "record_created_at": _dt(2024, 1, 1, 12), "emb_idx": 10},
            {"did": "user_1", "subject_uri": "p:d", "record_created_at": _dt(2024, 1, 1, 13), "emb_idx": 11},
            # user_2 likes 3/5 in Jan 1 bucket
            {"did": "user_2", "subject_uri": "p:a", "record_created_at": _dt(2024, 1, 1, 14), "emb_idx": 12},
            {"did": "user_2", "subject_uri": "p:c", "record_created_at": _dt(2024, 1, 1, 15), "emb_idx": 13},
            {"did": "user_2", "subject_uri": "p:e", "record_created_at": _dt(2024, 1, 1, 16), "emb_idx": 14},
            # user_2 likes ALL posts in Jan 2 bucket -> these should be dropped (no valid negatives)
            {"did": "user_2", "subject_uri": "p:f", "record_created_at": _dt(2024, 1, 2, 12), "emb_idx": 20},
            {"did": "user_2", "subject_uri": "p:g", "record_created_at": _dt(2024, 1, 2, 13), "emb_idx": 21},
            {"did": "user_2", "subject_uri": "p:h", "record_created_at": _dt(2024, 1, 2, 14), "emb_idx": 22},
        ]
    )

    liked_df = stage_target_posts_module._get_liked_target_posts(
        likes_df.lazy(), posts_df.lazy()
    ).collect()

    stage_out = stage_target_posts_module._get_negative_target_posts(
        args, posts_df.lazy(), liked_df.lazy(), dummy_logger, dummy_context
    ).collect()

    ref_out = _reference_negative_target_posts_old_impl(
        args=args, posts_df=posts_df, liked_target_posts_df=liked_df
    )

    stage_out = stage_out.sort(["target_did", "like_uri"])
    ref_out = ref_out.sort(["target_did", "like_uri"])

    assert stage_out.columns == ref_out.columns
    assert stage_out.height == ref_out.height
    assert stage_out.to_dicts() == ref_out.to_dicts()

    # Confirm the "all posts liked in bucket" case still drops those likes.
    assert "p:f" not in stage_out["like_uri"].to_list()
    assert "p:g" not in stage_out["like_uri"].to_list()
    assert "p:h" not in stage_out["like_uri"].to_list()


def test_apply_splits_matches_reference_holdout_assignment(
    stage_target_posts_module, dummy_logger
):
    df = pl.DataFrame(
        [
            # before train_start (potentially None for non-holdout users)
            {"seen_at": _dt(2023, 12, 31, 23), "target_did": "u1", "like_uri": "u1_p0", "like_emb_idx": 0, "like_author_did": "a1", "neg_uri": "u1_n0", "neg_emb_idx": 10, "neg_author_did": "a2"},
            # train / val / holdout_seen_users windows
            {"seen_at": _dt(2024, 1, 2, 0), "target_did": "u1", "like_uri": "u1_p1", "like_emb_idx": 1, "like_author_did": "a1", "neg_uri": "u1_n1", "neg_emb_idx": 11, "neg_author_did": "a2"},
            {"seen_at": _dt(2024, 1, 16, 0), "target_did": "u1", "like_uri": "u1_p2", "like_emb_idx": 2, "like_author_did": "a1", "neg_uri": "u1_n2", "neg_emb_idx": 12, "neg_author_did": "a2"},
            {"seen_at": _dt(2024, 1, 20, 0), "target_did": "u1", "like_uri": "u1_p3", "like_emb_idx": 3, "like_author_did": "a1", "neg_uri": "u1_n3", "neg_emb_idx": 13, "neg_author_did": "a2"},
            {"seen_at": _dt(2024, 1, 2, 0), "target_did": "u2", "like_uri": "u2_p1", "like_emb_idx": 1, "like_author_did": "a1", "neg_uri": "u2_n1", "neg_emb_idx": 11, "neg_author_did": "a2"},
            {"seen_at": _dt(2024, 1, 16, 0), "target_did": "u2", "like_uri": "u2_p2", "like_emb_idx": 2, "like_author_did": "a1", "neg_uri": "u2_n2", "neg_emb_idx": 12, "neg_author_did": "a2"},
            {"seen_at": _dt(2024, 1, 20, 0), "target_did": "u2", "like_uri": "u2_p3", "like_emb_idx": 3, "like_author_did": "a1", "neg_uri": "u2_n3", "neg_emb_idx": 13, "neg_author_did": "a2"},
            {"seen_at": _dt(2024, 1, 2, 0), "target_did": "u3", "like_uri": "u3_p1", "like_emb_idx": 1, "like_author_did": "a1", "neg_uri": "u3_n1", "neg_emb_idx": 11, "neg_author_did": "a2"},
            {"seen_at": _dt(2024, 1, 16, 0), "target_did": "u3", "like_uri": "u3_p2", "like_emb_idx": 2, "like_author_did": "a1", "neg_uri": "u3_n2", "neg_emb_idx": 12, "neg_author_did": "a2"},
            {"seen_at": _dt(2024, 1, 20, 0), "target_did": "u3", "like_uri": "u3_p3", "like_emb_idx": 3, "like_author_did": "a1", "neg_uri": "u3_n3", "neg_emb_idx": 13, "neg_author_did": "a2"},
            # beyond holdout_end -> should be None for everyone
            {"seen_at": _dt(2024, 1, 25, 0), "target_did": "u3", "like_uri": "u3_p4", "like_emb_idx": 4, "like_author_did": "a1", "neg_uri": "u3_n4", "neg_emb_idx": 14, "neg_author_did": "a2"},
        ]
    )

    args = _make_split_args(
        train_start="2024-01-01T00:00:00",
        val_start="2024-01-15T00:00:00",
        holdout_start="2024-01-18T00:00:00",
        holdout_end="2024-01-22T00:00:00",
        holdout_user_fraction=0.5,
        holdout_user_seed=42,
    )

    out, _ = _apply_splits_for_test(stage_target_posts_module, args, df, dummy_logger)

    holdout_dids = {
        did
        for did in df["target_did"].unique().to_list()
        if stage_target_posts_module._user_is_holdout(
            did, args.holdout_user_seed, args.holdout_user_fraction
        )
    }

    train_start = stage_target_posts_module.parse_one_ts_strict(args.train_start)
    val_start = stage_target_posts_module.parse_one_ts_strict(args.val_start)
    holdout_start = stage_target_posts_module.parse_one_ts(args.holdout_start)
    holdout_end = stage_target_posts_module.parse_one_ts(args.holdout_end)

    df_sorted = df.sort(["target_did", "seen_at", "like_uri"])
    expected = []
    for row in df_sorted.iter_rows(named=True):
        seen_at = row["seen_at"]
        did = row["target_did"]
        if holdout_end is not None and seen_at >= holdout_end:
            split = None
        elif did in holdout_dids and seen_at >= train_start and seen_at < val_start:
            split = "train_unseen_users"
        elif (
            did in holdout_dids
            and seen_at >= val_start
            and (holdout_start is None or seen_at < holdout_start)
        ):
            split = "val_unseen_users"
        elif did in holdout_dids and holdout_start is not None and seen_at >= holdout_start:
            split = "holdout_unseen_users"
        elif holdout_start is not None and seen_at >= holdout_start:
            split = "holdout_seen_users"
        elif seen_at >= train_start and seen_at < val_start:
            split = "train"
        elif seen_at >= val_start and (holdout_start is None or seen_at < holdout_start):
            split = "val"
        else:
            split = None
        expected.append(split)

    out = out.sort(["target_did", "seen_at", "like_uri"])

    assert out["target_did"].to_list() == df_sorted["target_did"].to_list()
    assert out["seen_at"].to_list() == df_sorted["seen_at"].to_list()
    assert out["like_uri"].to_list() == df_sorted["like_uri"].to_list()
    assert out["split"].to_list() == expected
