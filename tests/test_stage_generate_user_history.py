"""
Tests for stage_generate_user_history.py (directory-based user history)

Tests the _build_user_history_directory function which creates a mapping from
each target (target_did, like_uri) to prior liked embedding indices.

Target posts use a wide format:
  target_did | seen_at | like_uri | like_emb_idx | ... | neg_uri | neg_emb_idx | ... | split

The user history depends only on (target_did, seen_at), so one history list is
produced per (target_did, like_uri) pair.  Rows where the user has no prior
likes get an empty list.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest


# ---------------------------------------------------------------------------
# Load the production module by file path (03_user_history isn't a valid
# Python package name, so we use importlib like test_stage_get_data.py does).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def stage_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "utils" / "03_user_history" / "stage_generate_user_history.py"
    spec = importlib.util.spec_from_file_location("stage_generate_user_history", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage_generate_user_history"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def build_history(stage_module):
    """Shortcut to the production _build_user_history_directory function."""
    return stage_module._build_user_history_directory


def _make_test_logger() -> logging.Logger:
    """Create a simple logger for tests."""
    logger = logging.getLogger("test_user_history")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Helper to build a minimal wide-format target-posts LazyFrame
# ---------------------------------------------------------------------------

def _make_targets(
    target_dids: list[str],
    like_uris: list[str],
    seen_ats: list[datetime],
) -> pl.LazyFrame:
    """Build a minimal wide-format targets LazyFrame with required columns."""
    n = len(target_dids)
    return pl.DataFrame({
        "target_did": target_dids,
        "seen_at": seen_ats,
        "like_uri": like_uris,
        "like_emb_idx": [0] * n,
        "neg_uri": ["stub"] * n,
        "neg_emb_idx": [0] * n,
        "split": ["train"] * n,
    }).lazy()


def _make_likes(
    dids: list[str],
    timestamps: list[datetime],
    subject_uris: list[str],
    emb_idxs: list[int],
    author_dids: list[str] | None = None,
) -> pl.LazyFrame:
    """Build a minimal likes LazyFrame."""
    data = {
        "did": dids,
        "record_created_at": timestamps,
        "subject_uri": subject_uris,
        "emb_idx": emb_idxs,
    }
    if author_dids is not None:
        data["author_did"] = author_dids
    return pl.DataFrame(data).lazy()


# --- Tests ---

def test_directory_basic_creation(build_history):
    """Test basic directory creation with prior likes."""
    logger = _make_test_logger()

    # User u1 liked posts p1, p2, p3 at times t1, t2, t3
    # Target event is at time t4 (after all likes)
    likes_lf = _make_likes(
        ["u1", "u1", "u1"],
        [
            datetime(2024, 1, 1, 10, 0),  # earliest
            datetime(2024, 1, 1, 11, 0),  # middle
            datetime(2024, 1, 1, 12, 0),  # latest
        ],
        ["p1", "p2", "p3"],
        [100, 200, 300],
    )

    targets_lf = _make_targets(
        ["u1"], ["at://u1/like/4"], [datetime(2024, 1, 1, 13, 0)],
    )

    result = build_history(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    assert result.height == 1
    assert result["target_did"][0] == "u1"
    assert result["like_uri"][0] == "at://u1/like/4"
    # Should have 3 prior likes, sorted by recency (most recent first)
    prior = result["prior_emb_indices"][0].to_list()
    assert len(prior) == 3
    assert prior == [300, 200, 100]  # most recent first


def test_directory_recency_ordering(build_history):
    """Test that prior_emb_indices are correctly ordered by recency (descending)."""
    logger = _make_test_logger()

    # Likes in random timestamp order
    likes_lf = _make_likes(
        ["u1", "u1", "u1", "u1"],
        [
            datetime(2024, 1, 5, 0, 0),   # 3rd most recent
            datetime(2024, 1, 10, 0, 0),  # most recent
            datetime(2024, 1, 1, 0, 0),   # oldest
            datetime(2024, 1, 7, 0, 0),   # 2nd most recent
        ],
        ["p1", "p2", "p3", "p4"],
        [10, 20, 30, 40],
    )

    targets_lf = _make_targets(
        ["u1"], ["at://u1/like/5"], [datetime(2024, 1, 15, 0, 0)],
    )

    result = build_history(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    prior = result["prior_emb_indices"][0].to_list()
    # Expected order by recency: p2 (Jan 10) -> p4 (Jan 7) -> p1 (Jan 5) -> p3 (Jan 1)
    assert prior == [20, 40, 10, 30]


def test_directory_max_prior_likes_capping(build_history):
    """Test that max_prior_likes caps the number of prior likes."""
    logger = _make_test_logger()

    # User has 5 likes
    likes_lf = _make_likes(
        ["u1"] * 5,
        [datetime(2024, 1, i, 0, 0) for i in range(1, 6)],
        [f"p{i}" for i in range(1, 6)],
        list(range(1, 6)),
    )

    targets_lf = _make_targets(
        ["u1"], ["at://u1/like/6"], [datetime(2024, 1, 10, 0, 0)],
    )

    # Cap to 3 prior likes
    result = build_history(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=3,
        logger=logger,
    ).collect()

    prior = result["prior_emb_indices"][0].to_list()
    assert len(prior) == 3
    # Should be the 3 most recent: emb_idx 5, 4, 3
    assert prior == [5, 4, 3]


def test_directory_no_prior_history_returns_empty_list(build_history):
    """Test that targets with no prior likes get an empty list."""
    logger = _make_test_logger()

    # User u1 has likes, user u2 has no likes at all in the dataset
    likes_lf = _make_likes(
        ["u1"],
        [datetime(2024, 1, 1, 0, 0)],
        ["p1"],
        [100],
    )

    targets_lf = _make_targets(
        ["u1", "u2"],
        ["at://u1/like/1", "at://u2/like/1"],
        [datetime(2024, 1, 5, 0, 0), datetime(2024, 1, 5, 0, 0)],
    )

    result = build_history(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect().sort("target_did")

    assert result.height == 2

    # u1 has prior likes
    u1_row = result.filter(pl.col("target_did") == "u1")
    assert u1_row["prior_emb_indices"][0].to_list() == [100]

    # u2 has no prior likes (empty list, not null)
    u2_row = result.filter(pl.col("target_did") == "u2")
    assert u2_row["prior_emb_indices"][0].to_list() == []


def test_directory_first_like_produces_empty_history(build_history):
    """Test that a user's very first like in the dataset has empty history.

    This is the key scenario: target_posts did NOT filter out first likes,
    so this stage must produce an empty prior_emb_indices list for them.
    """
    logger = _make_test_logger()

    # User u1 has exactly one like at t=10:00
    likes_lf = _make_likes(
        ["u1"],
        [datetime(2024, 1, 1, 10, 0)],
        ["p1"],
        [42],
    )

    # The target row corresponds to that very first like (seen_at == like time)
    # Since we use strict < (not <=), a like at the same timestamp is NOT prior
    targets_lf = _make_targets(
        ["u1"], ["at://u1/like/1"], [datetime(2024, 1, 1, 10, 0)],
    )

    result = build_history(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    assert result.height == 1
    assert result["prior_emb_indices"][0].to_list() == []


def test_directory_excludes_future_likes(build_history):
    """Test that likes after the target timestamp are excluded."""
    logger = _make_test_logger()

    # User has likes before and after the target timestamp
    likes_lf = _make_likes(
        ["u1", "u1", "u1"],
        [
            datetime(2024, 1, 1, 0, 0),   # before target
            datetime(2024, 1, 5, 0, 0),   # before target
            datetime(2024, 1, 10, 0, 0),  # after target
        ],
        ["p1", "p2", "p3"],
        [1, 2, 3],
    )

    targets_lf = _make_targets(
        ["u1"], ["at://u1/like/x"], [datetime(2024, 1, 7, 0, 0)],
    )

    result = build_history(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    prior = result["prior_emb_indices"][0].to_list()
    # Only p1 and p2 should be included (before target ts)
    assert len(prior) == 2
    assert prior == [2, 1]  # most recent first


def test_directory_multiple_targets_same_user(build_history):
    """Test that each target gets correct prior likes based on its timestamp."""
    logger = _make_test_logger()

    likes_lf = _make_likes(
        ["u1", "u1", "u1"],
        [
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 3, 0, 0),
            datetime(2024, 1, 5, 0, 0),
        ],
        ["p1", "p2", "p3"],
        [10, 20, 30],
    )

    # Three targets at different times for the same user
    targets_lf = _make_targets(
        ["u1", "u1", "u1"],
        ["at://u1/like/early", "at://u1/like/mid", "at://u1/like/late"],
        [
            datetime(2024, 1, 2, 0, 0),   # only sees p1
            datetime(2024, 1, 4, 0, 0),   # sees p1, p2
            datetime(2024, 1, 10, 0, 0),  # sees p1, p2, p3
        ],
    )

    result = build_history(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    result_dict = {
        row["like_uri"]: list(row["prior_emb_indices"])
        for row in result.iter_rows(named=True)
    }

    assert result_dict["at://u1/like/early"] == [10]           # only p1 before
    assert result_dict["at://u1/like/mid"] == [20, 10]         # p2, p1 before
    assert result_dict["at://u1/like/late"] == [30, 20, 10]    # all three before


def test_directory_multiple_users(build_history):
    """Test that directory handles multiple users correctly."""
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

    targets_lf = _make_targets(
        ["u1", "u2"],
        ["at://u1/like/t1", "at://u2/like/t1"],
        [datetime(2024, 1, 5, 0, 0), datetime(2024, 1, 5, 0, 0)],
    )

    result = build_history(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    result_dict = {
        row["target_did"]: list(row["prior_emb_indices"])
        for row in result.iter_rows(named=True)
    }

    # u1's likes (emb_idx 1, 2)
    assert result_dict["u1"] == [2, 1]  # most recent first

    # u2's likes (emb_idx 11, 12)
    assert result_dict["u2"] == [12, 11]  # most recent first


def test_directory_output_schema(build_history):
    """Test that output has correct schema with expected columns."""
    logger = _make_test_logger()

    likes_lf = _make_likes(
        ["u1"],
        [datetime(2024, 1, 1, 0, 0)],
        ["p1"],
        [100],
    )

    targets_lf = _make_targets(
        ["u1"], ["at://u1/like/1"], [datetime(2024, 1, 5, 0, 0)],
    )

    result = build_history(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    # Check expected columns are present
    assert "target_did" in result.columns
    assert "like_uri" in result.columns
    assert "seen_at" in result.columns
    assert "prior_emb_indices" in result.columns
    assert "raw_prior_count" in result.columns

    # Internal columns should NOT be present
    assert "target_idx" not in result.columns
    assert "did" not in result.columns
    assert "post_id" not in result.columns

    # Check that prior_emb_indices is List[UInt32]
    assert result.schema["prior_emb_indices"] == pl.List(pl.UInt32)


def test_directory_raw_prior_count(build_history):
    """Test that raw_prior_count reflects uncapped count even when capping is applied."""
    logger = _make_test_logger()

    # User has 5 likes
    likes_lf = _make_likes(
        ["u1"] * 5,
        [datetime(2024, 1, i, 0, 0) for i in range(1, 6)],
        [f"p{i}" for i in range(1, 6)],
        list(range(1, 6)),
    )

    targets_lf = _make_targets(
        ["u1"], ["at://u1/like/6"], [datetime(2024, 1, 10, 0, 0)],
    )

    # Cap to 2 prior likes
    result = build_history(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=2,
        logger=logger,
    ).collect()

    # prior_emb_indices should be capped at 2
    prior = result["prior_emb_indices"][0].to_list()
    assert len(prior) == 2

    # raw_prior_count should reflect the uncapped count (5)
    assert result["raw_prior_count"][0] == 5


# ---------------------------------------------------------------------------
# prior_author_indices tests
# ---------------------------------------------------------------------------


def test_directory_author_indices_preserve_order_and_unknowns(build_history):
    """Author indices should align with prior_emb_indices and keep unknown authors."""
    logger = _make_test_logger()

    likes_lf = _make_likes(
        ["u1", "u1", "u1"],
        [
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
            datetime(2024, 1, 1, 12, 0),
        ],
        ["p1", "p2", "p3"],
        [100, 200, 300],
        author_dids=["author_a", "author_missing", "author_c"],
    )
    author_idx_lf = pl.DataFrame({
        "author_did": ["author_a", "author_c"],
        "author_idx": pl.Series([1, 3], dtype=pl.UInt32),
        "author_train_count": [5, 7],
    }).lazy()
    targets_lf = _make_targets(
        ["u1"], ["at://u1/like/4"], [datetime(2024, 1, 1, 13, 0)],
    )

    result = build_history(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
        author_idx_lf=author_idx_lf,
    ).collect()

    assert result["prior_emb_indices"][0].to_list() == [300, 200, 100]
    assert result["prior_author_indices"][0].to_list() == [3, None, 1]
    assert result["raw_prior_count"][0] == 3


def test_directory_author_indices_empty_for_no_history(build_history):
    """Rows with no prior likes should get empty author history lists."""
    logger = _make_test_logger()

    likes_lf = _make_likes(
        ["u1"],
        [datetime(2024, 1, 2, 10, 0)],
        ["p1"],
        [100],
        author_dids=["author_a"],
    )
    author_idx_lf = pl.DataFrame({
        "author_did": ["author_a"],
        "author_idx": pl.Series([1], dtype=pl.UInt32),
        "author_train_count": [5],
    }).lazy()
    targets_lf = _make_targets(
        ["u1"], ["at://u1/like/0"], [datetime(2024, 1, 1, 13, 0)],
    )

    result = build_history(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
        author_idx_lf=author_idx_lf,
    ).collect()

    assert result["prior_emb_indices"][0].to_list() == []
    assert result["prior_author_indices"][0].to_list() == []


def test_directory_without_author_mapping_omits_author_history(build_history):
    """Legacy inputs should not emit prior_author_indices."""
    logger = _make_test_logger()

    likes_lf = _make_likes(
        ["u1"],
        [datetime(2024, 1, 1, 10, 0)],
        ["p1"],
        [100],
    )
    targets_lf = _make_targets(
        ["u1"], ["at://u1/like/2"], [datetime(2024, 1, 1, 13, 0)],
    )

    result = build_history(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
    ).collect()

    assert "prior_author_indices" not in result.columns


# ---------------------------------------------------------------------------
# history_buffer_hours tests
# ---------------------------------------------------------------------------

def test_history_buffer_hours_excludes_recent_likes(build_history):
    """Test that history_buffer_hours excludes likes near the seen_at boundary.

    With a 2-hour buffer and seen_at at 12:00, only likes before 10:00 should
    be included (like_ts < seen_at - 2h).
    """
    logger = _make_test_logger()

    likes_lf = _make_likes(
        ["u1", "u1", "u1"],
        [
            datetime(2024, 1, 1, 8, 0),   # 4h before seen_at → included
            datetime(2024, 1, 1, 9, 30),   # 2.5h before → included
            datetime(2024, 1, 1, 11, 0),  # 1h before → EXCLUDED by 2h buffer
        ],
        ["p1", "p2", "p3"],
        [10, 20, 30],
    )

    targets_lf = _make_targets(
        ["u1"], ["at://u1/like/x"], [datetime(2024, 1, 1, 12, 0)],
    )

    result = build_history(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
        history_buffer_hours=2.0,
    ).collect()

    prior = result["prior_emb_indices"][0].to_list()
    # Only likes before 10:00 (= 12:00 - 2h): p1 (08:00) and p2 (09:30)
    assert len(prior) == 2
    assert prior == [20, 10]  # most recent first


def test_history_buffer_hours_zero_is_no_buffer(build_history):
    """Test that history_buffer_hours=0 behaves identically to None (no buffer)."""
    logger = _make_test_logger()

    likes_lf = _make_likes(
        ["u1", "u1"],
        [
            datetime(2024, 1, 1, 10, 0),
            datetime(2024, 1, 1, 11, 0),
        ],
        ["p1", "p2"],
        [10, 20],
    )

    targets_lf = _make_targets(
        ["u1"], ["at://u1/like/x"], [datetime(2024, 1, 1, 12, 0)],
    )

    # buffer=0 should include all prior likes (same as None)
    for buffer_val in [None, 0.0]:
        result = build_history(
            targets_lf=targets_lf,
            likes_lf=likes_lf,
            max_prior_likes=None,
            logger=logger,
            history_buffer_hours=buffer_val,
        ).collect()

        prior = result["prior_emb_indices"][0].to_list()
        assert prior == [20, 10], f"Failed for history_buffer_hours={buffer_val}"


def test_history_buffer_hours_excludes_all_likes(build_history):
    """Test that a very large buffer can exclude all prior likes, producing an empty list."""
    logger = _make_test_logger()

    likes_lf = _make_likes(
        ["u1"],
        [datetime(2024, 1, 1, 11, 0)],  # 1h before seen_at
        ["p1"],
        [10],
    )

    targets_lf = _make_targets(
        ["u1"], ["at://u1/like/x"], [datetime(2024, 1, 1, 12, 0)],
    )

    # 24h buffer: cutoff is 12:00 - 24h = Jan 0 12:00, nothing qualifies
    result = build_history(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=None,
        logger=logger,
        history_buffer_hours=24.0,
    ).collect()

    assert result["prior_emb_indices"][0].to_list() == []


def test_history_buffer_hours_with_capping(build_history):
    """Test that history_buffer_hours and max_prior_likes work together correctly."""
    logger = _make_test_logger()

    # 5 likes spread over time
    likes_lf = _make_likes(
        ["u1"] * 5,
        [
            datetime(2024, 1, 1, 1, 0),   # old, within buffer
            datetime(2024, 1, 1, 2, 0),   # old, within buffer
            datetime(2024, 1, 1, 3, 0),   # old, within buffer
            datetime(2024, 1, 1, 9, 0),   # recent, outside buffer (9:00 < 10:00 - 1h = 09:00? no, 9 < 9 is false, excluded)
            datetime(2024, 1, 1, 8, 30),  # 1.5h before seen_at, within 1h buffer cutoff (8:30 < 9:00, included)
        ],
        ["p1", "p2", "p3", "p4", "p5"],
        [1, 2, 3, 4, 5],
    )

    targets_lf = _make_targets(
        ["u1"], ["at://u1/like/x"], [datetime(2024, 1, 1, 10, 0)],
    )

    # 1h buffer (cutoff = 09:00), cap to 2
    result = build_history(
        targets_lf=targets_lf,
        likes_lf=likes_lf,
        max_prior_likes=2,
        logger=logger,
        history_buffer_hours=1.0,
    ).collect()

    prior = result["prior_emb_indices"][0].to_list()
    # Likes before 09:00: p5 (08:30), p3 (03:00), p2 (02:00), p1 (01:00) → 4 total
    # After capping to 2 most recent: p5 (08:30), p3 (03:00)
    assert len(prior) == 2
    assert prior == [5, 3]  # most recent first within the buffer

    # raw_prior_count should be 4 (uncapped)
    assert result["raw_prior_count"][0] == 4
