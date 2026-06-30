import importlib.util
import logging
import struct
import sys
import zlib
import base64
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest


@pytest.fixture(scope="session")
def stage_get_data_module():
    pytest.importorskip("google.cloud.storage")
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "utils/01_get_data/stage_get_data.py"
    spec = importlib.util.spec_from_file_location("stage_get_data", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["stage_get_data"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _encode_embedding(vec):
    raw = struct.pack(f"<{len(vec)}f", *vec)
    compressed = zlib.compress(raw)
    return base64.b85encode(compressed).decode()


def _decode_embedding(encoded_str):
    """Decode a base85+zlib encoded embedding string to float list."""
    bs = base64.b85decode(encoded_str.encode())
    try:
        bs = zlib.decompress(bs)
    except zlib.error:
        pass
    return list(struct.unpack(f'<{int(len(bs) / 4)}f', bs))


def _write_likes_parquet(tmp_path, rows):
    df = pl.DataFrame(rows)
    path = tmp_path / "likes.parquet"
    df.write_parquet(path)
    return path


def _scan_likes_lf(stage_get_data_module, likes_path, start_str, end_str):
    return stage_get_data_module.apply_time_filter(pl.scan_parquet(str(likes_path)), start_str, end_str)


def _write_posts_parquet(tmp_path, rows):
    df = pl.DataFrame(rows)
    path = tmp_path / "posts.parquet"
    df.write_parquet(path)
    return path


def _global_likes_for_posts(rows, count=100):
    if callable(count):
        counts = [count(row) for row in rows]
    elif isinstance(count, dict):
        counts = [count[row["at_uri"]] for row in rows]
    else:
        counts = [count for _ in rows]
    return pl.DataFrame({
        "subject_uri": [row["at_uri"] for row in rows],
        "global_like_count": counts,
    }).with_columns(
        pl.col("global_like_count").cast(pl.UInt64),
    )


def _split_kwargs(**overrides):
    kwargs = dict(
        train_start="2024-01-01T00:00:00",
        val_start="2024-01-03T00:00:00",
        holdout_start=None,
        holdout_end=None,
    )
    kwargs.update(overrides)
    return kwargs


def _make_posts_rows(embedding_model):
    return [
        {
            "at_uri": "post:1",
            "did": "user_a",
            "record_created_at": "2024-01-02T00:00:00",
            "record_text": "one",
            "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.1, 0.2, 0.3])}],
        },
        {
            "at_uri": "post:2",
            "did": "user_b",
            "record_created_at": "2024-01-02T00:00:00",
            "record_text": "two",
            "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.4, 0.5, 0.6])}],
        },
        {
            "at_uri": "post:3",
            "did": "user_c",
            "record_created_at": "2024-01-02T00:00:00",
            "record_text": "three",
            "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.7, 0.8, 0.9])}],
        },
        {
            "at_uri": "post:4",
            "did": "user_d",
            "record_created_at": "2024-01-02T00:00:00",
            "record_text": "four",
            "embeddings": [{"key": embedding_model, "value": _encode_embedding([1.0, 1.1, 1.2])}],
        },
        {
            "at_uri": "post:5",
            "did": "user_e",
            "record_created_at": "2024-01-02T00:00:00",
            "record_text": "five",
            "embeddings": [{"key": embedding_model, "value": _encode_embedding([1.3, 1.4, 1.5])}],
        },
    ]


def test_load_likes_filters_time_and_min_likes(tmp_path, stage_get_data_module):
    likes_rows = [
        {"did": "user_a", "subject_uri": "post:1", "record_created_at": "2024-01-02T00:00:00"},
        {"did": "user_a", "subject_uri": "post:2", "record_created_at": "2024-01-03T00:00:00"},
        {"did": "user_b", "subject_uri": "post:3", "record_created_at": "2024-01-02T00:00:00"},
        {"did": "user_c", "subject_uri": "post:4", "record_created_at": "2024-01-05T00:00:00"},
    ]
    likes_path = _write_likes_parquet(tmp_path, likes_rows)
    raw_likes_lf = _scan_likes_lf(
        stage_get_data_module,
        likes_path,
        "2024-01-02T00:00:00",
        "2024-01-04T00:00:00",
    )
    logger = logging.getLogger("test_stage_get_data.likes")

    likes_df, stats = stage_get_data_module._load_likes_core_polars(
        raw_likes_lf=raw_likes_lf,
        max_trainval_users=None,
        max_unseen_eval_users=0,
        max_likes_per_user=0,
        min_likes_per_user=2,
        random_seed=123,
        **_split_kwargs(),
        logger=logger,
    )

    assert likes_df.height == 2
    assert likes_df["did"].unique().to_list() == ["user_a"]
    assert likes_df.schema["record_created_at"] == pl.Datetime
    assert likes_df.schema["like_hour_bucket"] == pl.Datetime
    splits_by_uri = {
        row["subject_uri"]: row["split"]
        for row in likes_df.select(["subject_uri", "split"]).iter_rows(named=True)
    }
    assert splits_by_uri == {"post:1": "train", "post:2": "val"}
    assert likes_df['did'].n_unique() == 1
    buckets_by_uri = {
        row["subject_uri"]: row["like_hour_bucket"]
        for row in likes_df.select(["subject_uri", "like_hour_bucket"]).iter_rows(named=True)
    }
    assert buckets_by_uri["post:1"].isoformat() == "2024-01-02T00:00:00+00:00"
    assert buckets_by_uri["post:2"].isoformat() == "2024-01-03T00:00:00+00:00"


def test_load_likes_per_user_cap(tmp_path, stage_get_data_module):
    likes_rows = [
        {"did": "user_a", "subject_uri": f"post:{i}", "record_created_at": "2024-01-02T00:00:00"}
        for i in range(5)
    ] + [
        {"did": "user_b", "subject_uri": "post:99", "record_created_at": "2024-01-02T00:00:00"},
    ]
    likes_path = _write_likes_parquet(tmp_path, likes_rows)
    raw_likes_lf = _scan_likes_lf(
        stage_get_data_module,
        likes_path,
        "2024-01-01T00:00:00",
        "2024-01-03T00:00:00",
    )
    logger = logging.getLogger("test_stage_get_data.likes_cap")

    likes_df, _ = stage_get_data_module._load_likes_core_polars(
        raw_likes_lf=raw_likes_lf,
        max_trainval_users=None,
        max_unseen_eval_users=0,
        max_likes_per_user=2,
        min_likes_per_user=0,
        random_seed=42,
        **_split_kwargs(),
        logger=logger,
    )

    assert likes_df.filter(pl.col("did") == "user_a").height == 2
    assert likes_df.filter(pl.col("did") == "user_b").height == 1


def test_get_sampled_user_cohorts_deterministic_and_disjoint(stage_get_data_module):
    likes_rows = [
        {"did": f"user_{i}", "subject_uri": f"post:{i}", "record_created_at": "2024-01-02T00:00:00"}
        for i in range(10)
    ]
    likes_lf = pl.DataFrame(likes_rows).lazy()

    first, *_ = stage_get_data_module._get_sampled_user_cohorts_with_min_likes(
        raw_likes_lf=likes_lf,
        min_likes_per_user=1,
        max_trainval_users=5,
        max_unseen_eval_users=3,
        random_seed=7,
    )
    second, *_ = stage_get_data_module._get_sampled_user_cohorts_with_min_likes(
        raw_likes_lf=likes_lf,
        min_likes_per_user=1,
        max_trainval_users=5,
        max_unseen_eval_users=3,
        random_seed=7,
    )

    assert set(first["did"].to_list()) == set(second["did"].to_list())
    trainval = set(first.filter(pl.col("_user_cohort") == "trainval")["did"].to_list())
    unseen = set(first.filter(pl.col("_user_cohort") == "unseen_eval")["did"].to_list())
    assert len(trainval) == 5
    assert len(unseen) == 3
    assert trainval.isdisjoint(unseen)


def test_load_likes_unseen_eval_discards_train_window_and_labels_splits(tmp_path, stage_get_data_module):
    likes_rows = []
    for user_idx in range(8):
        for ts in [
            "2024-01-02T00:00:00",
            "2024-01-04T00:00:00",
            "2024-01-06T00:00:00",
        ]:
            likes_rows.append({
                "did": f"user_{user_idx}",
                "subject_uri": f"post:{user_idx}:{ts}",
                "record_created_at": ts,
            })
    likes_path = _write_likes_parquet(tmp_path, likes_rows)
    raw_likes_lf = _scan_likes_lf(
        stage_get_data_module,
        likes_path,
        "2024-01-01T00:00:00",
        "2024-01-07T00:00:00",
    )
    logger = logging.getLogger("test_stage_get_data.unseen_splits")

    likes_df, stats = stage_get_data_module._load_likes_core_polars(
        raw_likes_lf=raw_likes_lf,
        max_trainval_users=3,
        max_unseen_eval_users=2,
        max_likes_per_user=0,
        min_likes_per_user=1,
        random_seed=99,
        **_split_kwargs(
            val_start="2024-01-03T00:00:00",
            holdout_start="2024-01-05T00:00:00",
        ),
        logger=logger,
    )

    assert stats["n_trainval_users_sampled"] == 3
    assert stats["n_unseen_eval_users_sampled"] == 2
    assert set(likes_df["split"].unique().to_list()) == {
        "train",
        "val",
        "holdout_seen_users",
        "val_unseen_users",
        "holdout_unseen_users",
    }
    unseen_users = set(
        likes_df
        .filter(pl.col("split").is_in(["val_unseen_users", "holdout_unseen_users"]))
        ["did"]
        .unique()
        .to_list()
    )
    unseen_train_rows = likes_df.filter(
        pl.col("did").is_in(unseen_users)
        & (pl.col("split") == "train")
    )
    assert unseen_train_rows.height == 0


def test_load_likes_holdout_end_filters_rows(tmp_path, stage_get_data_module):
    likes_rows = [
        {"did": "user_a", "subject_uri": "post:1", "record_created_at": "2024-01-02T00:00:00"},
        {"did": "user_a", "subject_uri": "post:2", "record_created_at": "2024-01-04T00:00:00"},
        {"did": "user_a", "subject_uri": "post:3", "record_created_at": "2024-01-06T00:00:00"},
    ]
    likes_path = _write_likes_parquet(tmp_path, likes_rows)
    raw_likes_lf = _scan_likes_lf(
        stage_get_data_module,
        likes_path,
        "2024-01-01T00:00:00",
        "2024-01-07T00:00:00",
    )
    logger = logging.getLogger("test_stage_get_data.holdout_end")

    likes_df, _ = stage_get_data_module._load_likes_core_polars(
        raw_likes_lf=raw_likes_lf,
        max_trainval_users=None,
        max_unseen_eval_users=0,
        max_likes_per_user=0,
        min_likes_per_user=1,
        random_seed=99,
        **_split_kwargs(
            holdout_start="2024-01-05T00:00:00",
            holdout_end="2024-01-05T00:00:00",
        ),
        logger=logger,
    )

    assert set(likes_df["subject_uri"].to_list()) == {"post:1", "post:2"}


def test_exact_prior_cumulative_likes_excludes_same_hour_for_positives(stage_get_data_module):
    raw_likes_lf = pl.DataFrame([
        {"did": "user_a", "subject_uri": "post:forced", "record_created_at": "2024-01-02T00:10:00"},
        {"did": "user_b", "subject_uri": "post:forced", "record_created_at": "2024-01-02T00:20:00"},
        {"did": "user_c", "subject_uri": "post:forced", "record_created_at": "2024-01-02T01:05:00"},
        {"did": "user_d", "subject_uri": "post:sampled_only", "record_created_at": "2024-01-02T00:15:00"},
    ]).lazy()
    likes_df = pl.DataFrame({
        "did": ["target_same", "target_next", "target_later"],
        "subject_uri": ["post:forced", "post:forced", "post:forced"],
        "record_created_at": [
            "2024-01-02T00:30:00",
            "2024-01-02T01:30:00",
            "2024-01-02T02:30:00",
        ],
        "split": ["train", "train", "train"],
    }).with_columns(
        pl.col("record_created_at").str.to_datetime(time_zone="UTC")
    ).with_columns(
        pl.col("record_created_at").dt.truncate("1h").alias("like_hour_bucket")
    )
    needed_pairs_df = stage_get_data_module._build_needed_post_hours_df(
        likes_df,
        pl.DataFrame(schema={
            "at_uri": pl.String,
            "in_random_sample": pl.Boolean,
            "negative_hour_bucket": pl.Datetime(time_zone="UTC"),
        }),
    )
    prior_counts_df, stats = stage_get_data_module._build_exact_prior_cumulative_likes_df(
        raw_likes_lf=raw_likes_lf,
        needed_post_hours_df=needed_pairs_df,
    )
    enriched_likes_df = stage_get_data_module._add_prior_cumulative_likes_to_likes(
        likes_df,
        prior_counts_df,
    )

    counts_by_did = {
        row["did"]: row["prior_cumulative_likes"]
        for row in enriched_likes_df.select(["did", "prior_cumulative_likes"]).iter_rows(named=True)
    }
    assert counts_by_did == {
        "target_same": 0,
        "target_next": 2,
        "target_later": 3,
    }
    assert stats["n_needed_prior_count_pairs"] == 3
    assert stats["n_exact_prior_source_like_rows"] == 3
    assert prior_counts_df.height == 3


def test_global_negative_counts_respect_hash_seed_and_min_likes(stage_get_data_module):
    raw_likes_lf = pl.DataFrame([
        {"did": "user_a", "subject_uri": "post:forced", "record_created_at": "2024-01-02T00:10:00"},
        {"did": "user_b", "subject_uri": "post:forced", "record_created_at": "2024-01-02T00:20:00"},
        {"did": "user_c", "subject_uri": "post:sampled", "record_created_at": "2024-01-02T01:05:00"},
        {"did": "user_d", "subject_uri": "post:sampled", "record_created_at": "2024-01-02T01:15:00"},
        {"did": "user_e", "subject_uri": "post:low", "record_created_at": "2024-01-02T01:20:00"},
    ]).lazy()

    no_sample_df, no_sample_stats = stage_get_data_module._build_global_like_counts_df(
        raw_likes_lf=raw_likes_lf,
        random_seed=123,
        initial_negative_sampling_pct=0.0,
        min_likes_per_negative_post=1,
    )
    assert no_sample_df.height == 0
    assert no_sample_stats["n_global_negative_candidate_posts"] == 0

    counts_df, stats = stage_get_data_module._build_global_like_counts_df(
        raw_likes_lf=raw_likes_lf,
        random_seed=123,
        initial_negative_sampling_pct=1.0,
        min_likes_per_negative_post=2,
    )

    counts_by_uri = {
        row["subject_uri"]: row["global_like_count"]
        for row in counts_df.select(["subject_uri", "global_like_count"]).iter_rows(named=True)
    }
    assert counts_by_uri == {"post:forced": 2, "post:sampled": 2}
    assert stats["n_global_negative_candidate_posts_before_min_likes"] == 3
    assert stats["n_global_negative_candidate_posts"] == 2


def test_force_included_positive_post_is_not_sampled_as_negative(stage_get_data_module):
    forced_post_lf = pl.DataFrame([{
        "at_uri": "post:forced",
        "did": "author_forced",
        "record_created_at": "2024-01-01T12:00:00",
        "record_text": "forced",
    }]).lazy()
    forced_negative_df = stage_get_data_module._get_negative_sample_posts(
        posts_lf=forced_post_lf,
        global_like_counts_df=pl.DataFrame({
            "subject_uri": pl.Series([], dtype=pl.String),
            "global_like_count": pl.Series([], dtype=pl.UInt64),
        }),
        liked_post_uris_df=pl.DataFrame({"subject_uri": ["post:forced"]}),
        cols_metadata=["at_uri", "record_created_at", "did", "record_text"],
        negative_samples_per_hour=10,
        random_seed=123,
        negative_sampling_alpha=0.5,
        train_start="2024-01-01T00:00:00",
        posts_end=None,
        holdout_end=None,
    )
    assert forced_negative_df.height == 0


def test_load_posts_random_sample_all_metadata_only(tmp_path, stage_get_data_module):
    """Test that _load_posts_core_polars returns metadata WITHOUT emb_idx column.
    
    NOTE: emb_idx is no longer assigned by _load_posts_core_polars. It's assigned
    later by _write_embeddings_memmap to ensure only posts with valid embeddings
    get indices.
    """
    embedding_model = "test-model"
    posts_rows = _make_posts_rows(embedding_model)
    posts_path = _write_posts_parquet(tmp_path, posts_rows)
    liked_post_uris_df = pl.DataFrame({"subject_uri": ["post:1", "post:3"]})
    logger = logging.getLogger("test_stage_get_data.posts_all")

    # Returns 3 values: posts_df (without emb_idx), stats, embed_dim
    posts_df, stats, embed_dim = stage_get_data_module._load_posts_core_polars(
        start_str="2024-01-01T00:00:00",
        end_str="2024-01-03T00:00:00",
        liked_post_uris_df=liked_post_uris_df,
        paths=[str(posts_path)],
        global_like_counts_df=_global_likes_for_posts(posts_rows),
        negative_samples_per_hour=len(posts_rows),
        negative_sampling_alpha=0.5,
        embedding_model=embedding_model,
        random_seed=11,
        **_split_kwargs(),
        logger=logger,
    )

    assert posts_df.height == len(posts_rows) * 24
    assert posts_df["in_random_sample"].all()
    assert stats["n_random_sample"] == len(posts_rows) * 24
    assert stats["n_random_sample_unique_posts"] == len(posts_rows)
    assert stats["n_random_sample_buckets"] == 24
    assert stats["n_liked_posts"] == 2
    assert embed_dim == 3

    # emb_idx should NOT be present (added later by memmap write)
    assert "emb_idx" not in posts_df.columns
    # Embeddings should not be expanded
    assert "embeddings" not in posts_df.columns
    assert "post_emb_0" not in posts_df.columns
    
    # Should have metadata columns
    assert "at_uri" in posts_df.columns
    assert "is_liked" in posts_df.columns
    assert "in_random_sample" in posts_df.columns
    assert "negative_hour_bucket" in posts_df.columns
    assert "split_window" in posts_df.columns
    assert posts_df.schema["negative_hour_bucket"] == pl.Datetime
    assert posts_df["split_window"].unique().to_list() == ["train"]
    assert posts_df.filter(pl.col("at_uri") == "post:1")["is_liked"].all()
    assert posts_df.filter(pl.col("at_uri") == "post:1")["in_random_sample"].all()
    assert posts_df.filter(pl.col("at_uri") == "post:1")["negative_hour_bucket"].null_count() == 0
    assert posts_df.filter(pl.col("at_uri") == "post:1").height == 24


def test_negative_sampling_weights_by_global_like_counts(tmp_path, stage_get_data_module):
    embedding_model = "test-model"
    posts_rows = [
        {"at_uri": "post:low", "did": "author_low", "record_created_at": "2024-01-01T12:00:00", "record_text": "low", "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.1, 0.2, 0.3])}]},
        {"at_uri": "post:mid", "did": "author_mid", "record_created_at": "2024-01-01T12:00:00", "record_text": "mid", "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.1, 0.2, 0.3])}]},
        {"at_uri": "post:popular", "did": "author_popular", "record_created_at": "2024-01-01T12:00:00", "record_text": "popular", "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.1, 0.2, 0.3])}]},
    ]
    posts_path = _write_posts_parquet(tmp_path, posts_rows)
    global_negative_like_counts_df = pl.DataFrame({
        "subject_uri": ["post:mid", "post:popular"],
        "global_like_count": [6, 1_000_000],
    }).with_columns(pl.col("global_like_count").cast(pl.UInt64))

    posts_df = stage_get_data_module._get_negative_sample_posts(
        posts_lf=pl.scan_parquet(str(posts_path)),
        global_like_counts_df=global_negative_like_counts_df,
        liked_post_uris_df=pl.DataFrame({"subject_uri": []}, schema={"subject_uri": pl.String}),
        cols_metadata=["at_uri", "record_created_at", "did", "record_text"],
        negative_samples_per_hour=1,
        random_seed=33,
        negative_sampling_alpha=1.0,
        train_start="2024-01-01T00:00:00",
        posts_end=None,
        holdout_end=None,
    )

    assert set(posts_df["at_uri"].to_list()) == {"post:popular"}
    assert posts_df.height == 24
    assert posts_df["negative_hour_bucket"].min().isoformat() == "2024-01-01T12:00:00+00:00"
    assert posts_df["negative_hour_bucket"].max().isoformat() == "2024-01-02T11:00:00+00:00"
    assert set(posts_df["record_created_at"].to_list()) == {"2024-01-01T12:00:00"}


def test_negative_sampling_hashes_post_and_hour(tmp_path, stage_get_data_module):
    embedding_model = "test-model"
    posts_rows = [
        {"at_uri": "post:a", "did": "author_a", "record_created_at": "2024-01-01T12:00:00", "record_text": "a", "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.1, 0.2, 0.3])}]},
        {"at_uri": "post:b", "did": "author_b", "record_created_at": "2024-01-01T12:00:00", "record_text": "b", "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.1, 0.2, 0.3])}]},
    ]
    posts_path = _write_posts_parquet(tmp_path, posts_rows)
    global_negative_like_counts_df = pl.DataFrame({
        "subject_uri": ["post:a", "post:b"],
        "global_like_count": [10, 10],
    }).with_columns(pl.col("global_like_count").cast(pl.UInt64))

    posts_df = stage_get_data_module._get_negative_sample_posts(
        posts_lf=pl.scan_parquet(str(posts_path)),
        global_like_counts_df=global_negative_like_counts_df,
        liked_post_uris_df=pl.DataFrame({"subject_uri": []}, schema={"subject_uri": pl.String}),
        cols_metadata=["at_uri", "record_created_at", "did", "record_text"],
        negative_samples_per_hour=1,
        random_seed=3,
        negative_sampling_alpha=0.0,
        train_start="2024-01-01T00:00:00",
        posts_end=None,
        holdout_end=None,
    )

    sampled_by_bucket = {
        row["negative_hour_bucket"].isoformat(): row["at_uri"]
        for row in posts_df.select(["negative_hour_bucket", "at_uri"]).iter_rows(named=True)
    }
    assert len(sampled_by_bucket) == 24
    assert set(sampled_by_bucket.values()) == {"post:a", "post:b"}


def test_exact_prior_counts_are_added_to_sampled_negatives_without_refiltering(stage_get_data_module):
    raw_likes_lf = pl.DataFrame([
        {"did": "user_a", "subject_uri": "post:negative", "record_created_at": "2024-01-02T02:10:00"},
        {"did": "user_b", "subject_uri": "post:negative", "record_created_at": "2024-01-02T02:20:00"},
    ]).lazy()
    posts_df = pl.DataFrame({
        "at_uri": ["post:negative", "post:negative"],
        "record_created_at": [
            "2024-01-02T00:15:00",
            "2024-01-02T00:15:00",
        ],
        "did": ["author_negative", "author_negative"],
        "record_text": ["negative", "negative"],
        "is_liked": [False, False],
        "in_random_sample": [True, True],
        "negative_hour_bucket": [
            "2024-01-02T02:00:00",
            "2024-01-02T03:00:00",
        ],
    }).with_columns(
        pl.col("negative_hour_bucket").str.to_datetime(time_zone="UTC")
    )
    likes_df = pl.DataFrame(schema={
        "subject_uri": pl.String,
        "like_hour_bucket": pl.Datetime(time_zone="UTC"),
    })

    needed_pairs_df = stage_get_data_module._build_needed_post_hours_df(likes_df, posts_df)
    prior_counts_df, _ = stage_get_data_module._build_exact_prior_cumulative_likes_df(
        raw_likes_lf=raw_likes_lf,
        needed_post_hours_df=needed_pairs_df,
    )
    enriched_posts_df = stage_get_data_module._add_prior_cumulative_likes_to_posts(
        posts_df,
        prior_counts_df,
    )

    counts_by_bucket = {
        row["negative_hour_bucket"].isoformat(): row["prior_cumulative_likes"]
        for row in enriched_posts_df.select(["negative_hour_bucket", "prior_cumulative_likes"]).iter_rows(named=True)
    }
    assert counts_by_bucket == {
        "2024-01-02T02:00:00+00:00": 0,
        "2024-01-02T03:00:00+00:00": 2,
    }
    assert enriched_posts_df.height == 2


def test_load_posts_liked_always_included_with_null_negative_bucket(tmp_path, stage_get_data_module):
    """Test that liked posts are always included even with zero random sample."""
    embedding_model = "test-model"
    posts_rows = _make_posts_rows(embedding_model)
    posts_path = _write_posts_parquet(tmp_path, posts_rows)
    liked_post_uris_df = pl.DataFrame({"subject_uri": ["post:2", "post:5"]})
    logger = logging.getLogger("test_stage_get_data.posts_liked")

    posts_df, stats, embed_dim = stage_get_data_module._load_posts_core_polars(
        start_str="2024-01-01T00:00:00",
        end_str="2024-01-03T00:00:00",
        liked_post_uris_df=liked_post_uris_df,
        paths=[str(posts_path)],
        global_like_counts_df=_global_likes_for_posts(posts_rows),
        negative_samples_per_hour=0,
        negative_sampling_alpha=0.5,
        embedding_model=embedding_model,
        random_seed=21,
        **_split_kwargs(),
        logger=logger,
    )

    returned_uris = set(posts_df["at_uri"].to_list())
    assert set(["post:2", "post:5"]).issubset(returned_uris)
    assert posts_df.filter(pl.col("at_uri") == "post:2")["is_liked"].all()
    assert posts_df.filter(pl.col("at_uri") == "post:5")["is_liked"].all()
    assert posts_df["in_random_sample"].sum() == 0
    assert posts_df["negative_hour_bucket"].null_count() == posts_df.height
    assert stats["n_liked_posts"] == 2
    
    # emb_idx should NOT be present (added later by memmap write)
    assert "emb_idx" not in posts_df.columns


def test_load_posts_samples_approximately_per_hour(tmp_path, stage_get_data_module):
    embedding_model = "test-model"
    posts_rows = []
    for hour in [0, 1]:
        for idx in range(50):
            posts_rows.append({
                "at_uri": f"post:{hour}:{idx}",
                "did": f"user_{hour}_{idx}",
                "record_created_at": f"2024-01-02T{hour:02d}:15:00",
                "record_text": f"text {hour} {idx}",
                "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.1, 0.2, 0.3])}],
            })
    posts_path = _write_posts_parquet(tmp_path, posts_rows)
    liked_post_uris_df = pl.DataFrame({"subject_uri": ["missing:post"]})
    logger = logging.getLogger("test_stage_get_data.posts_per_hour")

    posts_df, stats, _ = stage_get_data_module._load_posts_core_polars(
        start_str="2024-01-02T00:00:00",
        end_str="2024-01-02T02:00:00",
        liked_post_uris_df=liked_post_uris_df,
        paths=[str(posts_path)],
        global_like_counts_df=_global_likes_for_posts(posts_rows),
        negative_samples_per_hour=10,
        negative_sampling_alpha=0.5,
        embedding_model=embedding_model,
        random_seed=33,
        **_split_kwargs(),
        logger=logger,
    )

    random_counts = (
        posts_df
        .filter(pl.col("in_random_sample"))
        .group_by("negative_hour_bucket")
        .len()
    )
    assert posts_df.height == 20
    assert stats["n_random_sample"] == posts_df.height
    assert stats["n_random_sample_buckets"] == 2
    assert random_counts["len"].min() == 10
    assert random_counts["len"].max() == 10


def test_load_posts_adds_split_window_and_filters_holdout_end(tmp_path, stage_get_data_module):
    embedding_model = "test-model"
    posts_rows = [
        {"at_uri": "post:pretrain", "did": "author_a", "record_created_at": "2024-01-01T00:00:00", "record_text": "pre", "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.1, 0.2, 0.3])}]},
        {"at_uri": "post:train", "did": "author_a", "record_created_at": "2024-01-02T00:00:00", "record_text": "train", "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.1, 0.2, 0.3])}]},
        {"at_uri": "post:val", "did": "author_b", "record_created_at": "2024-01-04T00:00:00", "record_text": "val", "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.1, 0.2, 0.3])}]},
        {"at_uri": "post:holdout", "did": "author_c", "record_created_at": "2024-01-05T00:00:00", "record_text": "holdout", "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.1, 0.2, 0.3])}]},
        {"at_uri": "post:after", "did": "author_d", "record_created_at": "2024-01-06T00:00:00", "record_text": "after", "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.1, 0.2, 0.3])}]},
    ]
    posts_path = _write_posts_parquet(tmp_path, posts_rows)
    liked_post_uris_df = pl.DataFrame({"subject_uri": []}, schema={"subject_uri": pl.String})
    logger = logging.getLogger("test_stage_get_data.posts_split_window")

    posts_df, _, _ = stage_get_data_module._load_posts_core_polars(
        start_str="2024-01-01T00:00:00",
        end_str="2024-01-07T00:00:00",
        liked_post_uris_df=liked_post_uris_df,
        paths=[str(posts_path)],
        global_like_counts_df=_global_likes_for_posts(posts_rows),
        negative_samples_per_hour=len(posts_rows),
        negative_sampling_alpha=0.5,
        embedding_model=embedding_model,
        random_seed=33,
        **_split_kwargs(
            train_start="2024-01-02T00:00:00",
            val_start="2024-01-03T00:00:00",
            holdout_start="2024-01-05T00:00:00",
            holdout_end="2024-01-06T00:00:00",
        ),
        logger=logger,
    )

    split_by_uri = {
        row["at_uri"]: row["split_window"]
        for row in posts_df.select(["at_uri", "split_window"]).iter_rows(named=True)
    }
    assert split_by_uri == {
        "post:train": "train",
        "post:val": "val",
        "post:holdout": "holdout",
    }


def test_load_posts_rejects_non_string_record_created_at(tmp_path, stage_get_data_module):
    embedding_model = "test-model"
    posts_path = _write_posts_parquet(tmp_path, [{
        "at_uri": "post:1",
        "did": "user_a",
        "record_created_at": datetime(2024, 1, 2, 0, 15, 0),
        "record_text": "one",
        "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.1, 0.2, 0.3])}],
    }])
    liked_post_uris_df = pl.DataFrame({"subject_uri": []}, schema={"subject_uri": pl.String})
    logger = logging.getLogger("test_stage_get_data.posts_bad_ts")

    with pytest.raises(ValueError, match="record_created_at"):
        stage_get_data_module._load_posts_core_polars(
                start_str=None,
                end_str=None,
                liked_post_uris_df=liked_post_uris_df,
                paths=[str(posts_path)],
                global_like_counts_df=_global_likes_for_posts([{
                    "at_uri": "post:1",
                }]),
                negative_samples_per_hour=1,
                negative_sampling_alpha=0.5,
                embedding_model=embedding_model,
            random_seed=33,
            **_split_kwargs(),
            logger=logger,
        )


def test_build_author_idx_counts_train_exposures_and_joins_to_core_tables(stage_get_data_module):
    posts_core_df = pl.DataFrame({
        "at_uri": ["post:1", "post:2", "post:3", "post:4", "post:5", "post:6"],
        "did": ["author_a", "author_a", "author_b", "author_b", "author_c", "author_d"],
        "in_random_sample": [True, True, False, True, True, True],
        "split_window": ["train", "train", "train", "val", "train", "holdout"],
    })
    likes_core_df = pl.DataFrame({
        "did": ["user_1", "user_2", "user_3", "user_4", "user_5", "user_6"],
        "subject_uri": ["post:1", "post:1", "post:2", "post:3", "post:4", "post:5"],
        "split": ["train", "train", "train", "train", "val", "holdout_seen_users"],
        "author_did": ["author_a", "author_a", "author_a", "author_b", "author_b", "author_c"],
    })
    logger = logging.getLogger("test_stage_get_data.author_idx")

    author_idx_df, stats = stage_get_data_module._build_author_idx_mapping(
        posts_core_df=posts_core_df,
        likes_core_df=likes_core_df,
        min_author_support=2,
        logger=logger,
    )

    assert author_idx_df.to_dicts() == [{
        "author_did": "author_a",
        "author_train_count": 5,
        "author_idx": 2,
    }]
    assert stats["n_author_random_train_exposures"] == 3
    assert stats["n_author_like_train_exposures"] == 4

    posts_with_author_idx, likes_with_author_idx = stage_get_data_module._join_author_idx_to_core_tables(
        posts_core_df=posts_core_df,
        likes_core_df=likes_core_df,
        author_idx_df=author_idx_df,
    )
    post_idx_by_uri = {
        row["at_uri"]: row["author_idx"]
        for row in posts_with_author_idx.select(["at_uri", "author_idx"]).iter_rows(named=True)
    }
    assert post_idx_by_uri["post:1"] == 2
    assert post_idx_by_uri["post:2"] == 2
    assert post_idx_by_uri["post:3"] is None
    assert likes_with_author_idx.filter(pl.col("author_did") == "author_a")["author_idx"].to_list() == [2, 2, 2]
    assert likes_with_author_idx.filter(pl.col("author_did") == "author_b")["author_idx"].null_count() == 2


def test_build_author_idx_assignment_is_deterministic_and_starts_at_two(stage_get_data_module):
    posts_core_df = pl.DataFrame({
        "at_uri": ["post:b", "post:a", "post:c"],
        "did": ["author_b", "author_a", "author_c"],
        "in_random_sample": [True, True, True],
        "split_window": ["train", "train", "train"],
    })
    likes_core_df = pl.DataFrame({
        "subject_uri": pl.Series([], dtype=pl.String),
        "split": pl.Series([], dtype=pl.String),
        "author_did": pl.Series([], dtype=pl.String),
    })
    logger = logging.getLogger("test_stage_get_data.author_idx_deterministic")

    author_idx_df, _ = stage_get_data_module._build_author_idx_mapping(
        posts_core_df=posts_core_df,
        likes_core_df=likes_core_df,
        min_author_support=1,
        logger=logger,
    )

    assert author_idx_df.select(["author_did", "author_idx"]).to_dicts() == [
        {"author_did": "author_a", "author_idx": 2},
        {"author_did": "author_b", "author_idx": 3},
        {"author_did": "author_c", "author_idx": 4},
    ]


def test_write_embeddings_memmap(tmp_path, stage_get_data_module):
    """Test that _write_embeddings_memmap creates a valid memmap file and returns uri_to_idx.
    
    The function now:
    1. Writes embeddings sequentially (not to pre-assigned indices)
    2. Returns uri_to_idx mapping and stats
    3. Accepts posts_core_df WITHOUT emb_idx column
    """
    embedding_model = "test-model"
    posts_rows = _make_posts_rows(embedding_model)
    posts_path = _write_posts_parquet(tmp_path, posts_rows)
    
    # Create posts_core_df WITHOUT emb_idx (that's now assigned by this function)
    posts_core_df = pl.DataFrame({
        "at_uri": ["post:1", "post:3", "post:5"],
    })
    
    embeddings_path = tmp_path / "embeddings.npy"
    logger = logging.getLogger("test_stage_get_data.memmap")
    embed_dim = 3
    
    # Function now returns (uri_to_idx, stats)
    uri_to_idx, stats = stage_get_data_module._write_embeddings_memmap(
        posts_paths=[str(posts_path)],
        posts_start="2024-01-01T00:00:00",
        posts_end="2024-01-03T00:00:00",
        posts_core_df=posts_core_df,
        embeddings_path=embeddings_path,
        embed_dim=embed_dim,
        embedding_model=embedding_model,
        logger=logger,
    )
    
    # Verify memmap was created
    assert embeddings_path.exists()
    
    # Verify uri_to_idx was returned
    assert len(uri_to_idx) == 3
    assert "post:1" in uri_to_idx
    assert "post:3" in uri_to_idx
    assert "post:5" in uri_to_idx
    
    # Verify stats
    assert stats["n_embeddings_valid"] == 3
    assert stats["n_embeddings_null"] == 0
    
    # Load and verify contents using the uri_to_idx mapping
    mmap = np.load(embeddings_path, mmap_mode="r")
    assert mmap.shape == (3, embed_dim)
    
    # post:1 has embedding [0.1, 0.2, 0.3]
    assert np.allclose(mmap[uri_to_idx["post:1"]], [0.1, 0.2, 0.3], atol=1e-5)
    # post:3 has embedding [0.7, 0.8, 0.9]
    assert np.allclose(mmap[uri_to_idx["post:3"]], [0.7, 0.8, 0.9], atol=1e-5)
    # post:5 has embedding [1.3, 1.4, 1.5]
    assert np.allclose(mmap[uri_to_idx["post:5"]], [1.3, 1.4, 1.5], atol=1e-5)
    
    del mmap  # Close memmap
    assert not embeddings_path.with_suffix(".tmp.npy").exists()


def test_write_embeddings_memmap_handles_missing_embeddings(tmp_path, stage_get_data_module):
    """Test that _write_embeddings_memmap correctly handles posts with missing embeddings.
    
    Posts with null/invalid embeddings should be skipped, and only valid embeddings
    should get indices. This ensures no gaps in the memmap file.
    """
    embedding_model = "test-model"
    # Create posts where some have null embeddings
    posts_rows = [
        {"at_uri": "post:1", "record_created_at": "2024-01-01T10:00:00", "did": "user:a", 
         "record_text": "text1", "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.1, 0.2, 0.3])}]},
        {"at_uri": "post:2", "record_created_at": "2024-01-01T11:00:00", "did": "user:b", 
         "record_text": "text2", "embeddings": None},  # NULL embedding
        {"at_uri": "post:3", "record_created_at": "2024-01-02T10:00:00", "did": "user:c", 
         "record_text": "text3", "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.7, 0.8, 0.9])}]},
    ]
    posts_path = _write_posts_parquet(tmp_path, posts_rows)
    
    # Request all 3 posts
    posts_core_df = pl.DataFrame({
        "at_uri": ["post:1", "post:2", "post:3"],
    })
    
    embeddings_path = tmp_path / "embeddings.npy"
    logger = logging.getLogger("test_stage_get_data.missing_emb")
    embed_dim = 3
    
    uri_to_idx, stats = stage_get_data_module._write_embeddings_memmap(
        posts_paths=[str(posts_path)],
        posts_start="2024-01-01T00:00:00",
        posts_end="2024-01-03T00:00:00",
        posts_core_df=posts_core_df,
        embeddings_path=embeddings_path,
        embed_dim=embed_dim,
        embedding_model=embedding_model,
        logger=logger,
    )
    
    # Only 2 posts should have valid embeddings
    assert len(uri_to_idx) == 2
    assert "post:1" in uri_to_idx
    assert "post:2" not in uri_to_idx  # NULL embedding
    assert "post:3" in uri_to_idx
    
    # Stats should reflect the null
    assert stats["n_embeddings_valid"] == 2
    assert stats["n_embeddings_null"] == 1
    assert stats["n_posts_dropped_no_embedding"] == 1
    
    # Memmap should have exactly 2 rows (no gaps)
    mmap = np.load(embeddings_path, mmap_mode="r")
    assert mmap.shape == (2, embed_dim)
    
    # Verify embeddings are correct using uri_to_idx
    assert np.allclose(mmap[uri_to_idx["post:1"]], [0.1, 0.2, 0.3], atol=1e-5)
    assert np.allclose(mmap[uri_to_idx["post:3"]], [0.7, 0.8, 0.9], atol=1e-5)
    
    del mmap
    assert not embeddings_path.with_suffix(".tmp.npy").exists()


def test_write_embeddings_memmap_skips_duplicate_uris_and_cleans_temp_file(tmp_path, stage_get_data_module):
    embedding_model = "test-model"
    posts_rows = [
        {"at_uri": "post:1", "record_created_at": "2024-01-01T10:00:00", "did": "user:a",
         "record_text": "text1", "embeddings": [{"key": embedding_model, "value": _encode_embedding([0.1, 0.2, 0.3])}]},
        {"at_uri": "post:1", "record_created_at": "2024-01-01T10:05:00", "did": "user:a",
         "record_text": "text1 duplicate", "embeddings": [{"key": embedding_model, "value": _encode_embedding([9.0, 9.0, 9.0])}]},
    ]
    posts_path = _write_posts_parquet(tmp_path, posts_rows)
    posts_core_df = pl.DataFrame({"at_uri": ["post:1"]})
    embeddings_path = tmp_path / "embeddings.npy"
    logger = logging.getLogger("test_stage_get_data.duplicate_emb")
    embed_dim = 3

    uri_to_idx, stats = stage_get_data_module._write_embeddings_memmap(
        posts_paths=[str(posts_path)],
        posts_start="2024-01-01T00:00:00",
        posts_end="2024-01-02T00:00:00",
        posts_core_df=posts_core_df,
        embeddings_path=embeddings_path,
        embed_dim=embed_dim,
        embedding_model=embedding_model,
        logger=logger,
    )

    assert uri_to_idx == {"post:1": 0}
    assert stats["n_embeddings_valid"] == 1
    assert stats["n_embeddings_duplicate_uri"] == 1
    assert not embeddings_path.with_suffix(".tmp.npy").exists()

    mmap = np.load(embeddings_path, mmap_mode="r")
    assert mmap.shape == (1, embed_dim)
    assert np.allclose(mmap[0], [0.1, 0.2, 0.3], atol=1e-5)
    del mmap


def test_write_embeddings_memmap_cleans_temp_file_when_no_valid_embeddings(tmp_path, stage_get_data_module):
    embedding_model = "test-model"
    posts_rows = [
        {"at_uri": "post:1", "record_created_at": "2024-01-01T10:00:00", "did": "user:a",
         "record_text": "text1", "embeddings": None},
    ]
    posts_path = _write_posts_parquet(tmp_path, posts_rows)
    posts_core_df = pl.DataFrame({"at_uri": ["post:1"]})
    embeddings_path = tmp_path / "embeddings.npy"
    logger = logging.getLogger("test_stage_get_data.no_valid_emb")

    with pytest.raises(ValueError, match="No valid embeddings"):
        stage_get_data_module._write_embeddings_memmap(
            posts_paths=[str(posts_path)],
            posts_start="2024-01-01T00:00:00",
            posts_end="2024-01-02T00:00:00",
            posts_core_df=posts_core_df,
            embeddings_path=embeddings_path,
            embed_dim=3,
            embedding_model=embedding_model,
            logger=logger,
        )

    assert not embeddings_path.exists()
    assert not embeddings_path.with_suffix(".tmp.npy").exists()


def test_get_embeddings_list_col_extracts_and_decodes(stage_get_data_module):
    target_model = "model_b"
    expected_vec = [0.1, 0.2, 0.3]

    df = pl.DataFrame(
        {
            "embeddings": [
                [
                    {"key": "model_a", "value": _encode_embedding([9.9])},
                    {"key": target_model, "value": _encode_embedding(expected_vec)},
                ],
                [{"key": "model_a", "value": _encode_embedding([1.0, 2.0])}],
                None,
            ]
        }
    )

    out = stage_get_data_module.get_embeddings_list_col_polars(df.lazy(), target_model).collect()
    got = out["_emb_vec"].to_list()

    assert got[0] == pytest.approx(expected_vec, rel=1e-6, abs=1e-6)
    assert got[1] is None
    assert got[2] is None
