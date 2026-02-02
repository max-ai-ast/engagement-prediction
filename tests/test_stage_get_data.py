import importlib.util
import logging
import struct
import sys
import zlib
import base64
from pathlib import Path

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


def _write_likes_parquet(tmp_path, rows):
    df = pl.DataFrame(rows)
    path = tmp_path / "likes.parquet"
    df.write_parquet(path)
    return path


def _write_posts_parquet(tmp_path, rows):
    df = pl.DataFrame(rows)
    path = tmp_path / "posts.parquet"
    df.write_parquet(path)
    return path


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
    logger = logging.getLogger("test_stage_get_data.likes")

    likes_df, stats = stage_get_data_module._load_likes_core_polars(
        start_str="2024-01-02T00:00:00",
        end_str="2024-01-04T00:00:00",
        paths=[str(likes_path)],
        max_liking_users=None,
        max_likes_per_user=0,
        min_likes_per_user=2,
        random_seed=123,
        logger=logger,
    )

    assert likes_df.height == 2
    assert likes_df["did"].unique().to_list() == ["user_a"]
    assert likes_df.schema["record_created_at"] == pl.Datetime
    assert likes_df['did'].n_unique() == 1


def test_load_likes_per_user_cap(tmp_path, stage_get_data_module):
    likes_rows = [
        {"did": "user_a", "subject_uri": f"post:{i}", "record_created_at": "2024-01-02T00:00:00"}
        for i in range(5)
    ] + [
        {"did": "user_b", "subject_uri": "post:99", "record_created_at": "2024-01-02T00:00:00"},
    ]
    likes_path = _write_likes_parquet(tmp_path, likes_rows)
    logger = logging.getLogger("test_stage_get_data.likes_cap")

    likes_df, _ = stage_get_data_module._load_likes_core_polars(
        start_str="2024-01-01T00:00:00",
        end_str="2024-01-03T00:00:00",
        paths=[str(likes_path)],
        max_liking_users=None,
        max_likes_per_user=2,
        min_likes_per_user=0,
        random_seed=42,
        logger=logger,
    )

    assert likes_df.filter(pl.col("did") == "user_a").height == 2
    assert likes_df.filter(pl.col("did") == "user_b").height == 1


def test_get_sampled_users_deterministic(stage_get_data_module):
    likes_rows = [
        {"did": f"user_{i}", "subject_uri": f"post:{i}", "record_created_at": "2024-01-02T00:00:00"}
        for i in range(10)
    ]
    likes_lf = pl.DataFrame(likes_rows).lazy()

    first, *_ = stage_get_data_module._get_sampled_users_with_min_likes(
        likes_lf=likes_lf,
        min_likes_per_user=1,
        max_liking_users=5,
        random_seed=7,
    )
    second, *_ = stage_get_data_module._get_sampled_users_with_min_likes(
        likes_lf=likes_lf,
        min_likes_per_user=1,
        max_liking_users=5,
        random_seed=7,
    )

    first_set = set(first["did"].to_list())
    second_set = set(second["did"].to_list())
    assert first_set == second_set
    assert len(first_set) <= 5


def test_load_posts_random_sample_all_and_embeddings(tmp_path, stage_get_data_module):
    embedding_model = "test-model"
    posts_rows = _make_posts_rows(embedding_model)
    posts_path = _write_posts_parquet(tmp_path, posts_rows)
    liked_post_uris_df = pl.DataFrame({"subject_uri": ["post:1", "post:3"]})
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    logger = logging.getLogger("test_stage_get_data.posts_all")

    posts_df, stats, embed_dim, posts_core_path = stage_get_data_module._load_posts_core_polars(
        start_str="2024-01-01T00:00:00",
        end_str="2024-01-03T00:00:00",
        liked_post_uris_df=liked_post_uris_df,
        paths=[str(posts_path)],
        negative_posts_sample=len(posts_rows),
        embedding_model=embedding_model,
        random_seed=11,
        logger=logger,
        out_dir=out_dir,
    )

    assert posts_df.height == len(posts_rows)
    assert posts_df["in_random_sample"].all()
    assert stats["n_random_sample"] == len(posts_rows)
    assert stats["n_liked_posts"] == 2
    assert embed_dim == 3

    parquet_df = pl.read_parquet(posts_core_path)
    assert "embeddings" not in parquet_df.columns
    assert "post_emb_0" in parquet_df.columns
    assert "post_emb_1" in parquet_df.columns
    assert "post_emb_2" in parquet_df.columns
    assert "post_emb_0" not in posts_df.columns


def test_load_posts_liked_always_included(tmp_path, stage_get_data_module):
    embedding_model = "test-model"
    posts_rows = _make_posts_rows(embedding_model)
    posts_path = _write_posts_parquet(tmp_path, posts_rows)
    liked_post_uris_df = pl.DataFrame({"subject_uri": ["post:2", "post:5"]})
    out_dir = tmp_path / "out2"
    out_dir.mkdir()
    logger = logging.getLogger("test_stage_get_data.posts_liked")

    posts_df, stats, *_ = stage_get_data_module._load_posts_core_polars(
        start_str="2024-01-01T00:00:00",
        end_str="2024-01-03T00:00:00",
        liked_post_uris_df=liked_post_uris_df,
        paths=[str(posts_path)],
        negative_posts_sample=0,
        embedding_model=embedding_model,
        random_seed=21,
        logger=logger,
        out_dir=out_dir,
    )

    returned_uris = set(posts_df["at_uri"].to_list())
    assert set(["post:2", "post:5"]).issubset(returned_uris)
    assert posts_df.filter(pl.col("at_uri") == "post:2")["is_liked"].all()
    assert posts_df.filter(pl.col("at_uri") == "post:5")["is_liked"].all()
    assert stats["n_liked_posts"] == 2
