import importlib.util
import logging
import struct
import sys
import zlib
import base64
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
        negative_posts_sample=len(posts_rows),
        embedding_model=embedding_model,
        random_seed=11,
        logger=logger,
    )

    assert posts_df.height == len(posts_rows)
    assert posts_df["in_random_sample"].all()
    assert stats["n_random_sample"] == len(posts_rows)
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


def test_load_posts_liked_always_included(tmp_path, stage_get_data_module):
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
        negative_posts_sample=0,
        embedding_model=embedding_model,
        random_seed=21,
        logger=logger,
    )

    returned_uris = set(posts_df["at_uri"].to_list())
    assert set(["post:2", "post:5"]).issubset(returned_uris)
    assert posts_df.filter(pl.col("at_uri") == "post:2")["is_liked"].all()
    assert posts_df.filter(pl.col("at_uri") == "post:5")["is_liked"].all()
    assert stats["n_liked_posts"] == 2
    
    # emb_idx should NOT be present (added later by memmap write)
    assert "emb_idx" not in posts_df.columns


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
    
    # post:1 has embedding [0.1, 0.2, 0.3]
    assert np.allclose(mmap[uri_to_idx["post:1"]], [0.1, 0.2, 0.3], atol=1e-5)
    # post:3 has embedding [0.7, 0.8, 0.9]
    assert np.allclose(mmap[uri_to_idx["post:3"]], [0.7, 0.8, 0.9], atol=1e-5)
    # post:5 has embedding [1.3, 1.4, 1.5]
    assert np.allclose(mmap[uri_to_idx["post:5"]], [1.3, 1.4, 1.5], atol=1e-5)
    
    del mmap  # Close memmap


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
    
    # Verify embeddings are correct using uri_to_idx
    assert np.allclose(mmap[uri_to_idx["post:1"]], [0.1, 0.2, 0.3], atol=1e-5)
    assert np.allclose(mmap[uri_to_idx["post:3"]], [0.7, 0.8, 0.9], atol=1e-5)
    
    del mmap
