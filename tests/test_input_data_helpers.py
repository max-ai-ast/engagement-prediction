import base64
import struct
import zlib

import polars as pl
import pytest

from model_serving.input_data_helpers import get_embeddings_list_col_polars, _extract_compressed_embedding_vector_from_struct


def _encode_embedding(vec: list[float]) -> str:
    raw = struct.pack(f"<{len(vec)}f", *vec)
    compressed = zlib.compress(raw)
    return base64.b85encode(compressed).decode()


def test_get_embedding_value_for_model_dicts():
    embeddings = [
        {"key": "other", "value": "x"},
        {"key": "target", "value": "y"},
    ]
    assert _extract_compressed_embedding_vector_from_struct(embeddings, "target") == "y"
    assert _extract_compressed_embedding_vector_from_struct(embeddings, "missing") is None


def test_get_embedding_value_for_model_list_or_tuple_items():
    embeddings_tuples = [
        ("other", "x"),
        ("target", "y"),
    ]
    assert _extract_compressed_embedding_vector_from_struct(embeddings_tuples, "target") == "y"

    embeddings_lists = [
        ["other", "x"],
        ["target", "y"],
    ]
    assert _extract_compressed_embedding_vector_from_struct(embeddings_lists, "target") == "y"


def test_get_embeddings_list_col_extracts_and_decodes():
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

    out = get_embeddings_list_col_polars(df.lazy(), target_model).collect()
    got = out["_emb_vec"].to_list()

    assert got[0] == pytest.approx(expected_vec, rel=1e-6, abs=1e-6)
    assert got[1] is None
    assert got[2] is None
