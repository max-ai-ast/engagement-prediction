import base64
import struct
import zlib

import pytest

from shared.input_data_helpers import (
    _extract_compressed_embedding_vector_from_struct,
    get_user_tower_input_from_raw_history_embeddings,
)


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


def test_get_user_tower_input_filters_padding_and_mask():
    embedding_model = "model_b"

    raw_history_embeddings = [
        [
            {"key": "other", "value": _encode_embedding([9.9])},
            {"key": embedding_model, "value": _encode_embedding([0.1, 0.2, 0.3])},
        ],
        [{"key": "other", "value": _encode_embedding([1.0, 2.0, 3.0])}],  # missing model -> dropped
        None,  # dropped
        [{"key": embedding_model, "value": _encode_embedding([0.4, 0.5, 0.6])}],
    ]

    padded, mask = get_user_tower_input_from_raw_history_embeddings(
        raw_history_embeddings=raw_history_embeddings,
        embedding_model=embedding_model,
        max_history_len=4,
        embed_dim=3,
    )

    assert padded.shape == (4, 3)
    assert padded.dtype == "float32"
    assert mask.shape == (4,)
    assert mask.dtype == bool

    assert padded[0].tolist() == pytest.approx([0.1, 0.2, 0.3], rel=1e-6, abs=1e-6)
    assert padded[1].tolist() == pytest.approx([0.4, 0.5, 0.6], rel=1e-6, abs=1e-6)
    assert padded[2].tolist() == pytest.approx([0.0, 0.0, 0.0], rel=0, abs=0)
    assert padded[3].tolist() == pytest.approx([0.0, 0.0, 0.0], rel=0, abs=0)

    assert mask.tolist() == [True, True, False, False]


def test_get_user_tower_input_truncates_to_max_history_len():
    embedding_model = "model_b"
    raw_history_embeddings = [
        [{"key": embedding_model, "value": _encode_embedding([float(i), float(i + 1), float(i + 2)])}]
        for i in range(10, 15)
    ]

    padded, mask = get_user_tower_input_from_raw_history_embeddings(
        raw_history_embeddings=raw_history_embeddings,
        embedding_model=embedding_model,
        max_history_len=3,
        embed_dim=3,
    )

    assert padded.shape == (3, 3)
    assert mask.tolist() == [True, True, True]
    assert padded[0].tolist() == pytest.approx([10.0, 11.0, 12.0], rel=1e-6, abs=1e-6)
    assert padded[1].tolist() == pytest.approx([11.0, 12.0, 13.0], rel=1e-6, abs=1e-6)
    assert padded[2].tolist() == pytest.approx([12.0, 13.0, 14.0], rel=1e-6, abs=1e-6)


def test_get_user_tower_input_raises_on_embed_dim_mismatch():
    embedding_model = "model_b"
    raw_history_embeddings = [
        [{"key": embedding_model, "value": _encode_embedding([1.0, 2.0])}],
    ]

    with pytest.raises(ValueError, match="embed_dim"):
        get_user_tower_input_from_raw_history_embeddings(
            raw_history_embeddings=raw_history_embeddings,
            embedding_model=embedding_model,
            max_history_len=3,
            embed_dim=3,
        )


def test_get_user_tower_input_all_missing_returns_zeros():
    embedding_model = "model_b"
    raw_history_embeddings = [
        None,
        [{"key": "other", "value": _encode_embedding([9.9, 9.8, 9.7])}],
    ]

    padded, mask = get_user_tower_input_from_raw_history_embeddings(
        raw_history_embeddings=raw_history_embeddings,
        embedding_model=embedding_model,
        max_history_len=2,
        embed_dim=3,
    )

    assert padded.tolist() == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert mask.tolist() == [False, False]


def test_get_user_tower_input_raises_on_non_zlib_embedding():
    embedding_model = "model_b"
    raw = struct.pack("<3f", 0.1, 0.2, 0.3)
    not_compressed = base64.b85encode(raw).decode()
    raw_history_embeddings = [[{"key": embedding_model, "value": not_compressed}]]

    with pytest.raises(zlib.error):
        get_user_tower_input_from_raw_history_embeddings(
            raw_history_embeddings=raw_history_embeddings,
            embedding_model=embedding_model,
            max_history_len=3,
            embed_dim=3,
        )
