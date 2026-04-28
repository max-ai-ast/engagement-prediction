import base64
import struct
import zlib

import pytest

from shared.input_data_helpers import (
    _decompress_and_unpack_embedding,
    _extract_compressed_embedding_vector_from_struct,
    get_embedding_dim_for_known_model,
    get_expanded_embedding_vector,
    get_padded_embedding_history_and_mask,
    get_padded_embedding_history_and_mask_batched,
)


def test_get_embedding_value_for_model_dicts():
    embeddings = [
        {"key": "other", "value": "x"},
        {"key": "target", "value": "y"},
    ]
    assert _extract_compressed_embedding_vector_from_struct(embeddings, "target") == "y"
    assert _extract_compressed_embedding_vector_from_struct(embeddings, "missing") is None
    assert _extract_compressed_embedding_vector_from_struct(None, "target") is None


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


def test_get_embedding_value_for_model_object_items():
    class _Item:
        def __init__(self, key: str, value: str):
            self.key = key
            self.value = value

    embeddings = [_Item("other", "x"), _Item("target", "y")]
    assert _extract_compressed_embedding_vector_from_struct(embeddings, "target") == "y"
    assert _extract_compressed_embedding_vector_from_struct(embeddings, "missing") is None


def test_get_embedding_dim_for_known_model_ok_and_unknown():
    assert get_embedding_dim_for_known_model("all-MiniLM-L6-v2") == 384
    with pytest.raises(ValueError, match="Unknown embedding model"):
        get_embedding_dim_for_known_model("does-not-exist")


def _encode_embedding_bytes(vec: list[float], *, compress: bool) -> str:
    raw = struct.pack(f"<{len(vec)}f", *vec)
    bs = zlib.compress(raw) if compress else raw
    return base64.b85encode(bs).decode()


def test_decompress_and_unpack_embedding_round_trip_compressed():
    s = _encode_embedding_bytes([0.1, 0.2, 0.3], compress=True)
    out = _decompress_and_unpack_embedding(s, decompress=True)
    assert out == pytest.approx([0.1, 0.2, 0.3], rel=1e-6, abs=1e-6)


def test_decompress_and_unpack_embedding_uncompressed_with_decompress_none_and_false():
    s = _encode_embedding_bytes([1.0, 2.0, 3.0], compress=False)
    out_none = _decompress_and_unpack_embedding(s, decompress=None)
    out_false = _decompress_and_unpack_embedding(s, decompress=False)
    assert out_none == pytest.approx([1.0, 2.0, 3.0], rel=1e-6, abs=1e-6)
    assert out_false == pytest.approx([1.0, 2.0, 3.0], rel=1e-6, abs=1e-6)


def test_decompress_and_unpack_embedding_raises_when_decompress_true_on_uncompressed():
    s = _encode_embedding_bytes([1.0, 2.0, 3.0], compress=False)
    with pytest.raises(zlib.error):
        _decompress_and_unpack_embedding(s, decompress=True)


def test_decompress_and_unpack_embedding_raises_on_bad_length():
    bad = base64.b85encode(b"abc").decode()
    with pytest.raises(ValueError, match="multiple of 4"):
        _decompress_and_unpack_embedding(bad, decompress=False)


def test_get_expanded_embedding_vector_extracts_target_and_returns_none():
    embedding_model = "model_b"
    embedding_input = [
        {"key": "other", "value": _encode_embedding_bytes([9.9], compress=True)},
        {"key": embedding_model, "value": _encode_embedding_bytes([0.4, 0.5, 0.6], compress=True)},
    ]
    vec = get_expanded_embedding_vector(embedding_input, embedding_model)
    assert vec == pytest.approx([0.4, 0.5, 0.6], rel=1e-6, abs=1e-6)
    assert get_expanded_embedding_vector(embedding_input, "missing") is None


def test_get_padded_embedding_history_and_mask_padding_and_truncation():
    hist = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    padded, mask = get_padded_embedding_history_and_mask(hist, max_history_len=5, embed_dim=2)
    assert padded.shape == (5, 2)
    assert padded.dtype == "float32"
    for got_row, exp_row in zip(padded[:3].tolist(), hist):
        assert got_row == pytest.approx(exp_row, rel=0, abs=0)
    assert padded[3:].tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert mask.tolist() == [True, True, True, False, False]

    padded2, mask2 = get_padded_embedding_history_and_mask(hist, max_history_len=2, embed_dim=2)
    assert padded2.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert mask2.tolist() == [True, True]


def test_get_padded_embedding_history_and_mask_raises_on_embed_dim_mismatch():
    with pytest.raises(ValueError, match="embed_dim"):
        get_padded_embedding_history_and_mask([[1.0, 2.0, 3.0]], max_history_len=2, embed_dim=2)


def test_get_padded_embedding_history_and_mask_batched_accepts_single_empty_history():
    padded, mask = get_padded_embedding_history_and_mask_batched(
        [],
        max_history_len=3,
        embed_dim=2,
    )
    assert padded == [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
    assert mask == [[False, False, False]]


def test_get_padded_embedding_history_and_mask_batched_accepts_single_history():
    padded, mask = get_padded_embedding_history_and_mask_batched(
        [[1.0, 2.0], [3.0, 4.0]],
        max_history_len=3,
        embed_dim=2,
    )
    assert padded == [[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]]
    assert mask == [[True, True, False]]


def test_get_padded_embedding_history_and_mask_batched_accepts_batched_histories_and_normalizes_empty_entries():
    padded, mask = get_padded_embedding_history_and_mask_batched(
        [[], [[1.0, 2.0], [3.0, 4.0]], [[]]],
        max_history_len=3,
        embed_dim=2,
    )
    assert padded == [
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
    ]
    assert mask == [
        [False, False, False],
        [True, True, False],
        [False, False, False],
    ]


def test_get_padded_embedding_history_and_mask_batched_rejects_non_list_top_level():
    with pytest.raises(ValueError, match="history_embeddings must be a list"):
        get_padded_embedding_history_and_mask_batched(
            "not-a-list",
            max_history_len=3,
            embed_dim=2,
        )


def test_get_padded_embedding_history_and_mask_batched_rejects_top_level_non_list_entries():
    with pytest.raises(ValueError, match="history_embeddings must be a list of lists"):
        get_padded_embedding_history_and_mask_batched(
            [1.0, 2.0],
            max_history_len=3,
            embed_dim=2,
        )

