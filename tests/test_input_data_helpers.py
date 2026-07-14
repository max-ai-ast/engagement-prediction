import base64
import struct
import zlib

import pytest
import shared

from shared.input_data_helpers import (
    AUTHOR_PAD_IDX,
    AUTHOR_UNK_IDX,
    _decompress_and_unpack_embedding,
    _extract_compressed_embedding_vector_from_struct,
    classify_history_embeddings_shape,
    get_embedding_dim_for_known_model,
    get_expanded_embedding_vector,
    get_padded_author_indices,
    get_padded_embedding_history_and_mask,
    get_padded_embedding_history_and_mask_batched,
    get_padded_history_time_deltas,
    get_padded_prior_cumulative_likes,
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

    embeddings_mixed = [
        None,
        ["too-short"],
        ("target", "y", "extra"),
    ]
    assert _extract_compressed_embedding_vector_from_struct(embeddings_mixed, "target") == "y"


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


def test_get_padded_author_indices_padding_and_truncation():
    padded = get_padded_author_indices([2, 3], max_history_len=4)
    assert padded.tolist() == [2, 3, AUTHOR_PAD_IDX, AUTHOR_PAD_IDX]

    truncated = get_padded_author_indices([2, 3, 4], max_history_len=2)
    assert truncated.tolist() == [2, 3]


def test_get_padded_history_time_deltas_padding_and_truncation():
    padded = get_padded_history_time_deltas([1.5, 2.25], max_history_len=4)
    assert padded.tolist() == pytest.approx([1.5, 2.25, 0.0, 0.0])

    truncated = get_padded_history_time_deltas([1.0, 2.0, 3.0], max_history_len=2)
    assert truncated.tolist() == pytest.approx([1.0, 2.0])


def test_get_padded_prior_cumulative_likes_padding_and_truncation():
    padded = get_padded_prior_cumulative_likes([10, 20], max_history_len=4)
    assert padded.tolist() == [10, 20, 0, 0]

    truncated = get_padded_prior_cumulative_likes([10, 20, 30], max_history_len=2)
    assert truncated.tolist() == [10, 20]


def test_classify_history_embeddings_shape_covers_public_shapes():
    assert classify_history_embeddings_shape([]) == "single_empty"
    assert classify_history_embeddings_shape([[]]) == "single_empty"
    assert classify_history_embeddings_shape([[1.0, 2.0], [3.0, 4.0]]) == "single_history"
    assert classify_history_embeddings_shape([[], [[1.0, 2.0]]]) == "batched_history"
    assert classify_history_embeddings_shape([[[1.0, 2.0]], [[3.0, 4.0]]]) == "batched_history"


def test_classify_history_embeddings_shape_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="history_embeddings must be a list"):
        classify_history_embeddings_shape("not-a-list")

    with pytest.raises(ValueError, match="history_embeddings must be a list of lists"):
        classify_history_embeddings_shape([1.0, 2.0])


def test_get_padded_embedding_history_and_mask_batched_accepts_single_empty_history():
    padded, mask, author_indices, time_deltas, prior_likes = get_padded_embedding_history_and_mask_batched(
        [],
        max_history_len=3,
        embed_dim=2,
        author_indices=[],
    )
    assert padded == [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]
    assert mask == [[False, False, False]]
    assert author_indices == [[AUTHOR_PAD_IDX, AUTHOR_PAD_IDX, AUTHOR_PAD_IDX]]
    assert time_deltas == [[0.0, 0.0, 0.0]]
    assert prior_likes == [[0, 0, 0]]


def test_get_padded_embedding_history_and_mask_batched_accepts_single_history():
    padded, mask, author_indices, time_deltas, prior_likes = get_padded_embedding_history_and_mask_batched(
        [[1.0, 2.0], [3.0, 4.0]],
        max_history_len=3,
        embed_dim=2,
        author_indices=[2, 3],
    )
    assert padded == [[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]]
    assert mask == [[True, True, False]]
    assert author_indices == [[2, 3, AUTHOR_PAD_IDX]]
    assert time_deltas == [[0.0, 0.0, 0.0]]
    assert prior_likes == [[0, 0, 0]]


def test_get_padded_embedding_history_and_mask_batched_accepts_single_history_time_deltas_and_prior_likes():
    padded, mask, author_indices, time_deltas, prior_likes = get_padded_embedding_history_and_mask_batched(
        [[1.0, 2.0], [3.0, 4.0]],
        max_history_len=3,
        embed_dim=2,
        author_indices=[2, 3],
        time_deltas_hours=[1.5, 2.25],
        prior_cumulative_likes=[10, 20],
    )
    assert padded == [[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]]
    assert mask == [[True, True, False]]
    assert author_indices == [[2, 3, AUTHOR_PAD_IDX]]
    assert time_deltas == [[1.5, 2.25, 0.0]]
    assert prior_likes == [[10, 20, 0]]


def test_get_padded_embedding_history_and_mask_batched_defaults_single_history_author_indices_to_unknown():
    padded, mask, author_indices, time_deltas, prior_likes = get_padded_embedding_history_and_mask_batched(
        [[1.0, 2.0], [3.0, 4.0]],
        max_history_len=3,
        embed_dim=2,
        author_indices=None,
    )
    assert padded == [[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]]
    assert mask == [[True, True, False]]
    assert author_indices == [[AUTHOR_UNK_IDX, AUTHOR_UNK_IDX, AUTHOR_PAD_IDX]]
    assert time_deltas == [[0.0, 0.0, 0.0]]
    assert prior_likes == [[0, 0, 0]]


def test_get_padded_embedding_history_and_mask_batched_accepts_batched_histories_and_normalizes_empty_entries():
    padded, mask, author_indices, time_deltas, prior_likes = get_padded_embedding_history_and_mask_batched(
        [[], [[1.0, 2.0], [3.0, 4.0]], [[]]],
        max_history_len=3,
        embed_dim=2,
        author_indices=[[], [2, 3], []],
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
    assert author_indices == [
        [AUTHOR_PAD_IDX, AUTHOR_PAD_IDX, AUTHOR_PAD_IDX],
        [2, 3, AUTHOR_PAD_IDX],
        [AUTHOR_PAD_IDX, AUTHOR_PAD_IDX, AUTHOR_PAD_IDX],
    ]
    assert time_deltas == [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert prior_likes == [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]


def test_get_padded_embedding_history_and_mask_batched_accepts_batched_time_deltas_and_prior_likes():
    padded, mask, author_indices, time_deltas, prior_likes = get_padded_embedding_history_and_mask_batched(
        [[], [[1.0, 2.0], [3.0, 4.0]], [[]]],
        max_history_len=3,
        embed_dim=2,
        author_indices=[[], [2, 3], []],
        time_deltas_hours=[[], [1.5, 2.25], []],
        prior_cumulative_likes=[[], [10, 20], []],
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
    assert author_indices == [
        [AUTHOR_PAD_IDX, AUTHOR_PAD_IDX, AUTHOR_PAD_IDX],
        [2, 3, AUTHOR_PAD_IDX],
        [AUTHOR_PAD_IDX, AUTHOR_PAD_IDX, AUTHOR_PAD_IDX],
    ]
    assert time_deltas == [
        [0.0, 0.0, 0.0],
        [1.5, 2.25, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert prior_likes == [
        [0, 0, 0],
        [10, 20, 0],
        [0, 0, 0],
    ]


def test_get_padded_embedding_history_and_mask_batched_defaults_batched_author_indices_to_unknown():
    padded, mask, author_indices, time_deltas, prior_likes = get_padded_embedding_history_and_mask_batched(
        [[], [[1.0, 2.0]], [[]]],
        max_history_len=2,
        embed_dim=2,
        author_indices=None,
    )
    assert padded == [
        [[0.0, 0.0], [0.0, 0.0]],
        [[1.0, 2.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 0.0]],
    ]
    assert mask == [
        [False, False],
        [True, False],
        [False, False],
    ]
    assert author_indices == [
        [AUTHOR_PAD_IDX, AUTHOR_PAD_IDX],
        [AUTHOR_UNK_IDX, AUTHOR_PAD_IDX],
        [AUTHOR_PAD_IDX, AUTHOR_PAD_IDX],
    ]
    assert time_deltas == [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
    assert prior_likes == [
        [0, 0],
        [0, 0],
        [0, 0],
    ]


def test_get_padded_embedding_history_and_mask_batched_accepts_single_nested_empty_history():
    padded, mask, author_indices, time_deltas, prior_likes = get_padded_embedding_history_and_mask_batched(
        [[]],
        max_history_len=2,
        embed_dim=2,
        author_indices=[],
    )
    assert padded == [[[0.0, 0.0], [0.0, 0.0]]]
    assert mask == [[False, False]]
    assert author_indices == [[AUTHOR_PAD_IDX, AUTHOR_PAD_IDX]]
    assert time_deltas == [[0.0, 0.0]]
    assert prior_likes == [[0, 0]]


def test_get_padded_embedding_history_and_mask_batched_rejects_non_list_top_level():
    with pytest.raises(ValueError, match="history_embeddings must be a list"):
        get_padded_embedding_history_and_mask_batched(
            "not-a-list",
            max_history_len=3,
            embed_dim=2,
            author_indices=[],
        )


def test_get_padded_embedding_history_and_mask_batched_rejects_top_level_non_list_entries():
    with pytest.raises(ValueError, match="history_embeddings must be a list of lists"):
        get_padded_embedding_history_and_mask_batched(
            [1.0, 2.0],
            max_history_len=3,
            embed_dim=2,
            author_indices=[],
        )


def test_get_padded_embedding_history_and_mask_batched_rejects_mixed_batch_shapes():
    with pytest.raises(ValueError, match="batched history_embeddings must be a list of user histories"):
        get_padded_embedding_history_and_mask_batched(
            [[], [1.0, 2.0]],
            max_history_len=3,
            embed_dim=2,
            author_indices=[[], [2]],
        )


def test_get_padded_embedding_history_and_mask_batched_rejects_author_batch_size_mismatch():
    with pytest.raises(ValueError, match="Batch size of history_embeddings and author_indices must match"):
        get_padded_embedding_history_and_mask_batched(
            [[[1.0, 2.0]], [[3.0, 4.0]]],
            max_history_len=3,
            embed_dim=2,
            author_indices=[[2]],
        )


def test_get_padded_embedding_history_and_mask_batched_rejects_author_history_length_mismatch():
    with pytest.raises(ValueError, match="Length of author_indices must match history length"):
        get_padded_embedding_history_and_mask_batched(
            [[1.0, 2.0], [3.0, 4.0]],
            max_history_len=3,
            embed_dim=2,
            author_indices=[2],
        )


def test_get_padded_embedding_history_and_mask_batched_rejects_time_delta_batch_size_mismatch():
    with pytest.raises(ValueError, match="Batch size of history_embeddings and time_deltas_hours must match"):
        get_padded_embedding_history_and_mask_batched(
            [[[1.0, 2.0]], [[3.0, 4.0]]],
            max_history_len=3,
            embed_dim=2,
            author_indices=[[2], [3]],
            time_deltas_hours=[[1.5]],
        )


def test_get_padded_embedding_history_and_mask_batched_rejects_time_delta_history_length_mismatch():
    with pytest.raises(ValueError, match="Length of time_deltas_hours must match history length"):
        get_padded_embedding_history_and_mask_batched(
            [[1.0, 2.0], [3.0, 4.0]],
            max_history_len=3,
            embed_dim=2,
            author_indices=[2, 3],
            time_deltas_hours=[1.5],
        )


def test_get_padded_embedding_history_and_mask_batched_rejects_prior_likes_batch_size_mismatch():
    with pytest.raises(ValueError, match="Batch size of history_embeddings and prior_cumulative_likes must match"):
        get_padded_embedding_history_and_mask_batched(
            [[[1.0, 2.0]], [[3.0, 4.0]]],
            max_history_len=3,
            embed_dim=2,
            author_indices=[[2], [3]],
            prior_cumulative_likes=[[10]],
        )


def test_get_padded_embedding_history_and_mask_batched_rejects_prior_likes_history_length_mismatch():
    with pytest.raises(ValueError, match="Length of prior_cumulative_likes must match history length"):
        get_padded_embedding_history_and_mask_batched(
            [[1.0, 2.0], [3.0, 4.0]],
            max_history_len=3,
            embed_dim=2,
            author_indices=[2, 3],
            prior_cumulative_likes=[10],
        )


def test_shared_package_re_exports_public_helpers():
    assert shared.__all__ == [
        "get_expanded_embedding_vector",
        "get_padded_embedding_history_and_mask",
        "get_padded_embedding_history_and_mask_batched",
        "get_embedding_dim_for_known_model",
        "classify_history_embeddings_shape",
    ]
    assert shared.get_expanded_embedding_vector is get_expanded_embedding_vector
    assert shared.get_padded_embedding_history_and_mask is get_padded_embedding_history_and_mask
    assert (
        shared.get_padded_embedding_history_and_mask_batched
        is get_padded_embedding_history_and_mask_batched
    )
    assert shared.get_embedding_dim_for_known_model is get_embedding_dim_for_known_model
    assert shared.classify_history_embeddings_shape is classify_history_embeddings_shape
