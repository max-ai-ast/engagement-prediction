import base64
import struct
import sys
import types
import zlib

import pytest

from shared.input_data_helpers import (
    _decompress_and_unpack_embedding,
    _extract_compressed_embedding_vector_from_struct,
    _is_embedding_struct,
    get_embedding_dim_for_known_model,
    get_expanded_embedding_vector,
    get_padded_embedding_history_and_mask,
    get_user_tower_input_from_single_raw_history_embeddings,
    query_user_tower_with_processed_history_embeddings,
    query_user_tower_with_raw_history_embeddings,
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


def test_is_embedding_struct_heuristics():
    assert _is_embedding_struct([{"key": "a", "value": "b"}]) is True
    assert _is_embedding_struct([("a", "b")]) is True
    assert _is_embedding_struct([["a", "b"]]) is True

    class _Item:
        def __init__(self):
            self.key = "a"
            self.value = "b"

    assert _is_embedding_struct([_Item()]) is True
    assert _is_embedding_struct([]) is False
    assert _is_embedding_struct(None) is False
    assert _is_embedding_struct([{"nope": 1}]) is False
    assert _is_embedding_struct(["a"]) is False


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

    padded, mask = get_user_tower_input_from_single_raw_history_embeddings(
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

    padded, mask = get_user_tower_input_from_single_raw_history_embeddings(
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
        get_user_tower_input_from_single_raw_history_embeddings(
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

    padded, mask = get_user_tower_input_from_single_raw_history_embeddings(
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
        get_user_tower_input_from_single_raw_history_embeddings(
            raw_history_embeddings=raw_history_embeddings,
            embedding_model=embedding_model,
            max_history_len=3,
            embed_dim=3,
        )


def test_query_user_tower_with_processed_history_embeddings_success(monkeypatch):
    inference_url = "http://example.test/predict"
    padded = [[[0.0, 0.0], [1.0, 1.0]]]
    mask = [[True, False]]

    class _Resp:
        status_code = 200
        text = "ok"

        def json(self):
            return {"outputs": [[0.1, 0.2]]}

    def _post(url, json, timeout):
        assert url == inference_url
        assert json == {"history_embeddings": padded, "history_mask": mask}
        assert timeout == 30
        return _Resp()

    requests_mod = types.ModuleType("requests")
    requests_mod.post = _post  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", requests_mod)

    out = query_user_tower_with_processed_history_embeddings(padded, mask, inference_url)
    assert out == [[0.1, 0.2]]


def test_query_user_tower_with_processed_history_embeddings_non_200(monkeypatch):
    inference_url = "http://example.test/predict"

    class _Resp:
        status_code = 500
        text = "boom"

        def json(self):
            return {"outputs": []}

    requests_mod = types.ModuleType("requests")
    requests_mod.post = lambda *_args, **_kwargs: _Resp()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", requests_mod)

    with pytest.raises(ValueError, match="status code 500"):
        query_user_tower_with_processed_history_embeddings([[[0.0]]], [[True]], inference_url)


def test_query_user_tower_with_processed_history_embeddings_invalid_json(monkeypatch):
    inference_url = "http://example.test/predict"

    class _Resp:
        status_code = 200
        text = "<html>not json</html>"

        def json(self):
            raise ValueError("no json")

    requests_mod = types.ModuleType("requests")
    requests_mod.post = lambda *_args, **_kwargs: _Resp()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", requests_mod)

    with pytest.raises(ValueError, match="not valid JSON"):
        query_user_tower_with_processed_history_embeddings([[[0.0]]], [[True]], inference_url)


def test_query_user_tower_with_raw_history_embeddings_single_and_batched(monkeypatch):
    import shared.input_data_helpers as helpers

    embedding_model = "model_b"
    max_history_len = 3
    embed_dim = 3
    inference_url = "http://example.test/predict"

    raw_single = [
        [{"key": embedding_model, "value": _encode_embedding([0.1, 0.2, 0.3])}],
        [{"key": embedding_model, "value": _encode_embedding([0.4, 0.5, 0.6])}],
    ]

    captured = {}

    def _stub_processed(padded_history_embeddings, history_mask, inference_url):
        captured["padded"] = padded_history_embeddings
        captured["mask"] = history_mask
        captured["url"] = inference_url
        # return [B, output_dim]
        return [[1.0, 2.0] for _ in padded_history_embeddings]

    monkeypatch.setattr(helpers, "query_user_tower_with_processed_history_embeddings", _stub_processed)

    out_single = query_user_tower_with_raw_history_embeddings(
        raw_single, embedding_model, max_history_len, embed_dim, inference_url
    )
    assert out_single == [[1.0, 2.0]]
    assert captured["url"] == inference_url
    assert captured["mask"] == [[True, True, False]]
    assert captured["padded"][0][0] == pytest.approx([0.1, 0.2, 0.3], rel=1e-6, abs=1e-6)
    assert captured["padded"][0][1] == pytest.approx([0.4, 0.5, 0.6], rel=1e-6, abs=1e-6)

    raw_batched = [
        raw_single,
        None,  # should turn into all-zero padded history and all-false mask
    ]
    out_batched = query_user_tower_with_raw_history_embeddings(
        raw_batched, embedding_model, max_history_len, embed_dim, inference_url
    )
    assert out_batched == [[1.0, 2.0], [1.0, 2.0]]
    assert captured["mask"][1] == [False, False, False]
    assert captured["padded"][1] == [[0.0, 0.0, 0.0]] * max_history_len


def test_query_user_tower_with_raw_history_embeddings_rejects_invalid_batched_shape():
    with pytest.raises(ValueError, match="Invalid batched input"):
        query_user_tower_with_raw_history_embeddings(
            [[123]], embedding_model="model_b", max_history_len=3, embed_dim=3, inference_url="http://x"
        )


def test_query_user_tower_with_raw_history_embeddings_rejects_empty():
    with pytest.raises(ValueError, match="non-empty list"):
        query_user_tower_with_raw_history_embeddings(
            [], embedding_model="model_b", max_history_len=3, embed_dim=3, inference_url="http://x"
        )
