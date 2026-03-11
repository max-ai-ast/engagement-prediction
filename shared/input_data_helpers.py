from __future__ import annotations

from typing import Any, Tuple, Dict, List, Optional, Union
import numpy as np
import base64
import struct
import zlib

# ----------------------------------------
# Embeddings helpers
# ----------------------------------------

# Known embedding model dimensions
EMBEDDING_MODEL_DIMS: Dict[str, int] = {
    "all_MiniLM_L6_v2": 384,
    "all_MiniLM_L12_v2": 384,
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "paraphrase-MiniLM-L6-v2": 384,
    "multi-qa-MiniLM-L6-cos-v1": 384,
}


def get_embedding_dim_for_known_model(embedding_model: str) -> int:
    """
    Get the embedding dimension for a known model name.
    
    Args:
        embedding_model: Name of the embedding model
        
    Returns:
        Embedding dimension (e.g., 384 for MiniLM models)
        
    Raises:
        ValueError: If model name is not in EMBEDDING_MODEL_DIMS
    """
    if embedding_model not in EMBEDDING_MODEL_DIMS:
        known_models = ", ".join(sorted(EMBEDDING_MODEL_DIMS.keys()))
        raise ValueError(
            f"Unknown embedding model '{embedding_model}'. "
            f"Known models: {known_models}. "
            f"Add new models to EMBEDDING_MODEL_DIMS in input_data_helpers.py."
        )
    return EMBEDDING_MODEL_DIMS[embedding_model]


def _extract_compressed_embedding_vector_from_struct(embeddings: Any, embedding_model: str) -> Optional[str]:
    """
    Extract the base85-encoded embedding string for a given model from a single row's
    `embeddings` value.

    This is intentionally pure-Python (non-Polars) so it can be used inside
    `map_elements()` without relying on Polars struct/list expressions.
    """
    if embeddings is None:
        return None

    for item in embeddings:
        if item is None:
            continue

        if isinstance(item, dict):
            if item.get("key") == embedding_model:
                return item.get("value")
            continue

        if isinstance(item, (tuple, list)) and len(item) >= 2:
            if item[0] == embedding_model:
                return item[1]
            continue

        key = getattr(item, "key", None)
        if key == embedding_model:
            return getattr(item, "value", None)

    return None


def _decompress_and_unpack_embedding(s: str, decompress: Optional[bool] = None) -> list[float]:
    """
    Convert an embedding from a base85-encoded string to a list of floats.

    If `decompress` is `True`, decompress with zlib and throw an error if decompression fails.

    If `decompress` is `False`, do not decompress before unpacking.

    If `decompress` is `None`, attempt decompression and silently fallback to an uncompressed string
    if decompression fails.
    """

    bs = base64.b85decode(s.encode())

    if decompress or decompress is None:
        try:
            bs = zlib.decompress(bs)
        except zlib.error:
            if decompress:
                raise
    
    if len(bs) % 4 != 0:
        raise ValueError(f"Byte length {len(bs)} is not a multiple of 4, cannot unpack into floats")
    return list(struct.unpack(f'<{len(bs) // 4}f', bs))


def get_expanded_embedding_vector(embedding_input: Any, embedding_model: str) -> Optional[list[float]]:
    """
    Takes a single raw embeddings input, which might have embeddings from multiple models. 
    Extracts the correct compressed embedding for the given model.
    Then decompresses it and unpacks it into a list of floats.
    """
    compressed_embedding = _extract_compressed_embedding_vector_from_struct(embedding_input, embedding_model)
    if compressed_embedding is None:
        return None
    return _decompress_and_unpack_embedding(compressed_embedding, decompress=True)


# ----------------------------------------
# Input data shape helpers
# ----------------------------------------

def get_padded_embedding_history_and_mask(
    history_embeddings: Any,
    max_history_len: int, 
    embed_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad/truncate a variable-length history of embedding vectors and build a mask.

    This helper is used when a model expects a fixed-length sequence input
    (e.g. a transformer-style user-history encoder), but the available user
    history is variable length.

    Args:
        history:
            Either a 2D numpy array with shape ``[T, embed_dim]`` or a sequence
            (e.g. list) of length ``T`` containing 1D arrays/lists of length
            ``embed_dim``.
        max_history_len:
            The fixed sequence length to emit. If ``T > max_history_len``,
            the history is truncated.
        embed_dim:
            Embedding dimension (width) for each history vector.

    Returns:
        padded:
            A float32 numpy array of shape ``[max_history_len, embed_dim]``.
            Entries beyond the available history are zero-padded.
        mask:
            A boolean numpy array of shape ``[max_history_len]`` where ``True``
            indicates a real (non-padding) history position.
    """
    hist_len = len(history_embeddings)

    # validate input data 
    if hist_len > 0:
        for h in history_embeddings:
            if len(h) != embed_dim:
                raise ValueError(
                    f"History embedding length ({len(h)}) and embed_dim ({embed_dim}) do not match"
                )
            
    seq_len = min(hist_len, max_history_len)
    
    # Initialize padded array
    padded = np.zeros((max_history_len, embed_dim), dtype=np.float32)
    mask = np.zeros(max_history_len, dtype=bool)

    if seq_len > 0:
        # Truncate to max_history_len if needed, load from memmap
        padded[:seq_len] = history_embeddings[: max_history_len]
        mask[:seq_len] = True

    return padded, mask


# ----------------------------------------
# Inference helpers
# ----------------------------------------

def get_user_tower_input_from_single_raw_history_embeddings(
    raw_history_embeddings: List[Any],
    embedding_model: str,
    max_history_len: int,
    embed_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Take a list of (reverse-chron-ordered) raw embedding inputs and get them in the format needed for input into the user tower model.
    First extracts the raw compressed embedding for the given model from the dict-like input.
    Then adds padding and builds a mask to get fixed-length numpy arrays for the user tower input.
    This is the function that should be used in the inference scenario.

    Args:
        raw_history_embeddings:
            A list of raw embedding inputs (e.g. from a Polars struct/list column) containing the compressed embedding data for each history item.
            The function will extract the relevant embedding for the specified `embedding_model`.
            Note: These should be ordered *most-recent-first* (reverse chronological order) since the user tower typically attends more to recent history, and the padding will be added to the end of the sequence.
        embedding_model: 
            The name of the embedding model to extract from the raw input (e.g. "all-MiniLM-L6-v2").
        max_history_len:
            The fixed sequence length to emit. If ``T > max_history_len``,
            the history is truncated.
        embed_dim:
            Embedding dimension (width) for each history vector.

    Returns:
        padded:
            A float32 numpy array of shape ``[max_history_len, embed_dim]``.
            Entries beyond the available history are zero-padded.
        mask:
            A boolean numpy array of shape ``[max_history_len]`` where ``True``
            indicates a real (non-padding) history position.

    """
    history_embeddings = [
        vec for vec in (get_expanded_embedding_vector(rhe, embedding_model) for rhe in raw_history_embeddings)
        if vec is not None
    ]
    return get_padded_embedding_history_and_mask(history_embeddings, max_history_len=max_history_len, embed_dim=embed_dim)


def _is_embedding_struct(x: Any) -> bool:
    """
    Heuristically detect whether `x` looks like the "embeddings struct" value used by
    `_extract_compressed_embedding_vector_from_struct`.

    Expected patterns:
      - iterable of dicts like {"key": ..., "value": ...}
      - iterable of tuples/lists like (key, value) or [key, value]
      - iterable of objects with `.key` / `.value` attrs
    """
    if x is None:
        return False
    if not isinstance(x, (list, tuple)):
        return False
    if len(x) == 0:
        return False

    for item in x:
        if item is None:
            continue
        if isinstance(item, dict):
            return "key" in item and "value" in item
        if isinstance(item, (tuple, list)) and len(item) >= 2 and isinstance(item[0], str):
            return True
        if getattr(item, "key", None) is not None:
            return True
    return False


def query_user_tower_with_processed_history_embeddings(
    padded_history_embeddings: Union[List[List[Any]], List[List[List[Any]]]],
    history_mask: List[List[Any]],
    inference_url: str,
) -> List[List[float]]:
    import requests
    
    payload = {
        "history_embeddings": padded_history_embeddings, 
        "history_mask": history_mask,
    }

    # hit api
    resp = requests.post(inference_url, json=payload, timeout=30)
    if resp.status_code != 200:
        raise ValueError(f"Request failed with status code {resp.status_code}: {resp.text}")

    try:
        data = resp.json()
    except ValueError:
        raise ValueError(f"Response was not valid JSON (status code {resp.status_code}): {resp.text}")
    return data["outputs"]


def query_user_tower_with_raw_history_embeddings(
    # Single: list of embeddings-structs (one per history item).
    # Batched: list of single-history lists.
    raw_history_embeddings: Union[List[Any], List[List[Any]]],
    embedding_model: str,
    max_history_len: int,
    embed_dim: int,
    inference_url: str,
) -> List[List[float]]:
    if not isinstance(raw_history_embeddings, list) or len(raw_history_embeddings) == 0:
        raise ValueError("Invalid input: raw_history_embeddings must be a non-empty list")

    batch_padded_history_embeddings = []
    batch_history_mask = []

    first_non_none = next((x for x in raw_history_embeddings if x is not None), None)
    if first_non_none is None or _is_embedding_struct(first_non_none):
        # Single input: raw_history_embeddings is a list of embeddings structs (or all Nones).
        padded_history_embeddings, history_mask = get_user_tower_input_from_single_raw_history_embeddings(
            raw_history_embeddings=raw_history_embeddings,  # type: ignore[arg-type]
            embedding_model=embedding_model,
            max_history_len=max_history_len,
            embed_dim=embed_dim,
        )
        batch_padded_history_embeddings.append(padded_history_embeddings.tolist())
        batch_history_mask.append(history_mask.tolist())
    else:
        # Batched input: raw_history_embeddings is a list of single-user histories.
        for single_raw_history_embeddings in raw_history_embeddings:  # type: ignore[assignment]
            if single_raw_history_embeddings is None:
                padded_history_embeddings, history_mask = get_user_tower_input_from_single_raw_history_embeddings(
                    raw_history_embeddings=[],
                    embedding_model=embedding_model,
                    max_history_len=max_history_len,
                    embed_dim=embed_dim,
                )
            else:
                if not isinstance(single_raw_history_embeddings, list):
                    raise ValueError("Invalid batched input: each batch element must be a list")

                inner_first_non_none = next((x for x in single_raw_history_embeddings if x is not None), None)
                if inner_first_non_none is not None and not _is_embedding_struct(inner_first_non_none):
                    raise ValueError(
                        "Invalid batched input: expected each batch element to be a list of embeddings-structs"
                    )

                padded_history_embeddings, history_mask = get_user_tower_input_from_single_raw_history_embeddings(
                    raw_history_embeddings=single_raw_history_embeddings,
                    embedding_model=embedding_model,
                    max_history_len=max_history_len,
                    embed_dim=embed_dim,
                )

            batch_padded_history_embeddings.append(padded_history_embeddings.tolist())
            batch_history_mask.append(history_mask.tolist())
    
    return query_user_tower_with_processed_history_embeddings(
        padded_history_embeddings=batch_padded_history_embeddings,
        history_mask=batch_history_mask,
        inference_url=inference_url,
    )
