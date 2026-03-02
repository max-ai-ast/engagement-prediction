from __future__ import annotations

from typing import Any, Tuple, Dict, List, Optional
import numpy as np
import polars as pl
import base64
import struct
import zlib


def get_padded_vector_and_mask(
    history: Any,
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

    Notes:
        - Truncation keeps the *first* ``max_history_len`` entries in ``history``.
          If you want the most recent entries, pass ``history[-max_history_len:]``.
    """
    hist_len = len(history)

    # validate input data 
    if hist_len > 0:
        for h in history:
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
        padded[:seq_len] = history[: max_history_len]
        mask[:seq_len] = True

    return padded, mask


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


def get_embedding_dim_for_model(embedding_model: str) -> int:
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
            f"Add new models to EMBEDDING_MODEL_DIMS in helpers.py."
        )
    return EMBEDDING_MODEL_DIMS[embedding_model]


def get_embeddings_list_col(lf: pl.LazyFrame, embedding_model: str) -> pl.LazyFrame:
    emb_str = (
        pl.col("embeddings")
        .list.eval(
            pl.when(pl.element().struct.field("key") == embedding_model)
              .then(pl.element().struct.field("value"))
        )
        .list.drop_nulls()
        .list.get(0)
    )
    emb_vec = emb_str.map_elements(
        lambda s: _decompress_and_unpack_embedding(s, decompress=True) if s is not None else None,
        return_dtype=pl.List(pl.Float32),
    )
    return lf.with_columns(emb_vec.alias("_emb_vec"))


def get_embed_dim(lf: pl.LazyFrame, embedding_model: str) -> int:
    lf_with_emb = get_embeddings_list_col(lf, embedding_model)
    return (
        lf_with_emb
        .select(pl.col("_emb_vec").list.len().alias("dim"))
        .filter(pl.col("dim").is_not_null())
        .head(1)
        .collect(engine="streaming")
        .item()
    )


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

    return list(struct.unpack(f'<{int(len(bs) / 4)}f', bs))