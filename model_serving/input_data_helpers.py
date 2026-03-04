from __future__ import annotations

from typing import Any, Tuple
import numpy as np

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
