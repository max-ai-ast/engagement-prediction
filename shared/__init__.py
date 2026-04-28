"""
Reusable helpers for model serving.

This package is intentionally lightweight (e.g. numpy-only helpers) so it can be
used in inference/serving contexts without pulling in the full training stack.
"""

from .input_data_helpers import (
    get_expanded_embedding_vector,
    get_padded_embedding_history_and_mask,
    get_padded_embedding_history_and_mask_batched,
    get_embedding_dim_for_known_model,
    classify_history_embeddings_shape,
)

__all__ = [
    "get_expanded_embedding_vector",
    "get_padded_embedding_history_and_mask",
    "get_padded_embedding_history_and_mask_batched",
    "get_embedding_dim_for_known_model",
    "classify_history_embeddings_shape",
]

