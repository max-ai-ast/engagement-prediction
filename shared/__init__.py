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
    get_user_tower_input_from_single_raw_history_embeddings,
    query_user_tower_with_processed_history_embeddings,
    get_user_tower_input_from_raw_history_embeddings
)

__all__ = [
    "get_expanded_embedding_vector",
    "get_padded_embedding_history_and_mask",
    "get_padded_embedding_history_and_mask_batched",
    "get_embedding_dim_for_known_model",
    "get_user_tower_input_from_single_raw_history_embeddings",
    "query_user_tower_with_processed_history_embeddings",
    "get_user_tower_input_from_raw_history_embeddings",
]

