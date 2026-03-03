"""
Reusable helpers for model serving.

This package is intentionally lightweight (e.g. numpy-only helpers) so it can be
used in inference/serving contexts without pulling in the full training stack.
"""

from .input_data_helpers import (
    get_padded_embedding_history_and_mask,
    get_embedding_dim_for_known_model,
    get_embeddings_list_col_polars,
    infer_embed_dim_from_first_row_polars,
    get_user_tower_input_from_raw_history_embeddings,
)

__all__ = [
    "get_padded_embedding_history_and_mask",
    "get_embedding_dim_for_known_model",
    "get_embeddings_list_col_polars",
    "infer_embed_dim_from_first_row_polars",
    "get_user_tower_input_from_raw_history_embeddings",
]

