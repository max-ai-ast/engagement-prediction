from __future__ import annotations

from typing import Any, Union
import numpy as np


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):

    def preprocess(self, body: Union[bytes, dict], state: dict, collect_custom_statistics_fn=None) -> Any:
        if not isinstance(body, dict):
            raise ValueError("Input was not a (correctly formatted) JSON dictionary!")

        # MLPModel forward signature is:
        #   forward(history_embeddings, history_mask, post_embeddings) -> probs
        #
        # Expected payload:
        #   {
        #     "history_embeddings": [B, T, D] (or [T, D] for a single example),
        #     "history_mask": [B, T] (or [T])  (optional only when T == 1),
        #     "post_embeddings": [B, D] (or [D])
        #   }

        # --- history embeddings ---
        if "history_embeddings" not in body:
            raise ValueError("Expected 'history_embeddings' in request body.")
        history_embeddings = np.asarray(body["history_embeddings"], dtype=np.float32)
        if history_embeddings.ndim == 2:
            history_embeddings = history_embeddings[None, ...]
        if history_embeddings.ndim != 3:
            raise ValueError(
                f"Expected history_embeddings with shape [batch, seq_len, embed_dim], but got: {history_embeddings.shape}"
            )

        # --- post embeddings ---
        if "post_embeddings" not in body:
            raise ValueError("Expected 'post_embeddings' in request body.")
        post_embeddings = np.asarray(body["post_embeddings"], dtype=np.float32)
        if post_embeddings.ndim == 1:
            post_embeddings = post_embeddings[None, ...]
        if post_embeddings.ndim != 2:
            raise ValueError(
                f"Expected post_embeddings with shape [batch, embed_dim], but got: {post_embeddings.shape}"
            )

        # --- history mask ---
        if "history_mask" in body:
            # Use int32 (0/1) mask for maximum compatibility across serving stacks.
            history_mask = np.asarray(body["history_mask"], dtype=np.int32)
            if history_mask.ndim == 1:
                history_mask = history_mask[None, ...]
        else:
            # If no mask is provided, allow only the summarized convention (seq_len == 1).
            if history_embeddings.shape[1] == 1:
                history_mask = np.ones((history_embeddings.shape[0], 1), dtype=np.int32)
            else:
                raise ValueError("history_mask is required when history_embeddings has seq_len > 1.")

        if history_mask.ndim != 2:
            raise ValueError(f"Expected history_mask with shape [batch, seq_len], but got: {history_mask.shape}")

        # --- shape checks ---
        if history_embeddings.shape[0] != history_mask.shape[0]:
            raise ValueError(
                f"Batch size mismatch: history_embeddings has batch={history_embeddings.shape[0]} but history_mask has batch={history_mask.shape[0]}"
            )
        if history_embeddings.shape[1] != history_mask.shape[1]:
            raise ValueError(
                f"Sequence length mismatch: history_embeddings has seq_len={history_embeddings.shape[1]} but history_mask has seq_len={history_mask.shape[1]}"
            )
        if history_embeddings.shape[0] != post_embeddings.shape[0]:
            raise ValueError(
                f"Batch size mismatch: history_embeddings has batch={history_embeddings.shape[0]} but post_embeddings has batch={post_embeddings.shape[0]}"
            )
        if history_embeddings.shape[2] != post_embeddings.shape[1]:
            raise ValueError(
                f"Embedding dim mismatch: history_embeddings has embed_dim={history_embeddings.shape[2]} but post_embeddings has embed_dim={post_embeddings.shape[1]}"
            )

        return [history_embeddings, history_mask, post_embeddings]
    
    def postprocess(self, data, state, collect_custom_statistics_fn=None):
        if isinstance(data, np.ndarray):
            return {"output": data.tolist()}
        if isinstance(data, (list, tuple)):
            return {"output": [d.tolist() if isinstance(d, np.ndarray) else d for d in data]}
        return {"output": data}
