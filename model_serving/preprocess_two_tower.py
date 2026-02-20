from __future__ import annotations

from typing import Any, Union
import numpy as np


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):

    def preprocess(self, body: Union[bytes, dict], state: dict, collect_custom_statistics_fn=None) -> Any:
        if not (isinstance(body, dict) and "history_embeddings" in body and "post_embeddings" in body):
            raise ValueError("Input was not a (correctly formatted) JSON dictionary!")

        history_embeddings = np.asarray(body["history_embeddings"], dtype=np.float32)
        post_embeddings = np.asarray(body["post_embeddings"], dtype=np.float32)

        # No batch dimension supplied for either, so add them:
        if (history_embeddings.ndim == 2) and (post_embeddings.ndim == 1):
            print("No batch dimension supplied, adding batch dimension of size 1.")
            history_embeddings = history_embeddings[None, ...]
            post_embeddings = post_embeddings[None, ...]

        if (history_embeddings.ndim) != 3:
            raise ValueError(f"Expected history_embeddings input with shape [batch, seq_len, embed_dim], but got: {history_embeddings.shape}")
        if (post_embeddings.ndim != 2):
            raise ValueError(f"Expected post_embeddings input with shape [batch, embed_dim], but got: {post_embeddings.shape}")
        if history_embeddings.shape[0] != post_embeddings.shape[0]:
            raise ValueError(f"Batch size of history_embeddings ({history_embeddings.shape[0]}) does not mach batch size of post_embeddings ({post_embeddings.shape[0]})!")
        # now each input is of the form [batch, seq_len, embed_dim]

        # handle history_mask
        if "history_mask" in body:
            # Use int32 (0/1) mask for maximum compatibility across serving stacks.
            history_mask = np.asarray(body["history_mask"], dtype=np.int32)
            # add batch dimension if necessary
            if history_mask.ndim == 1:
                history_mask = history_mask[None, ...]
        else:
            # in summarized mode, create dummy history mask with a one for the single summarized embedding per example
            if history_embeddings.shape[1] == 1:
                history_mask = np.ones((history_embeddings.shape[0], 1), dtype=np.int32)
            else:
                raise ValueError(f"history_embeddings have >1 length user histories and history_mask was not provided!")
        
        if history_embeddings.shape[1] != history_mask.shape[1]:
            raise ValueError(f"Sequence length of history_embeddings ({history_embeddings.shape[1]}) does not match sequence length of history_mask ({history_mask.shape[1]})!")
        if history_embeddings.shape[0] != history_mask.shape[0]:
            raise ValueError(f"Batch size of history_embeddings ({history_embeddings.shape[0]}) does not match batch size of history_mask ({history_mask.shape[0]})!")

        return [history_embeddings, history_mask, post_embeddings]
    
    def postprocess(self, data, state, collect_custom_statistics_fn=None):
        if isinstance(data, np.ndarray):
            return {"output": data.tolist()}
        if isinstance(data, (list, tuple)):
            return {"output": [d.tolist() if isinstance(d, np.ndarray) else d for d in data]}
        return {"output": data}
