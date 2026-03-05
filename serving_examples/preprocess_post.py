from typing import Any, Union
import numpy as np


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):

    def preprocess(self, body: Union[bytes, dict], state: dict, collect_custom_statistics_fn=None) -> Any:
        if not (isinstance(body, dict) and "post_embeddings" in body):
            raise ValueError("Input was not a (correctly formatted) JSON dictionary!")

        post_embeddings = np.asarray(body["post_embeddings"], dtype=np.float32)

        # No batch dimension supplied for either, so add them:
        if post_embeddings.ndim == 1:
            print("No batch dimension supplied, adding batch dimension of size 1.")
            post_embeddings = post_embeddings[None, ...]

        if (post_embeddings.ndim != 2):
            raise ValueError(f"Expected post_embeddings input with shape [batch, embed_dim], but got: {post_embeddings.shape}")
        # now each input is of the form [batch, seq_len, embed_dim]

        return post_embeddings
    
    def postprocess(self, data, state, collect_custom_statistics_fn=None):
        if isinstance(data, np.ndarray):
            return {"output": data.tolist()}
        if isinstance(data, (list, tuple)):
            return {"output": [d.tolist() if isinstance(d, np.ndarray) else d for d in data]}
        return {"output": data}
