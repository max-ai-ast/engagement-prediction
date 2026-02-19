from typing import Any, Union
import numpy as np


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):

    def preprocess(self, body: Union[bytes, dict], state: dict, collect_custom_statistics_fn=None) -> Any:
        if not (isinstance(body, dict) and "user_history" in body and "post_embed" in body):
            raise ValueError("Input was not a (correctly formatted) JSON dictionary!")

        user_history = np.asarray(body["user_history"], dtype=np.float32)
        post_embed = np.asarray(body["post_embed"], dtype=np.float32)

        # No batch dimension supplied for either, so add them:
        if (user_history.ndim == 2) and (post_embed.ndim == 1):
            print("No batch dimension supplied, adding batch dimension of size 1.")
            user_history = user_history[None, ...]
            post_embed = post_embed[None, ...]

        if (user_history.ndim) != 3:
            raise ValueError(f"Expected user_history input with shape [batch, seq_len, embed_dim], but got: {user_history.shape}")
        if (post_embed.ndim != 2):
            raise ValueError(f"Expected post_embed input with shape [batch, embed_dim], but got: {post_embed.shape}")
        if user_history.shape[2] != post_embed.shape[1]:
            raise ValueError(f"Embedding dim of user_history ({user_history.shape[2]}) does not mach embedding dim of post_embed ({post_embed.shape[1]})!")
        # now each input is of the form [B,T,D]

        return [user_history, post_embed]
    
    def postprocess(self, data, state, collect_custom_statistics_fn=None):
        if isinstance(data, np.ndarray):
            return {"output": data.tolist()}
        if isinstance(data, (list, tuple)):
            return {"output": [d.tolist() if isinstance(d, np.ndarray) else d for d in data]}
        return {"output": data}