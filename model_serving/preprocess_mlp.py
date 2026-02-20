from __future__ import annotations

from typing import Any, Union
import numpy as np


# Notice Preprocess class Must be named "Preprocess"
class Preprocess(object):

    def preprocess(self, body: Union[bytes, dict], state: dict, collect_custom_statistics_fn=None) -> Any:
        if not (isinstance(body, dict) and "user" in body and "post" in body):
            raise ValueError("Input was not a (correctly formatted) JSON dictionary!")

        user_embed = np.asarray(body["user"], dtype=np.float32)
        post_embed = np.asarray(body["post"], dtype=np.float32)

        # create a batch dimension (of size 1) if it doesn't exist
        if (user_embed.ndim == 1) and (post_embed.ndim == 1):
            user_embed = user_embed[None, ...]
            post_embed = post_embed[None, ...]
        if user_embed.shape[1] != post_embed.shape[1]:
            raise ValueError(f"User input and post input are not of the same size ({user_embed.shape[1]} vs {post_embed.shape[1]})")
        # now each input is of the form [B,D]

        # "horizontally stack to get of the form [B,2D]"
        return np.hstack([user_embed, post_embed])
    
    def postprocess(self, data, state, collect_custom_statistics_fn=None):
        if isinstance(data, np.ndarray):
            return {"output": data.tolist()}
        if isinstance(data, (list, tuple)):
            return {"output": [d.tolist() if isinstance(d, np.ndarray) else d for d in data]}
        return {"output": data}