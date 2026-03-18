from __future__ import annotations

import os

import numpy as np
import requests


def main() -> None:
    url = os.getenv("PREDICT_URL", "http://127.0.0.1:8000/models/post-tower/predict")

    # input dimensions
    batch_size = 3
    embed_dim = 384

    # generate inputs
    post_embedding = (np.random.random((batch_size, embed_dim,)) - 0.5).tolist()

    # Field name for post tower is "post_embedding".
    payload = {"post_embeddings": post_embedding}

    # hit api
    resp = requests.post(url, json=payload, timeout=30)
    print(f"Response status code: {resp.status_code}")
    if resp.status_code != 200:
        print(resp.text)
        return

    data = resp.json()
    print(len(data["outputs"]))
    print(len(data["outputs"][0]))


if __name__ == "__main__":
    main()
