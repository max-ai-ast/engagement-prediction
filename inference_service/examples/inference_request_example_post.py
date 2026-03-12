from __future__ import annotations

import os

import numpy as np
import requests


def main() -> None:
    url = os.getenv("PREDICT_URL", "http://localhost:8080/predict")

    # input dimensions
    batch_size = 3
    embed_dim = 384

    # generate inputs
    post_embeddings = (np.random.random((batch_size, embed_dim,)) - 0.5).tolist()

    payload = {"inputs": post_embeddings}

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