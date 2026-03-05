from __future__ import annotations

import requests
import numpy as np

def main():
    url = "http://127.0.0.1:8080/serve/post"
    
    # input dimensions
    batch_size = 3
    embed_dim = 384
    
    rng = np.random.default_rng(42)

    # generate inputs
    post_embeddings = (rng.random((batch_size, embed_dim)) - 0.5).tolist()

    payload = {
        "post_embeddings": post_embeddings
    }

    # hit api
    resp = requests.post(url, json=payload)
    print(f"Response status code: {resp.status_code}")
    print(resp.json())


if __name__ == "__main__":
    main()
