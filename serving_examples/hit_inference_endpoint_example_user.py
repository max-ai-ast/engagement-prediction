from __future__ import annotations

import requests
import numpy as np

def main():
    url = "http://127.0.0.1:8080/serve/user"
    
    # input dimensions
    batch_size = 3
    seq_len = 20
    embed_dim = 384
    
    rng = np.random.default_rng(42)

    # generate inputs
    history_embeddings = (rng.random((batch_size, seq_len, embed_dim)) - 0.5).tolist()
    history_mask = np.ones((batch_size, seq_len)).tolist()

    payload = {
        "history_embeddings": history_embeddings,
        "history_mask": history_mask,
    }

    # hit api
    resp = requests.post(url, json=payload)
    print(f"Response status code: {resp.status_code}")
    print(resp.json())


if __name__ == "__main__":
    main()
