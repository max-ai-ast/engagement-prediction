import requests
import numpy as np

def main():
    url = "http://127.0.0.1:8080/serve/mlp"
    
    # input dimensions
    batch_size = 3
    embed_dim = 384
    
    rng = np.random.default_rng(42)

    # generate inputs
    user_summary = (rng.random((batch_size, embed_dim)) - 0.5).tolist()
    post_embeddings = (rng.random((batch_size, embed_dim)) - 0.5).tolist()

    payload = {
        # summarized mode convention: pass [B, 1, D] history with summary at position 0
        "history_embeddings": [[u] for u in user_summary],
        # "history_mask" is optional when seq_len == 1
        "post_embeddings": post_embeddings,
    }

    # hit api
    resp = requests.post(url, json=payload)
    print(f"Response status code: {resp.status_code}")
    print(resp.json())


if __name__ == "__main__":
    main()
