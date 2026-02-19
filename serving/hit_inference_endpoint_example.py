import requests
import numpy as np

def main():
    url = "http://127.0.0.1:8080/serve/mlp"
    
    # input dimensions
    batch_size = 3
    embed_dim = 384
    
    # generate inputs
    user_embed = (np.random.random((batch_size, embed_dim,)) - 0.5).tolist()
    post_embed = (np.random.random((batch_size, embed_dim,)) - 0.5).tolist()

    payload = {
        "user": user_embed,
        "post": post_embed
    }

    # hit api
    resp = requests.post(url, json=payload)
    print(f"Response status code: {resp.status_code}")
    print(resp.json())


if __name__ == "__main__":
    main()
