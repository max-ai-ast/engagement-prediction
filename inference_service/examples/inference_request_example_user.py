from __future__ import annotations

import sys
from pathlib import Path
import requests
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from shared.input_data_helpers import (
    get_user_tower_input_from_raw_history_embeddings,
    query_user_tower_with_processed_history_embeddings,
)


def _encode_embedding(vec: list[float]) -> str:
    """
    Encode a float vector into the repo's "compressed embedding struct" format:
    `base85(zlib(struct.pack('<{D}f', ...)))`.

    This matches what `shared.input_data_helpers.get_expanded_embedding_vector(...)`
    expects when decoding history embeddings.
    """
    import base64
    import struct
    import zlib

    raw = struct.pack(f"<{len(vec)}f", *vec)
    compressed = zlib.compress(raw)
    return base64.b85encode(compressed).decode()


def main() -> None:
    """
    Example: user-tower inference from *raw*, variable-length user histories.

    This generates batched raw histories where each history item contains a list of
    {"key": <embedding_model>, "value": <compressed_embedding>} dicts, then calls
    `get_user_tower_input_from_raw_history_embeddings(...)` which:
      - extracts the requested embedding model
      - pads/truncates to `MAX_HISTORY_LEN`
      - builds the boolean mask
      - POSTs to the inference endpoint

    Notes:
      - The inference service must be configured with `GE_INFERENCE_MODELS` including "user-tower".
      - Defaults assume MiniLM-style embeddings (D=384).
    """
    inference_url = "http://127.0.0.1:8080/models/user-tower/predict"

    batch_size = 3
    max_history_len = 128
    embed_dim = 384
    # embedding_model = "all-MiniLM-L12-v2"

    rng = np.random.default_rng()

    batch_history_embeddings: list[list[list[float]]] = []
    for _ in range(batch_size):
        hist_len = int(rng.integers(low=1, high=max_history_len + 1))
        single_history_embedding: list[list[float]] = []
        for _ in range(hist_len):
            single_history_embedding.append((rng.random((embed_dim,)) - 0.5).astype(np.float32).tolist())

        batch_history_embeddings.append(single_history_embedding)

    print(f"Batch dimension: {len(batch_history_embeddings)}")
    print(f"History dimension: {len(batch_history_embeddings[0])}")
    print(f"Input example: {batch_history_embeddings[0][0]}")
    
    payload = {
        "history_embeddings": batch_history_embeddings, 
    }

    # hit api
    resp = requests.post(inference_url, json=payload, timeout=30, headers={"X-API-Key": "dave-dev-key"})
    if resp.status_code != 200:
        raise ValueError(f"Request failed with status code {resp.status_code}: {resp.text}")

    try:
        outputs = resp.json()['outputs']
    except ValueError:
        raise ValueError(f"Response was not valid JSON (status code {resp.status_code}): {resp.text}")

    print(f"Batch size: {len(outputs)}")
    print(f"Output dim: {len(outputs[0])}")
    print(f"Example output first 5: {outputs[0][:5]}")


if __name__ == "__main__":
    main()
