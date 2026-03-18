from __future__ import annotations

import os
import sys
from pathlib import Path

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
    inference_url = os.getenv("PREDICT_URL", "http://127.0.0.1:8000/models/user-tower/predict")

    batch_size = int(os.getenv("BATCH_SIZE", "3"))
    max_history_len = int(os.getenv("MAX_HISTORY_LEN", os.getenv("MAX_SEQ_LEN", "128")))
    embed_dim = int(os.getenv("EMBED_DIM", "384"))
    embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    rng = np.random.default_rng()

    batch_raw_history_embeddings: list[list[object]] = []
    for _ in range(batch_size):
        hist_len = int(rng.integers(low=1, high=max_history_len + 1))
        single_raw_history: list[object] = []
        for _ in range(hist_len):
            vec = (rng.random((embed_dim,)) - 0.5).astype(np.float32).tolist()
            single_raw_history.append(
                [
                    {"key": embedding_model, "value": _encode_embedding(vec)},
                    # Optional extra embedding entry to mirror real multi-model structs.
                    {"key": "other_model", "value": _encode_embedding([1.0, 2.0, 3.0])},
                ]
            )
        batch_raw_history_embeddings.append(single_raw_history)

    print(f"Batch dimension: {len(batch_raw_history_embeddings)}")
    print(f"History dimension: {len(batch_raw_history_embeddings[0])}")
    print(f"Input example: {batch_raw_history_embeddings[0][0]}")

    batch_padded_history_embeddings, batch_history_mask = get_user_tower_input_from_raw_history_embeddings(
        raw_history_embeddings=batch_raw_history_embeddings,
        embedding_model=embedding_model,
        max_history_len=max_history_len,
        embed_dim=embed_dim,
    )
    outputs = query_user_tower_with_processed_history_embeddings(
        batch_padded_history_embeddings,
        batch_history_mask,
        inference_url
    )
    print(f"Batch size: {len(outputs)}")
    print(f"Output dim: {len(outputs[0])}")
    print(f"Example output first 5: {outputs[0][:5]}")


if __name__ == "__main__":
    main()
