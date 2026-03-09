#!/usr/bin/env python3
"""
Online FIT shard query script.

This script performs query-time candidate retrieval by:
1) Loading prebuilt FIT shard embeddings from disk.
2) Encoding the requesting user history for each shard query index.
3) Scoring each shard and returning global top-k candidates.

Note:
- This script uses exact cosine search over saved shard vectors.
- For lower latency, you can replace per-shard exact search with ANN indexes.

Usage example:
  python utils/07_retrieval/query_fit_post_shards.py \
    --model-path outputs/.../04_train/.../checkpoints/two_tower_best.pth \
    --manifest-path outputs/.../07_retrieval/fit_shards/fit_shards_manifest.json \
    --history-npy /tmp/user_history.npy \
    --top-k 50
"""

from __future__ import annotations

# argparse is used to parse request/query parameters from CLI.
import argparse
# importlib is used to import training module with numeric path segments.
import importlib
# json is used for reading the shard manifest and writing output.
import json
# Path is used for filesystem-safe path operations.
from pathlib import Path
# Dict and List provide clear type hints for retrieval structures.
from typing import Dict, List

# numpy is used for vector math and top-k selection.
import numpy as np
# torch is used for model loading and user encoding.
import torch


def _load_two_tower_model(model_path: Path, device: str):
    """Load trained TwoTowerModel for user query encoding."""
    # Load checkpoint contents on the selected device.
    checkpoint = torch.load(str(model_path), map_location=device)

    # Import the module that defines TwoTowerModel.
    stage_train_two_tower = importlib.import_module("utils.04_train.stage_train_two_tower")
    # Extract the class from the imported module.
    TwoTowerModel = stage_train_two_tower.TwoTowerModel

    # Normalize checkpoint structure to state_dict.
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Read optional saved config.
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    # Detect FIT fields from weights if config is incomplete.
    fit_in_state = any(k.startswith("mqm.") for k in state_dict.keys())

    # Reconstruct model with saved hyperparameters.
    model = TwoTowerModel(
        post_embedding_dim=int(cfg.get("post_embedding_dim", checkpoint.get("post_embedding_dim", 384))),
        shared_dim=int(cfg.get("shared_dim", checkpoint.get("shared_dim", 128))),
        user_hidden_dim=int(cfg.get("user_hidden_dim", checkpoint.get("user_hidden_dim", 256))),
        post_hidden_dim=int(cfg.get("post_hidden_dim", checkpoint.get("post_hidden_dim", 256))),
        num_attention_heads=int(cfg.get("num_attention_heads", checkpoint.get("num_attention_heads", 4))),
        num_attention_layers=int(cfg.get("num_attention_layers", checkpoint.get("num_attention_layers", 2))),
        max_history_len=int(cfg.get("max_history_len", checkpoint.get("max_history_len", 50))),
        dropout_rate=float(cfg.get("dropout_rate", checkpoint.get("dropout_rate", 0.1))),
        user_encoder_type=str(cfg.get("user_encoder_type", checkpoint.get("user_encoder_type", "full_transformer"))),
        use_post_encoder=bool(cfg.get("use_post_encoder", checkpoint.get("use_post_encoder", True))),
        use_fit=bool(cfg.get("use_fit", fit_in_state)),
        fit_num_queries=int(cfg.get("fit_num_queries", checkpoint.get("fit_num_queries", 64))),
        fit_tau_init=float(cfg.get("fit_tau_init", checkpoint.get("fit_tau_init", 1.0))),
        fit_tau_min=float(cfg.get("fit_tau_min", checkpoint.get("fit_tau_min", 0.1))),
        fit_tau_decay=float(cfg.get("fit_tau_decay", checkpoint.get("fit_tau_decay", 0.9995))),
        fit_use_lss=bool(cfg.get("fit_use_lss", checkpoint.get("fit_use_lss", False))),
    )

    # Load trained weights.
    model.load_state_dict(state_dict)
    # Move model to target device.
    model.to(device)
    # Set model to eval mode.
    model.eval()

    # Return loaded model.
    return model


def _load_manifest(manifest_path: Path) -> Dict:
    """Load shard manifest JSON created by build_fit_post_shards.py."""
    # Read manifest text and decode from JSON.
    return json.loads(manifest_path.read_text())


def _encode_user_for_shard(model, history_embeddings: np.ndarray, q_idx: int, device: str) -> np.ndarray:
    """Encode user history conditioned on one FIT hard-query index."""
    # Convert history array [T, D] to torch tensor [1, T, D].
    history_tensor = torch.tensor(history_embeddings, dtype=torch.float32, device=device).unsqueeze(0)
    # Build dense validity mask [1, T] for this history sequence.
    history_mask = torch.ones(1, history_embeddings.shape[0], dtype=torch.bool, device=device)

    # Disable gradients for inference-time encoding.
    with torch.no_grad():
        # Recompute stabilized query group from MQM matrix.
        query_group = (model.mqm.meta_matrix @ model.mqm.meta_matrix.T) @ model.mqm.meta_matrix
        # Select this shard's query vector and add batch dimension.
        q_vec = query_group[int(q_idx)].unsqueeze(0)
        # Encode user with meta-query conditioning.
        user_emb = model.user_tower(history_tensor, history_mask, meta_query_vec=q_vec)

    # Return numpy array [1, shared_dim] for similarity scoring.
    return user_emb.cpu().numpy()


def _cosine_topk(query: np.ndarray, matrix: np.ndarray, k: int):
    """Return top-k cosine similarity scores/indices for one query vs matrix."""
    # Normalize query to unit length with numeric floor for stability.
    query_norm = query / np.clip(np.linalg.norm(query, axis=1, keepdims=True), a_min=1e-12, a_max=None)
    # Normalize matrix rows to unit length for cosine via dot product.
    matrix_norm = matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), a_min=1e-12, a_max=None)
    # Compute similarity scores [1, N].
    sims = query_norm @ matrix_norm.T

    # Cap k by available rows to avoid invalid indexing.
    k = min(int(k), int(matrix.shape[0]))
    # Fast partial top-k selection indices for descending similarities.
    idx_part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    # Gather selected similarity values.
    val_part = np.take_along_axis(sims, idx_part, axis=1)
    # Sort selected candidates by descending score.
    order = np.argsort(-val_part, axis=1)
    # Produce final sorted indices.
    idx = np.take_along_axis(idx_part, order, axis=1)
    # Produce final sorted scores.
    val = np.take_along_axis(val_part, order, axis=1)

    # Return indices and scores as 1D arrays for caller convenience.
    return idx[0], val[0]


def query_fit_shards(model_path: Path, manifest_path: Path, history_npy: Path, top_k: int, per_shard_k: int, device: str) -> Dict:
    """Run online retrieval over all FIT shards and return global top-k candidates."""
    # Load trained model needed for user query encoding.
    model = _load_two_tower_model(model_path=model_path, device=device)

    # Ensure checkpoint supports FIT path.
    if not bool(getattr(model, "use_fit", False)):
        raise ValueError("Checkpoint does not have FIT enabled; cannot query FIT shards.")

    # Load shard manifest generated by offline build step.
    manifest = _load_manifest(manifest_path=manifest_path)

    # Load user history embeddings array [T, D] for this request.
    history_embeddings = np.load(history_npy)

    # Ensure 2D shape for history matrix.
    if history_embeddings.ndim != 2:
        raise ValueError(f"Expected history_npy shape [T, D], got {history_embeddings.shape}")

    # Collect best score per post across shards.
    best_scores: Dict[str, float] = {}

    # Iterate each non-empty shard from manifest.
    for shard in manifest["shards"]:
        # Read shard ID used to condition user query.
        q_idx = int(shard["q_idx"])
        # Load encoded post vectors for this shard [N_shard, shared_dim].
        shard_vecs = np.load(shard["embeddings_path"])
        # Load aligned post IDs for this shard [N_shard].
        shard_post_ids = np.load(shard["post_ids_path"], allow_pickle=True)

        # Skip empty shard safely.
        if shard_vecs.shape[0] == 0:
            continue

        # Encode user specifically for this shard condition.
        user_q = _encode_user_for_shard(
            model=model,
            history_embeddings=history_embeddings,
            q_idx=q_idx,
            device=device,
        )

        # Score this shard and get local top candidates.
        local_idx, local_scores = _cosine_topk(user_q, shard_vecs, k=per_shard_k)

        # Merge local shard candidates into global best-score map.
        for i, score in zip(local_idx, local_scores):
            # Resolve post ID at this local row.
            post_id = str(shard_post_ids[int(i)])
            # Convert similarity to python float for JSON-safe storage.
            score_f = float(score)
            # Keep highest score if the post appears in multiple paths.
            prev = best_scores.get(post_id)
            if prev is None or score_f > prev:
                best_scores[post_id] = score_f

    # Sort all candidate posts by descending score.
    ranked = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
    # Truncate to requested global top-k.
    ranked = ranked[: int(top_k)]

    # Build structured output payload.
    result = {
        "top_k": int(top_k),
        "num_candidates_scored": int(len(best_scores)),
        "results": [
            {
                "rank": int(i + 1),
                "post_id": pid,
                "score": float(score),
            }
            for i, (pid, score) in enumerate(ranked)
        ],
    }

    # Return query results.
    return result


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags for query-time retrieval."""
    # Create parser for query script arguments.
    p = argparse.ArgumentParser(description="Query prebuilt FIT post shards")
    # Add checkpoint path for query encoding.
    p.add_argument("--model-path", type=Path, required=True, help="Path to two_tower checkpoint (.pth)")
    # Add manifest path for shard file discovery.
    p.add_argument("--manifest-path", type=Path, required=True, help="Path to fit_shards_manifest.json")
    # Add request user history embedding matrix path.
    p.add_argument("--history-npy", type=Path, required=True, help="Path to .npy file shaped [T, D]")
    # Add output size for final ranked list.
    p.add_argument("--top-k", type=int, default=50, help="Final number of candidates to return")
    # Add per-shard shortlist size before global merge.
    p.add_argument("--per-shard-k", type=int, default=200, help="Top candidates to keep per shard")
    # Add torch device selector.
    p.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda")
    # Add optional output JSON path.
    p.add_argument("--output-json", type=Path, default=None, help="Optional path to write JSON result")
    # Parse CLI and return namespace.
    return p.parse_args()


def main() -> None:
    """Entry point for command-line invocation."""
    # Parse input flags.
    args = _parse_args()

    # Execute retrieval query.
    result = query_fit_shards(
        model_path=args.model_path,
        manifest_path=args.manifest_path,
        history_npy=args.history_npy,
        top_k=int(args.top_k),
        per_shard_k=int(args.per_shard_k),
        device=args.device,
    )

    # Serialize result as pretty JSON text.
    payload = json.dumps(result, indent=2)

    # If output path provided, write JSON to disk.
    if args.output_json is not None:
        args.output_json.write_text(payload + "\n")

    # Always print JSON to stdout for immediate inspection.
    print(payload)


# Run main when script is executed directly.
if __name__ == "__main__":
    main()
