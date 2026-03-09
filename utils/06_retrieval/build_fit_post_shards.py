#!/usr/bin/env python3
"""
Offline FIT post-shard builder.

This script creates FIT shards from candidate posts by:
1) Encoding posts through the trained post tower.
2) Assigning each post to a FIT hard-query shard index.
3) Saving per-shard post embeddings and post IDs for online retrieval.

Usage example:
  python utils/07_retrieval/build_fit_post_shards.py \
    --model-path outputs/.../04_train/.../checkpoints/two_tower_best.pth \
    --bundle-path outputs/.../02_target_posts/.../embedding_bundle_*.pkl \
    --output-dir outputs/.../07_retrieval/fit_shards
"""

from __future__ import annotations

# argparse is used to parse CLI flags for this script.
import argparse
# importlib is used because the training module path contains numeric segments.
import importlib
# json is used for writing a machine-readable shard manifest.
import json
# pickle is used for loading the embedding bundle artifact.
import pickle
# defaultdict is used to append shard rows without repeated key checks.
from collections import defaultdict
# Path is used for robust filesystem path handling.
from pathlib import Path
# Dict and List provide type hints for clarity.
from typing import Dict, List

# numpy is used for numeric arrays and .npy persistence.
import numpy as np
# torch is used to load the model checkpoint and run tower inference.
import torch


def _load_two_tower_model(model_path: Path, device: str):
    """Load a trained TwoTowerModel checkpoint with FIT fields enabled when present."""
    # Load checkpoint from disk to the selected device.
    checkpoint = torch.load(str(model_path), map_location=device)

    # Dynamically import the training module that defines TwoTowerModel.
    stage_train_two_tower = importlib.import_module("utils.04_train.stage_train_two_tower")
    # Extract the class object from the imported module.
    TwoTowerModel = stage_train_two_tower.TwoTowerModel

    # Normalize checkpoint structure: handle raw state_dict or wrapped dict.
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    # Read config dict if checkpoint was saved with config metadata.
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

    # Infer FIT usage from weights if explicit config is missing.
    fit_in_state = any(k.startswith("mqm.") for k in state_dict.keys())

    # Recreate model using saved hyperparameters (with safe defaults as fallback).
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

    # Load trained weights into the reconstructed model.
    model.load_state_dict(state_dict)
    # Move model to the requested device.
    model.to(device)
    # Set eval mode so dropout/batchnorm are deterministic.
    model.eval()

    # Return model plus checkpoint metadata for optional diagnostics.
    return model, cfg


def _load_bundle(bundle_path: Path):
    """Load embedding bundle and return posts table metadata needed for sharding."""
    # Open the pickle bundle file in binary mode.
    with bundle_path.open("rb") as f:
        # Deserialize bundle object into memory.
        bundle = pickle.load(f)

    # Pull out posts embedding dataframe.
    posts_emb_df = bundle["posts_emb_df"]
    # Pull out post ID column name used by this run.
    join_post = bundle["join_post"]
    # Pull out base embedding dimension D (raw post embedding width).
    embedding_dim = int(bundle["embedding_dim"])

    # Return the fields needed for shard generation.
    return posts_emb_df, join_post, embedding_dim


def build_fit_shards(model_path: Path, bundle_path: Path, output_dir: Path, device: str, batch_size: int) -> None:
    """Create FIT shards and persist embeddings + post IDs per shard."""
    # Ensure output folder exists before writing shard artifacts.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load trained model checkpoint and reconstruct TwoTowerModel.
    model, cfg = _load_two_tower_model(model_path=model_path, device=device)

    # Enforce that this checkpoint supports FIT shard assignment.
    if not bool(getattr(model, "use_fit", False)):
        raise ValueError("Checkpoint does not have FIT enabled; cannot build FIT shards.")

    # Load post embeddings table from bundle.
    posts_emb_df, join_post, embedding_dim = _load_bundle(bundle_path=bundle_path)

    # Build list of raw embedding columns in the bundle dataframe.
    emb_cols = [f"post_emb_{i}" for i in range(embedding_dim)]

    # Keep post IDs grouped by shard index.
    shard_post_ids: Dict[int, List[str]] = defaultdict(list)
    # Keep encoded post vectors grouped by shard index.
    shard_post_vecs: Dict[int, List[np.ndarray]] = defaultdict(list)

    # Disable grad to reduce memory and speed up inference pass.
    with torch.no_grad():
        # Iterate rows in chunks to avoid loading all tensors at once.
        for start in range(0, len(posts_emb_df), batch_size):
            # Select current dataframe slice.
            batch_df = posts_emb_df.iloc[start : start + batch_size]
            # Collect post IDs as strings for stable serialization.
            batch_post_ids = batch_df[join_post].astype(str).tolist()
            # Read raw post embeddings as float32 numpy matrix [B, D].
            batch_raw = batch_df[emb_cols].values.astype(np.float32, copy=False)
            # Move batch to torch tensor on target device.
            batch_tensor = torch.tensor(batch_raw, dtype=torch.float32, device=device)

            # Encode posts through post tower to shared space [B, shared_dim].
            batch_post_encoded = model.encode_post(batch_tensor).cpu().numpy()

            # Compute hard FIT shard assignment for each post in the batch.
            # We use tau_min and hard=True to get deterministic shard IDs.
            _, batch_q_idx = model.mqm(
                batch_tensor,
                tau=float(getattr(model, "fit_tau_min", 0.1)),
                hard=True,
            )
            # Move shard IDs back to CPU numpy int64.
            batch_q_idx_np = batch_q_idx.detach().cpu().numpy().astype(np.int64, copy=False)

            # Store each row under its shard bucket.
            for i, q_idx in enumerate(batch_q_idx_np):
                # Convert shard index to plain Python int for dict key stability.
                shard = int(q_idx)
                # Append this post ID to its shard list.
                shard_post_ids[shard].append(batch_post_ids[i])
                # Append this encoded vector to its shard matrix list.
                shard_post_vecs[shard].append(batch_post_encoded[i])

    # Track summary stats for manifest writing.
    manifest = {
        "model_path": str(model_path.resolve()),
        "bundle_path": str(bundle_path.resolve()),
        "device": device,
        "embedding_dim": embedding_dim,
        "shared_dim": int(getattr(model, "shared_dim", -1)),
        "fit_num_queries": int(getattr(model, "fit_num_queries", -1)),
        "num_posts_total": int(sum(len(v) for v in shard_post_ids.values())),
        "num_shards_non_empty": int(len(shard_post_ids)),
        "shards": [],
        "config": cfg,
    }

    # Materialize each shard to disk as .npy files.
    for shard in sorted(shard_post_ids.keys()):
        # Convert list of vectors to dense matrix [N_shard, shared_dim].
        shard_vecs = np.vstack(shard_post_vecs[shard]).astype(np.float32, copy=False)
        # Convert post IDs list to numpy object array for easy round-trip.
        shard_ids = np.array(shard_post_ids[shard], dtype=object)

        # Build file path for encoded vectors of this shard.
        vec_path = output_dir / f"shard_{shard:03d}.embeddings.npy"
        # Build file path for post IDs aligned to vector rows.
        id_path = output_dir / f"shard_{shard:03d}.post_ids.npy"

        # Save encoded vectors to disk.
        np.save(vec_path, shard_vecs)
        # Save aligned post IDs to disk.
        np.save(id_path, shard_ids)

        # Append shard metadata to manifest.
        manifest["shards"].append(
            {
                "q_idx": int(shard),
                "num_posts": int(shard_vecs.shape[0]),
                "shared_dim": int(shard_vecs.shape[1]),
                "embeddings_path": str(vec_path.resolve()),
                "post_ids_path": str(id_path.resolve()),
            }
        )

    # Define manifest output path.
    manifest_path = output_dir / "fit_shards_manifest.json"
    # Write human-readable JSON manifest for later query-time loading.
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    # Print concise completion summary for operators.
    print(f"Built FIT shards at: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Posts: {manifest['num_posts_total']}, Shards: {manifest['num_shards_non_empty']}")


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for shard build job."""
    # Create argument parser with short description.
    p = argparse.ArgumentParser(description="Build FIT post shards from model + embedding bundle")
    # Add required path to trained two-tower checkpoint.
    p.add_argument("--model-path", type=Path, required=True, help="Path to two_tower checkpoint (.pth)")
    # Add required path to embedding bundle pickle.
    p.add_argument("--bundle-path", type=Path, required=True, help="Path to embedding_bundle_*.pkl")
    # Add required output directory for shard artifacts.
    p.add_argument("--output-dir", type=Path, required=True, help="Directory to write shard files")
    # Add optional device selector.
    p.add_argument("--device", type=str, default="cpu", help="Torch device, e.g. cpu or cuda")
    # Add optional batch size to control throughput/memory.
    p.add_argument("--batch-size", type=int, default=1024, help="Batch size for post encoding")
    # Parse and return arguments.
    return p.parse_args()


def main() -> None:
    """Entry point for CLI execution."""
    # Parse command-line inputs.
    args = _parse_args()
    # Run shard build pipeline with parsed arguments.
    build_fit_shards(
        model_path=args.model_path,
        bundle_path=args.bundle_path,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=int(args.batch_size),
    )


# Run main only when executed as a script.
if __name__ == "__main__":
    main()
