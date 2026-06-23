#!/usr/bin/env python3

"""Smoke-test BST TorchScript matrix scoring."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import torch


def load_bst_module():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    module_path = repo_root / "utils" / "03_train" / "stage_train_bst_ranker.py"
    spec = importlib.util.spec_from_file_location("stage_train_bst_ranker_smoke", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    torch.manual_seed(123)
    torch.set_printoptions(precision=5, sci_mode=False)

    bst_module = load_bst_module()
    model = bst_module.BSTRanker(
        post_embedding_dim=4,
        author_table_num_rows=8,
        author_embedding_dim=3,
        content_projection_dim=6,
        author_projection_dim=4,
        model_dim=5,
        time_embedding_dim=3,
        num_attention_heads=2,
        num_transformer_layers=1,
        transformer_ff_dim=16,
        dropout_rate=0.0,
        author_unknown_dropout_rate=0.0,
        norm_first=False,
        time_delta_bucket_boundaries_hours=[1.0, 3.0, 6.0],
        prediction_hidden_dims=[8],
    ).eval()

    torchscript_path = Path(tempfile.gettempdir()) / "bst_ranker_torchscript_smoke.pt"
    torch.jit.script(model).save(str(torchscript_path))
    loaded_model = torch.jit.load(str(torchscript_path))
    loaded_model.eval()

    history_embeddings = torch.randn(1, 3, 4)
    history_mask = torch.tensor([[True, True, False]], dtype=torch.bool)
    history_time_deltas_hours = torch.tensor([[0.5, 2.0, 0.0]], dtype=torch.float32)
    history_author_indices = torch.tensor([[2, 3, 0]], dtype=torch.long)

    candidate_post_embeddings = torch.randn(3, 4)
    candidate_post_author_idx = torch.tensor([2, 4, 1], dtype=torch.long)

    with torch.inference_mode():
        matrix_scores = loaded_model.score_candidate_matrix(
            history_embeddings,
            history_mask,
            history_time_deltas_hours,
            candidate_post_embeddings,
            history_author_indices,
            candidate_post_author_idx,
        )

        num_candidates = candidate_post_embeddings.size(0)
        forward_scores = loaded_model(
            history_embeddings.repeat_interleave(num_candidates, dim=0),
            history_mask.repeat_interleave(num_candidates, dim=0),
            history_time_deltas_hours.repeat_interleave(num_candidates, dim=0),
            candidate_post_embeddings,
            history_author_indices.repeat_interleave(num_candidates, dim=0),
            candidate_post_author_idx,
        )

    print(f"Saved TorchScript model to: {torchscript_path}")
    print("Matrix scoring output [1, C]:")
    print(matrix_scores)
    print("Regular forward output [C]:")
    print(forward_scores)
    print("Same scores:")
    print(torch.allclose(matrix_scores.squeeze(0), forward_scores, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    main()
