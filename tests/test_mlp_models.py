"""Tests for matrix-native MLP ranking models."""
import importlib

import pytest
import torch
import torch.nn as nn

from utils.matrix_ranking import ranking_rows_for_batch

stage_train_mlp = importlib.import_module("utils.03_train.stage_train_mlp")
MLPModel = stage_train_mlp.MLPModel
CrossAttentionPoolingEncoder = stage_train_mlp.CrossAttentionPoolingEncoder


def _make_mlp(
    *,
    embed_dim: int = 4,
    user_encoder_type: str = "summarized",
    user_summarization: str = "mean",
    use_author_embedding_table: bool = False,
) -> MLPModel:
    return MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[8],
        dropout_rate=0.0,
        user_hidden_dim=8,
        user_output_dim=embed_dim,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=4,
        attention_dropout=0.0,
        user_encoder_type=user_encoder_type,
        user_summarization=user_summarization,
        ema_alpha=0.5,
        use_author_embedding_table=use_author_embedding_table,
        author_table_num_rows=6 if use_author_embedding_table else None,
        author_embedding_dim=3 if use_author_embedding_table else None,
        author_unknown_dropout_rate=0.0,
    )


def _matrix_batch() -> dict:
    return {
        "history_embeddings": torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], [0.0, 0.0, 0.0, 0.0]],
                [[4.0, 3.0, 2.0, 1.0], [1.0, 3.0, 5.0, 7.0], [9.0, 9.0, 9.0, 9.0]],
            ],
            dtype=torch.float32,
        ),
        "history_mask": torch.tensor(
            [
                [True, True, False],
                [True, False, False],
            ],
            dtype=torch.bool,
        ),
        "candidate_post_embeddings": torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        ),
        "label_matrix": torch.tensor(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "user_id": ["u1", "u2"],
        "bucket": "2026-05-01T00:00:00Z",
    }


def _matrix_batch_with_authors() -> dict:
    batch = _matrix_batch()
    batch["history_author_indices"] = torch.tensor(
        [
            [2, 3, 0],
            [4, 0, 0],
        ],
        dtype=torch.long,
    )
    batch["candidate_post_author_idx"] = torch.tensor([2, 3, 5], dtype=torch.long)
    return batch


def test_summarized_mlp_initialization():
    model = _make_mlp(embed_dim=384)

    assert isinstance(model.mlp_head, nn.Sequential)
    linear_layers = [m for m in model.mlp_head.modules() if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 2


def test_mlp_score_matrix_shape_and_raw_logits():
    model = _make_mlp()
    model.eval()
    batch = _matrix_batch()

    logits = model.score_matrix(
        batch["history_embeddings"],
        batch["history_mask"],
        batch["candidate_post_embeddings"],
    )

    assert logits.shape == (2, 3)
    assert logits.dtype == torch.float32


def test_mlp_compute_loss_and_preds_matrix_multi_positive_rows():
    model = _make_mlp()
    batch = _matrix_batch()

    loss, scores = model.compute_loss_and_preds(batch, device="cpu", embed_dim=4)

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert scores.shape == batch["label_matrix"].shape


def test_summarized_mean_outputs_expected_masked_history_summary():
    model = _make_mlp(user_summarization="mean")
    batch = _matrix_batch()

    summary = model.encode_user(batch["history_embeddings"], batch["history_mask"])

    assert torch.allclose(summary[0], torch.tensor([1.5, 3.0, 4.5, 6.0]))
    assert torch.allclose(summary[1], torch.tensor([4.0, 3.0, 2.0, 1.0]))


def test_summarized_ema_outputs_expected_masked_history_summary():
    model = _make_mlp(user_summarization="ema")
    batch = _matrix_batch()

    summary = model.encode_user(batch["history_embeddings"], batch["history_mask"])

    expected = (0.5 * batch["history_embeddings"][0, 0] + 0.25 * batch["history_embeddings"][0, 1]) / 0.75
    assert torch.allclose(summary[0], expected)


def test_summarized_linear_recency_outputs_expected_masked_history_summary():
    model = _make_mlp(user_summarization="linear_recency")
    batch = _matrix_batch()

    summary = model.encode_user(batch["history_embeddings"], batch["history_mask"])

    expected = (3.0 * batch["history_embeddings"][0, 0] + 2.0 * batch["history_embeddings"][0, 1]) / 5.0
    assert torch.allclose(summary[0], expected)


def test_all_positive_candidate_rows_have_valid_ap_and_undefined_auc():
    batch = _matrix_batch()
    labels = torch.ones((2, 3), dtype=torch.float32)
    scores = torch.tensor(
        [
            [0.9, 0.2, 0.1],
            [0.4, 0.5, 0.6],
        ],
        dtype=torch.float32,
    )

    rows = ranking_rows_for_batch(batch, scores, labels, [1, 2])

    assert rows[0]["average_precision"] == pytest.approx(1.0)
    assert rows[0]["auc_roc"] is None
    assert rows[0]["recall@2"] == pytest.approx(2.0 / 3.0)


def test_cross_attention_mlp_uses_learned_sequence_encoder_for_matrix_batches():
    model = _make_mlp(user_encoder_type="cross_attention")
    batch = _matrix_batch()

    assert isinstance(model.user_encoder, CrossAttentionPoolingEncoder)
    loss, scores = model.compute_loss_and_preds(batch, "cpu", embed_dim=4)

    assert torch.isfinite(loss)
    assert scores.shape == (2, 3)


def test_mlp_author_embeddings_affect_history_and_candidate_paths():
    torch.manual_seed(0)
    model = _make_mlp(use_author_embedding_table=True)
    model.eval()
    batch = _matrix_batch_with_authors()

    scores_1 = model.score_matrix(
        batch["history_embeddings"],
        batch["history_mask"],
        batch["candidate_post_embeddings"],
        batch["history_author_indices"],
        batch["candidate_post_author_idx"],
    )
    changed_batch = dict(batch)
    changed_batch["history_author_indices"] = torch.tensor(
        [
            [5, 3, 0],
            [4, 0, 0],
        ],
        dtype=torch.long,
    )
    changed_batch["candidate_post_author_idx"] = torch.tensor([2, 5, 5], dtype=torch.long)
    scores_2 = model.score_matrix(
        changed_batch["history_embeddings"],
        changed_batch["history_mask"],
        changed_batch["candidate_post_embeddings"],
        changed_batch["history_author_indices"],
        changed_batch["candidate_post_author_idx"],
    )

    assert not torch.allclose(scores_1, scores_2)


def test_mlp_author_embeddings_require_author_indices():
    model = _make_mlp(use_author_embedding_table=True)
    batch = _matrix_batch()

    with pytest.raises(RuntimeError, match="history_author_indices"):
        model.compute_loss_and_preds(batch, device="cpu", embed_dim=4)


def test_mlp_compute_loss_and_preds_with_author_embeddings():
    model = _make_mlp(use_author_embedding_table=True)
    batch = _matrix_batch_with_authors()

    loss, scores = model.compute_loss_and_preds(batch, device="cpu", embed_dim=4)

    assert torch.isfinite(loss)
    assert scores.shape == batch["label_matrix"].shape


def test_summarized_mlp_torchscript():
    model = _make_mlp()
    model.eval()
    scripted = torch.jit.script(model)
    batch = _matrix_batch()

    output = scripted.forward(
        batch["history_embeddings"],
        batch["history_mask"],
        batch["candidate_post_embeddings"],
    )

    assert output.shape == (2, 3)


def test_author_aware_mlp_torchscript():
    model = _make_mlp(use_author_embedding_table=True)
    model.eval()
    scripted = torch.jit.script(model)
    batch = _matrix_batch_with_authors()

    output = scripted.forward(
        batch["history_embeddings"],
        batch["history_mask"],
        batch["candidate_post_embeddings"],
        batch["history_author_indices"],
        batch["candidate_post_author_idx"],
    )

    assert output.shape == (2, 3)


def test_mlp_compute_loss_rejects_rows_without_positives():
    model = _make_mlp()
    batch = _matrix_batch()
    batch["label_matrix"] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    with pytest.raises(RuntimeError, match="at least one positive"):
        model.compute_loss_and_preds(batch, device="cpu", embed_dim=4)
