"""Tests for the BST heavy ranker model components."""
import importlib

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


stage_train_bst_ranker = importlib.import_module("utils.03_train.stage_train_bst_ranker")
BSTRanker = stage_train_bst_ranker.BSTRanker
LinearPredictionHead = stage_train_bst_ranker.LinearPredictionHead
_compute_bst_listwise_loss_and_preds = stage_train_bst_ranker._compute_bst_listwise_loss_and_preds
run_bst_listwise_epoch = stage_train_bst_ranker.run_bst_listwise_epoch
train_bst_ranker_model = stage_train_bst_ranker.train_bst_ranker_model

DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS = [1.0, 3.0, 6.0, 12.0, 24.0, 72.0, 168.0, 720.0, 2160.0]


def _make_model(
    *,
    dropout_rate: float = 0.0,
    num_attention_heads: int = 2,
    num_transformer_layers: int = 1,
    norm_first: bool = False,
    prediction_hidden_dims=(8, 4),
) -> BSTRanker:
    torch.manual_seed(123)
    return BSTRanker(
        post_embedding_dim=4,
        author_table_num_rows=8,
        author_embedding_dim=3,
        content_projection_dim=6,
        author_projection_dim=4,
        model_dim=5,
        time_embedding_dim=3,
        num_attention_heads=num_attention_heads,
        num_transformer_layers=num_transformer_layers,
        transformer_ff_dim=16,
        dropout_rate=dropout_rate,
        author_unknown_dropout_rate=0.0,
        norm_first=norm_first,
        time_delta_bucket_boundaries_hours=DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS,
        prediction_hidden_dims=prediction_hidden_dims,
    )


def _batch() -> dict[str, torch.Tensor]:
    return {
        "history_embeddings": torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.5], [0.0, 1.0, 0.0, 0.5], [9.0, 9.0, 9.0, 9.0]],
                [[0.0, 0.0, 1.0, 0.5], [1.0, 1.0, 0.0, 0.5], [8.0, 8.0, 8.0, 8.0]],
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
        "history_time_deltas_hours": torch.tensor(
            [
                [2.0, 25.0, 999.0],
                [0.5, 777.0, 888.0],
            ],
            dtype=torch.float32,
        ),
        "candidate_post_embeddings": torch.tensor(
            [
                [0.25, 0.5, 0.75, 1.0],
                [1.0, 0.75, 0.5, 0.25],
            ],
            dtype=torch.float32,
        ),
        "history_author_indices": torch.tensor(
            [
                [2, 3, 7],
                [4, 6, 7],
            ],
            dtype=torch.long,
        ),
        "candidate_post_author_idx": torch.tensor([5, 6], dtype=torch.long),
    }


def _expected_matrix_scores(model: BSTRanker, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    num_users = batch["history_embeddings"].shape[0]
    num_candidates = batch["candidate_post_embeddings"].shape[0]
    return model(
        history_embeddings=batch["history_embeddings"].repeat_interleave(num_candidates, dim=0),
        history_mask=batch["history_mask"].repeat_interleave(num_candidates, dim=0),
        history_time_deltas_hours=batch["history_time_deltas_hours"].repeat_interleave(num_candidates, dim=0),
        candidate_post_embeddings=batch["candidate_post_embeddings"].repeat(num_users, 1),
        history_author_indices=batch["history_author_indices"].repeat_interleave(num_candidates, dim=0),
        candidate_post_author_idx=batch["candidate_post_author_idx"].repeat(num_users),
    ).reshape(num_users, num_candidates)


def _listwise_batch() -> dict[str, torch.Tensor]:
    batch = _batch()
    return {
        **batch,
        "label_matrix": torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
    }


class _SingleBatchDataset(Dataset):
    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.batch


def test_bst_ranker_forward_transformer_shape_and_builtin_transformer_encoder():
    model = _make_model()
    model.eval()
    batch = _batch()

    output = model._forward_transformer(**batch)

    assert isinstance(model.transformer_encoder, nn.TransformerEncoder)
    assert isinstance(model.prediction_head, LinearPredictionHead)
    assert output.shape == (2, model.transformer_input_dim)
    assert output.dtype == torch.float32


def test_bst_ranker_forward_returns_raw_logits():
    model = _make_model()
    model.eval()
    batch = _batch()

    logits = model(**batch)

    assert logits.shape == (2,)
    assert logits.dtype == torch.float32


@pytest.mark.parametrize("norm_first", [False, True])
def test_bst_ranker_score_candidate_matrix_one_layer_matches_repeated_path(norm_first):
    model = _make_model(norm_first=norm_first)
    model.eval()
    batch = _batch()

    with torch.inference_mode():
        expected = _expected_matrix_scores(model, batch)
        scores = model.score_candidate_matrix_one_layer(**batch)

    assert scores.shape == (2, 2)
    torch.testing.assert_close(scores, expected, atol=1e-6, rtol=1e-6)


def test_bst_ranker_score_candidate_matrix_one_layer_supports_training_gradients():
    model = _make_model()
    batch = _batch()

    scores = model.score_candidate_matrix_one_layer(**batch)
    loss = scores.square().sum()
    loss.backward()

    assert scores.shape == (2, 2)
    assert scores.requires_grad
    grad_sum = sum(
        param.grad.abs().sum()
        for param in model.parameters()
        if param.grad is not None
    )
    assert grad_sum > 0


def test_bst_ranker_score_candidate_matrix_one_layer_rejects_multi_layer_model():
    model = _make_model(num_transformer_layers=2)
    model.eval()
    batch = _batch()

    with pytest.raises(RuntimeError, match="exactly one transformer layer"):
        model.score_candidate_matrix_one_layer(**batch)


def test_bst_ranker_rejects_attention_head_mismatch():
    with pytest.raises(ValueError, match="divisible"):
        BSTRanker(
            post_embedding_dim=4,
            author_table_num_rows=8,
            author_embedding_dim=3,
            content_projection_dim=6,
            author_projection_dim=4,
            model_dim=5,
            time_embedding_dim=2,
            num_attention_heads=4,
            num_transformer_layers=1,
            transformer_ff_dim=16,
            dropout_rate=0.0,
            author_unknown_dropout_rate=0.0,
            norm_first=False,
            time_delta_bucket_boundaries_hours=DEFAULT_TIME_DELTA_BUCKET_BOUNDARIES_HOURS,
            prediction_hidden_dims=(7,),
        )


def test_bst_ranker_bucketizes_time_deltas_reserving_zero_and_clipping_tail():
    model = _make_model()
    deltas = torch.tensor([-2.0, 0.0, 0.5, 1.0, 1.1, 3.0, 2160.0, 2161.0])

    bucket_ids = model._bucketize_time_deltas_hours(deltas)

    assert bucket_ids.dtype == torch.long
    assert bucket_ids.tolist() == [0, 0, 1, 1, 2, 2, 9, 10]


def test_bst_ranker_masks_padded_history_positions():
    model = _make_model()
    model.eval()
    batch = _batch()

    output = model(**batch)
    changed_batch = {key: value.clone() for key, value in batch.items()}
    changed_batch["history_embeddings"][0, 2] = torch.tensor([1000.0, 1000.0, 1000.0, 1000.0])
    changed_batch["history_embeddings"][1, 1:] = torch.tensor(
        [[2000.0, 2000.0, 2000.0, 2000.0], [3000.0, 3000.0, 3000.0, 3000.0]]
    )
    changed_batch["history_time_deltas_hours"][0, 2] = 100000.0
    changed_batch["history_time_deltas_hours"][1, 1:] = torch.tensor([200000.0, 300000.0])
    changed_batch["history_author_indices"][0, 2] = 2
    changed_batch["history_author_indices"][1, 1:] = torch.tensor([3, 4])

    changed_output = model(**changed_batch)

    torch.testing.assert_close(changed_output, output, atol=1e-6, rtol=1e-6)


def test_bst_ranker_score_candidate_matrix_one_layer_masks_padded_history_positions():
    model = _make_model()
    model.eval()
    batch = _batch()

    output = model.score_candidate_matrix_one_layer(**batch)
    changed_batch = {key: value.clone() for key, value in batch.items()}
    changed_batch["history_embeddings"][0, 2] = torch.tensor([1000.0, 1000.0, 1000.0, 1000.0])
    changed_batch["history_embeddings"][1, 1:] = torch.tensor(
        [[2000.0, 2000.0, 2000.0, 2000.0], [3000.0, 3000.0, 3000.0, 3000.0]]
    )
    changed_batch["history_time_deltas_hours"][0, 2] = 100000.0
    changed_batch["history_time_deltas_hours"][1, 1:] = torch.tensor([200000.0, 300000.0])
    changed_batch["history_author_indices"][0, 2] = 2
    changed_batch["history_author_indices"][1, 1:] = torch.tensor([3, 4])

    changed_output = model.score_candidate_matrix_one_layer(**changed_batch)

    torch.testing.assert_close(changed_output, output, atol=1e-6, rtol=1e-6)


def test_bst_ranker_supports_candidate_only_sequence_with_zero_delta_bucket():
    model = _make_model()
    model.eval()
    history_time_deltas = torch.empty((2, 0), dtype=torch.float32)
    candidate_deltas = torch.zeros((2, 1), dtype=torch.float32)

    output = model(
        history_embeddings=torch.empty((2, 0, 4), dtype=torch.float32),
        history_mask=torch.empty((2, 0), dtype=torch.bool),
        history_time_deltas_hours=history_time_deltas,
        candidate_post_embeddings=torch.tensor(
            [
                [0.25, 0.5, 0.75, 1.0],
                [1.0, 0.75, 0.5, 0.25],
            ],
            dtype=torch.float32,
        ),
        history_author_indices=torch.empty((2, 0), dtype=torch.long),
        candidate_post_author_idx=torch.tensor([5, 6], dtype=torch.long),
    )

    assert model._bucketize_time_deltas_hours(candidate_deltas).tolist() == [[0], [0]]
    assert output.shape == (2,)


def test_bst_ranker_gradients_flow_through_post_time_transformer_and_head_parameters():
    model = _make_model()
    batch = _batch()

    output = model(**batch)
    loss = output.square().sum()
    loss.backward()

    assert model.post_feature_encoder.content_projection.weight.grad is not None
    assert model.post_feature_encoder.content_projection.weight.grad.abs().sum() > 0
    assert model.post_feature_encoder.author_projection.weight.grad is not None
    assert model.post_feature_encoder.author_projection.weight.grad.abs().sum() > 0
    assert model.post_feature_encoder.fusion_layer.weight.grad is not None
    assert model.post_feature_encoder.fusion_layer.weight.grad.abs().sum() > 0
    assert model.time_delta_embedding.weight.grad is not None
    assert model.time_delta_embedding.weight.grad.abs().sum() > 0
    transformer_grad_sum = sum(
        param.grad.abs().sum()
        for param in model.transformer_encoder.parameters()
        if param.grad is not None
    )
    assert transformer_grad_sum > 0
    prediction_head_grad_sum = sum(
        param.grad.abs().sum()
        for param in model.prediction_head.parameters()
        if param.grad is not None
    )
    assert prediction_head_grad_sum > 0


def test_bst_ranker_supports_direct_linear_prediction_head():
    model = _make_model(prediction_hidden_dims=())
    model.eval()
    batch = _batch()

    output = model(**batch)

    linear_layers = [m for m in model.prediction_head.modules() if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 1
    assert output.shape == (2,)


def test_bst_ranker_torchscript_forward_matches_eager():
    model = _make_model().eval()
    batch = _batch()

    with torch.no_grad():
        eager_output = model(**batch)
        scripted_model = torch.jit.script(model)
        scripted_output = scripted_model(
            batch["history_embeddings"],
            batch["history_mask"],
            batch["history_time_deltas_hours"],
            batch["candidate_post_embeddings"],
            batch["history_author_indices"],
            batch["candidate_post_author_idx"],
        )

    assert scripted_output.shape == eager_output.shape
    assert torch.allclose(scripted_output, eager_output, atol=1e-5)


def test_bst_ranker_torchscript_exports_matrix_scorer():
    model = _make_model().eval()
    batch = _batch()

    with torch.no_grad():
        expected = model.score_candidate_matrix_one_layer(**batch)
        scripted_model = torch.jit.script(model)
        scripted_scores = scripted_model.score_candidate_matrix(
            batch["history_embeddings"],
            batch["history_mask"],
            batch["history_time_deltas_hours"],
            batch["candidate_post_embeddings"],
            batch["history_author_indices"],
            batch["candidate_post_author_idx"],
        )

    assert scripted_scores.shape == expected.shape
    torch.testing.assert_close(scripted_scores, expected, atol=1e-5, rtol=1e-5)


def test_bst_ranker_rejects_invalid_prediction_hidden_dims():
    with pytest.raises(ValueError, match="hidden_dims"):
        _make_model(prediction_hidden_dims=[0])


def test_bst_ranker_rejects_invalid_prediction_head_output_shape():
    model = _make_model()
    model.prediction_head = nn.Linear(8, 2)
    batch = _batch()

    with pytest.raises(RuntimeError, match="prediction_head"):
        model(**batch)


def test_compute_bst_listwise_loss_and_preds_returns_finite_multi_positive_loss_and_gradients():
    model = _make_model()
    batch = _listwise_batch()

    loss, scores, labels = _compute_bst_listwise_loss_and_preds(model, batch, "cpu")
    loss.backward()

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert scores.shape == (2, 2)
    assert labels.tolist() == [[1.0, 0.0], [1.0, 1.0]]
    grad_sum = sum(
        param.grad.abs().sum()
        for param in model.parameters()
        if param.grad is not None
    )
    assert grad_sum > 0


def test_run_bst_listwise_epoch_computes_rank_metrics():
    model = _make_model()
    model.eval()
    loader = DataLoader(_SingleBatchDataset(_listwise_batch()), batch_size=None, shuffle=False)

    loss, metrics = run_bst_listwise_epoch(
        train=False,
        split_name="Validation",
        model=model,
        device="cpu",
        dataloader=loader,
        optimizer=None,
        disable_progress=True,
        gradient_clip_max_norm=1.0,
        metrics_top_ks=[1, 2],
    )

    assert loss >= 0.0
    assert metrics["loss"] == loss
    assert metrics["rank_metric_user_count"] == 2
    for metric_name in ("ndcg@1", "recall@1", "ndcg@2", "recall@2", "mean_average_precision"):
        assert metric_name in metrics
        assert 0.0 <= metrics[metric_name] <= 1.0


def test_train_bst_ranker_model_uses_val_unseen_ndcg_for_listwise_primary_metric_and_checkpoint(
    tmp_path,
    monkeypatch,
):
    torch.manual_seed(0)
    model = _make_model()
    loader = DataLoader(_SingleBatchDataset(_listwise_batch()), batch_size=None, shuffle=False)
    val_unseen_ndcg_values = [0.25, 0.75, 0.5]
    val_unseen_call_count = 0

    def fake_run_bst_listwise_epoch(**kwargs):
        nonlocal val_unseen_call_count
        split_name = kwargs["split_name"]
        if split_name == "Validation Unseen Users":
            ndcg = val_unseen_ndcg_values[val_unseen_call_count]
            val_unseen_call_count += 1
        elif split_name == "Validation":
            ndcg = 0.2
        else:
            ndcg = 0.1
        return 1.0, {
            "loss": 1.0,
            "ndcg@1": ndcg,
            "recall@1": ndcg,
            "mean_average_precision": ndcg,
            "rank_metric_user_count": 2,
        }

    monkeypatch.setattr(stage_train_bst_ranker, "run_bst_listwise_epoch", fake_run_bst_listwise_epoch)

    results = train_bst_ranker_model(
        model=model,
        train_loader=loader,
        val_loader=loader,
        val_unseen_loader=loader,
        device="cpu",
        epochs=3,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=10,
        early_stopping_min_delta=0.0,
        checkpoints_dir=tmp_path,
        disable_progress=True,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=2,
        gradient_clip_max_norm=1.0,
        metrics_top_ks=[1],
    )

    assert results["primary_metric_name"] == "val_unseen_ndcg@1"
    assert results["history"]["val_unseen_ndcg@1"] == val_unseen_ndcg_values
    assert results["best_val_metric"] == 0.75
    assert (tmp_path / "bst_ranker_best.pth").exists()
