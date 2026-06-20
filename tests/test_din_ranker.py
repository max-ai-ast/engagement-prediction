"""Tests for the DIN ranker model components."""
import importlib

import torch
from torch.utils.data import DataLoader, Dataset


stage_train_din_ranker = importlib.import_module("utils.03_train.stage_train_din_ranker")
DINRanker = stage_train_din_ranker.DINRanker
_compute_din_listwise_loss_and_preds = stage_train_din_ranker._compute_din_listwise_loss_and_preds
run_din_epoch = stage_train_din_ranker.run_din_epoch
train_din_ranker_model = stage_train_din_ranker.train_din_ranker_model


def _make_model(
    *,
    dropout_rate: float = 0.0,
    attention_hidden_dims=(8, 4),
    prediction_hidden_dims=(8, 4),
) -> DINRanker:
    torch.manual_seed(123)
    return DINRanker(
        post_embedding_dim=4,
        author_table_num_rows=8,
        author_embedding_dim=3,
        content_projection_dim=6,
        author_projection_dim=4,
        model_dim=5,
        attention_hidden_dims=attention_hidden_dims,
        prediction_hidden_dims=prediction_hidden_dims,
        dropout_rate=dropout_rate,
        author_unknown_dropout_rate=0.0,
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


def _expected_matrix_scores(model: DINRanker, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    num_users = batch["history_embeddings"].shape[0]
    num_candidates = batch["candidate_post_embeddings"].shape[0]
    return model(
        history_embeddings=batch["history_embeddings"].repeat_interleave(num_candidates, dim=0),
        history_mask=batch["history_mask"].repeat_interleave(num_candidates, dim=0),
        candidate_post_embeddings=batch["candidate_post_embeddings"].repeat(num_users, 1),
        history_author_indices=batch["history_author_indices"].repeat_interleave(num_candidates, dim=0),
        candidate_post_author_idx=batch["candidate_post_author_idx"].repeat(num_users),
    ).reshape(num_users, num_candidates)


def _listwise_batch() -> dict[str, torch.Tensor]:
    return {
        **_batch(),
        "label_matrix": torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32),
    }


class _SingleBatchDataset(Dataset):
    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.batch


def test_din_ranker_forward_returns_raw_logits():
    model = _make_model()
    model.eval()
    batch = _batch()

    logits = model(**batch)

    assert logits.shape == (2,)
    assert logits.dtype == torch.float32


def test_din_ranker_score_candidate_matrix_matches_repeated_path():
    model = _make_model()
    model.eval()
    batch = _batch()

    with torch.inference_mode():
        expected = _expected_matrix_scores(model, batch)
        scores = model.score_candidate_matrix(**batch)

    assert scores.shape == (2, 2)
    torch.testing.assert_close(scores, expected, atol=1e-6, rtol=1e-6)


def test_din_ranker_masks_padded_history_positions():
    model = _make_model()
    model.eval()
    batch = _batch()

    output = model.score_candidate_matrix(**batch)
    changed_batch = {key: value.clone() for key, value in batch.items()}
    changed_batch["history_embeddings"][0, 2] = torch.tensor([1000.0, 1000.0, 1000.0, 1000.0])
    changed_batch["history_embeddings"][1, 1:] = torch.tensor(
        [[2000.0, 2000.0, 2000.0, 2000.0], [3000.0, 3000.0, 3000.0, 3000.0]]
    )
    changed_batch["history_author_indices"][0, 2] = 2
    changed_batch["history_author_indices"][1, 1:] = torch.tensor([3, 4])

    changed_output = model.score_candidate_matrix(**changed_batch)

    torch.testing.assert_close(changed_output, output, atol=1e-6, rtol=1e-6)


def test_din_ranker_empty_history_rows_produce_finite_scores():
    model = _make_model()
    model.eval()

    scores = model.score_candidate_matrix(
        history_embeddings=torch.empty((2, 0, 4), dtype=torch.float32),
        history_mask=torch.empty((2, 0), dtype=torch.bool),
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

    assert scores.shape == (2, 2)
    assert torch.isfinite(scores).all()


def test_compute_din_listwise_loss_and_preds_returns_finite_multi_positive_loss_and_gradients():
    model = _make_model()
    batch = _listwise_batch()

    loss, scores, labels = _compute_din_listwise_loss_and_preds(model, batch, "cpu")
    loss.backward()

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert scores.shape == (2, 2)
    assert labels.tolist() == [[1.0, 0.0], [1.0, 1.0]]
    assert model.post_feature_encoder.content_projection.weight.grad is not None
    assert model.post_feature_encoder.content_projection.weight.grad.abs().sum() > 0
    attention_grad_sum = sum(
        param.grad.abs().sum()
        for param in model.attention_unit.parameters()
        if param.grad is not None
    )
    prediction_head_grad_sum = sum(
        param.grad.abs().sum()
        for param in model.prediction_head.parameters()
        if param.grad is not None
    )
    assert attention_grad_sum > 0
    assert prediction_head_grad_sum > 0


def test_run_din_epoch_computes_rank_metrics_without_classification_accumulation():
    model = _make_model()
    model.eval()
    loader = DataLoader(_SingleBatchDataset(_listwise_batch()), batch_size=None, shuffle=False)

    loss, metrics = run_din_epoch(
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
    assert "auc_roc" not in metrics
    assert "classification_average_precision" not in metrics


def test_train_din_ranker_model_uses_val_unseen_ndcg_for_primary_metric_and_checkpoint(
    tmp_path,
    monkeypatch,
):
    torch.manual_seed(0)
    model = _make_model()
    loader = DataLoader(_SingleBatchDataset(_listwise_batch()), batch_size=None, shuffle=False)
    val_unseen_ndcg_values = [0.25, 0.75, 0.5]
    val_unseen_call_count = 0

    def fake_run_din_epoch(**kwargs):
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

    monkeypatch.setattr(stage_train_din_ranker, "run_din_epoch", fake_run_din_epoch)

    results = train_din_ranker_model(
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
    assert "train_auc_roc" not in results["history"]
    assert "val_auc_roc" not in results["history"]
    assert "val_unseen_auc_roc" not in results["history"]
    assert "train_classification_average_precision" not in results["history"]
    assert "val_classification_average_precision" not in results["history"]
    assert "val_unseen_classification_average_precision" not in results["history"]
    assert results["best_val_metric"] == 0.75
    assert (tmp_path / "din_ranker_best.pth").exists()
