import importlib

import torch
from torch.utils.data import DataLoader, Dataset


stage_train_two_tower = importlib.import_module("utils.03_train.stage_train_two_tower")


class _RecordingTracker:
    def __init__(self) -> None:
        self.calls = []

    def log_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        self.calls.append(
            {
                "title": title,
                "series": series,
                "value": value,
                "iteration": iteration,
            }
        )


class _TinyBucketedDataset(Dataset):
    def __init__(self, embed_dim: int) -> None:
        self.batches = [
            {
                "history_embeddings": torch.tensor(
                    [
                        [[1.0, 0.5, 0.2, 0.1], [0.8, 0.2, 0.4, 0.7]],
                        [[0.2, 0.9, 0.7, 0.3], [0.3, 0.1, 0.9, 0.8]],
                    ],
                    dtype=torch.float32,
                ),
                "history_mask": torch.ones(2, 2, dtype=torch.bool),
                "candidate_post_embeddings": torch.tensor(
                    [
                        [1.0, 0.0, 0.2, 0.4],
                        [0.3, 0.8, 0.6, 0.1],
                        [0.2, 0.4, 0.9, 0.5],
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
            }
        ]
        assert self.batches[0]["history_embeddings"].shape[-1] == embed_dim

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, idx: int):
        return self.batches[idx]


def _scalar_calls_by_series(calls, series: str):
    return [call for call in calls if call["series"] == series]


def test_train_mlp_model_logs_epoch_metrics_to_tracker(tmp_path):
    stage_train_mlp = importlib.import_module("utils.03_train.stage_train_mlp")
    torch.manual_seed(0)
    embed_dim = 4
    tracker = _RecordingTracker()
    dataset = _TinyBucketedDataset(embed_dim=embed_dim)
    train_loader = DataLoader(dataset, batch_size=None, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=None, shuffle=False)
    val_unseen_loader = DataLoader(dataset, batch_size=None, shuffle=False)

    model = stage_train_mlp.MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[8],
        dropout_rate=0.0,
        user_hidden_dim=8,
        user_output_dim=embed_dim,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=4,
        attention_dropout=0.0,
        user_encoder_type="summarized",
        user_summarization="mean",
        ema_alpha=0.1,
    )

    results = stage_train_mlp.train_mlp_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_unseen_loader=val_unseen_loader,
        device="cpu",
        epochs=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=10,
        early_stopping_min_delta=0.0,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=2,
        checkpoints_dir=tmp_path,
        disable_progress=True,
        gradient_clip_max_norm=1.0,
        embed_dim=embed_dim,
        metrics_top_ks=[1, 2],
        experiment_tracker=tracker,
    )

    assert len(results["history"]["train_ndcg@1"]) == 2
    assert results["primary_metric_name"] == "ndcg@1"
    assert len(tracker.calls) == 48
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train Loss")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Loss")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Unseen Users Loss")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train ndcg@1")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation ndcg@1")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Unseen Users ndcg@1")] == [1, 2]


def test_train_two_tower_model_logs_epoch_metrics_to_tracker(tmp_path):
    torch.manual_seed(0)
    embed_dim = 4
    tracker = _RecordingTracker()
    dataset = _TinyBucketedDataset(embed_dim=embed_dim)
    train_loader = DataLoader(dataset, batch_size=None, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=None, shuffle=False)
    val_unseen_loader = DataLoader(dataset, batch_size=None, shuffle=False)

    model = stage_train_two_tower.TwoTowerModel(
        post_embedding_dim=embed_dim,
        shared_dim=embed_dim,
        user_hidden_dim=8,
        post_hidden_dim=8,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=4,
        dropout_rate=0.0,
        l2_normalize_embeddings=True,
        similarity_temperature=1.0,
        user_encoder_type="cross_attention",
        use_post_encoder=True,
    )

    results = stage_train_two_tower.train_two_tower_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_unseen_loader=val_unseen_loader,
        device="cpu",
        epochs=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        patience=10,
        early_stopping_min_delta=0.0,
        checkpoints_dir=tmp_path,
        disable_progress=True,
        lr_scheduler_factor=0.5,
        lr_scheduler_patience=2,
        gradient_clip_max_norm=1.0,
        embed_dim=embed_dim,
        metrics_top_ks=[1, 2],
        experiment_tracker=tracker,
    )

    assert len(results["history"]["train_ndcg@1"]) == 2
    assert results["primary_metric_name"] == "ndcg@1"
    assert (tmp_path / "two_tower_best.pth").exists()
    assert (tmp_path / "engagement_user_tower_best.pt").exists()
    assert (tmp_path / "engagement_post_tower_best.pt").exists()
    assert len(tracker.calls) == 84
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train Loss")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Loss")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Unseen Users Loss")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train ndcg@1")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation ndcg@1")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Unseen Users ndcg@1")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train NDCG@1")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation NDCG@1")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Unseen Users NDCG@1")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train NDCG@2")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation NDCG@2")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Unseen Users NDCG@2")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train Recall@1")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Recall@1")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Unseen Users Recall@1")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train Recall@2")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Recall@2")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Unseen Users Recall@2")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train Baseline NDCG@1")] == [1]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Baseline NDCG@1")] == [1]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Unseen Users Baseline NDCG@1")] == [1]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train Baseline Recall@1")] == [1]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Baseline Recall@1")] == [1]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Unseen Users Baseline Recall@1")] == [1]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train Zero-History NDCG@1")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Zero-History Recall@2")] == [1, 2]
    zero_history_count_calls = _scalar_calls_by_series(tracker.calls, "Validation Unseen Users Zero-History User Count")
    assert [call["iteration"] for call in zero_history_count_calls] == [1, 2]
    assert [call["value"] for call in zero_history_count_calls] == [0.0, 0.0]


def test_two_tower_logs_final_classification_metrics_by_split():
    tracker = _RecordingTracker()

    stage_train_two_tower.log_final_classification_metrics(
        tracker,
        {
            "train": {"auc_roc": 0.75, "classification_average_precision": 0.50},
            "val": {"auc_roc": None, "classification_average_precision": float("nan")},
            "holdout_unseen_users": {"auc_roc": 0.80},
        },
        iteration=3,
    )

    assert len(tracker.calls) == 3
    assert tracker.calls[0] == {
        "title": "Final AUC-ROC by Split",
        "series": "Train AUC-ROC",
        "value": 0.75,
        "iteration": 3,
    }
    assert tracker.calls[1] == {
        "title": "Final Classification Average Precision by Split",
        "series": "Train Classification Average Precision",
        "value": 0.50,
        "iteration": 3,
    }
    assert tracker.calls[2] == {
        "title": "Final AUC-ROC by Split",
        "series": "Holdout Unseen Users AUC-ROC",
        "value": 0.80,
        "iteration": 3,
    }


def test_mlp_logs_final_classification_metrics_by_split():
    stage_train_mlp = importlib.import_module("utils.03_train.stage_train_mlp")
    tracker = _RecordingTracker()

    stage_train_mlp.log_final_classification_metrics(
        tracker,
        {
            "val_unseen_users": {"auc_roc": 0.81, "classification_average_precision": 0.62},
        },
        iteration=2,
    )

    assert tracker.calls == [
        {
            "title": "Final AUC-ROC by Split",
            "series": "Val Unseen Users AUC-ROC",
            "value": 0.81,
            "iteration": 2,
        },
        {
            "title": "Final Classification Average Precision by Split",
            "series": "Val Unseen Users Classification Average Precision",
            "value": 0.62,
            "iteration": 2,
        },
    ]


def test_two_tower_stage_info_metric_lines_include_final_classification_metrics():
    lines = stage_train_two_tower.stage_info_metric_lines({
        "train": {
            "auc_roc": 0.75,
            "classification_average_precision": 0.50,
            "zero_history_rank_metric_user_count": 3,
            "zero_history_dcg@1": 0.90,
            "zero_history_ndcg@1": 0.40,
            "zero_history_recall@1": 0.30,
            "zero_history_mean_average_precision": 0.60,
        },
        "val": {"auc_roc": None, "classification_average_precision": float("nan")},
        "holdout_unseen_users": {"auc_roc": 0.80},
    })

    assert lines == [
        "train_auc_roc: 0.7500",
        "train_classification_average_precision: 0.5000",
        "train_zero_history_rank_metric_user_count: 3",
        "train_zero_history_ndcg@1: 0.4000",
        "train_zero_history_recall@1: 0.3000",
        "train_zero_history_mean_average_precision: 0.6000",
        "holdout_unseen_users_auc_roc: 0.8000",
    ]
