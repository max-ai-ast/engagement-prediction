import importlib

import torch
from torch.utils.data import DataLoader, Dataset


stage_train_mlp = importlib.import_module("utils.04_train.stage_train_mlp")
stage_train_two_tower = importlib.import_module("utils.04_train.stage_train_two_tower")


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


class _TinySummarizedDataset(Dataset):
    def __init__(self, embed_dim: int) -> None:
        self.features = torch.tensor(
            [
                [1.0, 0.5, 0.2, 0.1, 0.9, 0.3, 0.1, 0.0],
                [0.8, 0.2, 0.4, 0.7, 0.1, 0.9, 0.3, 0.2],
                [0.2, 0.9, 0.7, 0.3, 0.4, 0.2, 0.8, 0.6],
                [0.3, 0.1, 0.9, 0.8, 0.6, 0.7, 0.2, 0.4],
            ],
            dtype=torch.float32,
        )
        self.labels = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)
        self.user_ids = ["user1", "user1", "user2", "user2"]
        assert self.features.shape[1] == 2 * embed_dim

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {
            "features": self.features[idx],
            "label": self.labels[idx],
            "user_id": self.user_ids[idx],
        }


def _scalar_calls_by_series(calls, series: str):
    return [call for call in calls if call["series"] == series]


def test_train_mlp_model_logs_epoch_metrics_to_tracker(tmp_path):
    torch.manual_seed(0)
    embed_dim = 4
    tracker = _RecordingTracker()
    dataset = _TinySummarizedDataset(embed_dim=embed_dim)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)

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
    )

    results = stage_train_mlp.train_mlp_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
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
        experiment_tracker=tracker,
    )

    assert len(results["history"]["train_auc"]) == 2
    assert len(tracker.calls) == 8
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train Loss")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Loss")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train AUC")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation AUC")] == [1, 2]


def test_train_two_tower_model_logs_epoch_metrics_to_tracker(tmp_path):
    torch.manual_seed(0)
    embed_dim = 4
    tracker = _RecordingTracker()
    dataset = _TinySummarizedDataset(embed_dim=embed_dim)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
    val_unseen_loader = DataLoader(dataset, batch_size=2, shuffle=False)

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
        user_encoder_type="summarized",
        use_post_encoder=False,
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

    assert len(results["history"]["train_auc"]) == 2
    assert len(tracker.calls) == 24
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train Loss")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Loss")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Unseen Users Loss")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train AUC")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation AUC")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Unseen Users AUC")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train NDCG@1")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation NDCG@1")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Unseen Users NDCG@1")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Train NDCG@2")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation NDCG@2")] == [1, 2]
    assert [call["iteration"] for call in _scalar_calls_by_series(tracker.calls, "Validation Unseen Users NDCG@2")] == [1, 2]
