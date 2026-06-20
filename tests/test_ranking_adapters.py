import importlib
import json

import pytest
import torch

from utils.ranking_adapters import BstPthAdapter, DinPthAdapter, TwoTowerPthAdapter

stage_train_two_tower = importlib.import_module("utils.03_train.stage_train_two_tower")
stage_train_bst_ranker = importlib.import_module("utils.03_train.stage_train_bst_ranker")
stage_train_din_ranker = importlib.import_module("utils.03_train.stage_train_din_ranker")


def _two_tower_config():
    return {
        "model_type": "two_tower",
        "user_encoder_type": "cross_attention",
        "use_post_encoder": False,
        "post_embedding_dim": 4,
        "shared_dim": 4,
        "user_hidden_dim": 4,
        "post_hidden_dim": 5,
        "num_attention_heads": 2,
        "num_attention_layers": 1,
        "max_history_len": 3,
        "dropout_rate": 0.0,
        "l2_normalize_embeddings": False,
        "similarity_temperature": 1.0,
        "use_author_embedding_table": False,
        "author_embedding_dim": None,
        "author_unknown_dropout_rate": None,
        "author_table_num_rows": None,
    }


def _make_two_tower_model(config):
    return stage_train_two_tower.TwoTowerModel(
        post_embedding_dim=config["post_embedding_dim"],
        shared_dim=config["shared_dim"],
        user_hidden_dim=config["user_hidden_dim"],
        post_hidden_dim=config["post_hidden_dim"],
        num_attention_heads=config["num_attention_heads"],
        num_attention_layers=config["num_attention_layers"],
        max_history_len=config["max_history_len"],
        dropout_rate=config["dropout_rate"],
        l2_normalize_embeddings=config["l2_normalize_embeddings"],
        similarity_temperature=config["similarity_temperature"],
        user_encoder_type=config["user_encoder_type"],
        use_post_encoder=config["use_post_encoder"],
        use_author_embedding_table=config["use_author_embedding_table"],
        author_table_num_rows=config["author_table_num_rows"],
        author_embedding_dim=config["author_embedding_dim"],
        author_unknown_dropout_rate=config["author_unknown_dropout_rate"] or 0.0,
    )


def _bst_config():
    return {
        "model_type": "bst-ranker",
        "post_embedding_dim": 4,
        "model_dim": 4,
        "content_projection_dim": 5,
        "author_projection_dim": 3,
        "time_embedding_dim": 2,
        "num_attention_heads": 2,
        "num_transformer_layers": 1,
        "transformer_ff_dim": 8,
        "dropout_rate": 0.0,
        "norm_first": False,
        "time_delta_bucket_boundaries_hours": [1.0, 3.0],
        "prediction_hidden_dims": [],
        "max_history_len": 3,
        "use_author_embedding_table": True,
        "author_embedding_dim": 2,
        "author_unknown_dropout_rate": 0.0,
        "author_table_num_rows": 6,
    }


def _make_bst_model(config):
    return stage_train_bst_ranker.BSTRanker(
        post_embedding_dim=config["post_embedding_dim"],
        author_table_num_rows=config["author_table_num_rows"],
        author_embedding_dim=config["author_embedding_dim"],
        content_projection_dim=config["content_projection_dim"],
        author_projection_dim=config["author_projection_dim"],
        model_dim=config["model_dim"],
        time_embedding_dim=config["time_embedding_dim"],
        num_attention_heads=config["num_attention_heads"],
        num_transformer_layers=config["num_transformer_layers"],
        transformer_ff_dim=config["transformer_ff_dim"],
        dropout_rate=config["dropout_rate"],
        author_unknown_dropout_rate=config["author_unknown_dropout_rate"],
        norm_first=config["norm_first"],
        time_delta_bucket_boundaries_hours=config["time_delta_bucket_boundaries_hours"],
        prediction_hidden_dims=config["prediction_hidden_dims"],
    )


def _din_config():
    return {
        "model_type": "din-ranker",
        "post_embedding_dim": 4,
        "model_dim": 4,
        "content_projection_dim": 5,
        "author_projection_dim": 3,
        "attention_hidden_dims": [6],
        "prediction_hidden_dims": [],
        "dropout_rate": 0.0,
        "max_history_len": 3,
        "use_author_embedding_table": True,
        "author_embedding_dim": 2,
        "author_unknown_dropout_rate": 0.0,
        "author_table_num_rows": 6,
    }


def _make_din_model(config):
    return stage_train_din_ranker.DINRanker(
        post_embedding_dim=config["post_embedding_dim"],
        author_table_num_rows=config["author_table_num_rows"],
        author_embedding_dim=config["author_embedding_dim"],
        content_projection_dim=config["content_projection_dim"],
        author_projection_dim=config["author_projection_dim"],
        model_dim=config["model_dim"],
        attention_hidden_dims=config["attention_hidden_dims"],
        prediction_hidden_dims=config["prediction_hidden_dims"],
        dropout_rate=config["dropout_rate"],
        author_unknown_dropout_rate=config["author_unknown_dropout_rate"],
    )


def test_two_tower_pth_adapter_scores_bucketed_batch(tmp_path):
    torch.manual_seed(11)
    config = _two_tower_config()
    model = _make_two_tower_model(config)
    model.eval()
    checkpoint_path = tmp_path / "two_tower.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_path)
    batch = {
        "history_embeddings": torch.randn(2, 3, 4),
        "history_mask": torch.tensor([[True, True, False], [True, False, False]]),
        "candidate_post_embeddings": torch.randn(3, 4),
        "label_matrix": torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    }

    adapter = TwoTowerPthAdapter(checkpoint_path)
    adapter.prepare_for_eval("cpu")
    scores = adapter.score_batch(batch, "cpu").scores

    with torch.inference_mode():
        expected = model(
            batch["history_embeddings"],
            batch["history_mask"],
            batch["candidate_post_embeddings"],
        )
    torch.testing.assert_close(scores, expected)


def test_two_tower_pth_adapter_loads_config_from_stage_training_config(tmp_path):
    torch.manual_seed(12)
    config = _two_tower_config()
    model = _make_two_tower_model(config)
    checkpoint_dir = tmp_path / "stage" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "two_tower_best.pth"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    (tmp_path / "stage" / "training_config.json").write_text(json.dumps(config) + "\n")

    adapter = TwoTowerPthAdapter(checkpoint_path)
    adapter.prepare_for_eval("cpu")

    assert adapter.config == config


def test_bst_pth_adapter_scores_bucketed_batch_in_candidate_chunks(tmp_path):
    torch.manual_seed(13)
    config = _bst_config()
    model = _make_bst_model(config)
    model.eval()
    checkpoint_path = tmp_path / "bst_ranker.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_path)
    batch = {
        "history_embeddings": torch.randn(2, 3, 4),
        "history_mask": torch.tensor([[True, True, False], [True, False, False]]),
        "history_time_deltas_hours": torch.tensor([[0.5, 2.0, 0.0], [4.0, 0.0, 0.0]]),
        "candidate_post_embeddings": torch.randn(3, 4),
        "history_author_indices": torch.tensor([[2, 3, 0], [4, 0, 0]]),
        "candidate_post_author_idx": torch.tensor([2, 3, 5]),
        "label_matrix": torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    }

    adapter = BstPthAdapter(checkpoint_path, candidate_chunk_size=2)
    adapter.prepare_for_eval("cpu")
    scores = adapter.score_batch(batch, "cpu").scores

    num_users = batch["history_embeddings"].shape[0]
    num_candidates = batch["candidate_post_embeddings"].shape[0]
    with torch.inference_mode():
        expected = model(
            history_embeddings=batch["history_embeddings"].repeat_interleave(num_candidates, dim=0),
            history_mask=batch["history_mask"].repeat_interleave(num_candidates, dim=0),
            history_time_deltas_hours=batch["history_time_deltas_hours"].repeat_interleave(num_candidates, dim=0),
            candidate_post_embeddings=batch["candidate_post_embeddings"].repeat(num_users, 1),
            history_author_indices=batch["history_author_indices"].repeat_interleave(num_candidates, dim=0),
            candidate_post_author_idx=batch["candidate_post_author_idx"].repeat(num_users),
        ).reshape(num_users, num_candidates)
    torch.testing.assert_close(scores, expected)


def test_bst_pth_adapter_requires_bucketed_bst_fields(tmp_path):
    torch.manual_seed(14)
    config = _bst_config()
    model = _make_bst_model(config)
    checkpoint_path = tmp_path / "bst_ranker.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_path)
    adapter = BstPthAdapter(checkpoint_path, candidate_chunk_size=2)

    with pytest.raises(RuntimeError, match="history_time_deltas_hours"):
        adapter.score_batch({
            "history_embeddings": torch.randn(1, 3, 4),
            "history_mask": torch.ones((1, 3), dtype=torch.bool),
            "candidate_post_embeddings": torch.randn(2, 4),
            "history_author_indices": torch.ones((1, 3), dtype=torch.long),
            "candidate_post_author_idx": torch.ones(2, dtype=torch.long),
        }, "cpu")


def test_din_pth_adapter_scores_bucketed_batch_in_candidate_chunks(tmp_path):
    torch.manual_seed(15)
    config = _din_config()
    model = _make_din_model(config)
    model.eval()
    checkpoint_path = tmp_path / "din_ranker.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_path)
    batch = {
        "history_embeddings": torch.randn(2, 3, 4),
        "history_mask": torch.tensor([[True, True, False], [True, False, False]]),
        "history_time_deltas_hours": torch.tensor([[0.5, 2.0, 0.0], [4.0, 0.0, 0.0]]),
        "candidate_post_embeddings": torch.randn(3, 4),
        "history_author_indices": torch.tensor([[2, 3, 0], [4, 0, 0]]),
        "candidate_post_author_idx": torch.tensor([2, 3, 5]),
        "label_matrix": torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
    }

    adapter = DinPthAdapter(checkpoint_path, candidate_chunk_size=2)
    adapter.prepare_for_eval("cpu")
    scores = adapter.score_batch(batch, "cpu").scores

    with torch.inference_mode():
        expected = model.score_candidate_matrix(
            history_embeddings=batch["history_embeddings"],
            history_mask=batch["history_mask"],
            candidate_post_embeddings=batch["candidate_post_embeddings"],
            history_author_indices=batch["history_author_indices"],
            candidate_post_author_idx=batch["candidate_post_author_idx"],
        )
    torch.testing.assert_close(scores, expected)


def test_din_pth_adapter_requires_bucketed_din_fields(tmp_path):
    torch.manual_seed(16)
    config = _din_config()
    model = _make_din_model(config)
    checkpoint_path = tmp_path / "din_ranker.pth"
    torch.save({"model_state_dict": model.state_dict(), "config": config}, checkpoint_path)
    adapter = DinPthAdapter(checkpoint_path, candidate_chunk_size=2)

    with pytest.raises(RuntimeError, match="candidate_post_author_idx"):
        adapter.score_batch({
            "history_embeddings": torch.randn(1, 3, 4),
            "history_mask": torch.ones((1, 3), dtype=torch.bool),
            "candidate_post_embeddings": torch.randn(2, 4),
            "history_author_indices": torch.ones((1, 3), dtype=torch.long),
        }, "cpu")
