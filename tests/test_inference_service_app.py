import os

import pytest

from fastapi.testclient import TestClient
from pydantic import ValidationError
import torch

_prev_max_history_len = os.environ.get("GE_INFERENCE_MAX_HISTORY_LEN")
os.environ["GE_INFERENCE_MAX_HISTORY_LEN"] = "2"
try:
    import inference_service.app as app_module
finally:
    if _prev_max_history_len is None:
        os.environ.pop("GE_INFERENCE_MAX_HISTORY_LEN", None)
    else:
        os.environ["GE_INFERENCE_MAX_HISTORY_LEN"] = _prev_max_history_len


def _install_dummy_models(monkeypatch: pytest.MonkeyPatch) -> None:
    def user_model(history_embeddings: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        # history_embeddings: [B, T, D] (float)
        # history_mask: [B, T] (bool/int)
        _ = history_mask
        return history_embeddings.sum(dim=(1, 2), keepdim=True)

    def post_model(post_embeddings: torch.Tensor) -> torch.Tensor:
        # post_embeddings: [B, D]
        return post_embeddings.mean(dim=1, keepdim=True)

    user_entry = app_module.LoadedModel(model_type="user-tower", signature="history")
    user_entry.module = user_model
    user_entry.device = torch.device("cpu")

    post_entry = app_module.LoadedModel(model_type="post-tower", signature="vector")
    post_entry.module = post_model
    post_entry.device = torch.device("cpu")

    monkeypatch.setattr(app_module, "_models_initialized", True)
    monkeypatch.setattr(app_module, "_models_init_error", None)
    monkeypatch.setattr(app_module, "_models", {"user-tower": user_entry, "post-tower": post_entry})


def test_user_tower_request_accepts_unbatched_and_batched():
    req1 = app_module.UserTowerPredictRequest(
        history_embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        history_mask=[1, 0],
    )
    assert isinstance(req1, app_module.UserTowerPredictRequest)

    req2 = app_module.UserTowerPredictRequest(
        history_embeddings=[
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ],
        history_mask=[[1, 0], [0, 1]],
    )
    assert isinstance(req2, app_module.UserTowerPredictRequest)


def test_user_tower_request_rejects_ragged_history_embeddings():
    with pytest.raises(ValidationError, match="rectangular"):
        app_module.UserTowerPredictRequest(
            history_embeddings=[[1.0, 2.0], [3.0]],
            history_mask=[1, 1],
        )


def test_user_tower_request_rejects_mask_shape_mismatch():
    with pytest.raises(ValidationError, match="history_mask must have shape"):
        app_module.UserTowerPredictRequest(
            history_embeddings=[
                [[1.0, 2.0], [3.0, 4.0]],
                [[5.0, 6.0], [7.0, 8.0]],
            ],  # [B=2,T=2,D=2]
            history_mask=[1, 0],  # [T] not allowed when B>1
        )

    with pytest.raises(ValidationError, match="history_mask must have shape"):
        app_module.UserTowerPredictRequest(
            history_embeddings=[[1.0, 2.0], [3.0, 4.0]],  # [T=2,D=2]
            history_mask=[[1, 0], [1, 0]],  # [B=2,T] mismatches implicit B=1
        )


def test_user_tower_request_enforces_embed_dim_and_seq_len_and_max_batch(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app_module, "GE_INFERENCE_EMBED_DIM", 3)
    monkeypatch.setattr(app_module, "GE_INFERENCE_MAX_HISTORY_LEN", 2)
    monkeypatch.setattr(app_module, "GE_INFERENCE_MAX_BATCH", 1)

    # ok: [T=2,D=3]
    app_module.UserTowerPredictRequest(
        history_embeddings=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        history_mask=[1, 1],
    )

    with pytest.raises(ValidationError, match="expected D=3"):
        app_module.UserTowerPredictRequest(
            history_embeddings=[[1.0, 2.0], [3.0, 4.0]],
            history_mask=[1, 1],
        )

    with pytest.raises(ValidationError, match="expected T == 2"):
        app_module.UserTowerPredictRequest(
            history_embeddings=[[1.0, 2.0, 3.0]],
            history_mask=[1],
        )

    with pytest.raises(ValidationError, match="batch too large"):
        app_module.UserTowerPredictRequest(
            history_embeddings=[
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            ],
            history_mask=[[1, 1], [1, 1]],
        )


def test_post_tower_request_accepts_unbatched_and_batched():
    app_module.PostTowerPredictRequest(post_embeddings=[1.0, 2.0, 3.0])
    app_module.PostTowerPredictRequest(post_embeddings=[[1.0, 2.0], [3.0, 4.0]])


def test_post_tower_request_rejects_ragged_batched_vectors():
    with pytest.raises(ValidationError, match="same length"):
        app_module.PostTowerPredictRequest(post_embeddings=[[1.0, 2.0], [3.0]])


def test_post_tower_request_enforces_embed_dim(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app_module, "GE_INFERENCE_EMBED_DIM", 2)
    app_module.PostTowerPredictRequest(post_embeddings=[1.0, 2.0])
    with pytest.raises(ValidationError, match="expected D=2"):
        app_module.PostTowerPredictRequest(post_embeddings=[1.0, 2.0, 3.0])


def test_predict_user_tower_endpoint_coerces_unbatched_and_returns_outputs(monkeypatch: pytest.MonkeyPatch):
    _install_dummy_models(monkeypatch)
    with TestClient(app_module.app) as client:
        r = client.post(
            "/models/user-tower/predict",
            json={"history_embeddings": [[1, 2], [3, 4]], "history_mask": [1, 0]},
        )
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["model_type"] == "user-tower"
        assert payload["outputs"] == [[[10.0]]]


def test_predict_post_tower_endpoint_coerces_unbatched_and_returns_outputs(monkeypatch: pytest.MonkeyPatch):
    _install_dummy_models(monkeypatch)
    with TestClient(app_module.app) as client:
        r = client.post("/models/post-tower/predict", json={"post_embeddings": [1, 2, 3]})
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["model_type"] == "post-tower"
        assert payload["outputs"] == [[2.0]]


def test_predict_returns_404_for_unknown_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(app_module, "_models_initialized", True)
    monkeypatch.setattr(app_module, "_models_init_error", None)
    monkeypatch.setattr(app_module, "_models", {})
    with TestClient(app_module.app) as client:
        r = client.post("/models/nope/predict", json={"post_embeddings": [1, 2, 3]})
        assert r.status_code == 404


def test_predict_rejects_request_type_mismatch(monkeypatch: pytest.MonkeyPatch):
    _install_dummy_models(monkeypatch)
    with TestClient(app_module.app) as client:
        # Body is discriminated as post-tower but the URL targets the user-tower model.
        r = client.post("/models/user-tower/predict", json={"post_embeddings": [1, 2, 3]})
        assert r.status_code == 422


def test_predict_returns_500_when_registry_init_failed(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GE_INFERENCE_MODELS", raising=False)
    monkeypatch.setattr(app_module, "_models_initialized", False)
    monkeypatch.setattr(app_module, "_models_init_error", None)
    monkeypatch.setattr(app_module, "_models", {})

    with TestClient(app_module.app) as client:
        r = client.post(
            "/models/user-tower/predict",
            json={"history_embeddings": [[1, 2], [3, 4]], "history_mask": [1, 0]},
        )
        assert r.status_code == 500
        assert "Model registry init failed" in r.text


def test_ready_returns_503_and_registry_error_when_unconfigured(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("GE_INFERENCE_MODELS", raising=False)
    monkeypatch.setattr(app_module, "_models_initialized", False)
    monkeypatch.setattr(app_module, "_models_init_error", None)
    monkeypatch.setattr(app_module, "_models", {})

    with TestClient(app_module.app) as client:
        r = client.get("/ready")
        assert r.status_code == 503
        payload = r.json()
        assert payload["ready"] is False
        assert payload["registry_error"]
