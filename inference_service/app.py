import os
import threading
import time
from dataclasses import dataclass
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from hashlib import sha256
from typing import Any, Dict, List, Literal, Optional, Union, Annotated, assert_never, get_args
from urllib.parse import urlparse
import logging

import torch
from clearml import Model
from fastapi import FastAPI, HTTPException, Security, Body
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Discriminator, Tag, model_validator


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    ensure_models_loaded()
    yield


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)
app = FastAPI(lifespan=lifespan)


def _require_api_key(api_key: str = Security(_api_key_header)) -> None:
    if api_key != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# Config
# -------------------------
GE_INFERENCE_MAX_BATCH = int(os.getenv("GE_INFERENCE_MAX_BATCH", "1024"))
GE_INFERENCE_PREFER_CUDA = os.getenv("GE_INFERENCE_PREFER_CUDA", "1") == "1"
GE_INFERENCE_WARMUP = os.getenv("GE_INFERENCE_WARMUP", "1") == "1"
_API_KEY: str | None = os.environ.get("GE_INFERENCE_API_KEY") or None

# If you know these shapes, set them to validate and to create dummy warmup.
GE_INFERENCE_EMBED_DIM = int(os.getenv("GE_INFERENCE_EMBED_DIM", "0")) # 0 means unknown/skip dim validation
GE_INFERENCE_MAX_HISTORY_LEN = int(os.getenv("GE_INFERENCE_MAX_HISTORY_LEN", "0")) 
if GE_INFERENCE_MAX_HISTORY_LEN <= 0:
    raise ValueError("Must supply a valid GE_INFERENCE_MAX_HISTORY_LEN!")

DTYPE = torch.float32

# -------------------------
# State
# -------------------------
ModelType = Literal["user-tower", "post-tower"]
ModelSignature = Literal["vector", "history"]


@dataclass
class LoadedModel:
    model_type: ModelType
    signature: ModelSignature
    configured_model_path: Optional[str] = None
    configured_model_uri: Optional[str] = None
    configured_clearml_model_id: Optional[str] = None

    module: Optional[torch.jit.ScriptModule] = None
    device: Optional[torch.device] = None
    resolved_model_path: Optional[str] = None
    resolved_model_id: Optional[str] = None

    load_error: Optional[str] = None
    load_started_at: Optional[float] = None
    load_finished_at: Optional[float] = None


_models_lock = threading.Lock()
_models_initialized = False
_models_init_error: Optional[str] = None
_models: Dict[str, LoadedModel] = {}


# -------------------------
# API schema
# -------------------------
class UserTowerPredictRequest(BaseModel):
    # history_embeddings: [T, D] or [B, T, D]
    history_embeddings: Union[List[List[float]], List[List[List[float]]]]
    # history_mask: [T] or [B, T]
    history_mask: Union[List[Union[int, bool, float]], List[List[Union[int, bool, float]]]]

    @model_validator(mode="after")
    def _validate_history(self) -> "UserTowerPredictRequest":
        he = self.history_embeddings
        if not isinstance(he, list) or len(he) == 0:
            raise ValueError("'history_embeddings' must be a non-empty list")

        # Determine shape: [T, D] vs [B, T, D]
        if isinstance(he[0], list) and len(he[0]) > 0 and not isinstance(he[0][0], list):
            # [T, D]
            seq_len = len(he)
            d0 = len(he[0])
            if d0 == 0:
                raise ValueError("history_embeddings must have shape [T, D] with D>0")
            if not all(isinstance(row, list) and len(row) == d0 for row in he):
                raise ValueError("history_embeddings must be rectangular with shape [T, D]")
            batch = 1
        else:
            # [B, T, D]
            batch = len(he)
            if batch > GE_INFERENCE_MAX_BATCH:
                raise ValueError(f"batch too large (max={GE_INFERENCE_MAX_BATCH})")
            if not (isinstance(he[0], list) and len(he[0]) > 0 and isinstance(he[0][0], list)):
                raise ValueError("history_embeddings must have shape [T, D] or [B, T, D]")
            seq_len = len(he[0])
            d0 = len(he[0][0])
            if seq_len == 0 or d0 == 0:
                raise ValueError("history_embeddings must have shape [B, T, D] with T>0 and D>0")
            for b in he:
                if not (isinstance(b, list) and len(b) == seq_len):
                    raise ValueError("history_embeddings must be rectangular with shape [B, T, D]")
                for row in b:
                    if not (isinstance(row, list) and len(row) == d0):
                        raise ValueError("history_embeddings must be rectangular with shape [B, T, D]")

        if seq_len != GE_INFERENCE_MAX_HISTORY_LEN:
            raise ValueError(f"expected T == {GE_INFERENCE_MAX_HISTORY_LEN}, got T={seq_len}")
        if GE_INFERENCE_EMBED_DIM and d0 != GE_INFERENCE_EMBED_DIM:
            raise ValueError(f"expected D={GE_INFERENCE_EMBED_DIM}, got D={d0}")

        mask = self.history_mask

        if not isinstance(mask, list) or len(mask) == 0:
            raise ValueError("'history_mask' must be a non-empty list")

        is_mask_batched = isinstance(mask[0], list)
        if is_mask_batched:
            mb = mask  # type: ignore[assignment]
            if len(mb) != batch:
                raise ValueError(f"history_mask must have shape [B, T] = [{batch}, {seq_len}]. got B={len(mb)}")
            for row in mb:
                if not (isinstance(row, list) and len(row) == seq_len):
                    raise ValueError(f"history_mask must have shape [B, T] = [{batch}, {seq_len}]")
        else:
            mv = mask  # type: ignore[assignment]
            if batch != 1:
                raise ValueError(f"history_mask must have shape [B, T] = [{batch}, {seq_len}]")
            if len(mv) != seq_len:
                raise ValueError(f"history_mask must have shape [T] = [{seq_len}]")

        return self


class PostTowerPredictRequest(BaseModel):
    # post_embeddings: [D] or [B, D]
    post_embeddings: Union[List[float], List[List[float]]]

    @model_validator(mode="after")
    def _validate_post_embeddings(self) -> "PostTowerPredictRequest":
        pe = self.post_embeddings
        if not isinstance(pe, list) or len(pe) == 0:
            raise ValueError("'post_embeddings' must be a non-empty list")

        is_batched = isinstance(pe[0], list)
        if is_batched:
            batch = pe  # type: ignore[assignment]
            if len(batch) > GE_INFERENCE_MAX_BATCH:
                raise ValueError(f"batch too large (max={GE_INFERENCE_MAX_BATCH})")
            d0 = len(batch[0]) if len(batch) > 0 else 0
            if d0 == 0:
                raise ValueError("each post_embeddings vector must be non-empty")
            if not all(isinstance(v, list) and len(v) == d0 for v in batch):
                raise ValueError("all post_embeddings vectors must have the same length")
            if GE_INFERENCE_EMBED_DIM and d0 != GE_INFERENCE_EMBED_DIM:
                raise ValueError(f"expected D={GE_INFERENCE_EMBED_DIM}, got D={d0}")
        else:
            vec = pe  # type: ignore[assignment]
            if len(vec) == 0:
                raise ValueError("'post_embeddings' must be non-empty")
            if GE_INFERENCE_EMBED_DIM and len(vec) != GE_INFERENCE_EMBED_DIM:
                raise ValueError(f"expected D={GE_INFERENCE_EMBED_DIM}, got D={len(vec)}")
        return self


def _predict_request_discriminator(value: Any) -> str:
    if isinstance(value, dict):
        if "post_embeddings" in value:
            return "post-tower"
        if "history_embeddings" in value and "history_mask" in value:
            return "user-tower"
    raise ValueError("Request must contain one of: 'post_embeddings' or 'history_embeddings'+'history_mask'.")


PredictRequest = Annotated[
    Union[
        Annotated[UserTowerPredictRequest, Tag("user-tower")],
        Annotated[PostTowerPredictRequest, Tag("post-tower")],
    ],
    Discriminator(_predict_request_discriminator),
]


# -------------------------
# Helpers
# -------------------------
def _choose_device() -> torch.device:
    if GE_INFERENCE_PREFER_CUDA and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _find_model_file(path: str) -> str:
    if os.path.isfile(path):
        return path
    raise RuntimeError(f"Model path is not a file: {path}")


def _download_gcs_uri_to_local(gs_uri: str) -> str:
    """
    Download a single GCS object (gs://bucket/path) to a local cache and return its path.
    Uses Application Default Credentials.
    """
    parsed = urlparse(gs_uri)
    if parsed.scheme != "gs":
        raise ValueError(f"Expected gs:// URI, got: {gs_uri}")

    bucket_name = parsed.netloc
    blob_name = parsed.path.lstrip("/")
    if not bucket_name or not blob_name:
        raise ValueError(f"Invalid gs:// URI (missing bucket or object): {gs_uri}")

    model_cache_dir = os.getenv("GE_INFERENCE_MODEL_CACHE_DIR", "/tmp/model_cache")
    os.makedirs(model_cache_dir, exist_ok=True)

    blob_basename = os.path.basename(blob_name) or "model"
    key = sha256(gs_uri.encode("utf-8")).hexdigest()[:16]
    local_path = os.path.join(model_cache_dir, f"{key}-{blob_basename}")

    if os.path.exists(local_path):
        return local_path

    # Import lazily so non-GCS paths don't require this dependency at import time.
    from google.cloud import storage  # type: ignore[import-not-found]

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

    return local_path


def _validate_model_type(model_type: str) -> ModelType:
    if model_type in get_args(ModelType):
        return model_type # type: ignore[return-value]
    raise RuntimeError(f"Unsupported model type: '{model_type}'")


def _tensor_from_nested_list(name: str, value: Any, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    if value is None:
        raise HTTPException(status_code=400, detail=f"Missing required field '{name}'")
    try:
        t = torch.tensor(value, dtype=dtype, device=device)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid '{name}': {e}")
    if t.numel() == 0:
        raise HTTPException(status_code=400, detail=f"'{name}' must be non-empty")
    return t


def _model_env_key(model_type: str) -> str:
    # Model names may contain "-" which isn't valid in env vars.
    # Example: "user-tower" -> "USER_TOWER"
    return "".join((c if c.isalnum() else "_") for c in model_type).upper()


def _read_model_env(model_type: str, suffix: str) -> Optional[str]:
    key = _model_env_key(model_type)
    return os.getenv(f"GE_INFERENCE_{key}_{suffix}")


def _infer_signature(model_type: ModelType) -> ModelSignature:
    match model_type:
        case "user-tower":
            return "history"
        case "post-tower":
            return "vector"
        case _:
            assert_never(model_type)


def _coerce_input(name: str, value: Any, dtype: torch.dtype, device: torch.device, non_batched_dim: int) -> torch.Tensor:
    t = _tensor_from_nested_list(name, value, dtype, device)
    if t.dim() == non_batched_dim:
        t = t.unsqueeze(0) # add a batch dimension of size 1 at the beginning
    return t


def _to_python(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_python(x) for x in obj]
    if isinstance(obj, dict):
        out: Dict[Any, Any] = {}
        for k, v in obj.items():
            out[k] = _to_python(v)
        return out
    return obj


def _warmup_entry(entry: LoadedModel) -> None:
    """Best-effort warmup to initialize CUDA context and validate the forward pass."""
    if entry.device is None or entry.module is None:
        return
    device = entry.device
    model = entry.module
    if device.type != "cuda":
        return
    if not GE_INFERENCE_WARMUP:
        return

    with torch.inference_mode():
        # Keep warmup short.
        if entry.signature == "vector":
            if GE_INFERENCE_EMBED_DIM <= 0:
                return
            dummy = torch.zeros((1, GE_INFERENCE_EMBED_DIM), dtype=DTYPE, device=device)
            _ = model(dummy)
            return

        if entry.signature == "history":
            if GE_INFERENCE_EMBED_DIM <= 0:
                return
            history_embeddings = torch.zeros(
                (1, GE_INFERENCE_MAX_HISTORY_LEN, GE_INFERENCE_EMBED_DIM), dtype=DTYPE, device=device
            )
            history_mask = torch.ones((1, GE_INFERENCE_MAX_HISTORY_LEN), dtype=torch.bool, device=device)
            _ = model(history_embeddings, history_mask)
            return


def _init_registry() -> None:
    global _models_initialized, _models_init_error, _models
    if _models_initialized:
        return

    with _models_lock:
        if _models_initialized:
            return

        try:
            models: Dict[str, LoadedModel] = {}

            models_env = os.getenv("GE_INFERENCE_MODELS", "").strip()
            if not models_env:
                raise RuntimeError(
                    "No models configured. Set GE_INFERENCE_MODELS (e.g. 'user-tower,post-tower') "
                    "and per-model GE_INFERENCE_{MODEL_TYPE}_MODEL_PATH/GE_INFERENCE_{MODEL_TYPE}_MODEL_URI/GE_INFERENCE_{MODEL_TYPE}_CLEARML_MODEL_ID env vars."
                )

            env_model_types: List[str] = []
            env_model_types: List[str] = models_env.split(",")
            if len(env_model_types) > 2:
                raise RuntimeError(f"Too many models configured ({len(env_model_types)}). Max is 2.")

            seen: set[str] = set()
            for env_model_type in env_model_types:
                if env_model_type in seen:
                    continue
                seen.add(env_model_type)

                model_type: ModelType = _validate_model_type(env_model_type)
                signature: ModelSignature = _infer_signature(model_type)

                # Per-model sources.
                model_path = _read_model_env(model_type, "MODEL_PATH")
                model_uri = _read_model_env(model_type, "MODEL_URI")
                clearml_id = _read_model_env(model_type, "CLEARML_MODEL_ID")

                if not (model_path or model_uri or clearml_id):
                    raise RuntimeError(
                        f"Model '{model_type}' is missing a source. "
                        f"Set one of: GE_INFERENCE_{_model_env_key(model_type)}_MODEL_PATH | "
                        f"GE_INFERENCE_{_model_env_key(model_type)}_MODEL_URI | "
                        f"GE_INFERENCE_{_model_env_key(model_type)}_CLEARML_MODEL_ID"
                    )

                models[model_type] = LoadedModel(
                    model_type=model_type,
                    signature=signature,
                    configured_model_path=model_path,
                    configured_model_uri=model_uri,
                    configured_clearml_model_id=clearml_id,
                )

            _models = models
            _models_init_error = None
        except Exception as e:
            _models = {}
            _models_init_error = str(e)
        finally:
            _models_initialized = True


def _resolve_model_file(entry: LoadedModel) -> tuple[str, Optional[str]]:
    model_id = None
    if entry.configured_model_path:
        model_file = _find_model_file(entry.configured_model_path)
    elif entry.configured_model_uri:
        parsed = urlparse(entry.configured_model_uri)
        if parsed.scheme == "gs":
            model_file = _find_model_file(_download_gcs_uri_to_local(entry.configured_model_uri))
        else:
            model_file = _find_model_file(entry.configured_model_uri)
    else:
        model_id = entry.configured_clearml_model_id
        if not model_id:
            model_env_key = _model_env_key(entry.model_type)
            raise RuntimeError(
                f"Model '{entry.model_type}' is missing a source (GE_INFERENCE_{model_env_key}_MODEL_PATH | GE_INFERENCE_{model_env_key}_MODEL_URI | GE_INFERENCE_{model_env_key}_CLEARML_MODEL_ID)"
            )
        cm = Model(model_id=model_id)
        local_copy = cm.get_local_copy()
        model_file = _find_model_file(local_copy)
    return model_file, model_id


def _load_entry(entry: LoadedModel) -> None:
    device = _choose_device()
    model_file, model_id = _resolve_model_file(entry)

    m = torch.jit.load(model_file, map_location=device)
    m.eval()

    entry.module = m
    entry.device = device
    entry.resolved_model_path = model_file
    entry.resolved_model_id = model_id

    _warmup_entry(entry)

    logger.info(
        "Model loaded | type=%s | signature=%s | model_id=%s | model_path=%s | device=%s",
        entry.model_type,
        entry.signature,
        model_id,
        model_file,
        device,
    )


def ensure_models_loaded() -> None:
    """Concurrency-safe, idempotent load of all configured models."""
    _init_registry()
    if _models_init_error is not None:
        logger.error("Model registry init failed: %s", _models_init_error)
        return

    with _models_lock:
        for entry in _models.values():
            if entry.module is not None:
                continue
            if entry.load_started_at is not None and entry.load_finished_at is None:
                continue

            entry.load_started_at = time.time()
            try:
                _load_entry(entry)
                entry.load_error = None
            except Exception as e:
                entry.load_error = str(e)
                logger.exception("Model load failed | type=%s | error=%s", entry.model_type, entry.load_error)
            finally:
                entry.load_finished_at = time.time()


def _get_entry_or_404(model_name: str) -> LoadedModel:
    _init_registry()
    if _models_init_error is not None:
        raise HTTPException(status_code=500, detail=f"Model registry init failed: {_models_init_error}")

    entry = _models.get(model_name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Unknown model '{model_name}'")
    return entry


def _require_ready(entry: LoadedModel) -> None:
    if entry.module is not None and entry.device is not None:
        return
    ensure_models_loaded()
    if entry.module is None or entry.device is None:
        raise HTTPException(
            status_code=503,
            detail={"model_type": entry.model_type, "ready": False, "load_error": entry.load_error},
        )

def _predict_with_entry(entry: LoadedModel, req: PredictRequest) -> Any:
    _require_ready(entry)
    assert entry.module is not None and entry.device is not None

    # Enforce schema against registered model type.
    match entry.model_type:
        case "user-tower":
            if not isinstance(req, UserTowerPredictRequest):
                raise HTTPException(
                    status_code=422,
                    detail=f"Model type '{entry.model_type}' expects a user-tower request body with 'history_embeddings'",
                )
        case "post-tower":
            if not isinstance(req, PostTowerPredictRequest):
                raise HTTPException(
                    status_code=422,
                    detail=f"Model type '{entry.model_type}' expects a post-tower request body with 'post_embeddings'",
                )
        case _:
            assert_never(entry.model_type)

    with torch.inference_mode():
        if entry.model_type == "user-tower":
            assert isinstance(req, UserTowerPredictRequest)
            history_embeddings = _coerce_input(
                value=req.history_embeddings,
                name="history_embeddings",
                dtype=DTYPE,
                device=entry.device,
                non_batched_dim=2,
            )
            history_mask = _coerce_input(
                value=req.history_mask,
                name="history_mask",
                dtype=torch.int64,
                device=entry.device,
                non_batched_dim=1,
            )
            y = entry.module(history_embeddings, history_mask)
            return y

        if entry.model_type == "post-tower":
            assert isinstance(req, PostTowerPredictRequest)
            post_embeddings = _coerce_input(
                value=req.post_embeddings,
                name="post_embeddings",
                dtype=DTYPE,
                device=entry.device,
                non_batched_dim=1,
            )
            y = entry.module(post_embeddings)
            return y

    raise HTTPException(status_code=500, detail=f"Unsupported model type: {entry.model_type}")


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health() -> dict:
    # Process is up.
    return {"ok": True}


@app.get("/ready", dependencies=[Security(_require_api_key)])
def ready():
    _init_registry()
    ensure_models_loaded()

    models_payload: List[dict[str, Any]] = []
    all_ready = _models_init_error is None and len(_models) > 0
    for entry in _models.values():
        model_ready = entry.module is not None and entry.device is not None and entry.load_error is None
        all_ready = all_ready and model_ready
        models_payload.append(
            {
                "type": entry.model_type,
                "signature": entry.signature,
                "ready": model_ready,
                "device": str(entry.device) if entry.device else None,
                "model_path": entry.resolved_model_path,
                "model_id": entry.resolved_model_id,
                "load_error": entry.load_error,
                "load_started_at": entry.load_started_at,
                "load_finished_at": entry.load_finished_at,
            }
        )

    payload = {
        "ready": all_ready,
        "registry_error": _models_init_error,
        "embed_dim": GE_INFERENCE_EMBED_DIM if GE_INFERENCE_EMBED_DIM > 0 else None,
        "max_seq_len": GE_INFERENCE_MAX_HISTORY_LEN,
        "models": models_payload,
    }

    status = 200 if all_ready else 503
    return JSONResponse(content=payload, status_code=status)

@app.get("/models", dependencies=[Security(_require_api_key)])
def list_models() -> dict:
    _init_registry()
    ensure_models_loaded()
    models_payload: List[dict[str, Any]] = []
    for entry in _models.values():
        models_payload.append(
            {
                "type": entry.model_type,
                "signature": entry.signature,
                "ready": entry.module is not None and entry.device is not None and entry.load_error is None,
                "device": str(entry.device) if entry.device else None,
                "model_path": entry.resolved_model_path,
                "model_id": entry.resolved_model_id,
                "load_error": entry.load_error,
                "load_started_at": entry.load_started_at,
                "load_finished_at": entry.load_finished_at,
            }
        )
    return {"models": models_payload, "registry_error": _models_init_error}


@app.post("/models/{model_name}/predict", dependencies=[Security(_require_api_key)])
def predict_model(model_name: str, req: PredictRequest = Body(...)) -> dict:
    entry = _get_entry_or_404(model_name)
    try:
        y = _predict_with_entry(entry, req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")
    return {"outputs": _to_python(y), "model_type": entry.model_type}
