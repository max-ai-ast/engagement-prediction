import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from hashlib import sha256
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import logging

import torch
from clearml import Model
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    ensure_model_loaded()
    yield


app = FastAPI(lifespan=lifespan)

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
GE_INFERENCE_WARMUP_SECONDS_BUDGET = float(os.getenv("GE_INFERENCE_WARMUP_SECONDS_BUDGET", "5.0"))

# If you know D, set GE_INFERENCE_INPUT_DIM to validate and to create dummy warmup.
GE_INFERENCE_INPUT_DIM = int(os.getenv("GE_INFERENCE_INPUT_DIM", "0"))  # 0 means unknown/skip dim validation

DTYPE = torch.float32

# -------------------------
# State
# -------------------------
_model_lock = threading.Lock()
_model: Optional[torch.jit.ScriptModule] = None
_device: Optional[torch.device] = None
_model_path: Optional[str] = None

_loaded_event = threading.Event()
_load_error: Optional[str] = None
_load_started_at: Optional[float] = None
_load_finished_at: Optional[float] = None


# -------------------------
# API schema
# -------------------------
class PredictRequest(BaseModel):
    # Accept [D] or [B, D]
    inputs: Union[List[float], List[List[float]]]


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


def _coerce_inputs_to_batched_tensor(
    inputs: Union[List[float], List[List[float]]],
    device: torch.device,
) -> torch.Tensor:
    if len(inputs) == 0:
        raise HTTPException(status_code=400, detail="inputs must be non-empty")

    is_batched = isinstance(inputs[0], list)  # type: ignore[index]
    if is_batched:
        batch: List[List[float]] = inputs  # type: ignore[assignment]
        if len(batch) > GE_INFERENCE_MAX_BATCH:
            raise HTTPException(status_code=400, detail=f"batch too large (max={GE_INFERENCE_MAX_BATCH})")
        d0 = len(batch[0])
        if d0 == 0:
            raise HTTPException(status_code=400, detail="each input vector must be non-empty")
        if not all(len(v) == d0 for v in batch):
            raise HTTPException(status_code=400, detail="all input vectors must have the same length")
        if GE_INFERENCE_INPUT_DIM and d0 != GE_INFERENCE_INPUT_DIM:
            raise HTTPException(status_code=400, detail=f"expected D={GE_INFERENCE_INPUT_DIM}, got D={d0}")
        return torch.tensor(batch, dtype=DTYPE, device=device)

    vec: List[float] = inputs  # type: ignore[assignment]
    if GE_INFERENCE_INPUT_DIM and len(vec) != GE_INFERENCE_INPUT_DIM:
        raise HTTPException(status_code=400, detail=f"expected D={GE_INFERENCE_INPUT_DIM}, got D={len(vec)}")
    x = torch.tensor(vec, dtype=DTYPE, device=device)
    return x.unsqueeze(0)  # [1, D]


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


def _warmup_model(model: torch.jit.ScriptModule, device: torch.device) -> None:
    """
    Warmup does two things:
      - validates that the model can run on the target device
      - forces CUDA context init and (sometimes) JIT internal setup
    """
    if device.type != "cuda":
        return
    if not GE_INFERENCE_WARMUP:
        return
    if GE_INFERENCE_INPUT_DIM <= 0:
        # Without knowing D, we can't safely warm up.
        return

    start = time.time()
    with torch.inference_mode():
        # A couple of warmup passes; keep it short.
        dummy = torch.zeros((1, GE_INFERENCE_INPUT_DIM), dtype=DTYPE, device=device)
        _ = model(dummy)
        if time.time() - start < GE_INFERENCE_WARMUP_SECONDS_BUDGET:
            _ = model(dummy)


def _load_model_inner() -> None:
    global _model, _device, _model_path

    device = _choose_device()

    model_id = None
    model_path_env = os.getenv("GE_INFERENCE_MODEL_PATH")
    model_uri_env = os.getenv("GE_INFERENCE_MODEL_URI")

    if model_path_env:
        model_file = _find_model_file(model_path_env)
    elif model_uri_env:
        parsed = urlparse(model_uri_env)
        if parsed.scheme == "gs":
            model_file = _find_model_file(_download_gcs_uri_to_local(model_uri_env))
        else:
            # Treat as local path (supports relative paths too).
            model_file = _find_model_file(model_uri_env)
    else:
        model_id = os.getenv("GE_INFERENCE_CLEARML_MODEL_ID")
        if not model_id:
            raise RuntimeError("Either MODEL_PATH, MODEL_URI, or CLEARML_MODEL_ID env var is required")

        cm = Model(model_id=model_id)
        local_copy = cm.get_local_copy()
        model_file = _find_model_file(local_copy)

    m = torch.jit.load(model_file, map_location=device)
    m.eval()

    _warmup_model(m, device)

    _model = m
    _device = device
    _model_path = model_file

    logger.info(
        "Model loaded successfully | model_id=%s | model_path=%s | device=%s",
        model_id,
        model_file,
        device,
    )


def ensure_model_loaded() -> None:
    """
    Concurrency-safe, idempotent load.
    Also sets readiness state & captures errors for /ready.
    """
    global _load_error, _load_started_at, _load_finished_at

    if _model is not None:
        _loaded_event.set()
        return

    with _model_lock:
        if _model is not None:
            _loaded_event.set()
            return

        _load_started_at = time.time()
        try:
            _load_model_inner()
            _load_error = None
        except Exception as e:
            _load_error = str(e)
            raise
        finally:
            _load_finished_at = time.time()
            if _model is not None:
                _loaded_event.set()


# -------------------------
# Endpoints
# -------------------------
@app.get("/health")
def health() -> dict:
    # Process is up.
    return {"ok": True}


@app.get("/ready")
def ready():
    ready = _loaded_event.is_set()

    payload = {
        "ready": ready,
        "device": str(_device) if _device else None,
        "load_error": _load_error,
        "load_started_at": _load_started_at,
        "load_finished_at": _load_finished_at
    }

    status = 200 if ready else 503
    return JSONResponse(content=payload, status_code=status)


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    # If startup load hasn't completed, do a guarded sync load here.
    if _model is None:
        try:
            ensure_model_loaded()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model not available: {e}")

    assert _model is not None and _device is not None

    x = _coerce_inputs_to_batched_tensor(req.inputs, device=_device)
    try:
        with torch.inference_mode():
            y = _model(x)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference failed: {e}")

    return {"outputs": _to_python(y)}
