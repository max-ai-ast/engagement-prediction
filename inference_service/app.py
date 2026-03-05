import os
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union
import logging

import torch
from clearml import Model
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    blocking_startup = os.getenv("BLOCKING_STARTUP", "0") == "1"
    if blocking_startup:
        # Useful for Cloud Run if you prefer "fail fast" at startup and avoid serving
        # requests before the model is available.
        ensure_model_loaded()
    else:
        # Start background model download+load immediately.
        t = threading.Thread(target=_background_startup_load, daemon=True)
        t.start()
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
MAX_BATCH = int(os.getenv("MAX_BATCH", "1024"))
PREFER_CUDA = os.getenv("PREFER_CUDA", "1") == "1"
WARMUP = os.getenv("WARMUP", "1") == "1"
WARMUP_SECONDS_BUDGET = float(os.getenv("WARMUP_SECONDS_BUDGET", "5.0"))

# If you know D, set INPUT_DIM to validate and to create dummy warmup.
INPUT_DIM = int(os.getenv("INPUT_DIM", "0"))  # 0 means unknown/skip dim validation

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
    if PREFER_CUDA and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _find_model_file(path: str) -> str:
    if os.path.isfile(path):
        return path
    raise RuntimeError(f"ClearML local copy path is not a file: {path}")


def _coerce_inputs_to_batched_tensor(
    inputs: Union[List[float], List[List[float]]],
    device: torch.device,
) -> torch.Tensor:
    if len(inputs) == 0:
        raise HTTPException(status_code=400, detail="inputs must be non-empty")

    is_batched = isinstance(inputs[0], list)  # type: ignore[index]
    if is_batched:
        batch: List[List[float]] = inputs  # type: ignore[assignment]
        if len(batch) > MAX_BATCH:
            raise HTTPException(status_code=400, detail=f"batch too large (max={MAX_BATCH})")
        d0 = len(batch[0])
        if d0 == 0:
            raise HTTPException(status_code=400, detail="each input vector must be non-empty")
        if not all(len(v) == d0 for v in batch):
            raise HTTPException(status_code=400, detail="all input vectors must have the same length")
        if INPUT_DIM and d0 != INPUT_DIM:
            raise HTTPException(status_code=400, detail=f"expected D={INPUT_DIM}, got D={d0}")
        return torch.tensor(batch, dtype=DTYPE, device=device)

    vec: List[float] = inputs  # type: ignore[assignment]
    if INPUT_DIM and len(vec) != INPUT_DIM:
        raise HTTPException(status_code=400, detail=f"expected D={INPUT_DIM}, got D={len(vec)}")
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
    if not WARMUP:
        return
    if INPUT_DIM <= 0:
        # Without knowing D, we can't safely warm up.
        return

    start = time.time()
    with torch.inference_mode():
        # A couple of warmup passes; keep it short.
        dummy = torch.zeros((1, INPUT_DIM), dtype=DTYPE, device=device)
        _ = model(dummy)
        if time.time() - start < WARMUP_SECONDS_BUDGET:
            _ = model(dummy)


def _load_model_inner() -> None:
    global _model, _device, _model_path

    device = _choose_device()

    model_id = None
    model_path_env = os.getenv("MODEL_PATH")

    if model_path_env:
        model_file = _find_model_file(model_path_env)
    else:
        model_id = os.getenv("CLEARML_MODEL_ID")
        if not model_id:
            raise RuntimeError("Either MODEL_PATH or CLEARML_MODEL_ID env var is required")

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
    Also sets readiness state & captures errors for /readyz.
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


def _background_startup_load() -> None:
    """
    Runs once at startup in a thread. If it fails, the service still starts,
    but /readyz remains false and /predict will fail until load succeeds.
    """
    try:
        ensure_model_loaded()
    except Exception:
        # Intentionally swallow here; surfaced via /readyz.
        pass


# -------------------------
# Endpoints
# -------------------------
@app.get("/healthz")
def healthz() -> dict:
    # Process is up.
    return {"ok": True}


@app.get("/readyz")
def readyz():
    ready = _loaded_event.is_set()

    payload = {
        "ready": ready,
        "device": str(_device) if _device else None,
        "model_path": _model_path,
        "load_error": _load_error,
        "load_started_at": _load_started_at,
        "load_finished_at": _load_finished_at,
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


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)