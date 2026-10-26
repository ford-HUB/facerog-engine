import logging
from pathlib import Path

import onnxruntime as ort

from app.core.exceptions import ModelLoadError

logger = logging.getLogger(__name__)

GPU_PROVIDERS = ("CUDAExecutionProvider", "DmlExecutionProvider")
CPU_PROVIDER = "CPUExecutionProvider"
ONNX_LOG_ERRORS_ONLY = 3


def select_providers(use_gpu: bool) -> list[str]:
    """Return the execution providers to use, best first, always ending with CPU."""
    available = ort.get_available_providers()
    providers = [p for p in GPU_PROVIDERS if use_gpu and p in available]
    return [*providers[:1], CPU_PROVIDER]


def create_session(model_path: Path, use_gpu: bool) -> ort.InferenceSession:
    """Load an ONNX model, failing loudly if the weights file is missing or corrupt."""
    if not model_path.is_file():
        raise ModelLoadError(
            f"Model file not found: {model_path}. Run `python scripts/download_weights.py`."
        )
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # The ArcFace export declares batch size 1 but runs batches fine; silence that shape warning.
    options.log_severity_level = ONNX_LOG_ERRORS_ONLY
    try:
        session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=select_providers(use_gpu)
        )
    except (RuntimeError, ort.capi.onnxruntime_pybind11_state.Fail) as err:
        raise ModelLoadError(f"Could not load {model_path}: {err}") from err
    logger.info("Loaded %s with providers %s", model_path.name, session.get_providers())
    return session
