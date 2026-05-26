import datetime
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health():
    """Basic liveness check."""
    return {"status": "ok", "timestamp": datetime.datetime.now().isoformat()}


@router.get("/ready")
def ready():
    """Readiness check — trả 200 chỉ khi VLM model đã được load vào VRAM."""
    try:
        from backend.vlm.vlm_analyze import ModelSingleton
        loaded = bool(ModelSingleton()._models)
    except Exception:
        loaded = False
    return {"ready": loaded}
