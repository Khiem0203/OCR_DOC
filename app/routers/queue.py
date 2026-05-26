from fastapi import APIRouter, HTTPException
from celery.result import AsyncResult
from pydantic import BaseModel
from app.celery_app import celery_app

router = APIRouter(prefix="/queue", tags=["queue"])


class QueueStatus(BaseModel):
    queue_name:     str
    active_tasks:   int
    reserved_tasks: int
    status:         str   # "idle" | "busy"


@router.get("/status", response_model=QueueStatus)
def get_queue_status():
    """Tổng trạng thái hàng đợi parse."""
    try:
        inspect = celery_app.control.inspect()
        active   = inspect.active()   or {}
        reserved = inspect.reserved() or {}

        total_active   = sum(len(t) for t in active.values())
        total_reserved = sum(len(t) for t in reserved.values())

        return QueueStatus(
            queue_name="celery",
            active_tasks=total_active,
            reserved_tasks=total_reserved,
            status="busy" if (total_active + total_reserved) > 0 else "idle",
        )
    except Exception as exc:
        raise HTTPException(500, f"Cannot inspect queue: {exc}")


@router.get("/task/{task_id}")
def get_task_detail(task_id: str):
    """Chi tiết một task theo task_id (state + info)."""
    r = AsyncResult(task_id, app=celery_app)
    info = r.info if not isinstance(r.info, Exception) else str(r.info)
    return {
        "task_id": task_id,
        "state":   r.state,
        "info":    info,
    }
