import subprocess
import sys
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.celery_app import celery_app

router = APIRouter(prefix="/workers", tags=["workers"])


class WorkerInfo(BaseModel):
    name:         str
    active_tasks: int
    status:       str   # "idle" | "busy"


class WorkerListResponse(BaseModel):
    total:   int
    workers: list[WorkerInfo]


class WorkerStartRequest(BaseModel):
    concurrency: int = 1
    queue:       str = "celery"
    loglevel:    str = "info"


@router.get("/status", response_model=WorkerListResponse)
def worker_status():
    """Danh sách worker đang online và số task đang chạy."""
    try:
        inspect = celery_app.control.inspect()
        active  = inspect.active() or {}
        ping    = inspect.ping()   or {}

        workers = []
        for name in ping:
            tasks = active.get(name, [])
            workers.append(WorkerInfo(
                name=name,
                active_tasks=len(tasks),
                status="busy" if tasks else "idle",
            ))

        return WorkerListResponse(total=len(workers), workers=workers)
    except Exception as exc:
        raise HTTPException(500, f"Cannot inspect workers: {exc}")


@router.post("/start")
def start_worker(req: WorkerStartRequest):
    """Khởi động thêm một Celery worker process (chạy nền)."""
    cmd = [
        sys.executable, "-m", "celery",
        "-A", "app.celery_app", "worker",
        f"--loglevel={req.loglevel}",
        f"--concurrency={req.concurrency}",
        f"--queues={req.queue}",
    ]
    try:
        proc = subprocess.Popen(cmd, start_new_session=True)
        return {"status": "started", "pid": proc.pid, "queue": req.queue}
    except Exception as exc:
        raise HTTPException(500, f"Failed to start worker: {exc}")


@router.post("/shutdown")
def shutdown_workers():
    """Gửi lệnh shutdown tất cả worker đang online."""
    try:
        celery_app.control.broadcast("shutdown")
        return {"status": "shutdown broadcast sent"}
    except Exception as exc:
        raise HTTPException(500, f"Shutdown failed: {exc}")
