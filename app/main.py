import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import parse, queue, workers, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    # FastAPI process KHÔNG load model — Celery worker process mới load.
    # Chỉ preload ở đây khi FastAPI và worker chạy cùng process (dev mode).
    if settings.enable_vlm_preload:
        try:
            from cli.vlm_preload import maybe_preload_vlm_model
            maybe_preload_vlm_model(
                enable_vlm_preload=True,
                model_kwargs={"gpu_memory_utilization": settings.gpu_memory_utilization},
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"VLM preload skipped: {exc}")
    yield


app = FastAPI(
    title="MinerU VLM Backend",
    description="Async PDF parsing service powered by MinerU VLM engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(parse.router,   prefix="/api/v1")   # /api/v1/parse
app.include_router(queue.router,   prefix="/api/v1")   # /api/v1/queue
app.include_router(workers.router, prefix="/api/v1")   # /api/v1/workers
app.include_router(health.router)                       # /health  /ready


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,      
        reload=False,
    )
