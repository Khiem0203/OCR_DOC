from celery import Celery
from celery.signals import worker_process_init
from app.config import settings
from app.model_config import apply_model_config

# Áp dụng model config ngay khi module được import (cả FastAPI lẫn worker process)
apply_model_config(settings.vlm_model_dir, settings.mineru_model_source)

celery_app = Celery(
    "mineru",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,        # bật state STARTED
    worker_concurrency=1,           # bắt buộc: 1 GPU, 1 worker process
    task_acks_late=True,            # ack sau khi xử lý xong (an toàn hơn)
    worker_prefetch_multiplier=1,   # không pre-fetch, tránh giữ task lâu trong RAM
    result_expires=3600,            # TTL kết quả trong PostgreSQL (giây)
)

celery_app.autodiscover_tasks(["app"])


@worker_process_init.connect
def preload_model(**kwargs):
    # Gọi lại apply_model_config trong worker subprocess vì os.environ không được
    # kế thừa từ parent process khi dùng spawn/forkserver (Windows mặc định spawn)
    apply_model_config(settings.vlm_model_dir, settings.mineru_model_source)

    if settings.enable_vlm_preload:
        try:
            from cli.vlm_preload import maybe_preload_vlm_model
            maybe_preload_vlm_model(
                enable_vlm_preload=True,
                model_kwargs={"gpu_memory_utilization": settings.gpu_memory_utilization},
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"VLM preload failed: {exc}")
