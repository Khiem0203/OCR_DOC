from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    output_dir: str = "/tmp/mineru_output"

    # Celery
    celery_broker_url: str
    celery_result_backend: str

    # VLM — model
    vlm_model_dir: str = ""                    # đường dẫn tuyệt đối đến folder model
    mineru_model_source: str = "local"         # "local" | "huggingface" | "modelscope"

    # VLM — runtime
    enable_vlm_preload: bool = False
    gpu_memory_utilization: float = 0.9

    class Config:
        env_file = ".env"


settings = Settings()
