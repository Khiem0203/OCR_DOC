"""
MinerU model-path bridge.

pydantic-settings đọc .env vào Settings nhưng KHÔNG tự propagate vào os.environ.
Thư viện MinerU đọc os.getenv('MINERU_MODEL_SOURCE') và ~/mineru.json trực tiếp,
nên cần code này làm cầu nối khi app hoặc worker khởi động.

Gọi apply_model_config() sớm nhất có thể — trước khi thư viện import bất kỳ
VLM module nào.
"""
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def apply_model_config(vlm_model_dir: str, mineru_model_source: str = "local") -> None:
    """Propagate model settings từ .env ra os.environ và ~/mineru.json.

    Args:
        vlm_model_dir:        Đường dẫn tuyệt đối đến folder model đã tải về.
                              Để trống ("") nếu muốn để MinerU tự tải từ HuggingFace.
        mineru_model_source:  "local" | "huggingface" | "modelscope"
    """
    # 1. Đặt MINERU_MODEL_SOURCE vào os.environ để thư viện đọc được
    os.environ.setdefault("MINERU_MODEL_SOURCE", mineru_model_source)

    # 2. Nếu dùng local model, đảm bảo ~/mineru.json có đường dẫn đúng
    if mineru_model_source == "local" and vlm_model_dir:
        _write_mineru_json(vlm_model_dir)
    elif mineru_model_source == "local" and not vlm_model_dir:
        logger.warning(
            "MINERU_MODEL_SOURCE=local nhưng VLM_MODEL_DIR chưa được đặt. "
            "Thư viện sẽ đọc ~/mineru.json hiện có hoặc báo lỗi."
        )


def _write_mineru_json(vlm_model_dir: str) -> None:
    """Ghi (hoặc cập nhật) ~/mineru.json với đường dẫn model VLM.

    Giữ nguyên các key khác đã có trong file (bucket_info, latex-delimiter-config, v.v.)
    """
    # Đường dẫn file config — có thể override qua env MINERU_TOOLS_CONFIG_JSON
    config_file_name = os.getenv("MINERU_TOOLS_CONFIG_JSON", "mineru.json")
    if os.path.isabs(config_file_name):
        config_path = Path(config_file_name)
    else:
        config_path = Path.home() / config_file_name

    # Đọc config hiện có (nếu có)
    cfg: dict = {}
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Không đọc được {config_path}: {exc} — sẽ ghi đè.")

    # Cập nhật đường dẫn model VLM
    cfg.setdefault("models-dir", {})["vlm"] = vlm_model_dir

    try:
        config_path.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        logger.info(f"Đã ghi model config → {config_path}  (vlm: {vlm_model_dir})")
    except Exception as exc:
        logger.error(f"Không thể ghi {config_path}: {exc}")
