import asyncio, base64, json, os, gc, shutil, tempfile
from pathlib import Path
import torch
from celery.utils.log import get_task_logger
from app.celery_app import celery_app
from app.config import settings

from cli.common import aio_do_parse

from pre.image_preprocess import maybe_preprocess
from post.vi_ocr_correct import correct_md

celery_logger = get_task_logger(__name__)

@celery_app.task(bind=True, name="app.tasks.parse_task")
def parse_task(self, pdf_b64: str, req_dict: dict) -> dict:
    """
    Celery task — chạy đồng bộ, wrap aio_do_parse bằng asyncio.run().
    pdf_b64 : PDF bytes được encode base64 (để truyền qua JSON serializer).
    req_dict: ParseRequest dưới dạng dict.
    """
    task_id   = self.request.id
    file_name = req_dict["file_name"]
    stem      = Path(file_name).stem

    task_output_dir = os.path.join(settings.output_dir, task_id)
    output_dir_path = Path(task_output_dir) / stem  

    celery_logger.info(f"parse_task ({task_id}): start — file={file_name}")

    try:
        pdf_bytes = base64.b64decode(pdf_b64)

        # [1] Tiền xử lý ảnh/PDF
        if req_dict.get("preprocess"):
            with tempfile.NamedTemporaryFile(
                suffix=Path(file_name).suffix, delete=False
            ) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            processed_path, _ = maybe_preprocess(
                tmp_path,
                contrast=req_dict.get("preprocess_contrast", 1.0),
                sharpness=req_dict.get("preprocess_sharpness", 1.0),
                use_clahe=req_dict.get("preprocess_use_clahe", True),
                use_deskew=req_dict.get("preprocess_use_deskew", True),
                use_denoise=req_dict.get("preprocess_use_denoise", True),
            )
            pdf_bytes = Path(processed_path).read_bytes()
            for p in {tmp_path, processed_path}:   # set: tránh xóa trùng khi passthrough
                try:
                    os.unlink(p)
                except OSError:
                    pass

        # [2] VLM parse — async chạy trong sync context
        asyncio.run(aio_do_parse(
            pdf_bytes_list=[pdf_bytes],
            pdf_file_names=[file_name],
            output_dir=task_output_dir,      
            backend="vlm-auto-engine",
            start_page_id=req_dict.get("start_page") or 0,
            end_page_id=req_dict.get("end_page"),
            formula_enable=req_dict.get("formula_enable", True),
            table_enable=req_dict.get("table_enable", True),
            f_make_md_mode=req_dict.get("make_mode", "mm_markdown"),
            f_dump_md=req_dict.get("dump_md", True),
            f_dump_content_list=req_dict.get("dump_content_list", True),
            f_dump_middle_json=req_dict.get("dump_middle_json", False),
            f_dump_model_output=req_dict.get("dump_model_output", False),
            p_lang_list=["latin"],
        ))

        # [3] Hậu xử lý tiếng Việt
        if req_dict.get("vi_correct"):
            md_path = Path(task_output_dir) / stem / "vlm" / f"{stem}.md"
            if md_path.exists():
                text = md_path.read_text(encoding="utf-8")
                fixed = correct_md(text)
                if fixed != text:
                    md_path.write_text(fixed, encoding="utf-8")

        # [4] Thu thập output → trả về để Celery lưu vào PostgreSQL
        result = _collect(stem, task_output_dir)
        celery_logger.info(f"parse_task ({task_id}): done — collected output")
        return result

    except Exception as exc:
        celery_logger.error(f"parse_task ({task_id}): failed — {exc}", exc_info=True)
        raise

    finally:
        # [5] Dọn dẹp folder tạm sau khi kết quả đã được collect
        if Path(task_output_dir).exists():
            try:
                shutil.rmtree(str(Path(task_output_dir)))
                celery_logger.info(f"parse_task ({task_id}): cleaned up {task_output_dir}")
            except Exception as cleanup_err:
                celery_logger.error(f"parse_task ({task_id}): cleanup failed — {cleanup_err}")

        # Giải phóng VRAM và RAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def _collect(stem: str, task_output_dir: str) -> dict:
    base = Path(task_output_dir) / stem / "vlm"
    out: dict = {"markdown": None, "content_list": None, "images": {}}

    md = base / f"{stem}.md"
    if md.exists():
        out["markdown"] = md.read_text(encoding="utf-8")

    cl = base / f"{stem}_content_list.json"
    if cl.exists():
        out["content_list"] = json.loads(cl.read_text(encoding="utf-8"))

    img_dir = base / "images"
    if img_dir.exists():
        for img in img_dir.iterdir():
            if img.is_file():
                out["images"][img.name] = base64.b64encode(
                    img.read_bytes()
                ).decode()

    return out