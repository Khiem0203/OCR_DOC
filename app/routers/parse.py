import base64
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from celery.result import AsyncResult
from app.celery_app import celery_app
from app.tasks import parse_task
from app.schemas.request import ParseRequest, MakeMode
from app.schemas.response import ParseResponse, JobStatus

router = APIRouter(prefix="/parse", tags=["parse"])

# Celery state → JobStatus
_STATE_MAP = {
    "PENDING": JobStatus.pending,
    "STARTED": JobStatus.processing,
    "RETRY":   JobStatus.processing,
    "SUCCESS": JobStatus.done,
    "FAILURE": JobStatus.error,
}


@router.post("", response_model=ParseResponse, status_code=202)
async def post_parse(
    file:                   UploadFile    = File(...),
    formula_enable:         bool          = Form(True),
    table_enable:           bool          = Form(True),
    start_page:             Optional[int] = Form(None),
    end_page:               Optional[int] = Form(None),
    make_mode:              MakeMode      = Form(MakeMode.mm_md),
    preprocess:             bool          = Form(False),
    preprocess_contrast:    float         = Form(1.0),
    preprocess_sharpness:   float         = Form(1.0),
    preprocess_use_clahe:   bool          = Form(False),
    preprocess_use_deskew:  bool          = Form(False),
    preprocess_use_denoise: bool          = Form(False),
    vi_correct:             bool          = Form(False),
    dump_md:                bool          = Form(True),
    dump_content_list:      bool          = Form(True),
    dump_middle_json:       bool          = Form(False),
    dump_model_output:      bool          = Form(False),
):
    """Submit a PDF file for async parsing.  Returns a `task_id` to poll later."""
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file")

    req = ParseRequest(
        file_name=file.filename or "upload.pdf",
        formula_enable=formula_enable,
        table_enable=table_enable,
        start_page=start_page,
        end_page=end_page,
        make_mode=make_mode,
        preprocess=preprocess,
        preprocess_contrast=preprocess_contrast,
        preprocess_sharpness=preprocess_sharpness,
        preprocess_use_clahe=preprocess_use_clahe,
        preprocess_use_deskew=preprocess_use_deskew,
        preprocess_use_denoise=preprocess_use_denoise,
        vi_correct=vi_correct,
        dump_md=dump_md,
        dump_content_list=dump_content_list,
        dump_middle_json=dump_middle_json,
        dump_model_output=dump_model_output,
    )

    # PDF bytes → base64 để truyền qua JSON serializer của Celery
    pdf_b64 = base64.b64encode(data).decode()
    task = parse_task.delay(pdf_b64, req.model_dump())

    return ParseResponse(task_id=task.id, status=JobStatus.pending)


@router.get("/{task_id}", response_model=ParseResponse)
def get_parse(task_id: str):
    """Poll the result of a previously submitted parse job."""
    result = AsyncResult(task_id, app=celery_app)
    status = _STATE_MAP.get(result.state, JobStatus.pending)

    if result.state == "SUCCESS":
        r = result.result or {}
        return ParseResponse(
            task_id=task_id,
            status=status,
            markdown=r.get("markdown"),
            content_list=r.get("content_list"),
            images=r.get("images"),
        )

    if result.state == "FAILURE":
        return ParseResponse(
            task_id=task_id,
            status=status,
            error=str(result.result),
        )

    return ParseResponse(task_id=task_id, status=status)
