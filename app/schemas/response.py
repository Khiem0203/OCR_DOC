from pydantic import BaseModel
from typing import Optional, Dict
from enum import Enum


class JobStatus(str, Enum):
    pending    = "pending"
    processing = "processing"
    done       = "done"
    error      = "error"


class ParseResponse(BaseModel):
    task_id:      str
    status:       JobStatus
    markdown:     Optional[str]            = None
    content_list: Optional[list]           = None
    images:       Optional[Dict[str, str]] = None
    error:        Optional[str]            = None
