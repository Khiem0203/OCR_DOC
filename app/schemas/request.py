from pydantic import BaseModel
from typing import Optional
from enum import Enum

class MakeMode(str, Enum):
    mm_md           = "mm_markdown"
    nlp_md          = "nlp_markdown"
    content_list    = "content_list"
    content_list_v2 = "content_list_v2"

class ParseRequest(BaseModel):
    file_name: str
    start_page: Optional[int] = None
    end_page:   Optional[int] = None
    make_mode:  MakeMode = MakeMode.mm_md
    formula_enable: bool = True
    table_enable:   bool = True

    preprocess: bool = False
    preprocess_contrast:   float = 1.0
    preprocess_sharpness:  float = 1.0
    preprocess_use_clahe:  bool  = False
    preprocess_use_deskew: bool  = False
    preprocess_use_denoise: bool = False

    vi_correct: bool = False

    dump_md:           bool = True
    dump_content_list: bool = True
    dump_middle_json:  bool = False
    dump_model_output: bool = False