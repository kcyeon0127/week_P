from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import hashlib

def make_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]

class Doc(BaseModel):
    doc_id: str
    url: Optional[str] = None
    title: str
    lang: str = "ko"
    doctype: str = "web" # web|pdf|faq|artifact
    section: Optional[str] = None
    published_at: Optional[str] = None
    last_seen: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d"))
    text: str
    tables: Optional[List[Dict[str, str]]] = None
    breadcrumbs: Optional[List[str]] = None
    license_note: Optional[str] = None

class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    url: Optional[str] = None
    section: Optional[str] = None
    doctype: str = "web"
    lang: str = "ko"
    chunk_index: int
    char_range: List[int]
    text: str
    meta: Dict[str, Any] = Field(default_factory=dict)
