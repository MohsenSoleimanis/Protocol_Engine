"""
Minimal ingestion models — self-contained fallback when old ingestion.src.models
is not importable. These mirror the essential classes from the old code.
"""
from __future__ import annotations

import hashlib
from enum import Enum
from pydantic import BaseModel, Field


class ExtractionMethod(str, Enum):
    PYMUPDF = "pymupdf"
    PDFPLUMBER = "pdfplumber"
    LLM_REPAIR = "llm_repair"
    HEURISTIC = "heuristic"
    MERGED = "merged"


class RowType(str, Enum):
    HEADER = "header"
    WINDOW = "window"
    GROUP_HEADER = "group_header"
    DATA = "data"
    SEPARATOR = "separator"


class Source(BaseModel):
    page: int
    bbox: tuple[float, float, float, float] | None = None
    extraction_method: ExtractionMethod = ExtractionMethod.PYMUPDF
    confidence: float = 1.0
    raw_text_hash: str | None = None

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class TableCell(BaseModel):
    row: int
    col: int
    text: str
    row_span: int = 1
    col_span: int = 1
    footnote_refs: list[str] = Field(default_factory=list)
    source: Source | None = None


class RowMetadata(BaseModel):
    row_index: int
    row_type: RowType = RowType.DATA
    source_page: int = 0
    group_label: str | None = None


class ColumnGroup(BaseModel):
    label: str
    column_indices: list[int]
    source: Source | None = None


class ExtractedTable(BaseModel):
    id: str = ""
    caption: str | None = None
    caption_source: Source | None = None
    section_id: str | None = None
    page_range: list[int] = Field(default_factory=list)
    is_continuation_merged: bool = False
    column_groups: list[ColumnGroup] = Field(default_factory=list)
    column_headers: list[str] = Field(default_factory=list)
    column_count: int = 0
    row_metadata: list[RowMetadata] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    cells: list[TableCell] = Field(default_factory=list)
    footnotes: dict[str, str] = Field(default_factory=dict)
    extraction_method: ExtractionMethod = ExtractionMethod.PDFPLUMBER
    confidence: float = 0.9
    source_hash: str | None = None
