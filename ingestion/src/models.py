"""
Data models for Hybrid PDF Parser v3.

Design principle: EVERY extracted element carries its provenance —
page number, bounding box, extraction method, and confidence score.
This creates an audit trail so downstream LLMs can ground claims
and reviewers can verify in seconds.
"""

from __future__ import annotations
import hashlib
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────────────

class ExtractionMethod(str, Enum):
    PYMUPDF = "pymupdf"
    PDFPLUMBER = "pdfplumber"
    LLM_REPAIR = "llm_repair"
    HEURISTIC = "heuristic"
    MERGED = "merged"  # continuation merge


class Verdict(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    USE_PDFPLUMBER = "USE_PDFPLUMBER"
    NEEDS_LLM = "NEEDS_LLM"
    EMPTY = "EMPTY"


class RowType(str, Enum):
    HEADER = "header"
    WINDOW = "window"           # e.g. "±3 days" row in SoA
    GROUP_HEADER = "group_header"
    DATA = "data"
    SEPARATOR = "separator"


class ContentBlockType(str, Enum):
    PARAGRAPH = "paragraph"
    LIST = "list"
    DEFINITION = "definition"
    CAPTION = "caption"
    FOOTNOTE_BLOCK = "footnote_block"


# ── Source provenance (attached to everything) ───────────────────────────────

class Source(BaseModel):
    """Provenance for any extracted element."""
    page: int
    bbox: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1)
    extraction_method: ExtractionMethod = ExtractionMethod.PYMUPDF
    confidence: float = 1.0
    raw_text_hash: str | None = None  # SHA-256 of raw extracted text

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── Low-level extraction ─────────────────────────────────────────────────────

class TextSpan(BaseModel):
    """A single text span from PyMuPDF with full font metadata."""
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font_name: str = ""
    font_size: float = 0.0
    is_bold: bool = False
    is_italic: bool = False
    is_superscript: bool = False
    is_subscript: bool = False
    color: int = 0  # RGB as integer
    page_num: int = 0


class DrawingLine(BaseModel):
    """A ruling line (horizontal or vertical) from PDF drawings."""
    x0: float
    y0: float
    x1: float
    y1: float
    orientation: str = "unknown"  # "horizontal" | "vertical"
    page_num: int = 0


# ── Inline formatting (within body text) ─────────────────────────────────────

class InlineFormat(BaseModel):
    """A formatting range within a text string."""
    start: int    # char offset
    end: int      # char offset (exclusive)
    bold: bool = False
    italic: bool = False
    superscript: bool = False
    subscript: bool = False
    font_size: float | None = None


# ── Cross-reference ──────────────────────────────────────────────────────────

class CrossReference(BaseModel):
    """A cross-reference found in text (e.g., 'see Section 8.2.1')."""
    text: str            # the reference text as it appears
    target_type: str     # "section" | "table" | "figure" | "appendix"
    target_id: str       # e.g. "8.2.1", "Table 5", "Appendix E"
    char_start: int = 0  # position in parent text
    char_end: int = 0


# ── List item (within body text) ─────────────────────────────────────────────

class ListItem(BaseModel):
    """A single item in a structured list."""
    marker: str = ""         # "1.", "a)", "•", "-"
    text: str = ""
    level: int = 0           # 0 = top level, 1 = sub-item
    sub_items: list[ListItem] = Field(default_factory=list)
    source: Source | None = None


# ── Content block (paragraph, list, definition) ─────────────────────────────

class ContentBlock(BaseModel):
    """A block of content within a section — paragraph, list, or definition."""
    type: ContentBlockType
    text: str = ""                # full text (for paragraphs)
    inline_formats: list[InlineFormat] = Field(default_factory=list)
    list_items: list[ListItem] = Field(default_factory=list)  # for LIST type
    cross_references: list[CrossReference] = Field(default_factory=list)
    source: Source | None = None


# ── Table models (with full grounding) ───────────────────────────────────────

class ColumnGroup(BaseModel):
    """A group of columns under a shared header (multi-level headers)."""
    label: str
    column_indices: list[int]   # which columns belong to this group
    source: Source | None = None


class TableCell(BaseModel):
    """A single cell with full provenance."""
    row: int
    col: int
    text: str
    row_span: int = 1
    col_span: int = 1
    footnote_refs: list[str] = Field(default_factory=list)  # ["a", "c"]
    source: Source | None = None


class RowMetadata(BaseModel):
    """Metadata about a table row."""
    row_index: int
    row_type: RowType = RowType.DATA
    source_page: int = 0
    group_label: str | None = None  # which group header this row falls under


class TableFootnote(BaseModel):
    """A footnote belonging to a specific table."""
    marker: str          # "a", "b", "1", "*"
    text: str
    source: Source | None = None


class ExtractedTable(BaseModel):
    """A fully extracted table with all semantic metadata and grounding."""
    id: str = ""                    # auto-generated: "table_p20_1"
    caption: str | None = None
    caption_source: Source | None = None
    section_id: str | None = None   # which section this table belongs to
    page_range: list[int] = Field(default_factory=list)
    is_continuation_merged: bool = False

    # Structure
    column_groups: list[ColumnGroup] = Field(default_factory=list)
    column_headers: list[str] = Field(default_factory=list)
    column_count: int = 0
    row_metadata: list[RowMetadata] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    cells: list[TableCell] = Field(default_factory=list)

    # Footnotes scoped to this table
    footnotes: dict[str, str] = Field(default_factory=dict)  # marker → text
    footnote_sources: list[TableFootnote] = Field(default_factory=list)

    # Provenance
    extraction_method: ExtractionMethod = ExtractionMethod.PDFPLUMBER
    confidence: float = 0.9
    source_hash: str | None = None  # hash of all raw cell text combined


# ── Figure placeholder ───────────────────────────────────────────────────────

class Figure(BaseModel):
    """A detected figure/image region (content not extractable)."""
    id: str = ""
    caption: str | None = None
    page: int = 0
    bbox: tuple[float, float, float, float] | None = None
    caption_source: Source | None = None


# ── Section tree ─────────────────────────────────────────────────────────────

class Section(BaseModel):
    """A section in the document hierarchy."""
    id: str                     # "sec_1_2_1"
    number: str                 # "1.2.1"
    title: str
    level: int                  # depth: 0 = top, 1, 2, 3...
    parent_id: str | None = None
    page_range: list[int] = Field(default_factory=list)
    content_blocks: list[ContentBlock] = Field(default_factory=list)
    source: Source | None = None


# ── Document structure profile (what we discovered) ──────────────────────────

class DocumentStructureProfile(BaseModel):
    """Metadata about the document's discovered structure conventions."""
    header_patterns: list[str] = Field(default_factory=list)
    footer_patterns: list[str] = Field(default_factory=list)
    page_number_format: str = ""                  # e.g. "{n} of {total}"
    total_pages_declared: int | None = None
    section_numbering_scheme: str = ""            # "numeric_dotted" | "roman" | "letter"
    body_font_size: float | None = None
    heading_min_font_size: float | None = None
    footnote_marker_style: str = ""               # "lowercase_letter" | "number" | "symbol"
    tables_found: int = 0
    continuation_merges_performed: int = 0
    abbreviation_definitions_found: int = 0
    cross_references_found: int = 0
    figures_found: int = 0
    lists_found: int = 0


# ── Reference index ──────────────────────────────────────────────────────────

class ReferenceEntry(BaseModel):
    """An entry in the document's cross-reference index."""
    target_type: str      # "section" | "table" | "figure" | "appendix"
    target_id: str        # "8.2.1", "Table 5"
    title: str = ""
    page: int | None = None


# ── Page result ──────────────────────────────────────────────────────────────

class PageResult(BaseModel):
    """Extraction result for a single page."""
    page_num: int
    page_display: str = ""
    width: float
    height: float
    is_landscape: bool = False
    verdict: Verdict = Verdict.SUFFICIENT
    tables: list[ExtractedTable] = Field(default_factory=list)
    content_blocks: list[ContentBlock] = Field(default_factory=list)
    figures: list[Figure] = Field(default_factory=list)
    raw_spans_count: int = 0
    filtered_spans_count: int = 0
    lines_count: int = 0
    llm_calls: int = 0


# ── Full document output ─────────────────────────────────────────────────────

class DocumentResult(BaseModel):
    """Complete parsed document with full grounding."""
    # Metadata
    filename: str
    total_pages: int
    extraction_timestamp: str = ""  # ISO format

    # Discovered structure
    structure_profile: DocumentStructureProfile = Field(
        default_factory=DocumentStructureProfile
    )

    # Reference index (for grounding downstream LLM)
    reference_index: dict[str, ReferenceEntry] = Field(default_factory=dict)

    # Abbreviations discovered from inline definitions + abbreviation tables
    abbreviations: dict[str, str] = Field(default_factory=dict)

    # Section tree
    sections: list[Section] = Field(default_factory=list)

    # Tables (after continuation merging)
    tables: list[ExtractedTable] = Field(default_factory=list)

    # Figures
    figures: list[Figure] = Field(default_factory=list)

    # Per-page results (for debugging / audit)
    pages: list[PageResult] = Field(default_factory=list)

    # Stats
    total_tables: int = 0
    total_llm_calls: int = 0
