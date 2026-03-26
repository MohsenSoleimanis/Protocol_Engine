"""
Quality gate: decides extraction strategy per page.

Verdicts:
- SUFFICIENT: text-only page, no tables → PyMuPDF spans only
- USE_PDFPLUMBER: has ruling lines → deterministic table extraction
- NEEDS_LLM: pdfplumber output has issues → send compact skeleton to GPT-4o-mini
- EMPTY: no content
"""

from __future__ import annotations
from .models import Verdict, TextSpan, DrawingLine, ExtractedTable


def assess_page_verdict(
    spans: list[TextSpan],
    lines: list[DrawingLine],
    page_width: float,
    page_height: float,
    is_landscape: bool = False,
) -> Verdict:
    """
    Page-level verdict BEFORE extraction.
    """
    if not spans:
        return Verdict.EMPTY

    # Has ruling lines → table page → use pdfplumber
    h_lines = [l for l in lines if l.orientation == "horizontal"]
    v_lines = [l for l in lines if l.orientation == "vertical"]
    
    if len(h_lines) >= 3 and len(v_lines) >= 2:
        return Verdict.USE_PDFPLUMBER
    
    # Dense lines (even without clear h/v distinction)
    if len(lines) > 20:
        return Verdict.USE_PDFPLUMBER

    return Verdict.SUFFICIENT


def validate_pdfplumber_output(table: ExtractedTable) -> Verdict:
    """
    Post-extraction validation of pdfplumber results.
    If quality is low, escalate to LLM repair.
    """
    if not table.rows:
        return Verdict.NEEDS_LLM
    
    issues = 0
    total_cells = sum(len(row) for row in table.rows)
    
    if total_cells == 0:
        return Verdict.NEEDS_LLM
    
    # Check: too many empty cells
    empty_cells = sum(1 for row in table.rows for c in row if not c.strip())
    if total_cells > 0 and empty_cells / total_cells > 0.7:
        issues += 1
    
    # Check: inconsistent column counts
    col_counts = set(len(row) for row in table.rows)
    if len(col_counts) > 2:
        issues += 1
    
    # Check: very few rows (might be a misdetection)
    if len(table.rows) < 2:
        issues += 1
    
    # Check: single column (probably not a real table)
    if table.column_count <= 1:
        issues += 1
    
    if issues >= 2:
        return Verdict.NEEDS_LLM
    
    return Verdict.SUFFICIENT
