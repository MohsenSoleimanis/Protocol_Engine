"""
Dual output renderer.

1. {name}_structured.json — full semantic structure with grounding for machine consumption
2. {name}_readable.md — human-readable view rendered from the JSON

The JSON is the source of truth. The markdown is a view.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from .models import (
    DocumentResult, ExtractedTable, Section, ContentBlock,
    ContentBlockType, RowType, Figure,
)


# ═══════════════════════════════════════════════════════════════════════════════
# JSON OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def render_json(result: DocumentResult, output_path: str) -> str:
    """Render full structured JSON with grounding."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use Pydantic's serialization
    data = result.model_dump(mode="json", exclude_none=True)
    
    # Write with pretty formatting
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return str(path)


# ═══════════════════════════════════════════════════════════════════════════════
# MARKDOWN OUTPUT (rendered from the model)
# ═══════════════════════════════════════════════════════════════════════════════

def render_markdown(result: DocumentResult, output_path: str) -> str:
    """Render human-readable markdown from the structured model."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    lines: list[str] = []
    
    # ── Document header ──────────────────────────────────────────────────────
    lines.append(f"# {result.filename}")
    lines.append("")
    lines.append(f"**Pages:** {result.total_pages}")
    lines.append(f"**Tables:** {result.total_tables}")
    lines.append(f"**Extracted:** {result.extraction_timestamp}")
    lines.append("")
    
    # ── Structure profile ────────────────────────────────────────────────────
    sp = result.structure_profile
    if sp.header_patterns or sp.footer_patterns:
        lines.append("## Document Structure Profile")
        lines.append("")
        if sp.page_number_format:
            lines.append(f"**Page numbers:** {sp.page_number_format}")
        if sp.section_numbering_scheme:
            lines.append(f"**Section numbering:** {sp.section_numbering_scheme}")
        if sp.body_font_size:
            lines.append(f"**Body font:** {sp.body_font_size}pt")
        lines.append(f"**Tables found:** {sp.tables_found}")
        if sp.continuation_merges_performed:
            lines.append(f"**Continuation merges:** {sp.continuation_merges_performed}")
        lines.append(f"**Abbreviations found:** {sp.abbreviation_definitions_found}")
        lines.append(f"**Cross-references found:** {sp.cross_references_found}")
        lines.append("")
    
    # ── Abbreviations ────────────────────────────────────────────────────────
    if result.abbreviations:
        lines.append("## Abbreviations")
        lines.append("")
        for abbr, defn in sorted(result.abbreviations.items()):
            lines.append(f"- **{abbr}**: {defn}")
        lines.append("")
    
    # ── Reference index ──────────────────────────────────────────────────────
    if result.reference_index:
        lines.append("## Reference Index")
        lines.append("")
        for key, ref in sorted(result.reference_index.items()):
            page_str = f" (p.{ref.page})" if ref.page else ""
            lines.append(f"- **{key}**: {ref.title}{page_str}")
        lines.append("")
    
    # ── Sections ─────────────────────────────────────────────────────────────
    if result.sections:
        lines.append("## Document Sections")
        lines.append("")
        for section in result.sections:
            indent = "  " * section.level
            page_str = f" [p.{section.page_range[0]}]" if section.page_range else ""
            lines.append(f"{indent}- **{section.number}** {section.title}{page_str}")
        lines.append("")
    
    # ── Tables ───────────────────────────────────────────────────────────────
    if result.tables:
        lines.append("## Tables")
        lines.append("")
        
        for table in result.tables:
            _render_table(table, lines)
    
    # ── Figures ──────────────────────────────────────────────────────────────
    if result.figures:
        lines.append("## Figures")
        lines.append("")
        for fig in result.figures:
            caption = fig.caption or "(no caption)"
            lines.append(f"- **{fig.id}**: {caption} [p.{fig.page}]")
        lines.append("")
    
    # ── Per-page content ─────────────────────────────────────────────────────
    lines.append("## Page Content")
    lines.append("")
    
    for page in result.pages:
        lines.append(f"### Page {page.page_num}")
        if page.page_display:
            lines.append(f"*{page.page_display}*")
        
        layout = "landscape" if page.is_landscape else "portrait"
        lines.append(
            f"*{layout} | verdict: {page.verdict.value} | "
            f"spans: {page.filtered_spans_count} | lines: {page.lines_count}*"
        )
        lines.append("")
        
        # Content blocks
        for block in page.content_blocks:
            if block.type == ContentBlockType.LIST:
                for item in block.list_items:
                    indent = "  " * item.level
                    lines.append(f"{indent}{item.marker} {item.text}")
                    for sub in item.sub_items:
                        lines.append(f"  {indent}{sub.marker} {sub.text}")
                lines.append("")
            elif block.type == ContentBlockType.PARAGRAPH:
                lines.append(block.text)
                lines.append("")
        
        # Page tables (reference to merged table)
        for table in page.tables:
            lines.append(f"*[Table: {table.id}]*")
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Write
    content = "\n".join(lines)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return str(path)


def _render_table(table: ExtractedTable, lines: list[str]):
    """Render a single table as markdown."""
    # Header
    page_str = f"pp. {table.page_range[0]}-{table.page_range[-1]}" if len(table.page_range) > 1 else f"p. {table.page_range[0]}" if table.page_range else ""
    merged_str = " (merged)" if table.is_continuation_merged else ""
    
    lines.append(f"### {table.id}{merged_str}")
    if table.caption:
        lines.append(f"**{table.caption}**")
    lines.append(f"*{page_str} | {table.extraction_method.value} | confidence: {table.confidence:.2f}*")
    lines.append("")
    
    if not table.rows:
        lines.append("*(empty table)*")
        lines.append("")
        return
    
    # Column groups
    if table.column_groups:
        group_labels = []
        for cg in table.column_groups:
            group_labels.append(f"{cg.label} (cols {cg.column_indices})")
        lines.append(f"Column groups: {', '.join(group_labels)}")
        lines.append("")
    
    # Markdown table
    max_cols = table.column_count or max(len(r) for r in table.rows)
    
    # Header row
    if table.column_headers:
        header = table.column_headers[:max_cols]
        while len(header) < max_cols:
            header.append("")
        lines.append("| " + " | ".join(_escape_md(h) for h in header) + " |")
        lines.append("| " + " | ".join("---" for _ in range(max_cols)) + " |")
    
    # Data rows with row type annotation
    for i, row in enumerate(table.rows):
        # Skip header row if already rendered above
        rm = next((m for m in table.row_metadata if m.row_index == i), None)
        if rm and rm.row_type == RowType.HEADER and table.column_headers:
            continue
        
        padded = list(row[:max_cols])
        while len(padded) < max_cols:
            padded.append("")
        
        row_prefix = ""
        if rm:
            if rm.row_type == RowType.GROUP_HEADER:
                row_prefix = "**"
            elif rm.row_type == RowType.WINDOW:
                row_prefix = "*"
        
        cells = []
        for c in padded:
            escaped = _escape_md(c)
            if row_prefix and c.strip():
                escaped = f"{row_prefix}{escaped}{row_prefix}"
            cells.append(escaped)
        
        lines.append("| " + " | ".join(cells) + " |")
    
    lines.append("")
    
    # Footnotes
    if table.footnotes:
        lines.append("**Footnotes:**")
        for marker, text in sorted(table.footnotes.items()):
            lines.append(f"  {marker}. {text}")
        lines.append("")


def _escape_md(text: str) -> str:
    """Escape pipe characters for markdown tables."""
    return text.replace("|", "\\|").replace("\n", " ")
