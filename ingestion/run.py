#!/usr/bin/env python3
"""
Hybrid PDF Parser v3 — Generic Document Intelligence Engine.

Pipeline:
  1. Scan document for header/footer patterns
  2. For each page:
     a. Extract raw spans + lines (PyMuPDF)
     b. Filter headers/footers/page numbers
     c. Assess verdict: SUFFICIENT (text) / USE_PDFPLUMBER (tables) / NEEDS_LLM
     d. Extract tables via pdfplumber (deterministic, fast, free)
     e. Validate pdfplumber output → optional LLM repair (compact skeleton, <$0.001)
     f. Extract body text with inline formatting, lists, cross-references
  3. Cross-page analysis:
     a. Merge continuation tables
     b. Discover section hierarchy
     c. Associate captions → tables
     d. Associate footnotes → tables
     e. Discover abbreviation definitions
     f. Build cross-reference index
  4. Output:
     a. {name}_structured.json — full semantic model with grounding
     b. {name}_readable.md — human-readable view

Usage:
    python run.py input.pdf                     # Full pipeline (no API needed!)
    python run.py input.pdf --with-llm          # Enable LLM repair for low-quality tables
    python run.py input.pdf --pages 19-26       # Specific pages
    python run.py input.pdf -o output/result     # Custom output prefix
    python run.py input.pdf --verbose            # Show detailed per-page info
"""

from __future__ import annotations
import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

import fitz  # PyMuPDF
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table as RichTable

from src.models import (
    Verdict, ExtractionMethod, Source,
    PageResult, DocumentResult, DocumentStructureProfile,
    ExtractedTable, Section, ContentBlock, ContentBlockType,
    Figure, TextSpan, InlineFormat, CrossReference,
)
from src.extractor import (
    extract_text_spans, extract_drawing_lines, extract_images_info,
    reconstruct_line_text,
)
from src.header_footer_filter import (
    detect_header_footer_patterns, extract_page_meta,
    filter_header_footer_spans,
)
from src.table_extractor import extract_tables_from_page
from src.validator import assess_page_verdict, validate_pdfplumber_output
from src.llm_repair import repair_table_with_llm
from src.continuation_merger import merge_continuation_tables
from src.structure_discovery import (
    discover_section_hierarchy, detect_body_font_size,
    assign_blocks_to_sections,
    discover_table_captions, discover_footnotes_for_table,
    discover_abbreviations, discover_cross_references,
    discover_lists_in_spans, extract_inline_formats,
    build_reference_index,
)
from src.output_renderer import render_json, render_markdown

console = Console()
logger = logging.getLogger(__name__)


def parse_page_range(page_str: str, total_pages: int) -> list[int]:
    """Parse '1-5' or '1,3,5' or '1-3,7' into 0-indexed page numbers."""
    pages: set[int] = set()
    for part in page_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start_i = max(0, int(start) - 1)
            end_i = min(total_pages, int(end))
            pages.update(range(start_i, end_i))
        else:
            idx = int(part) - 1
            if 0 <= idx < total_pages:
                pages.add(idx)
    return sorted(pages)


def process_document(
    pdf_path: str,
    output_prefix: str | None = None,
    page_range: str | None = None,
    with_llm: bool = False,
    verbose: bool = False,
) -> DocumentResult:
    """Main pipeline."""
    load_dotenv()

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        console.print(f"[bold red]Error:[/] File not found: {pdf_path}")
        sys.exit(1)

    if output_prefix is None:
        output_prefix = f"output/{pdf_path.stem}"

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    if page_range:
        page_indices = parse_page_range(page_range, total_pages)
    else:
        page_indices = list(range(total_pages))

    llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    llm_status = f"[green]enabled[/] ({llm_model})" if with_llm else "[dim]disabled[/] (deterministic only)"

    console.print(f"\n[bold cyan]Hybrid PDF Parser v3 — Generic Document Intelligence[/]")
    console.print(f"  File: {pdf_path.name} ({total_pages} pages)")
    console.print(f"  Processing: {len(page_indices)} pages")
    console.print(f"  LLM repair: {llm_status}")
    console.print()

    t_start = time.time()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Document-level scan
    # ══════════════════════════════════════════════════════════════════════════
    console.print("  [dim]Phase 1: Document scan...[/]", end=" ")

    hf_patterns, page_num_format = detect_header_footer_patterns(
        doc, sample_pages=min(30, total_pages)
    )
    console.print(f"found {len(hf_patterns)} header/footer patterns")

    if verbose and hf_patterns:
        for text, y in hf_patterns.items():
            console.print(f"    [dim]y={y:.0f}: \"{text[:60]}\"[/]")

    # Collect all spans for body font detection
    sample_spans: list[TextSpan] = []
    for i in page_indices[:20]:
        page = doc[i]
        spans = extract_text_spans(page, i + 1)
        sample_spans.extend(spans)

    body_font_size = detect_body_font_size(sample_spans)
    console.print(f"  [dim]Body font size: {body_font_size}pt[/]")
    console.print()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Per-page extraction
    # ══════════════════════════════════════════════════════════════════════════
    console.print("  [dim]Phase 2: Per-page extraction...[/]")

    page_results: list[PageResult] = []
    all_page_spans: dict[int, list[TextSpan]] = {}
    tables_by_page: dict[int, list[ExtractedTable]] = defaultdict(list)
    page_heights: dict[int, float] = {}
    total_llm_calls = 0
    figures_all: list[Figure] = []

    for page_idx in page_indices:
        page = doc[page_idx]
        page_num = page_idx + 1
        pw, ph = page.rect.width, page.rect.height
        is_landscape = pw > ph
        page_heights[page_num] = ph

        # 2a. Raw extraction
        raw_spans = extract_text_spans(page, page_num)
        lines = extract_drawing_lines(page, page_num)

        # 2b. Filter headers/footers
        filtered_spans, removed_spans, meta = filter_header_footer_spans(
            raw_spans, ph, hf_patterns,
        )

        # 2c. Detect figures (image regions)
        images = extract_images_info(page, page_num)
        for img in images:
            fig = Figure(
                id=f"fig_p{page_num}_{len(figures_all) + 1}",
                page=page_num,
                bbox=img["bbox"],
            )
            figures_all.append(fig)

        # Store for cross-page analysis
        all_page_spans[page_num] = filtered_spans

        # 2d. Assess page
        verdict = assess_page_verdict(filtered_spans, lines, pw, ph, is_landscape)

        if verbose:
            console.print(
                f"    Page {page_num:3d}: {len(filtered_spans):4d} spans, "
                f"{len(lines):3d} lines, "
                f"{'L' if is_landscape else 'P'} → "
                f"[{'green' if verdict == Verdict.SUFFICIENT else 'yellow'}]"
                f"{verdict.value}[/]"
            )

        # 2e. Extract tables (pdfplumber path)
        page_tables: list[ExtractedTable] = []

        if verdict == Verdict.USE_PDFPLUMBER:
            page_tables = extract_tables_from_page(
                str(pdf_path), page_idx, page_num
            )

            # Validate each table
            validated_tables: list[ExtractedTable] = []
            for table in page_tables:
                table_verdict = validate_pdfplumber_output(table)
                if table_verdict == Verdict.NEEDS_LLM and with_llm:
                    if verbose:
                        console.print(f"      [yellow]→ LLM repair: {table.id}[/]")
                    table = repair_table_with_llm(table, page_num)
                    total_llm_calls += 1
                validated_tables.append(table)
            
            page_tables = validated_tables
            tables_by_page[page_num] = page_tables

        # 2f. Extract body text (non-table spans)
        content_blocks: list[ContentBlock] = []

        non_table_spans = _exclude_table_spans(filtered_spans, page_tables)

        if non_table_spans:
            # Detect lists
            lists = discover_lists_in_spans(non_table_spans, body_font_size, page_num)
            content_blocks.extend(lists)

            # Remaining text as paragraphs
            paragraphs = _group_into_paragraphs(
                non_table_spans, body_font_size, page_num
            )
            content_blocks.extend(paragraphs)

        page_results.append(PageResult(
            page_num=page_num,
            page_display=meta.page_num_display,
            width=pw,
            height=ph,
            is_landscape=is_landscape,
            verdict=verdict,
            tables=page_tables,
            content_blocks=content_blocks,
            figures=[f for f in figures_all if f.page == page_num],
            raw_spans_count=len(raw_spans),
            filtered_spans_count=len(filtered_spans),
            lines_count=len(lines),
            llm_calls=total_llm_calls,
        ))

    doc.close()

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3: Cross-page analysis
    # ══════════════════════════════════════════════════════════════════════════
    console.print()
    console.print("  [dim]Phase 3: Cross-page analysis...[/]")

    # 3a. Merge continuation tables
    all_tables, merge_count = merge_continuation_tables(tables_by_page, page_heights)
    console.print(f"    Continuation merges: {merge_count}")

    # 3b. Discover section hierarchy
    sections, numbering_scheme = discover_section_hierarchy(
        all_page_spans, body_font_size
    )
    console.print(f"    Sections discovered: {len(sections)} ({numbering_scheme})")

    # 3b.5. Assign content blocks to sections by position
    blocks_assigned = assign_blocks_to_sections(sections, page_results)
    console.print(f"    Content blocks assigned to sections: {blocks_assigned}")

    # 3c. Associate captions → tables
    captions = discover_table_captions(all_page_spans, tables_by_page)
    for table_id, (caption_text, caption_source) in captions.items():
        for table in all_tables:
            if table.id == table_id:
                table.caption = caption_text
                table.caption_source = caption_source
    console.print(f"    Captions associated: {len(captions)}")

    # 3d. Discover footnotes for each table
    fn_count = 0
    for page_num, page_tables in tables_by_page.items():
        spans = all_page_spans.get(page_num, [])
        for table in page_tables:
            footnotes = discover_footnotes_for_table(
                spans, table, body_font_size, page_num
            )
            if footnotes:
                for marker, fn in footnotes.items():
                    table.footnotes[marker] = fn.text
                    table.footnote_sources.append(fn)
                fn_count += len(footnotes)
    console.print(f"    Footnotes discovered: {fn_count}")

    # 3e. Discover abbreviations
    abbreviations = discover_abbreviations(all_page_spans, all_tables)
    console.print(f"    Abbreviations: {len(abbreviations)}")

    # 3f. Count cross-references
    xref_count = 0
    for page in page_results:
        for block in page.content_blocks:
            if block.text:
                refs = discover_cross_references(block.text)
                block.cross_references = refs
                xref_count += len(refs)
    console.print(f"    Cross-references: {xref_count}")

    # 3g. Build reference index
    reference_index = build_reference_index(sections, all_tables, figures_all)
    console.print(f"    Reference index entries: {len(reference_index)}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 4: Assemble output
    # ══════════════════════════════════════════════════════════════════════════
    console.print()
    console.print("  [dim]Phase 4: Rendering output...[/]")

    profile = DocumentStructureProfile(
        header_patterns=list(hf_patterns.keys()),
        footer_patterns=[p for p, y in hf_patterns.items()
                         if y > min(page_heights.values(), default=800) * 0.8],
        page_number_format=page_num_format,
        section_numbering_scheme=numbering_scheme,
        body_font_size=body_font_size,
        tables_found=len(all_tables),
        continuation_merges_performed=merge_count,
        abbreviation_definitions_found=len(abbreviations),
        cross_references_found=xref_count,
        figures_found=len(figures_all),
    )

    result = DocumentResult(
        filename=pdf_path.name,
        total_pages=total_pages,
        extraction_timestamp=datetime.now(timezone.utc).isoformat(),
        structure_profile=profile,
        reference_index=reference_index,
        abbreviations=abbreviations,
        sections=sections,
        tables=all_tables,
        figures=figures_all,
        pages=page_results,
        total_tables=len(all_tables),
        total_llm_calls=total_llm_calls,
    )

    # Render dual output
    json_path = render_json(result, f"{output_prefix}_structured.json")
    md_path = render_markdown(result, f"{output_prefix}_readable.md")

    elapsed = time.time() - t_start

    # ── Summary ──────────────────────────────────────────────────────────────
    console.print()
    summary = RichTable(title="Pipeline Summary", show_header=False, border_style="cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Value")
    summary.add_row("Pages processed", str(len(page_results)))
    summary.add_row("Tables extracted", str(len(all_tables)))
    summary.add_row("Continuation merges", str(merge_count))
    summary.add_row("Sections discovered", str(len(sections)))
    summary.add_row("Abbreviations", str(len(abbreviations)))
    summary.add_row("Cross-references", str(xref_count))
    summary.add_row("Figures detected", str(len(figures_all)))
    summary.add_row("LLM calls", str(total_llm_calls))

    verdict_counts: dict[str, int] = {}
    for p in page_results:
        v = p.verdict.value
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    summary.add_row("Page verdicts", str(verdict_counts))
    summary.add_row("Time", f"{elapsed:.2f}s")
    summary.add_row("JSON output", json_path)
    summary.add_row("Markdown output", md_path)
    console.print(summary)
    console.print()

    return result


def _exclude_table_spans(
    spans: list[TextSpan],
    tables: list[ExtractedTable],
) -> list[TextSpan]:
    """Remove spans that fall inside any table's bounding box."""
    if not tables:
        return spans

    table_bboxes: list[tuple[float, float, float, float]] = []
    for table in tables:
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        for cell in table.cells:
            if cell.source and cell.source.bbox:
                b = cell.source.bbox
                min_x = min(min_x, b[0])
                min_y = min(min_y, b[1])
                max_x = max(max_x, b[2])
                max_y = max(max_y, b[3])
        if min_x < float('inf'):
            table_bboxes.append((min_x - 5, min_y - 5, max_x + 5, max_y + 5))

    if not table_bboxes:
        return spans

    result: list[TextSpan] = []
    for span in spans:
        inside = False
        for tb in table_bboxes:
            if (span.x0 >= tb[0] and span.y0 >= tb[1]
                    and span.x1 <= tb[2] and span.y1 <= tb[3]):
                inside = True
                break
        if not inside:
            result.append(span)
    return result


def _group_into_paragraphs(
    spans: list[TextSpan],
    body_font_size: float,
    page_num: int,
) -> list[ContentBlock]:
    """Group text spans into paragraph blocks by vertical proximity."""
    if not spans:
        return []

    sorted_spans = sorted(spans, key=lambda s: (round(s.y0, 1), s.x0))

    paragraphs: list[ContentBlock] = []
    current_spans: list[TextSpan] = [sorted_spans[0]]
    prev_y = sorted_spans[0].y1

    for span in sorted_spans[1:]:
        gap = span.y0 - prev_y
        if gap > body_font_size * 1.5:
            if current_spans:
                text, formats = extract_inline_formats(current_spans, body_font_size)
                xrefs = discover_cross_references(text)
                first = current_spans[0]
                last = current_spans[-1]
                paragraphs.append(ContentBlock(
                    type=ContentBlockType.PARAGRAPH,
                    text=text.strip(),
                    inline_formats=formats,
                    cross_references=xrefs,
                    source=Source(
                        page=page_num,
                        bbox=(first.x0, first.y0, last.x1, last.y1),
                        extraction_method=ExtractionMethod.PYMUPDF,
                        confidence=0.90,
                        raw_text_hash=Source.hash_text(text.strip()),
                    ),
                ))
            current_spans = []
        current_spans.append(span)
        prev_y = span.y1

    if current_spans:
        text, formats = extract_inline_formats(current_spans, body_font_size)
        xrefs = discover_cross_references(text)
        first = current_spans[0]
        last = current_spans[-1]
        paragraphs.append(ContentBlock(
            type=ContentBlockType.PARAGRAPH,
            text=text.strip(),
            inline_formats=formats,
            cross_references=xrefs,
            source=Source(
                page=page_num,
                bbox=(first.x0, first.y0, last.x1, last.y1),
                extraction_method=ExtractionMethod.PYMUPDF,
                confidence=0.90,
                raw_text_hash=Source.hash_text(text.strip()),
            ),
        ))

    return paragraphs


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid PDF Parser v3 — Generic Document Intelligence Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("-o", "--output", help="Output prefix (without extension)")
    parser.add_argument("--pages", help="Page range: '1-5', '19-26', '1,3,5'")
    parser.add_argument("--with-llm", action="store_true",
                        help="Enable LLM repair for low-quality tables")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed per-page extraction info")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(name)s | %(levelname)s | %(message)s",
    )

    process_document(
        pdf_path=args.pdf,
        output_prefix=args.output,
        page_range=args.pages,
        with_llm=args.with_llm,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
