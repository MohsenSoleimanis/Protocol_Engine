"""
Ingestion Agent — Agentic pipeline with nodes for each parser phase.

Instead of calling process_document() as a black box, the Ingestion Agent
orchestrates each phase as a visible node, making decisions at each step:

  Node 1: Document Scan     → discovers patterns, detects body font
  Node 2: Per-Page Extract  → per-page verdicts, triggers LLM repair where needed
  Node 3: Cross-Page        → merges tables, discovers sections/footnotes/abbreviations
  Node 4: Output            → writes _structured.json + _readable.md
  Node 5: Build Manifest    → domain classification, section mapping for retrieval

Each node reports progress via on_step callback so the UI shows live updates.
"""

import sys
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from typing import Callable, Optional

# Add this directory to sys.path so parser's internal imports work
_ingestion_dir = str(Path(__file__).parent)
if _ingestion_dir not in sys.path:
    sys.path.insert(0, _ingestion_dir)

import fitz  # PyMuPDF
from dotenv import load_dotenv

from src.models import (
    Verdict, ExtractionMethod, Source,
    PageResult, DocumentResult, DocumentStructureProfile,
    ExtractedTable, Section, ContentBlock, ContentBlockType,
    Figure, TextSpan, InlineFormat, CrossReference,
)
from src.extractor import (
    extract_text_spans, extract_drawing_lines, extract_images_info,
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
    discover_table_captions, discover_footnotes_for_table,
    discover_abbreviations, discover_cross_references,
    discover_lists_in_spans, extract_inline_formats,
    build_reference_index,
)
from src.output_renderer import render_json, render_markdown

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Agentic ingestion pipeline. Each phase is a node that:
    - Runs the actual parser logic (your code, untouched)
    - Reports progress via on_step callback
    - Makes decisions (which pages need LLM, which tables to merge)
    - Passes state to the next node
    """

    def __init__(self, on_step: Callable[[dict], None] | None = None):
        self.on_step = on_step
        self._steps = []

    def _report(self, phase: str, detail: str, status: str = "running"):
        step = {"phase": phase, "detail": detail, "status": status, "ts": time.time()}
        self._steps.append(step)
        if self.on_step:
            self.on_step(step)
        logger.info(f"[{phase}] {detail}")

    def run(
        self,
        pdf_path: str,
        output_prefix: str,
        with_llm: bool = True,
        verbose: bool = False,
    ) -> tuple:
        """
        Run the full agentic ingestion pipeline.

        Returns: (DocumentResult, json_path, steps)
        """
        load_dotenv()
        t_start = time.time()
        self._steps = []

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        page_indices = list(range(total_pages))

        self._report("init", f"Opened {pdf_path.name}: {total_pages} pages")

        # ═══ NODE 1: Document Scan ═══════════════════════════════════════
        self._report("phase1", "Scanning for header/footer patterns...")

        hf_patterns, page_num_format = detect_header_footer_patterns(
            doc, sample_pages=min(30, total_pages)
        )

        sample_spans = []
        for i in page_indices[:20]:
            page = doc[i]
            spans = extract_text_spans(page, i + 1)
            sample_spans.extend(spans)

        body_font_size = detect_body_font_size(sample_spans)

        self._report("phase1",
            f"Found {len(hf_patterns)} header/footer patterns, body font: {body_font_size}pt",
            "done")

        # ═══ NODE 2: Per-Page Extraction ═════════════════════════════════
        self._report("phase2", f"Extracting {total_pages} pages...")

        page_results = []
        all_page_spans = {}
        tables_by_page = defaultdict(list)
        page_heights = {}
        total_llm_calls = 0
        figures_all = []
        verdict_counts = defaultdict(int)

        for page_idx in page_indices:
            page = doc[page_idx]
            page_num = page_idx + 1
            pw, ph = page.rect.width, page.rect.height
            is_landscape = pw > ph
            page_heights[page_num] = ph

            # Extract raw spans + lines
            raw_spans = extract_text_spans(page, page_num)
            lines = extract_drawing_lines(page, page_num)

            # Filter headers/footers
            filtered_spans, removed_spans, meta = filter_header_footer_spans(
                raw_spans, ph, hf_patterns,
            )

            # Detect figures
            images = extract_images_info(page, page_num)
            for img in images:
                fig = Figure(
                    id=f"fig_p{page_num}_{len(figures_all) + 1}",
                    page=page_num, bbox=img["bbox"],
                )
                figures_all.append(fig)

            all_page_spans[page_num] = filtered_spans

            # Verdict decision
            verdict = assess_page_verdict(filtered_spans, lines, pw, ph, is_landscape)
            verdict_counts[verdict.value] += 1

            # Table extraction
            page_tables = []
            if verdict == Verdict.USE_PDFPLUMBER:
                page_tables = extract_tables_from_page(str(pdf_path), page_idx, page_num)

                validated_tables = []
                for table in page_tables:
                    table_verdict = validate_pdfplumber_output(table)
                    # Agent decision: trigger LLM repair if needed
                    if table_verdict == Verdict.NEEDS_LLM and with_llm:
                        self._report("phase2",
                            f"Page {page_num}: table {table.id} → NEEDS_LLM → calling GPT-4o-mini")
                        table = repair_table_with_llm(table, page_num)
                        total_llm_calls += 1
                    validated_tables.append(table)

                page_tables = validated_tables
                tables_by_page[page_num] = page_tables

            # Body text extraction
            content_blocks = []
            non_table_spans = self._exclude_table_spans(filtered_spans, page_tables)
            if non_table_spans:
                lists = discover_lists_in_spans(non_table_spans, body_font_size, page_num)
                content_blocks.extend(lists)
                paragraphs = self._group_into_paragraphs(non_table_spans, body_font_size, page_num)
                content_blocks.extend(paragraphs)

            page_results.append(PageResult(
                page_num=page_num, page_display=meta.page_num_display,
                width=pw, height=ph, is_landscape=is_landscape,
                verdict=verdict, tables=page_tables,
                content_blocks=content_blocks,
                figures=[f for f in figures_all if f.page == page_num],
                raw_spans_count=len(raw_spans),
                filtered_spans_count=len(filtered_spans),
                lines_count=len(lines), llm_calls=total_llm_calls,
            ))

        doc.close()

        self._report("phase2",
            f"Done: {total_pages} pages, verdicts: {dict(verdict_counts)}, "
            f"{sum(len(t) for t in tables_by_page.values())} raw tables, "
            f"{total_llm_calls} LLM repairs",
            "done")

        # ═══ NODE 3: Cross-Page Analysis ═════════════════════════════════
        self._report("phase3", "Running cross-page analysis...")

        # Continuation merging
        all_tables, merge_count = merge_continuation_tables(tables_by_page, page_heights)
        self._report("phase3", f"Continuation merges: {merge_count}")

        # Section hierarchy
        sections, numbering_scheme = discover_section_hierarchy(all_page_spans, body_font_size)
        self._report("phase3", f"Sections: {len(sections)} ({numbering_scheme})")

        # Captions
        captions = discover_table_captions(all_page_spans, tables_by_page)
        for table_id, (caption_text, caption_source) in captions.items():
            for table in all_tables:
                if table.id == table_id:
                    table.caption = caption_text
                    table.caption_source = caption_source

        # Footnotes
        fn_count = 0
        for page_num, page_tables in tables_by_page.items():
            spans = all_page_spans.get(page_num, [])
            for table in page_tables:
                footnotes = discover_footnotes_for_table(spans, table, body_font_size, page_num)
                if footnotes:
                    for marker, fn in footnotes.items():
                        table.footnotes[marker] = fn.text
                        table.footnote_sources.append(fn)
                    fn_count += len(footnotes)

        # Abbreviations
        abbreviations = discover_abbreviations(all_page_spans, all_tables)

        # Cross-references
        xref_count = 0
        for page in page_results:
            for block in page.content_blocks:
                if block.text:
                    refs = discover_cross_references(block.text)
                    block.cross_references = refs
                    xref_count += len(refs)

        # Reference index
        reference_index = build_reference_index(sections, all_tables, figures_all)

        self._report("phase3",
            f"Done: {merge_count} merges, {len(sections)} sections, "
            f"{len(captions)} captions, {fn_count} footnotes, "
            f"{len(abbreviations)} abbreviations, {xref_count} cross-refs, "
            f"{len(figures_all)} figures",
            "done")

        # ═══ NODE 4: Output Rendering ════════════════════════════════════
        self._report("phase4", "Rendering structured JSON + markdown...")

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

        json_path = render_json(result, f"{output_prefix}_structured.json")
        md_path = render_markdown(result, f"{output_prefix}_readable.md")

        elapsed = time.time() - t_start
        self._report("phase4",
            f"Output: {json_path} + {md_path} in {elapsed:.1f}s",
            "done")

        return result, json_path, self._steps

    # ── Helpers (exact logic from run.py) ────────────────────────────────

    @staticmethod
    def _exclude_table_spans(spans, tables):
        if not tables:
            return spans
        table_bboxes = []
        for table in tables:
            min_x = min_y = float('inf')
            max_x = max_y = float('-inf')
            for cell in table.cells:
                if cell.source and cell.source.bbox:
                    b = cell.source.bbox
                    min_x, min_y = min(min_x, b[0]), min(min_y, b[1])
                    max_x, max_y = max(max_x, b[2]), max(max_y, b[3])
            if min_x < float('inf'):
                table_bboxes.append((min_x - 5, min_y - 5, max_x + 5, max_y + 5))
        if not table_bboxes:
            return spans
        return [s for s in spans if not any(
            s.x0 >= tb[0] and s.y0 >= tb[1] and s.x1 <= tb[2] and s.y1 <= tb[3]
            for tb in table_bboxes
        )]

    @staticmethod
    def _group_into_paragraphs(spans, body_font_size, page_num):
        if not spans:
            return []
        sorted_spans = sorted(spans, key=lambda s: (round(s.y0, 1), s.x0))
        paragraphs = []
        current_spans = [sorted_spans[0]]
        prev_y = sorted_spans[0].y1
        for span in sorted_spans[1:]:
            gap = span.y0 - prev_y
            if gap > body_font_size * 1.5:
                if current_spans:
                    text, formats = extract_inline_formats(current_spans, body_font_size)
                    xrefs = discover_cross_references(text)
                    first, last = current_spans[0], current_spans[-1]
                    paragraphs.append(ContentBlock(
                        type=ContentBlockType.PARAGRAPH, text=text.strip(),
                        inline_formats=formats, cross_references=xrefs,
                        source=Source(page=page_num,
                            bbox=(first.x0, first.y0, last.x1, last.y1),
                            extraction_method=ExtractionMethod.PYMUPDF,
                            confidence=0.90,
                            raw_text_hash=Source.hash_text(text.strip())),
                    ))
                current_spans = []
            current_spans.append(span)
            prev_y = span.y1
        if current_spans:
            text, formats = extract_inline_formats(current_spans, body_font_size)
            xrefs = discover_cross_references(text)
            first, last = current_spans[0], current_spans[-1]
            paragraphs.append(ContentBlock(
                type=ContentBlockType.PARAGRAPH, text=text.strip(),
                inline_formats=formats, cross_references=xrefs,
                source=Source(page=page_num,
                    bbox=(first.x0, first.y0, last.x1, last.y1),
                    extraction_method=ExtractionMethod.PYMUPDF,
                    confidence=0.90,
                    raw_text_hash=Source.hash_text(text.strip())),
            ))
        return paragraphs


def process_protocol(
    pdf_path: str,
    output_prefix: str,
    with_llm: bool = True,
    verbose: bool = False,
    on_step: Callable[[dict], None] | None = None,
) -> tuple:
    """
    Run the agentic ingestion pipeline.

    Returns: (DocumentResult, json_path, steps)
    """
    pipeline = IngestionPipeline(on_step=on_step)
    return pipeline.run(
        pdf_path=pdf_path,
        output_prefix=output_prefix,
        with_llm=with_llm,
        verbose=verbose,
    )
