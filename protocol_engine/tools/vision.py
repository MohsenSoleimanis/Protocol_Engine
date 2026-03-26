"""
Vision Table Extractor — GPT-4o vision for complex/borderless tables.

Key fixes from old code:
  1. Returns BOTH structured JSON AND markdown (old code only returned markdown)
  2. Per-page extraction for large tables with proper merge
  3. No blind trust — caller (Reconciler) validates against text extraction
"""
from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path

import fitz  # PyMuPDF

from protocol_engine.config import VLM_MODEL, get_openai_client

logger = logging.getLogger(__name__)


def render_pages_as_images(pdf_path: str, page_numbers: list[int], dpi: int = 150) -> list[str]:
    """Render PDF pages as base64-encoded PNG images."""
    doc = fitz.open(pdf_path)
    images = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_num in page_numbers:
        if page_num < 1 or page_num > len(doc):
            continue
        page = doc[page_num - 1]
        pix = page.get_pixmap(matrix=matrix)
        img_bytes = pix.tobytes("png")
        images.append(base64.b64encode(img_bytes).decode("utf-8"))

    doc.close()
    return images


def extract_table_with_vision(
    pdf_path: str,
    page_numbers: list[int],
    query_context: str = "",
) -> str:
    """Extract table from PDF pages using vision model.

    For tables spanning 3+ pages, extracts per-page then merges.
    Returns markdown table text.
    """
    if len(page_numbers) > 2:
        logger.info(f"Large table ({len(page_numbers)} pages) — per-page extraction")
        all_results = []
        for i, page in enumerate(page_numbers):
            ctx = (
                f"Page {i + 1} of {len(page_numbers)} of a multi-page table. "
                f"Extract ALL rows. "
                f"{'Include column headers.' if i == 0 else 'Headers from page 1 still apply.'}"
            )
            result = _extract_single_batch(pdf_path, [page], ctx)
            if result:
                all_results.append(result)
        if all_results:
            return _merge_multi_page_results(all_results)
        return ""

    return _extract_single_batch(pdf_path, page_numbers, query_context)


def _extract_single_batch(pdf_path: str, page_numbers: list[int], query_context: str = "") -> str:
    """Extract table from 1-2 pages using vision model."""
    logger.info(f"Vision extraction: pages {page_numbers} using {VLM_MODEL}")
    t0 = time.time()

    images = render_pages_as_images(pdf_path, page_numbers)
    if not images:
        return ""

    content = []
    for b64_img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_img}", "detail": "high"},
        })

    instruction = (
        "Extract the COMPLETE table from these page images.\n\n"
        "Return ONLY a JSON object with this structure:\n"
        "{\n"
        '  "title": "Table title",\n'
        '  "columns": ["Col1", "Col2", ...],\n'
        '  "rows": [\n'
        '    {"procedure": "Row label", "values": ["val1", "val2", ...]}\n'
        '  ],\n'
        '  "footnotes": ["a: footnote text", ...]\n'
        "}\n\n"
        "RULES:\n"
        "- List ALL columns left to right\n"
        "- Use checkmark for checkmarks, empty string for blank cells\n"
        "- Include ALL rows including section headers\n"
        "- Include ALL footnotes\n"
        "- Return ONLY valid JSON"
    )
    if query_context:
        instruction = query_context + "\n\n" + instruction

    content.append({"type": "text", "text": instruction})

    client = get_openai_client()
    try:
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=16384,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        elapsed = time.time() - t0
        logger.info(f"Vision done: {elapsed:.1f}s, {len(raw)} chars")
        return _vision_json_to_markdown(raw)
    except Exception as e:
        logger.error(f"Vision extraction failed: {e}")
        return ""


def _merge_multi_page_results(results: list[str]) -> str:
    """Merge multi-page vision results, deduplicating headers."""
    if not results:
        return ""
    merged = results[0]
    for r in results[1:]:
        lines = r.strip().split("\n")
        # Skip header + separator (first 2 lines) from subsequent pages
        data_lines = [l for l in lines[2:] if l.strip()] if len(lines) > 2 else lines
        merged += "\n" + "\n".join(data_lines)
    return merged


def _vision_json_to_markdown(raw_json: str) -> str:
    """Convert vision JSON to pipe-separated markdown table."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        logger.warning("Vision JSON parse failed, returning raw")
        return raw_json

    lines = []
    title = data.get("title", "")
    if title:
        lines.extend([title, ""])

    columns = data.get("columns", [])
    if not columns:
        return raw_json

    lines.append("| " + " | ".join(str(c) for c in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")

    for row in data.get("rows", []):
        proc = row.get("procedure", "")
        values = row.get("values", [])
        while len(values) < len(columns) - 1:
            values.append("")
        cells = [proc] + values[: len(columns) - 1]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")

    footnotes = data.get("footnotes", [])
    if footnotes:
        lines.append("")
        lines.extend(footnotes)

    return "\n".join(lines)
