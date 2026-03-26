"""
Vision Table Extractor — Uses GPT-4o to extract complex tables from page images.

Triggered when a table has cells > 200 chars (indicating nested bullets,
case definitions, or other complex content that pdfplumber flattens).

Renders the table's pages as images and sends to GPT-4o vision.
Returns clean markdown text preserving bullets, line breaks, and structure.

Usage:
    from extraction.vision_table import extract_table_with_vision, is_complex_table
    
    if is_complex_table(table):
        markdown = extract_table_with_vision(pdf_path, table["page_range"])
"""
from __future__ import annotations
import base64
import logging
import time
from pathlib import Path

import fitz  # PyMuPDF

from config import VLM_MODEL, get_openai_client

logger = logging.getLogger(__name__)

# Threshold: if any cell exceeds this, table is "complex"
COMPLEX_CELL_CHARS = 200


def is_complex_table(table: dict) -> bool:
    """Check if a table has complex content that benefits from vision extraction."""
    for row in table.get("rows", []):
        for cell in row:
            if len(str(cell)) > COMPLEX_CELL_CHARS:
                return True
    # Also flag multi-page tables with many rows
    pages = table.get("page_range", [])
    if len(pages) >= 3 and len(table.get("rows", [])) > 15:
        return True
    return False


def render_pages_as_images(pdf_path: str, page_numbers: list[int], dpi: int = 150) -> list[str]:
    """Render PDF pages as base64-encoded PNG images."""
    doc = fitz.open(pdf_path)
    images = []
    
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    
    for page_num in page_numbers:
        if page_num < 1 or page_num > len(doc):
            continue
        page = doc[page_num - 1]  # 0-indexed
        pix = page.get_pixmap(matrix=matrix)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        images.append(b64)
    
    doc.close()
    return images


def extract_table_with_vision(
    pdf_path: str,
    page_numbers: list[int],
    query_context: str = "",
) -> str:
    """
    Extract a complex table using GPT-4o vision.
    
    For tables spanning 3+ pages, extracts each page separately to avoid
    hitting the output token limit, then merges results.
    
    Args:
        pdf_path: Path to the PDF file
        page_numbers: Pages containing the table
        query_context: Optional context about what we're extracting
    """
    # For large tables (3+ pages), extract per-page and merge
    if len(page_numbers) > 2:
        logger.info(f"  Large table ({len(page_numbers)} pages) — extracting per-page then merging")
        all_results = []
        for i, page in enumerate(page_numbers):
            ctx = f"This is page {i+1} of {len(page_numbers)} of a multi-page table. " \
                  f"Extract ALL rows on this page. " \
                  f"{'Include column headers.' if i == 0 else 'The column headers from page 1 apply here too — include them.'}"
            result = _extract_single_batch(pdf_path, [page], ctx)
            if result:
                all_results.append(result)
        if all_results:
            merged = all_results[0]
            for r in all_results[1:]:
                # Skip header line from subsequent pages (first 2 lines = header + separator)
                lines = r.strip().split("\n")
                data_lines = [l for l in lines[2:] if l.strip()] if len(lines) > 2 else lines
                merged += "\n" + "\n".join(data_lines)
            return merged
        return ""
    
    # For 1-2 page tables, extract all at once
    return _extract_single_batch(pdf_path, page_numbers, query_context)


def _extract_single_batch(
    pdf_path: str,
    page_numbers: list[int],
    query_context: str = "",
) -> str:
    """Extract table from a batch of pages (1-2 pages) using GPT-4o vision."""
    logger.info(
        f"🔭 Vision extraction: {len(page_numbers)} pages "
        f"({page_numbers}) using {VLM_MODEL}"
    )
    
    t0 = time.time()
    
    # Render pages as images
    images = render_pages_as_images(pdf_path, page_numbers)
    if not images:
        logger.warning("No images rendered — falling back to text")
        return ""
    
    # Build the message with images
    content = []
    
    # Add each page image
    for i, b64_img in enumerate(images):
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64_img}",
                "detail": "high"
            }
        })
    
    # Add the extraction instruction — force STRUCTURED JSON output
    instruction = (
        "Extract the COMPLETE table from these page images.\n\n"
        "Return ONLY a JSON object with this EXACT structure:\n"
        "{\n"
        '  "title": "Table title from the image",\n'
        '  "columns": ["Procedure", "Day 1", "Day 29", "Day 57", ...],\n'
        '  "rows": [\n'
        '    {"procedure": "Medical history", "values": ["✓", "", "✓", "", ...]},\n'
        '    {"procedure": "Vital signs", "values": ["✓", "✓", "✓", "✓", ...]},\n'
        '    ...\n'
        '  ],\n'
        '  "footnotes": ["a: Not a study site visit", "b: If urine test positive..."]\n'
        "}\n\n"
        "RULES:\n"
        "- columns: list ALL visit/day columns from left to right, including windows\n"
        "- For each row: procedure name + one value per column\n"
        "- Values: use ✓ for checkmarks, X for X marks, empty string for blank cells\n"
        "- If a cell has text like 'X (predose)', use that exact text\n"
        "- If a cell spans multiple columns (continuous arrow), put '↔' in each spanned cell\n"
        "- Include ALL rows: procedures, efficacy, immunogenicity, safety sections\n"
        "- Include section headers as rows with empty values (e.g., 'Efficacy assessments')\n"
        "- Include ALL footnotes at the bottom\n"
        "- Return ONLY valid JSON, no markdown, no explanation"
    )
    if query_context:
        instruction = query_context + "\n\n" + instruction
    
    content.append({"type": "text", "text": instruction})
    
    # Call GPT-4o vision
    client = get_openai_client()
    
    try:
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[
                {"role": "user", "content": content}
            ],
            max_tokens=16384,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        
        raw_result = response.choices[0].message.content
        elapsed = time.time() - t0
        
        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0
        
        logger.info(
            f"✅ Vision extraction done: {elapsed:.1f}s, "
            f"{tokens_in} tokens in, {tokens_out} tokens out, "
            f"{len(raw_result)} chars output"
        )
        
        # Convert JSON to pipe-separated markdown that UI can render
        result = _vision_json_to_markdown(raw_result)
        return result
    
    except Exception as e:
        logger.error(f"Vision extraction failed: {e}")
        return ""


def _vision_json_to_markdown(raw_json: str) -> str:
    """
    Convert structured JSON from vision LLM to pipe-separated markdown.
    
    Input JSON:
      {"columns": ["Procedure","Day 1","Day 29"],
       "rows": [{"procedure":"Vital signs","values":["✓","✓"]}],
       "footnotes": ["a: Not a site visit"]}
    
    Output markdown:
      | Procedure | Day 1 | Day 29 |
      |---|---|---|
      | Vital signs | ✓ | ✓ |
      
      a: Not a site visit
    """
    from shared.json_parser import parse_llm_json
    
    data = parse_llm_json(raw_json)
    if not data:
        logger.warning("Vision JSON parse failed, returning raw text")
        return raw_json
    
    lines = []
    
    # Title
    title = data.get("title", "")
    if title:
        lines.append(title)
        lines.append("")
    
    # Header row
    columns = data.get("columns", [])
    if not columns:
        return raw_json
    
    lines.append("| " + " | ".join(str(c) for c in columns) + " |")
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    
    # Data rows
    for row in data.get("rows", []):
        proc = row.get("procedure", "")
        values = row.get("values", [])
        
        # Pad values to match column count (minus 1 for procedure column)
        while len(values) < len(columns) - 1:
            values.append("")
        
        cells = [proc] + values[:len(columns) - 1]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    
    # Footnotes
    footnotes = data.get("footnotes", [])
    if footnotes:
        lines.append("")
        for fn in footnotes:
            lines.append(fn)
    
    return "\n".join(lines)
