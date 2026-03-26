"""
LLM-based table repair using compact skeleton.

Instead of sending an image (200KB, ~30s), we send a structured skeleton:
  Grid: 22 rows × 12 columns
  Headers: [Procedure, Day 1, Day 8, ...]
  Row 3: [Vital signs, X, , , ...]

This costs ~500 tokens (~$0.0005) and takes <2 seconds.
"""

from __future__ import annotations
import json
import os
import logging
from typing import Any

from .models import (
    ExtractedTable, TableCell, RowMetadata, RowType,
    Source, ExtractionMethod,
)

logger = logging.getLogger(__name__)


def build_table_skeleton(table: ExtractedTable) -> str:
    """
    Build a compact text representation of a table for LLM repair.
    """
    lines: list[str] = []
    lines.append(f"Grid: {len(table.rows)} rows × {table.column_count} columns")
    
    if table.caption:
        lines.append(f"Caption: {table.caption}")
    
    if table.column_headers:
        lines.append(f"Headers: {table.column_headers}")
    
    # Include all rows with their raw data
    for i, row in enumerate(table.rows):
        # Truncate very long cell text
        cells = [c[:50] if len(c) > 50 else c for c in row]
        lines.append(f"Row {i}: {cells}")
    
    # Include footnotes if any
    if table.footnotes:
        lines.append(f"Footnotes: {table.footnotes}")
    
    return "\n".join(lines)


def repair_table_with_llm(
    table: ExtractedTable,
    page_num: int,
) -> ExtractedTable:
    """
    Send compact skeleton to GPT-4o-mini for repair.
    
    The LLM fixes:
    - Misaligned columns
    - Split cells that should be merged
    - Garbled text from poor extraction
    
    Returns a new ExtractedTable with repaired data.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("CHEAP_MODEL", "gpt-4o-mini")  # Always cheap for table repair
    
    if not api_key:
        logger.warning("No OPENAI_API_KEY — skipping LLM repair")
        return table
    
    skeleton = build_table_skeleton(table)
    
    prompt = f"""You are a table repair specialist. Below is a table extracted from a PDF.
The extraction may have issues: misaligned columns, split cells, garbled text.

Your job: fix the table and return clean JSON.

RULES:
- Preserve ALL original cell text exactly (don't interpret or summarize)
- Fix column alignment (ensure each row has exactly the right number of columns)
- Merge cells that were incorrectly split across rows
- Keep empty cells as empty strings ""
- Return ONLY valid JSON, no explanation

INPUT TABLE:
{skeleton}

Return JSON format:
{{
  "headers": ["col1", "col2", ...],
  "rows": [
    ["cell1", "cell2", ...],
    ...
  ]
}}"""

    try:
        from config import get_openai_client
        client = get_openai_client()
        
        logger.info(f"LLM CALL [Parser/TableRepair] model={model} table={table.id} input={len(prompt)} chars")
        t0 = __import__('time').time()
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4000,
            response_format={"type": "json_object"},
        )
        
        elapsed = __import__('time').time() - t0
        raw = response.choices[0].message.content or ""
        logger.info(f"LLM DONE [Parser/TableRepair] {elapsed:.1f}s output={len(raw)} chars")
        
        parsed = json.loads(raw)
        
        # Build repaired table
        headers = parsed.get("headers", table.column_headers)
        rows = parsed.get("rows", table.rows)
        
        # Build new cells with grounding
        new_cells: list[TableCell] = []
        for r_idx, row in enumerate(rows):
            for c_idx, cell_text in enumerate(row):
                text = str(cell_text).strip()
                new_cells.append(TableCell(
                    row=r_idx,
                    col=c_idx,
                    text=text,
                    source=Source(
                        page=page_num,
                        extraction_method=ExtractionMethod.LLM_REPAIR,
                        confidence=0.80,
                        raw_text_hash=Source.hash_text(text) if text else None,
                    ),
                ))
        
        repaired = table.model_copy(update={
            "column_headers": headers,
            "rows": rows,
            "cells": new_cells,
            "column_count": len(headers) if headers else table.column_count,
            "extraction_method": ExtractionMethod.LLM_REPAIR,
            "confidence": 0.80,
        })
        
        logger.info(f"LLM repair successful: {len(rows)} rows × {len(headers)} cols")
        return repaired
        
    except Exception as e:
        logger.error(f"LLM repair failed: {e}")
        return table
