"""MCP tool: vision_extract — extract complex tables via GPT-4o vision."""
from __future__ import annotations

import base64
import json
import logging
import time

import fitz

from langchain_core.tools import tool
from protocol_engine.config import VLM_MODEL, get_openai_client

logger = logging.getLogger(__name__)


def make_vision_tool(pdf_path: str, gathered: dict, budget: list[int]):
    """Create a vision extraction tool bound to a PDF."""
    calls = [0]

    @tool
    def vision_extract(pages: list[int]) -> str:
        """Extract complex table from PDF page images using vision model."""
        if calls[0] >= 6:
            return "Vision call limit reached."
        calls[0] += 1
        result = _extract_table_vision(pdf_path, pages)
        if result:
            key = f"vision_{pages}"
            gathered[key] = {"text": result, "pages": pages, "chars": len(result), "type": "table"}
            budget[0] += len(result)
        return result or "Vision returned no content."

    return vision_extract


def _extract_table_vision(pdf_path: str, page_numbers: list[int]) -> str:
    """Extract table from pages using vision LLM. Returns markdown."""
    if len(page_numbers) > 2:
        parts = []
        for i, p in enumerate(page_numbers):
            ctx = f"Page {i+1} of {len(page_numbers)}. Extract ALL rows."
            r = _single_batch(pdf_path, [p], ctx)
            if r:
                parts.append(r)
        if not parts:
            return ""
        merged = parts[0]
        for r in parts[1:]:
            lines = r.strip().split("\n")
            data = [l for l in lines[2:] if l.strip()] if len(lines) > 2 else lines
            merged += "\n" + "\n".join(data)
        return merged
    return _single_batch(pdf_path, page_numbers, "")


def _single_batch(pdf_path: str, page_numbers: list[int], ctx: str) -> str:
    logger.info(f"Vision: pages {page_numbers}")
    doc = fitz.open(pdf_path)
    images = []
    for p in page_numbers:
        if p < 1 or p > len(doc):
            continue
        pix = doc[p - 1].get_pixmap(matrix=fitz.Matrix(2, 2))
        images.append(base64.b64encode(pix.tobytes("png")).decode())
    doc.close()
    if not images:
        return ""

    content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}", "detail": "high"}} for b in images]
    instruction = (
        "Extract the COMPLETE table as a markdown pipe table.\n"
        "Use ✓ for checkmarks, empty for blanks. Include ALL rows and footnotes.\n"
        "Return ONLY the markdown table, no JSON, no explanation."
    )
    if ctx:
        instruction = ctx + "\n\n" + instruction
    content.append({"type": "text", "text": instruction})

    try:
        t0 = time.time()
        resp = get_openai_client().chat.completions.create(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": content}],
            max_tokens=16384, temperature=0.1,
        )
        result = resp.choices[0].message.content or ""
        logger.info(f"Vision done: {time.time()-t0:.1f}s, {len(result)} chars")
        return result.strip()
    except Exception as e:
        logger.error(f"Vision failed: {e}")
        return ""
