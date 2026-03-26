"""
Block Index — Fine-grained content blocks with BM25 scoring.

LEGACY: Used by ProtocolStore for /api/search and /api/section endpoints.
Primary search is now via LlamaIndex retriever
in knowledge_base/llamaindex_retriever.py.

Reads _structured.json and splits 215 giant blocks into ~1200
typed blocks using:
  - Bold boundaries (from inline_formats)
  - Numbered patterns (1, 2, 3...)
  - Bullet patterns (•, -, ○)
  - Tables stay WHOLE (rows need headers for context)
  - Footnotes become own blocks

Each block: {id, type, text, page, section_id, is_bold, refs}
BM25 index: precomputed TF-IDF for instant scoring.
"""
from __future__ import annotations
import re
import math
import logging
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

# ── Block types ──────────────────────────────────────────────────────

BLOCK_TYPES = {
    "section_heading", "bold_heading", "paragraph", "numbered_item",
    "bullet_item", "table", "footnote", "definition",
}


def build_block_index(data: dict) -> tuple[list[dict], dict]:
    """
    Build fine-grained block index from structured JSON.
    
    Returns:
        blocks: list of block dicts
        bm25: precomputed BM25 data {idf, avgdl, N}
    """
    blocks = []
    
    # Phase 1: Split text blocks using bold + patterns
    for page in data.get("pages", []):
        page_num = page["page_num"]
        section_id = _find_section_for_page(data, page_num)
        
        for block in page.get("content_blocks", []):
            text = block.get("text", "")
            if not text.strip():
                continue
            # Filter out "Page XX" noise blocks (leftover page numbers)
            if re.match(r'^Page\s+\d+$', text.strip()):
                continue
            
            formats = block.get("inline_formats", [])
            refs = block.get("cross_references", [])
            source = block.get("source", {})
            
            sub_blocks = _split_block(text, formats, page_num, section_id, refs, source)
            blocks.extend(sub_blocks)
    
    # Phase 2: Add tables as whole blocks
    for table in data.get("tables", []):
        table_text = _format_table_markdown(table)
        if not table_text.strip():
            continue
        
        pages = table.get("page_range", [])
        page_num = pages[0] if pages else 0
        section_id = _find_section_for_page(data, page_num)
        caption = table.get("caption", "") or table["id"]
        
        blocks.append({
            "id": f"tbl_{table['id']}",
            "type": "table",
            "text": table_text,
            "page": page_num,
            "page_range": pages,
            "section_id": section_id,
            "is_bold": False,
            "caption": caption[:100],
            "table_id": table["id"],
            "refs": [],
            "footnotes": table.get("footnotes", {}),
        })
    
    # Phase 3: Build BM25 index
    bm25 = _build_bm25(blocks)
    
    logger.info(
        f"📦 Block index: {len(blocks)} blocks "
        f"({sum(1 for b in blocks if b['type']=='table')} tables, "
        f"{sum(1 for b in blocks if b['type']=='numbered_item')} numbered, "
        f"{sum(1 for b in blocks if b['type']=='bold_heading')} bold headings, "
        f"{sum(1 for b in blocks if b['type']=='paragraph')} paragraphs)"
    )
    
    return blocks, bm25


# ── Block splitting ──────────────────────────────────────────────────

def _split_block(
    text: str,
    formats: list[dict],
    page: int,
    section_id: str,
    refs: list[dict],
    source: dict,
) -> list[dict]:
    """Split a single content block into sub-blocks.
    
    Clinical protocols have a consistent pattern:
      Bold category header → numbered criteria → bold header → more criteria
    
    E.g.: "Medical Conditions 1 History of... 2 History of... Prior/Concomitant Therapy 11 Receipt of..."
    
    We split on:
      1. Bold text boundaries (category headers like "Medical Conditions")
      2. Numbered item patterns ("1 History of", "11 Receipt of")
      3. Bullet patterns ("• item", "- item")
    """
    if len(text) < 80:
        return [_make_block(text, "paragraph", page, section_id, formats, refs)]

    splits = []

    # Bold boundaries → category headers
    bold_formats = [f for f in formats if f.get("bold")]
    for bf in bold_formats:
        start = bf["start"]
        bold_text = text[start:bf["end"]].strip()
        # Skip section numbers like "5.2" but keep headings like "Medical Conditions"
        if re.match(r'^\d+\.?\d*\.?\d*$', bold_text):
            continue
        if len(bold_text) >= 3:
            splits.append((start, "bold_heading"))

    # Numbered items: "1 History", "11 Receipt" — preceded by space/newline
    # Must be: (boundary)(digit)(space)(uppercase letter)
    # But NOT inside words like "phase 3" (lowercase before digit)
    for m in re.finditer(r'(?:^|\s)(\d{1,2})\s+([A-Z][a-z])', text):
        pos = m.start()
        if pos > 0:
            pos += 1  # skip the leading whitespace, point to the digit
        splits.append((pos, "numbered_item"))

    # Bullet patterns
    for m in re.finditer(r'(?:^|\n)\s*[•\-○–]\s+', text):
        pos = m.start()
        if pos > 0:
            splits.append((pos, "bullet_item"))

    if not splits:
        return [_make_block(text, "paragraph", page, section_id, formats, refs)]

    # Sort and deduplicate nearby splits (within 3 chars)
    splits.sort(key=lambda x: x[0])
    deduped = [splits[0]]
    for pos, btype in splits[1:]:
        if pos - deduped[-1][0] > 3:
            deduped.append((pos, btype))
        elif btype == "bold_heading":
            # Bold heading takes priority over numbered_item at same position
            deduped[-1] = (pos, btype)
    splits = deduped

    # Build sub-blocks
    result = []
    prev_pos = 0
    prev_type = "paragraph"

    for i, (pos, btype) in enumerate(splits):
        if pos > prev_pos:
            chunk = text[prev_pos:pos].strip()
            if chunk and len(chunk) > 10:
                result.append(_make_block(chunk, prev_type, page, section_id, formats, refs))
        prev_pos = pos
        prev_type = btype

    # Last chunk
    if prev_pos < len(text):
        chunk = text[prev_pos:].strip()
        if chunk and len(chunk) > 10:
            result.append(_make_block(chunk, prev_type, page, section_id, formats, refs))

    if not result:
        return [_make_block(text, "paragraph", page, section_id, formats, refs)]

    return result


def _detect_type(text: str, pos: int, formats: list[dict]) -> str:
    """Detect block type from content and formatting."""
    # Check if this position has bold formatting
    for f in formats:
        if f.get("bold") and f["start"] <= pos < f["end"]:
            if f.get("font_size", 0) > 12:
                return "section_heading"
            return "bold_heading"
    
    # Check for numbered pattern at start
    if re.match(r'\s*\d{1,2}[.\s]+[A-Z]', text):
        return "numbered_item"
    
    if re.match(r'\s*[•\-○–]\s+', text):
        return "bullet_item"
    
    return "paragraph"


def _make_block(
    text: str,
    block_type: str,
    page: int,
    section_id: str,
    formats: list[dict],
    refs: list[dict],
) -> dict:
    """Create a block dict."""
    # Find refs that fall within this text
    block_refs = []
    for r in refs:
        ref_text = r.get("text", "")
        if ref_text and ref_text in text:
            block_refs.append({
                "text": ref_text,
                "target_id": r.get("target_id", ""),
                "target_type": r.get("target_type", ""),
            })
    
    # Check if any part of this block is bold
    is_bold = any(
        f.get("bold") for f in formats
        if f["start"] < len(text) and f["end"] > 0
    )
    
    return {
        "id": f"blk_p{page}_{hash(text[:50]) % 10000:04d}",
        "type": block_type,
        "text": text.strip(),
        "page": page,
        "section_id": section_id,
        "is_bold": is_bold,
        "refs": block_refs,
        "char_count": len(text.strip()),
    }


# ── Table formatting ─────────────────────────────────────────────────

def _format_table_markdown(table: dict) -> str:
    """Format a table as markdown. Removes empty columns from merged-cell artifacts."""
    lines = []
    caption = table.get("caption", "") or ""
    if caption:
        lines.append(f"**{caption}**")
        lines.append("")
    
    headers = table.get("column_headers", [])
    rows = table.get("rows", [])
    
    # Detect and remove columns that are empty in >80% of rows
    if rows and headers:
        num_cols = max(len(headers), max(len(r) for r in rows))
        non_empty_count = [0] * num_cols
        for row in rows:
            for j, cell in enumerate(row):
                if str(cell).strip():
                    if j < num_cols:
                        non_empty_count[j] += 1
        
        # Keep columns that have content in >20% of rows
        threshold = max(1, len(rows) * 0.2)
        keep_cols = [j for j in range(num_cols) if non_empty_count[j] >= threshold]
        
        if keep_cols and len(keep_cols) < num_cols:
            # Filter headers
            filtered_headers = [headers[j] for j in keep_cols if j < len(headers)]
            filtered_headers = [h for h in filtered_headers if str(h).strip()]
            
            if filtered_headers:
                lines.append("| " + " | ".join(str(h) for h in filtered_headers) + " |")
                lines.append("|" + "|".join("---" for _ in filtered_headers) + "|")
            
            # Filter rows
            for row in rows:
                cells = [str(row[j]) if j < len(row) else "" for j in keep_cols]
                # Skip rows where all kept cells are empty
                if any(c.strip() for c in cells):
                    lines.append("| " + " | ".join(cells) + " |")
        else:
            # No filtering needed
            if headers:
                lines.append("| " + " | ".join(str(h) for h in headers) + " |")
                lines.append("|" + "|".join("---" for _ in headers) + "|")
            for row in rows:
                cells = [str(c) for c in row]
                lines.append("| " + " | ".join(cells) + " |")
    else:
        # No headers or no rows
        for row in rows:
            cells = [str(c) for c in row]
            if any(c.strip() for c in cells):
                lines.append("| " + " | ".join(cells) + " |")
    
    # Footnotes
    footnotes = table.get("footnotes", {})
    if footnotes:
        lines.append("")
        for marker, text in footnotes.items():
            lines.append(f"^{marker}: {text}")
    
    return "\n".join(lines)


# ── BM25 Index ───────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return [w.lower() for w in re.findall(r'[a-zA-Z0-9]+(?:\.[0-9]+)*', text) if len(w) > 1]


def _build_bm25(blocks: list[dict]) -> dict:
    """Precompute BM25 components: IDF, average doc length."""
    N = len(blocks)
    if N == 0:
        return {"idf": {}, "avgdl": 1, "N": 0}
    
    # Document frequency per term
    df = Counter()
    doc_lengths = []
    
    for block in blocks:
        tokens = _tokenize(block["text"])
        doc_lengths.append(len(tokens))
        unique_tokens = set(tokens)
        for t in unique_tokens:
            df[t] += 1
    
    avgdl = sum(doc_lengths) / N if N > 0 else 1
    
    # IDF: log((N - n + 0.5) / (n + 0.5) + 1)
    idf = {}
    for term, n in df.items():
        idf[term] = math.log((N - n + 0.5) / (n + 0.5) + 1)
    
    return {"idf": idf, "avgdl": avgdl, "N": N}


def bm25_score(query: str, block: dict, bm25_data: dict, k1: float = 1.5, b: float = 0.75) -> float:
    """Score a single block against a query using BM25."""
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(block["text"])
    
    if not query_tokens or not doc_tokens:
        return 0.0
    
    idf = bm25_data["idf"]
    avgdl = bm25_data["avgdl"]
    dl = len(doc_tokens)
    
    # Term frequency in this block
    tf = Counter(doc_tokens)
    
    score = 0.0
    for qt in query_tokens:
        if qt not in idf:
            continue
        f = tf.get(qt, 0)
        numerator = f * (k1 + 1)
        denominator = f + k1 * (1 - b + b * dl / avgdl)
        score += idf[qt] * numerator / denominator
    
    return score


def search_blocks(
    query: str,
    blocks: list[dict],
    bm25_data: dict,
    manifest: dict = None,
    query_domain: str = None,
    top_k: int = 20,
) -> list[tuple[dict, float]]:
    """
    Search all blocks with BM25 + domain boost + type boost.
    
    Returns: [(block, score), ...] sorted by score descending.
    """
    # Build domain map from manifest
    section_domains = {}
    if manifest:
        for s in manifest.get("section_map", []):
            section_domains[s["id"]] = s.get("domain", "")
    
    scored = []
    for block in blocks:
        base_score = bm25_score(query, block, bm25_data)
        
        if base_score <= 0:
            continue
        
        # Domain boost
        if query_domain and section_domains:
            block_domain = section_domains.get(block["section_id"], "")
            if block_domain == query_domain:
                base_score += 0.3
            elif block_domain == "overview":
                base_score -= 0.2  # Synopsis penalty
        
        # Type boost
        if block["type"] == "table":
            base_score += 0.1
        elif block["type"] == "numbered_item":
            base_score += 0.05
        elif block["type"] == "definition":
            base_score += 0.05
        
        scored.append((block, base_score))
    
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


# ── Helpers ──────────────────────────────────────────────────────────

def _find_section_for_page(data: dict, page_num: int) -> str:
    """
    Find which section a page belongs to.
    
    When multiple sections start on the same page (e.g., §2.3.1, §2.3.2, §2.3.3, §3 
    all at page 30), we pick the one that comes LATEST in document order.
    This is because content AFTER §3's heading on page 30 belongs to §3, not §2.3.1.
    """
    # Collect all sections that start at or before this page
    candidates = []
    for section in data.get("sections", []):
        pages = section.get("page_range", [])
        if pages and pages[0] <= page_num:
            candidates.append(section)
    
    if not candidates:
        return ""
    
    # Sort by start page descending to find the nearest section
    candidates.sort(key=lambda s: s.get("page_range", [0])[0], reverse=True)
    
    # The nearest section(s) have the highest start page <= page_num
    nearest_page = candidates[0].get("page_range", [0])[0]
    
    # Among sections on the same nearest page, pick the one with 
    # the highest section number (latest in document order)
    same_page = [s for s in candidates if s.get("page_range", [0])[0] == nearest_page]
    
    if len(same_page) == 1:
        return same_page[0].get("number", "")
    
    # Sort by section number to find the "latest" one
    # "3" > "2.3.3" > "2.3.2" > "2.3.1" in document order
    # Use a key that puts higher top-level numbers first
    def section_sort_key(s):
        num = s.get("number", "")
        parts = []
        for p in num.replace(".", " ").split():
            try:
                parts.append(int(p))
            except ValueError:
                parts.append(999)  # non-numeric sections go last
        return parts
    
    same_page.sort(key=section_sort_key, reverse=True)
    return same_page[0].get("number", "")


def get_section_blocks(data: dict, section_id: str) -> list[dict]:
    """Get properly typed blocks for a section using its content_blocks.
    
    Uses the parser's content_blocks assignment (y-position based) which
    correctly handles shared pages. Falls back to page-range with smart
    extension if no content_blocks exist (old JSONs).
    """
    # Find the section
    sections = data.get("sections", [])
    target = None
    for s in sections:
        if s.get("number") == section_id:
            target = s
            break
    if not target:
        return []

    blocks = []
    
    # PRIMARY: use content_blocks if available (new parser)
    content_blocks = target.get("content_blocks", [])
    if content_blocks:
        for cb in content_blocks:
            text = cb.get("text", "")
            if not text.strip():
                continue
            formats = cb.get("inline_formats", [])
            refs = cb.get("cross_references", [])
            source = cb.get("source", {})
            page = source.get("page", target.get("page_range", [0])[0])
            
            sub_blocks = _split_block(text, formats, page, section_id, refs, source)
            blocks.extend(sub_blocks)
        return blocks

    # FALLBACK: reconstruct from page content with smart page extension
    page_range = target.get("page_range", [])
    if not page_range:
        return []
    
    start_page = min(page_range)
    
    # Find end page: start of next section (same logic as ProtocolStore)
    all_starts = sorted(set(
        min(s.get("page_range", [999]))
        for s in sections
        if s.get("page_range") and min(s.get("page_range", [999])) > start_page
    ))
    end_page = all_starts[0] if all_starts else start_page + 5
    extended_range = list(range(start_page, end_page + 1))
    
    for page_data in data.get("pages", []):
        pnum = page_data["page_num"]
        if pnum not in extended_range:
            continue
        for cb in page_data.get("content_blocks", []):
            text = cb.get("text", "")
            if not text.strip():
                continue
            formats = cb.get("inline_formats", [])
            refs = cb.get("cross_references", [])
            source = cb.get("source", {})
            sub_blocks = _split_block(text, formats, pnum, section_id, refs, source)
            blocks.extend(sub_blocks)
    
    return blocks


def render_section(data: dict, section_id: str) -> str:
    """Render a section as markdown+XML blocks for LLM consumption.
    
    This is the function that connects the parser's structured output
    to the LLM extraction pipeline:
    
      Structured JSON → get_section_blocks() → render_blocks_for_llm() → LLM
    
    Preserves: bold headings, numbered items, cross-references, page numbers.
    """
    from knowledge_base.block_renderer import render_blocks_for_llm
    blocks = get_section_blocks(data, section_id)
    if not blocks:
        return ""
    return render_blocks_for_llm(blocks)


def render_table(data: dict, table_id: str) -> str:
    """Render a table as markdown with footnotes for LLM consumption."""
    for table in data.get("tables", []):
        if table.get("id") == table_id:
            pages = table.get("page_range", [])
            page_str = f"pp. {pages[0]}-{pages[-1]}" if len(pages) > 1 else f"p. {pages[0]}" if pages else ""
            header = f'<table id="{table_id}" pages="{page_str}">\n'
            body = _format_table_markdown(table)
            return header + body + "\n</table>"
    return ""
