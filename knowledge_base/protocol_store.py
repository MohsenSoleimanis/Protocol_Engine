"""
Protocol Store — Page-indexed + Block-indexed content store.

Two levels of retrieval:
  - Page level: get_section(), get_pages() — for complete section content
  - Block level: search(), get_section_blocks() — for fine-grained scored retrieval

Design: JSON is the source of truth. inline_formats (bold), 
cross_references, and table structure are all preserved and used.
"""
from __future__ import annotations
import json
import re
import hashlib
import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


class ProtocolStore:
    """
    In-memory store for a parsed protocol.
    Provides tool-like methods that the Retrieval Agent calls.
    """

    def __init__(self, structured_json_path: str | Path):
        with open(structured_json_path, encoding="utf-8") as f:
            self._data = json.load(f)

        # Core indexes
        self._section_index: dict[str, dict] = {}
        self._page_content: dict[int, list[dict]] = {}
        self._table_index: dict[str, dict] = {}
        self._page_tables: dict[int, list[str]] = {}

        # Block index (fine-grained, BM25-scored)
        self._blocks: list[dict] = []
        self._bm25: dict = {}
        self._section_blocks: dict[str, list[dict]] = {}  # section_id → blocks
        self._pdf_path: str | None = None  # Set by discover_bookmarks

        self._build_indexes()
        
        # Block index for legacy API endpoints (optional)
        # Primary search is now via LlamaIndex retriever
        # Block index disabled — LlamaIndex retriever handles search
        # try:
        #     self._build_block_index()
        # except ImportError:
        #     logger.info("Block index not available")

    def _build_indexes(self):
        for section in self._data.get("sections", []):
            self._section_index[section["number"]] = section

        for page in self._data.get("pages", []):
            self._page_content[page["page_num"]] = page.get("content_blocks", [])

        for table in self._data.get("tables", []):
            self._table_index[table["id"]] = table
            for p in table.get("page_range", []):
                self._page_tables.setdefault(p, []).append(table["id"])

    def _build_block_index(self):
        """Build fine-grained block index with BM25 scoring."""
        from knowledge_base.block_index import build_block_index
        self._blocks, self._bm25 = build_block_index(self._data)
        
        # Index blocks by section
        for block in self._blocks:
            sid = block.get("section_id", "")
            if sid:
                self._section_blocks.setdefault(sid, []).append(block)

    # ── Tool: read_manifest ────────────────────────────────────────────
    # (manifest is separate — see ManifestBuilder)

    # ── Tool: get_section ──────────────────────────────────────────────

    def get_section(self, section_id: str) -> dict | None:
        """
        Retrieve a section's COMPLETE content.

        If the parser assigned content_blocks to this section (position-based),
        use those directly — this correctly handles shared pages where two
        sections split a page.

        Otherwise, fall back to page-range reconstruction (legacy behavior).
        """
        section = self._section_index.get(section_id)
        if not section:
            return None

        # PRIMARY PATH: use assigned content_blocks (parser phase 3b.5)
        blocks = section.get("content_blocks", [])
        if blocks:
            text_parts = []
            block_pages = set()
            for block in blocks:
                text = block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                source = block.get("source", {}) if isinstance(block, dict) else {}
                page = source.get("page")
                if page is not None:
                    block_pages.add(page)
                if text.strip():
                    text_parts.append(text)

            # Also include tables on the section's pages
            all_pages = sorted(block_pages) if block_pages else section.get("page_range", [])
            for p in all_pages:
                for tid in self._page_tables.get(p, []):
                    table = self._table_index.get(tid)
                    if table:
                        table_text = self._format_table_as_text(table)
                        text_parts.append(f"[Table: {table.get('caption', tid)}]\n{table_text}")

            content = "\n\n".join(text_parts)
            return {
                "section_id": section.get("number", section_id),
                "title": section.get("title", ""),
                "level": section.get("level", 0),
                "pages": sorted(block_pages) if block_pages else all_pages,
                "content": content,
                "char_count": len(content),
            }

        # FALLBACK: page-range reconstruction (for old JSONs without content_blocks)
        pages = section.get("page_range", [])
        if not pages:
            return None

        start_page = pages[0]
        end_page, next_section = self._find_section_end_page(section_id, start_page)

        section_source = section.get("source", {})
        section_page = section_source.get("page", start_page)
        section_y_top = section_source.get("bbox", [0, 0, 0, 0])[1]

        all_text_parts = []
        all_pages = list(range(start_page, end_page))
        
        for p in all_pages:
            if p == start_page and section_page == start_page and section_y_top > 20:
                page_text = self._get_page_text_after(p, section_y_top)
            else:
                page_text = self._get_page_text(p)
            
            if page_text.strip():
                all_text_parts.append(f"[Page {p}]\n{page_text}")

            for tid in self._page_tables.get(p, []):
                table = self._table_index.get(tid)
                if table:
                    table_text = self._format_table_as_text(table)
                    all_text_parts.append(f"[Table: {table.get('caption', tid)}]\n{table_text}")

        # Handle overflow: content on end_page above next section's heading
        if next_section and end_page not in all_pages:
            overflow = self._get_page_text_before(end_page, next_section)
            if overflow.strip():
                all_text_parts.append(f"[Page {end_page}]\n{overflow}")
                all_pages.append(end_page)

        content = "\n\n".join(all_text_parts)

        return {
            "section_id": section.get("number", section_id),
            "title": section.get("title", ""),
            "level": section.get("level", 0),
            "pages": all_pages,
            "content": content,
            "char_count": len(content),
        }

    def _find_section_end_page(self, section_id: str, start_page: int) -> tuple[int, dict | None]:
        """Find where this section ends (= where next REAL section starts).
        Returns: (end_page, next_section_dict or None)
        """
        all_sections = sorted(
            self._section_index.values(),
            key=lambda s: s.get("page_range", [0])[0] if s.get("page_range") else 9999
        )

        found = False
        for s in all_sections:
            if s.get("number") == section_id:
                found = True
                continue
            if found and s.get("page_range"):
                next_start = s["page_range"][0]
                # Skip sections on the SAME page (subsections, table bookmarks)
                if next_start > start_page:
                    return next_start, s

        # Last section — go 10 pages max
        return min(start_page + 10, self._data.get("total_pages", start_page + 10)), None

    def _get_page_text_before(self, page_num: int, next_section: dict) -> str:
        """Get text from blocks on a page that appear ABOVE the next section's heading.
        
        Uses bbox y-position: blocks whose top edge is above the next section's
        heading top edge belong to the previous section (overflow content).
        """
        blocks = self._page_content.get(page_num, [])
        if not blocks:
            return ""
        
        # Get next section's heading y-position on this page
        next_source = next_section.get("source", {})
        next_page = next_source.get("page", -1)
        next_y_top = next_source.get("bbox", [0, 9999, 0, 0])[1]  # bbox = [x0, y0, x1, y1]
        
        # Only apply if next section's heading is actually on this page
        if next_page != page_num:
            return ""
        
        # Include blocks whose y-top is above the next section heading
        overflow_parts = []
        for block in blocks:
            block_source = block.get("source", {})
            block_y_top = block_source.get("bbox", [0, 9999, 0, 0])[1]
            
            # Block starts above the next section's heading → belongs to previous section
            if block_y_top < next_y_top - 5:  # 5pt tolerance
                overflow_parts.append(block.get("text", ""))
        
        return "\n".join(t for t in overflow_parts if t)

    # ── Tool: search (BM25 over blocks) ──────────────────────────────

    def search(self, keyword: str, max_results: int = 20, query_domain: str = None, manifest: dict = None) -> list[dict]:
        """
        BM25 keyword search across fine-grained blocks.
        Legacy — primary search is now via LlamaIndex retriever.
        """
        if not self._blocks:
            return []
        
        try:
            from knowledge_base.block_index import search_blocks
        except ImportError:
            return []
        
        results = search_blocks(
            query=keyword,
            blocks=self._blocks,
            bm25_data=self._bm25,
            manifest=manifest,
            query_domain=query_domain,
            top_k=max_results,
        )
        
        return [
            {
                "block_id": block["id"],
                "score": round(score, 3),
                "text": block["text"][:200],
                "page": block["page"],
                "section_id": block["section_id"],
                "type": block["type"],
                "snippet": f"...{block['text'][:120]}...",
            }
            for block, score in results
        ]

    # ── Tool: get_section_blocks ───────────────────────────────────

    def get_section_blocks(self, section_id: str) -> list[dict]:
        """
        Get ALL blocks from a section — complete content, no keyword filtering.
        Used for structural routing (manifest says "fetch everything from §3").
        Returns blocks sorted by page order.
        """
        blocks = self._section_blocks.get(section_id, [])
        
        # Also include blocks from sub-sections
        for sid, sblocks in self._section_blocks.items():
            if sid.startswith(section_id + ".") and sid != section_id:
                blocks.extend(sblocks)
        
        # Deduplicate by block ID
        seen = set()
        unique = []
        for b in blocks:
            if b["id"] not in seen:
                seen.add(b["id"])
                unique.append(b)
        
        return sorted(unique, key=lambda b: (b["page"], b["id"]))

    # ── Tool: search (old page-level — kept for backward compat) ───

    def search_pages(self, keyword: str, max_results: int = 20) -> list[dict]:
        """
        BM25-style keyword search across ALL pages.
        Returns pages where keyword appears with context snippets.
        """
        keyword_lower = keyword.lower()
        results = []

        for page_num in sorted(self._page_content.keys()):
            blocks = self._page_content.get(page_num, [])
            page_text = "\n".join(b.get("text", "") for b in blocks if b.get("text"))

            # Also search tables on this page
            for tid in self._page_tables.get(page_num, []):
                table = self._table_index.get(tid, {})
                for row in table.get("rows", []):
                    page_text += " " + " ".join(str(c) for c in row)

            if keyword_lower in page_text.lower():
                # Extract snippet around match
                idx = page_text.lower().find(keyword_lower)
                snippet_start = max(0, idx - 80)
                snippet_end = min(len(page_text), idx + len(keyword) + 80)
                snippet = page_text[snippet_start:snippet_end].strip()

                section_id = self._find_section_for_page(page_num)

                results.append({
                    "page": page_num,
                    "section_id": section_id,
                    "snippet": f"...{snippet}...",
                })

                if len(results) >= max_results:
                    break

        return results

    # ── Tool: get_pages ────────────────────────────────────────────────

    def get_pages(self, page_numbers: list[int]) -> str:
        """
        Fetch content from specific pages.
        Legacy — orchestrator uses _fetch_raw_pages() instead.
        """
        # Collect all blocks from requested pages
        page_set = set(page_numbers)
        page_blocks = [b for b in self._blocks if b["page"] in page_set]
        
        if page_blocks:
            try:
                from knowledge_base.block_renderer import render_blocks_for_llm
                page_blocks.sort(key=lambda b: (b["page"], b["id"]))
                return render_blocks_for_llm(page_blocks)
            except ImportError:
                pass
        
        # Fallback: raw text if no blocks found
        parts = []
        for p in sorted(page_set):
            page_text = self._get_page_text(p)
            section_id = self._find_section_for_page(p)
            if page_text.strip():
                parts.append(
                    f'<block page="{p}" section="{section_id}" type="paragraph">\n'
                    f'{page_text}\n</block>'
                )

            for tid in self._page_tables.get(p, []):
                table = self._table_index.get(tid)
                if table:
                    table_content = self.get_table_content(tid)
                    parts.append(
                        f'<block page="{p}" section="{section_id}" type="table">\n'
                        f'### {table.get("caption", tid)}\n'
                        f'{table_content}\n</block>'
                    )

        return "\n\n".join(parts)

    # ── Tool: get_table ────────────────────────────────────────────────

    def get_table(self, table_id: str) -> dict | None:
        """Retrieve a specific table with headers, rows, footnotes."""
        table = self._table_index.get(table_id)
        if not table:
            return None
        return {
            "table_id": table["id"],
            "caption": table.get("caption", ""),
            "pages": table.get("page_range", []),
            "headers": table.get("column_headers", []),
            "rows": table.get("rows", []),
            "footnotes": table.get("footnotes", {}),
            "column_count": table.get("column_count", 0),
        }

    def get_table_content(self, table_id: str, use_vision: bool = True) -> str:
        """
        Get table content as text. For complex tables (cells > 200 chars,
        indicating nested bullets/criteria), uses GPT-4o vision on page images
        to preserve structure. Falls back to text extraction.
        """
        table = self._table_index.get(table_id)
        if not table:
            return f"[NOT FOUND] Table '{table_id}' not found."

        # Check if table is complex
        if use_vision and self._pdf_path:
            from extraction.vision_table import is_complex_table, extract_table_with_vision
            if is_complex_table(table):
                pages = table.get("page_range", [])
                caption = table.get("caption", "")
                logger.info(f"Complex table {table_id} — using vision extraction")
                vision_result = extract_table_with_vision(
                    self._pdf_path, pages,
                    query_context=f"Table: {caption}"
                )
                if vision_result:
                    return f"[Vision extraction of {table_id}, pages {pages}]\n\n{vision_result}"

        # Fallback: text-based extraction (no truncation)
        return self._format_table_as_text(table)

    # ── Tool: get_abbreviations ────────────────────────────────────────

    def get_abbreviations(self) -> dict[str, str]:
        return self._data.get("abbreviations", {})

    # ── Bookmark discovery ─────────────────────────────────────────────

    def discover_bookmarks(self, pdf_path: str) -> list[dict]:
        """Read PDF bookmarks and merge with existing sections."""
        self._pdf_path = pdf_path  # Store for vision extraction
        try:
            doc = fitz.open(pdf_path)
            toc = doc.get_toc()
            doc.close()
        except Exception as e:
            logger.warning(f"Could not read bookmarks: {e}")
            return []

        if len(toc) < 3:
            return []

        logger.info(f"Found {len(toc)} PDF bookmarks")
        sections = []
        existing = set(self._section_index.keys())

        for level, title, page in toc:
            title = title.strip()
            if not title or page < 1:
                continue

            # Extract section number
            m = re.match(r'^(\d+(?:\.\d+)*)\.?\s+(.*)', title)
            if m:
                number = m.group(1)
                clean_title = m.group(2).strip()
            else:
                # Skip navigational bookmarks (TOC, list of tables, etc.)
                if any(skip in title.upper() for skip in [
                    "TABLE OF CONTENTS", "LIST OF TABLES", "LIST OF FIGURES",
                    "LIST OF ABBREVIATIONS", "SIGNATURE PAGE"
                ]):
                    continue
                # Table/Figure bookmarks → skip (not section boundaries)
                if re.match(r'^(Table|Figure)\s+\d', title, re.IGNORECASE):
                    continue
                number = f"bm_p{page}"
                clean_title = title

            clean_title = re.sub(r'\s*\.{3,}\s*\d*$', '', clean_title).strip()
            if not clean_title:
                clean_title = title

            if number not in existing:
                section_data = {
                    "number": number,
                    "title": clean_title,
                    "level": level - 1,
                    "page_range": [page],
                }
                self._section_index[number] = section_data
                self._data.setdefault("sections", []).append(section_data)
                sections.append(section_data)
                existing.add(number)

        logger.info(f"Injected {len(sections)} new bookmark sections → {len(self._section_index)} total")
        return sections

    # ── Properties ─────────────────────────────────────────────────────

    @property
    def metadata(self) -> dict:
        return {
            "filename": self._data.get("filename", ""),
            "total_pages": self._data.get("total_pages", 0),
            "total_sections": len(self._section_index),
            "total_tables": len(self._table_index),
            "abbreviations": len(self._data.get("abbreviations", {})),
        }

    @property
    def all_section_ids(self) -> list[str]:
        return sorted(
            self._section_index.keys(),
            key=lambda x: [int(p) for p in re.findall(r'\d+', x)] or [9999]
        )

    @property
    def all_table_ids(self) -> list[str]:
        return sorted(self._table_index.keys())

    # ── Helpers ─────────────────────────────────────────────────────────

    def _get_page_text(self, page_num: int) -> str:
        blocks = self._page_content.get(page_num, [])
        return "\n".join(b.get("text", "") for b in blocks if b.get("text"))

    def _get_page_text_after(self, page_num: int, y_cutoff: float) -> str:
        """Get text from blocks on a page AT or BELOW a y-position.
        Used for start pages when a section starts mid-page."""
        blocks = self._page_content.get(page_num, [])
        parts = []
        for block in blocks:
            block_source = block.get("source", {})
            block_y_top = block_source.get("bbox", [0, 0, 0, 0])[1]
            # Block starts at or below the section heading
            if block_y_top >= y_cutoff - 5:  # 5pt tolerance
                text = block.get("text", "")
                if text:
                    parts.append(text)
        return "\n".join(parts)

    def _find_section_for_page(self, page_num: int) -> str:
        """Find which section owns a page. When multiple sections share a page,
        picks the latest in document order (§3 over §2.3.1 on same page)."""
        candidates = []
        for section in self._data.get("sections", []):
            pages = section.get("page_range", [])
            if pages and pages[0] <= page_num:
                candidates.append(section)
        
        if not candidates:
            return ""
        
        candidates.sort(key=lambda s: s.get("page_range", [0])[0], reverse=True)
        nearest_page = candidates[0].get("page_range", [0])[0]
        same_page = [s for s in candidates if s.get("page_range", [0])[0] == nearest_page]
        
        if len(same_page) == 1:
            return same_page[0].get("number", "")
        
        def section_sort_key(s):
            num = s.get("number", "")
            parts = []
            for p in num.replace(".", " ").split():
                try: parts.append(int(p))
                except ValueError: parts.append(999)
            return parts
        
        same_page.sort(key=section_sort_key, reverse=True)
        return same_page[0].get("number", "")

    def _format_table_as_text(self, table: dict) -> str:
        lines = []
        headers = table.get("column_headers", [])
        rows = table.get("rows", [])
        
        # Detect and remove empty columns (merged cell artifacts)
        if rows:
            num_cols = max(len(headers), max((len(r) for r in rows), default=0))
            non_empty = [0] * num_cols
            for row in rows:
                for j, cell in enumerate(row):
                    if j < num_cols and str(cell).strip():
                        non_empty[j] += 1
            
            keep = [j for j in range(num_cols) if non_empty[j] >= max(1, len(rows) * 0.2)]
            
            if keep and len(keep) < num_cols:
                fh = [str(headers[j]) for j in keep if j < len(headers) and str(headers[j]).strip()]
                if fh:
                    lines.append(" | ".join(fh))
                    lines.append("-" * 40)
                for row in rows:
                    cells = [str(row[j]) if j < len(row) else "" for j in keep]
                    if any(c.strip() for c in cells):
                        lines.append(" | ".join(cells))
            else:
                if headers:
                    lines.append(" | ".join(str(h) for h in headers))
                    lines.append("-" * 40)
                for row in rows:
                    lines.append(" | ".join(str(c) for c in row))
        
        if table.get("footnotes"):
            lines.append("")
            for marker, text in table["footnotes"].items():
                lines.append(f"  {marker}. {text}")
        return "\n".join(lines)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
