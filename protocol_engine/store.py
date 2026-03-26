"""
Protocol Store — Self-contained content store for the new package.

Wraps JSON data from the ingestion pipeline and provides section/table
lookup methods used by the retrieval engine and graph nodes.

This replaces the old knowledge_base/protocol_store.py import with a
clean, self-contained version that has NO old-code dependencies.
"""
from __future__ import annotations

import json
import re
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ProtocolStore:
    """In-memory store for a parsed protocol document."""

    def __init__(self, data: dict | str | Path):
        """Initialize from a dict or JSON file path."""
        if isinstance(data, (str, Path)):
            with open(data, encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = data

        self._section_index: dict[str, dict] = {}
        self._page_content: dict[int, list[dict]] = {}
        self._table_index: dict[str, dict] = {}
        self._page_tables: dict[int, list[str]] = {}
        self._pdf_path: str | None = None

        self._build_indexes()

    def _build_indexes(self):
        for section in self._data.get("sections", []):
            number = section.get("number", section.get("id", ""))
            if number:
                self._section_index[number] = section

        for page in self._data.get("pages", []):
            self._page_content[page["page_num"]] = page.get("content_blocks", [])

        for table in self._data.get("tables", []):
            tid = table.get("id", "")
            if tid:
                self._table_index[tid] = table
                for p in table.get("page_range", []):
                    self._page_tables.setdefault(p, []).append(tid)

    def get_section(self, section_id: str) -> dict | None:
        """Retrieve a section's complete content."""
        section = self._section_index.get(section_id)
        if not section:
            return None

        # Use content_blocks if available
        blocks = section.get("content_blocks", [])
        if blocks:
            text_parts = []
            block_pages = set()
            for block in blocks:
                text = block.get("text", "") if isinstance(block, dict) else ""
                source = block.get("source", {}) if isinstance(block, dict) else {}
                page = source.get("page")
                if page is not None:
                    block_pages.add(page)
                if text.strip():
                    text_parts.append(text)

            all_pages = sorted(block_pages) if block_pages else section.get("page_range", [])
            for p in all_pages:
                for tid in self._page_tables.get(p, []):
                    table = self._table_index.get(tid)
                    if table:
                        table_text = self._format_table(table)
                        text_parts.append(f"[Table: {table.get('caption', tid)}]\n{table_text}")

            content = "\n\n".join(text_parts)
            return {
                "section_id": section.get("number", section_id),
                "title": section.get("title", ""),
                "level": section.get("level", 0),
                "pages": sorted(block_pages) if block_pages else all_pages,
                "content": content,
            }

        # Fallback: page-range reconstruction
        pages = section.get("page_range", [])
        if not pages:
            return None

        text_parts = []
        for p in pages:
            page_text = self._get_page_text(p)
            if page_text.strip():
                text_parts.append(f"[Page {p}]\n{page_text}")
            for tid in self._page_tables.get(p, []):
                table = self._table_index.get(tid)
                if table:
                    text_parts.append(f"[Table: {table.get('caption', tid)}]\n{self._format_table(table)}")

        content = "\n\n".join(text_parts)
        return {
            "section_id": section.get("number", section_id),
            "title": section.get("title", ""),
            "level": section.get("level", 0),
            "pages": pages,
            "content": content,
        }

    def get_table(self, table_id: str) -> dict | None:
        """Retrieve a specific table."""
        return self._table_index.get(table_id)

    @property
    def all_section_ids(self) -> list[str]:
        return sorted(
            self._section_index.keys(),
            key=lambda x: [int(p) for p in re.findall(r'\d+', x)] or [9999],
        )

    @property
    def all_table_ids(self) -> list[str]:
        return sorted(self._table_index.keys())

    @property
    def metadata(self) -> dict:
        return {
            "filename": self._data.get("filename", ""),
            "total_pages": self._data.get("total_pages", 0),
            "total_sections": len(self._section_index),
            "total_tables": len(self._table_index),
        }

    def _get_page_text(self, page_num: int) -> str:
        blocks = self._page_content.get(page_num, [])
        return "\n".join(b.get("text", "") for b in blocks if b.get("text"))

    def _format_table(self, table: dict) -> str:
        lines = []
        headers = table.get("column_headers", [])
        rows = table.get("rows", [])
        if headers:
            lines.append(" | ".join(str(h) for h in headers))
            lines.append("-" * 40)
        for row in rows:
            lines.append(" | ".join(str(c) for c in row))
        footnotes = table.get("footnotes", {})
        if footnotes:
            lines.append("")
            for marker, text in footnotes.items():
                lines.append(f"  {marker}. {text}")
        return "\n".join(lines)
