"""
Protocol Store — Content store for parsed protocol data.

Wraps JSON from ingestion, provides section/table lookup.
"""
from __future__ import annotations

import json
import re
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ProtocolStore:
    def __init__(self, data: dict | str | Path):
        if isinstance(data, (str, Path)):
            with open(data, encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = data

        self._sections: dict[str, dict] = {}
        self._pages: dict[int, list[dict]] = {}
        self._tables: dict[str, dict] = {}
        self._page_tables: dict[int, list[str]] = {}
        self._build()

    def _build(self):
        for s in self._data.get("sections", []):
            num = s.get("number", s.get("id", ""))
            if num:
                self._sections[num] = s
        for p in self._data.get("pages", []):
            self._pages[p["page_num"]] = p.get("content_blocks", [])
        for t in self._data.get("tables", []):
            tid = t.get("id", "")
            if tid:
                self._tables[tid] = t
                for p in t.get("page_range", []):
                    self._page_tables.setdefault(p, []).append(tid)

    def get_section(self, section_id: str) -> dict | None:
        section = self._sections.get(section_id)
        if not section:
            return None

        blocks = section.get("content_blocks", [])
        if blocks:
            parts, block_pages = [], set()
            for b in blocks:
                text = b.get("text", "") if isinstance(b, dict) else ""
                pg = (b.get("source", {}) if isinstance(b, dict) else {}).get("page")
                if pg is not None:
                    block_pages.add(pg)
                if text.strip():
                    parts.append(text)
            pages = sorted(block_pages) if block_pages else section.get("page_range", [])
            for p in pages:
                for tid in self._page_tables.get(p, []):
                    t = self._tables.get(tid)
                    if t:
                        parts.append(f"[Table: {t.get('caption', tid)}]\n{self._fmt_table(t)}")
            return {"section_id": section.get("number", section_id),
                    "title": section.get("title", ""), "pages": pages,
                    "content": "\n\n".join(parts)}

        pages = section.get("page_range", [])
        if not pages:
            return None
        parts = []
        for p in pages:
            pt = "\n".join(b.get("text", "") for b in self._pages.get(p, []) if b.get("text"))
            if pt.strip():
                parts.append(f"[Page {p}]\n{pt}")
            for tid in self._page_tables.get(p, []):
                t = self._tables.get(tid)
                if t:
                    parts.append(f"[Table: {t.get('caption', tid)}]\n{self._fmt_table(t)}")
        return {"section_id": section.get("number", section_id),
                "title": section.get("title", ""), "pages": pages,
                "content": "\n\n".join(parts)}

    @property
    def all_section_ids(self) -> list[str]:
        return sorted(self._sections.keys(),
                      key=lambda x: [int(p) for p in re.findall(r'\d+', x)] or [9999])

    @property
    def all_table_ids(self) -> list[str]:
        return sorted(self._tables.keys())

    @property
    def metadata(self) -> dict:
        return {"filename": self._data.get("filename", ""),
                "total_pages": self._data.get("total_pages", 0),
                "sections": len(self._sections), "tables": len(self._tables)}

    def _fmt_table(self, t: dict) -> str:
        lines = []
        hdrs = t.get("column_headers", [])
        if hdrs:
            lines.append(" | ".join(str(h) for h in hdrs))
            lines.append("-" * 40)
        for row in t.get("rows", []):
            lines.append(" | ".join(str(c) for c in row))
        for marker, text in t.get("footnotes", {}).items():
            lines.append(f"  {marker}. {text}")
        return "\n".join(lines)
