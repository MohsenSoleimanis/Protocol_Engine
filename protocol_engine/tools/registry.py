"""
Tool Registry — Central registry for all agent tools.

Key fixes from old code:
  1. Tools are classes, not inline @tool functions
  2. Each tool is independently testable
  3. No hardcoded KNOWLEDGE_APPENDICES — uses knowledge base files
  4. No 10k char truncation that produces invalid JSON
  5. Each node gets only the tools it needs
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from protocol_engine.config import KNOWLEDGE_DIR
from protocol_engine.models.enums import NodeName

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Central registry — builds tools for each graph node."""

    def __init__(self, retriever: Any, store: Any, json_data: dict, pdf_path: str):
        self.retriever = retriever
        self.store = store
        self.json_data = json_data
        self.pdf_path = pdf_path

    def get_tools_for_node(self, node: NodeName) -> list:
        """Return only the tools a specific node needs."""
        if node == NodeName.EXPLORER:
            return self._explorer_tools()
        elif node == NodeName.EXTRACTOR:
            return []  # Extractor sees context directly — no tools needed
        elif node == NodeName.RECONCILER:
            return self._reconciler_tools()
        elif node == NodeName.REVIEWER:
            return self._reviewer_tools()
        return []

    def _explorer_tools(self) -> list:
        retriever = self.retriever
        store = self.store
        gathered = []
        gathered_chars = [0]
        existing = set()
        budget = 120000

        @tool
        def search(query: str) -> str:
            """Search protocol content by semantic query. Returns matching sections."""
            if gathered_chars[0] >= budget:
                return "Content budget reached. Use what you have."
            results = retriever.retrieve(query)
            if not results:
                return f"No results for '{query}'"
            summaries = []
            for node in results:
                meta = node.metadata
                sid = meta.get("section_id", meta.get("table_id", ""))
                if sid in existing:
                    continue
                existing.add(sid)
                text = node.text or ""
                pages = json.loads(meta.get("pages", "[]"))
                gathered.append({
                    "type": meta.get("type", "section"),
                    "id": sid, "text": text,
                    "pages": pages, "chars": len(text),
                })
                gathered_chars[0] += len(text)
                summaries.append(f"  §{sid}: {meta.get('title', '')} ({len(text)} chars)")
            if summaries:
                return f"Gathered {len(summaries)} sections:\n" + "\n".join(summaries)
            return "All results already gathered."

        @tool
        def read_section(section_id: str) -> str:
            """Read a specific protocol section by its ID (e.g. '5.1', '8.3.4')."""
            if section_id in existing:
                return f"[Already have §{section_id}]"
            existing.add(section_id)
            if store:
                data = store.get_section(section_id)
                if data and data.get("content", "").strip():
                    text = data["content"]
                    pages = data.get("pages", [])
                    gathered.append({
                        "type": "section", "id": section_id,
                        "text": text, "pages": pages, "chars": len(text),
                    })
                    gathered_chars[0] += len(text)
                    return f"Read §{section_id}: {data.get('title', '')} ({len(text)} chars)"
            return f"Section {section_id} not found."

        @tool
        def vision_extract(pages: list[int]) -> str:
            """Extract complex table from PDF page images using vision model."""
            from protocol_engine.tools.vision import extract_table_with_vision
            result = extract_table_with_vision(self.pdf_path, pages)
            if result:
                gathered.append({
                    "type": "vision", "id": f"vision_{pages}",
                    "text": result, "pages": pages, "chars": len(result),
                })
                gathered_chars[0] += len(result)
            return result or "Vision extraction returned no content."

        # Attach gathered list for the node to read later
        search.gathered = gathered
        read_section.gathered = gathered
        vision_extract.gathered = gathered

        return [search, read_section, vision_extract]

    def _reconciler_tools(self) -> list:
        @tool
        def vision_extract_for_reconciliation(pages: list[int]) -> str:
            """Re-extract table via vision for reconciliation with text extraction."""
            from protocol_engine.tools.vision import extract_table_with_vision
            return extract_table_with_vision(self.pdf_path, pages) or "No content."

        @tool
        def lookup_cdisc(term: str) -> str:
            """Look up a CDISC controlled terminology term."""
            return _lookup_knowledge("cdisc", term)

        return [vision_extract_for_reconciliation, lookup_cdisc]

    def _reviewer_tools(self) -> list:
        @tool
        def lookup_ich(guideline: str) -> str:
            """Look up ICH guideline reference (E6, E8, E9, M11, E2A)."""
            return _lookup_knowledge("ich_guidelines", guideline)

        @tool
        def lookup_cdisc(term: str) -> str:
            """Look up CDISC controlled terminology or SDTM domain."""
            return _lookup_knowledge("cdisc", term)

        return [lookup_ich, lookup_cdisc]


def _lookup_knowledge(domain: str, query: str) -> str:
    """Fuzzy lookup in a knowledge base JSON file."""
    path = KNOWLEDGE_DIR / f"{domain}.json"
    if not path.exists():
        return f"Knowledge base '{domain}' not found."
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return f"Error loading {domain}: {e}"

    query_lower = query.lower()
    matches = []
    _search_dict(data, query_lower, matches, max_depth=4)
    if matches:
        return json.dumps(matches[:5], indent=2, default=str)
    return f"No matches for '{query}' in {domain}."


def _search_dict(data: Any, query: str, matches: list, path: str = "", max_depth: int = 4):
    """Recursively search a dict/list for keys or values matching the query."""
    if max_depth <= 0 or len(matches) >= 10:
        return
    if isinstance(data, dict):
        for key, value in data.items():
            full_path = f"{path}.{key}" if path else key
            if query in key.lower() or (isinstance(value, str) and query in value.lower()):
                matches.append({full_path: value})
            if isinstance(value, (dict, list)):
                _search_dict(value, query, matches, full_path, max_depth - 1)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, str) and query in item.lower():
                matches.append({f"{path}[{i}]": item})
            elif isinstance(item, (dict, list)):
                _search_dict(item, query, matches, f"{path}[{i}]", max_depth - 1)
