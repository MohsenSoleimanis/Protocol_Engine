"""MCP tool: read_section — read a specific protocol section by ID."""
from __future__ import annotations

from langchain_core.tools import tool


def make_read_section_tool(store, gathered: dict, budget: list[int]):
    """Create a read_section tool bound to a store instance."""

    @tool
    def read_section(section_id: str) -> str:
        """Read a specific protocol section by its ID (e.g. '5.1', '8.3.4')."""
        if section_id in gathered:
            return f"[Already have §{section_id}]"
        if not store:
            return f"Section {section_id} not found."
        data = store.get_section(section_id)
        if not data or not data.get("content", "").strip():
            return f"Section {section_id} not found."
        text = data["content"]
        pages = data.get("pages", [])
        gathered[section_id] = {"text": text, "pages": pages, "chars": len(text), "type": "section"}
        budget[0] += len(text)
        return f"Read §{section_id}: {data.get('title', '')} ({len(text)} chars)"

    return read_section
