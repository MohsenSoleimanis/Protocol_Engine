"""MCP tool: search_protocol — semantic search over protocol content."""
from __future__ import annotations

import json
from langchain_core.tools import tool


def make_search_tool(retriever, gathered: dict, budget: list[int]):
    """Create a search tool bound to a retriever instance.

    Args:
        retriever: Hybrid retriever
        gathered: Mutable dict collecting {id: {text, pages, chars, type}}
        budget: Mutable [chars_used] list for budget tracking
    """

    @tool
    def search_protocol(query: str) -> str:
        """Search protocol content by semantic query. Returns matching section summaries."""
        if budget[0] >= 150_000:
            return "Content budget reached."
        results = retriever.retrieve(query)
        if not results:
            return f"No results for '{query}'"
        added = []
        for node in results:
            meta = node.metadata
            sid = meta.get("section_id", meta.get("table_id", ""))
            if not sid or sid in gathered:
                continue
            text = node.text or ""
            try:
                pages = json.loads(meta.get("pages", "[]"))
            except (json.JSONDecodeError, TypeError):
                pages = []
            ntype = meta.get("type", "section")
            gathered[sid] = {"text": text, "pages": pages, "chars": len(text), "type": ntype}
            budget[0] += len(text)
            added.append(f"  §{sid}: {meta.get('title', '')} ({len(text)} chars)")
        if added:
            return f"Found {len(added)} sections:\n" + "\n".join(added)
        return "All results already gathered."

    return search_protocol
