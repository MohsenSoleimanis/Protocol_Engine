"""MCP tool: lookup_knowledge — query CDISC/ICH/SDTM knowledge bases."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from protocol_engine.config import KNOWLEDGE_DIR


@tool
def lookup_knowledge(domain: str, term: str) -> str:
    """Look up clinical terminology. domain: 'cdisc', 'ich_guidelines', or 'sdtm_mappings'. term: search query."""
    path = KNOWLEDGE_DIR / f"{domain}.json"
    if not path.exists():
        return f"Unknown domain '{domain}'. Use: cdisc, ich_guidelines, sdtm_mappings"
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        return f"Error: {e}"
    matches = []
    _search(data, term.lower(), matches, max_depth=4)
    if matches:
        return json.dumps(matches[:5], indent=2, default=str)
    return f"No matches for '{term}' in {domain}."


def _search(data: Any, query: str, matches: list, path: str = "", max_depth: int = 4):
    if max_depth <= 0 or len(matches) >= 10:
        return
    if isinstance(data, dict):
        for k, v in data.items():
            fp = f"{path}.{k}" if path else k
            if query in k.lower() or (isinstance(v, str) and query in v.lower()):
                matches.append({fp: v})
            if isinstance(v, (dict, list)):
                _search(v, query, matches, fp, max_depth - 1)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, str) and query in item.lower():
                matches.append({f"{path}[{i}]": item})
            elif isinstance(item, (dict, list)):
                _search(item, query, matches, f"{path}[{i}]", max_depth - 1)
