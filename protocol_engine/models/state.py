"""
Graph state — 10 essential fields, no ghosts.

Every field here is READ by at least one downstream node.
Diagnostic data goes into the `steps` list, not separate fields.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict, Any

from protocol_engine.models.enums import EdgeSignal


def _merge_dicts(current: dict, update: dict) -> dict:
    if current is None:
        return update or {}
    if update is None:
        return current
    merged = dict(current)
    merged.update(update)
    return merged


class ProtocolState(TypedDict):
    # Input
    query: str
    query_type: str
    pdf_path: str

    # Content (Explorer writes, Extractor reads)
    sections_content: Annotated[dict, _merge_dicts]
    tables_content: Annotated[dict, _merge_dicts]
    assembled_context: str

    # Extraction (Extractor writes, Reviewer reads)
    extracted_data: dict
    validation: dict

    # Review (Reviewer writes, edge routing reads)
    signals: Annotated[list, operator.add]

    # Control
    edge_signal: str
    cycle_count: int
    steps: Annotated[list, operator.add]


@dataclass
class RuntimeContext:
    retriever: Any = None
    store: Any = None
    json_data: dict = field(default_factory=dict)
    event_bus: Any = None


def get_runtime(config: dict) -> RuntimeContext:
    return config.get("configurable", {}).get("runtime", RuntimeContext())
