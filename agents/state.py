"""
Protocol Intelligence State.

ProtocolState: Serializable graph state (flows between nodes).
RuntimeContext: Non-serializable objects passed via config["configurable"]["runtime"].
"""
from __future__ import annotations
import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict, Any


def _merge_dicts(current: dict, update: dict) -> dict:
    if current is None: return update or {}
    if update is None: return current
    merged = dict(current)
    merged.update(update)
    return merged


class ProtocolState(TypedDict):
    query: str
    query_type: str
    pdf_path: str

    # Gatherer output
    sections_content: Annotated[dict, _merge_dicts]
    tables_content: Annotated[dict, _merge_dicts]
    sections_read: Annotated[list, operator.add]

    # Extractor output
    extracted_data: dict
    validation: dict

    # Reviewer output
    signals: list[dict]

    # Metadata
    steps: Annotated[list, operator.add]
    error: str


@dataclass
class RuntimeContext:
    retriever: Any = None
    store: Any = None
    json_data: dict = field(default_factory=dict)
    event_bus: Any = None


def get_runtime(config: dict) -> RuntimeContext:
    return config.get("configurable", {}).get("runtime", RuntimeContext())
