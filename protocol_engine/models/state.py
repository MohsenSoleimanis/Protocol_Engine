"""
Protocol Engine — LangGraph State.

Key fixes from old code:
  1. `signals` now has operator.add reducer (old code dropped all but last)
  2. `retrieved_sections` has append reducer (old code replaced on each cycle)
  3. Token-based budgets instead of char-based
  4. Typed EdgeSignal instead of string-based "NEED_MORE:"
  5. Consistent node naming (no "Gatherer" vs "Explorer" mismatch)
"""
from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict, Any

from protocol_engine.models.enums import EdgeSignal, QueryType, NodeName


# ── Reducers ─────────────────────────────────────────────────────────────────

def _merge_dicts(current: dict, update: dict) -> dict:
    """Merge dicts — used as reducer for sections_content/tables_content."""
    if current is None:
        return update or {}
    if update is None:
        return current
    merged = dict(current)
    merged.update(update)
    return merged


# ── Sub-task (from Planner) ──────────────────────────────────────────────────

class SubTask(TypedDict):
    """A decomposed sub-task from the Planner node."""
    query: str
    query_type: str
    description: str
    completed: bool


# ── Validation result ────────────────────────────────────────────────────────

class ValidationResult(TypedDict, total=False):
    """Deterministic validation output."""
    verified: int
    flagged: int
    failed: int
    total: int
    details: list[dict]


# ── Signal (from Reviewer) ───────────────────────────────────────────────────

class Signal(TypedDict, total=False):
    """A review signal flagging an issue."""
    signal_type: str    # ambiguity, completeness, consistency, cross_reference, hallucination
    severity: str       # critical, major, minor
    title: str
    description: str


# ── Main Graph State ─────────────────────────────────────────────────────────

class ProtocolState(TypedDict):
    """LangGraph state flowing between all 7 nodes.

    Uses Annotated reducers so that:
    - signals APPEND across nodes (not replace)
    - sections_content MERGES (not replaces)
    - steps APPEND (not replace)
    - retrieved_sections APPEND (not replace)
    """
    # ── Query ────────────────────────────────────────────────────────────
    query: str
    query_type: str                               # QueryType value
    pdf_path: str

    # ── Planner output ───────────────────────────────────────────────────
    sub_tasks: list[SubTask]
    current_task_index: int

    # ── Explorer output (content gathering) ──────────────────────────────
    sections_content: Annotated[dict, _merge_dicts]    # sid → {text, pages, chars}
    tables_content: Annotated[dict, _merge_dicts]      # tid → {text, pages, chars}
    sections_read: Annotated[list, operator.add]       # list of section/table IDs

    # ── Context Assembler output ─────────────────────────────────────────
    assembled_context: str                             # relevance-scored, token-budgeted
    context_tokens_used: int
    context_sections_included: int
    context_relevance_scores: dict                     # sid → float

    # ── Extractor output ─────────────────────────────────────────────────
    extracted_data: dict
    extraction_history: Annotated[list, operator.add]  # all extraction attempts

    # ── Validation (deterministic) ───────────────────────────────────────
    validation: dict                                   # ValidationResult

    # ── Reviewer output ──────────────────────────────────────────────────
    signals: Annotated[list, operator.add]             # FIXED: was list without reducer

    # ── Control flow ─────────────────────────────────────────────────────
    edge_signal: str                                   # EdgeSignal value
    edge_detail: str                                   # what's needed (replaces NEED_MORE: string)
    cycle_count: int
    steps: Annotated[list, operator.add]               # execution trace
    error: str


# ── Runtime Context (non-serializable, passed via config) ────────────────────

@dataclass
class RuntimeContext:
    """Non-serializable objects passed via config['configurable']['runtime']."""
    retriever: Any = None
    store: Any = None
    json_data: dict = field(default_factory=dict)
    event_bus: Any = None


def get_runtime(config: dict) -> RuntimeContext:
    """Extract RuntimeContext from LangGraph config."""
    return config.get("configurable", {}).get("runtime", RuntimeContext())
