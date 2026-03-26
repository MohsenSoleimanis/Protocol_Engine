"""
Graph Edges — Typed conditional routing using EdgeSignal enum.

Key fixes from old code:
  1. No string-based "NEED_MORE:" parsing
  2. Typed EdgeSignal enum for all routing decisions
  3. Consistent cycle counting (no "Gatherer" vs "Explorer" mismatch)
  4. Empty extraction → error retry (not silent END)
  5. Reviewer → Extractor edge (re-extract with same content)
"""
from __future__ import annotations

import logging

from langgraph.graph import END

from protocol_engine.config import MAX_CYCLES
from protocol_engine.models.enums import EdgeSignal, NodeName

logger = logging.getLogger(__name__)


def route_after_router(state: dict) -> str:
    """Router → Planner (always, planner handles simple passthrough)."""
    return NodeName.PLANNER


def route_after_planner(state: dict) -> str:
    """Planner → Explorer (always)."""
    return NodeName.EXPLORER


def route_after_explorer(state: dict) -> str:
    """Explorer → Context Assembler (always)."""
    return NodeName.CONTEXT_ASSEMBLER


def route_after_context_assembler(state: dict) -> str:
    """Context Assembler → Extractor (always)."""
    return NodeName.EXTRACTOR


def route_after_extractor(state: dict) -> str:
    """Extractor → Reconciler or error handling."""
    edge_signal = state.get("edge_signal", "")
    extracted = state.get("extracted_data", {})

    if edge_signal == EdgeSignal.ERROR_RETRY:
        cycle_count = state.get("cycle_count", 0)
        if cycle_count < MAX_CYCLES:
            logger.info("Extractor error → retry Explorer")
            return NodeName.EXPLORER
        logger.warning("Extractor error → max retries reached → END")
        return END

    if edge_signal == EdgeSignal.ERROR_FATAL:
        return END

    if not extracted:
        logger.warning("Empty extraction → END")
        return END

    return NodeName.RECONCILER


def route_after_reconciler(state: dict) -> str:
    """Reconciler → Reviewer or back to Explorer for more content."""
    edge_signal = state.get("edge_signal", "")

    if edge_signal == EdgeSignal.NEED_MORE_CONTENT:
        cycle_count = state.get("cycle_count", 0)
        if cycle_count < MAX_CYCLES:
            logger.info(f"Reconciler needs more content → Explorer (cycle {cycle_count + 1})")
            return "increment_cycle"
        logger.info("Reconciler needs more but max cycles reached → Reviewer")
        return NodeName.REVIEWER

    if edge_signal == EdgeSignal.NEED_VISION:
        return NodeName.EXPLORER  # Explorer handles vision

    return NodeName.REVIEWER


def route_after_reviewer(state: dict) -> str:
    """Reviewer → END, or back to Explorer for critical issues."""
    edge_signal = state.get("edge_signal", "")

    if edge_signal == EdgeSignal.NEED_MORE_CONTENT:
        cycle_count = state.get("cycle_count", 0)
        if cycle_count < MAX_CYCLES:
            logger.info(f"Reviewer needs more content → Explorer (cycle {cycle_count + 1})")
            return "increment_cycle"
        logger.info("Reviewer needs more but max cycles reached → END")

    if edge_signal == EdgeSignal.NEED_REEXTRACT:
        cycle_count = state.get("cycle_count", 0)
        if cycle_count < MAX_CYCLES:
            logger.info("Reviewer wants re-extraction → Extractor")
            return "increment_cycle_extractor"

    return END


def increment_cycle(state: dict) -> dict:
    """Increment cycle count before routing back to Explorer."""
    return {
        "cycle_count": state.get("cycle_count", 0) + 1,
        "edge_signal": EdgeSignal.NEED_MORE_CONTENT,
    }


def increment_cycle_extractor(state: dict) -> dict:
    """Increment cycle count before routing back to Extractor."""
    return {
        "cycle_count": state.get("cycle_count", 0) + 1,
        "edge_signal": EdgeSignal.NEED_REEXTRACT,
    }
