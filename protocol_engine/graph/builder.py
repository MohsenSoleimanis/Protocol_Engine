"""
Graph Builder — 3-node pipeline with one cycle edge.

    START → Explorer → Extractor → Reviewer → END
                ↑                       |
                └───── (need_more) ─────┘

That's it. No router, planner, context assembler, or reconciler.
Context assembly is inside Explorer. Cross-field checks are inside Extractor.
"""
from __future__ import annotations

import logging

from langgraph.graph import StateGraph, START, END

from protocol_engine.models.state import ProtocolState
from protocol_engine.models.enums import EdgeSignal
from protocol_engine.config import MAX_CYCLES
from protocol_engine.graph.nodes.explorer import explorer_node
from protocol_engine.graph.nodes.extractor import extractor_node
from protocol_engine.graph.nodes.reviewer import reviewer_node

logger = logging.getLogger(__name__)


def _route_after_extractor(state: dict) -> str:
    """Extractor → Reviewer if we have data, else END."""
    if state.get("extracted_data"):
        return "reviewer"
    return END


def _route_after_reviewer(state: dict) -> str:
    """Reviewer → Explorer (cycle) or END."""
    if state.get("edge_signal") == EdgeSignal.NEED_MORE:
        cycle = state.get("cycle_count", 0)
        if cycle < MAX_CYCLES:
            logger.info(f"Reviewer: need more → Explorer (cycle {cycle + 1})")
            return "increment_cycle"
        logger.info("Reviewer: max cycles reached → END")
    return END


def _increment_cycle(state: dict) -> dict:
    """Bump cycle count before re-entering Explorer."""
    return {
        "cycle_count": state.get("cycle_count", 0) + 1,
        "edge_signal": EdgeSignal.NEED_MORE,
    }


def build_graph():
    g = StateGraph(ProtocolState)

    g.add_node("explorer", explorer_node)
    g.add_node("extractor", extractor_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("increment_cycle", _increment_cycle)

    g.add_edge(START, "explorer")
    g.add_edge("explorer", "extractor")
    g.add_conditional_edges("extractor", _route_after_extractor,
                            {"reviewer": "reviewer", END: END})
    g.add_conditional_edges("reviewer", _route_after_reviewer,
                            {"increment_cycle": "increment_cycle", END: END})
    g.add_edge("increment_cycle", "explorer")

    return g.compile()


protocol_graph = build_graph()
