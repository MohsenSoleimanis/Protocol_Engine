"""
Graph Builder — Constructs the 7-node LangGraph pipeline.

    START → Router → Planner → Explorer → Context Assembler → Extractor
              → Reconciler → Reviewer → END

With cycle edges:
  Reconciler → Explorer (need more content)
  Reviewer → Explorer (critical completeness gaps)
  Reviewer → Extractor (re-extract with same content)
  Extractor → Explorer (extraction error retry)
"""
from __future__ import annotations

import logging

from langgraph.graph import StateGraph, START, END

from protocol_engine.models.state import ProtocolState
from protocol_engine.models.enums import NodeName
from protocol_engine.graph.nodes.router import router_node
from protocol_engine.graph.nodes.planner import planner_node
from protocol_engine.graph.nodes.explorer import explorer_node
from protocol_engine.graph.nodes.context_assembler import context_assembler_node
from protocol_engine.graph.nodes.extractor import extractor_node
from protocol_engine.graph.nodes.reconciler import reconciler_node
from protocol_engine.graph.nodes.reviewer import reviewer_node
from protocol_engine.graph.edges import (
    route_after_router,
    route_after_planner,
    route_after_explorer,
    route_after_context_assembler,
    route_after_extractor,
    route_after_reconciler,
    route_after_reviewer,
    increment_cycle,
    increment_cycle_extractor,
)

logger = logging.getLogger(__name__)


def build_graph():
    """Build and compile the protocol extraction graph."""
    g = StateGraph(ProtocolState)

    # Add all 7 nodes
    g.add_node(NodeName.ROUTER, router_node)
    g.add_node(NodeName.PLANNER, planner_node)
    g.add_node(NodeName.EXPLORER, explorer_node)
    g.add_node(NodeName.CONTEXT_ASSEMBLER, context_assembler_node)
    g.add_node(NodeName.EXTRACTOR, extractor_node)
    g.add_node(NodeName.RECONCILER, reconciler_node)
    g.add_node(NodeName.REVIEWER, reviewer_node)

    # Helper nodes for cycle increment
    g.add_node("increment_cycle", increment_cycle)
    g.add_node("increment_cycle_extractor", increment_cycle_extractor)

    # START → Router
    g.add_edge(START, NodeName.ROUTER)

    # Router → Planner (always)
    g.add_conditional_edges(
        NodeName.ROUTER, route_after_router,
        {NodeName.PLANNER: NodeName.PLANNER},
    )

    # Planner → Explorer (always)
    g.add_conditional_edges(
        NodeName.PLANNER, route_after_planner,
        {NodeName.EXPLORER: NodeName.EXPLORER},
    )

    # Explorer → Context Assembler (always)
    g.add_conditional_edges(
        NodeName.EXPLORER, route_after_explorer,
        {NodeName.CONTEXT_ASSEMBLER: NodeName.CONTEXT_ASSEMBLER},
    )

    # Context Assembler → Extractor (always)
    g.add_conditional_edges(
        NodeName.CONTEXT_ASSEMBLER, route_after_context_assembler,
        {NodeName.EXTRACTOR: NodeName.EXTRACTOR},
    )

    # Extractor → Reconciler | Explorer (retry) | END
    g.add_conditional_edges(
        NodeName.EXTRACTOR, route_after_extractor,
        {
            NodeName.RECONCILER: NodeName.RECONCILER,
            NodeName.EXPLORER: NodeName.EXPLORER,
            END: END,
        },
    )

    # Reconciler → Reviewer | increment_cycle (→ Explorer)
    g.add_conditional_edges(
        NodeName.RECONCILER, route_after_reconciler,
        {
            NodeName.REVIEWER: NodeName.REVIEWER,
            "increment_cycle": "increment_cycle",
        },
    )

    # Reviewer → END | increment_cycle (→ Explorer) | increment_cycle_extractor (→ Extractor)
    g.add_conditional_edges(
        NodeName.REVIEWER, route_after_reviewer,
        {
            END: END,
            "increment_cycle": "increment_cycle",
            "increment_cycle_extractor": "increment_cycle_extractor",
        },
    )

    # Cycle helper → Explorer/Extractor
    g.add_edge("increment_cycle", NodeName.EXPLORER)
    g.add_edge("increment_cycle_extractor", NodeName.EXTRACTOR)

    return g.compile()


# Pre-built graph instance
protocol_graph = build_graph()
