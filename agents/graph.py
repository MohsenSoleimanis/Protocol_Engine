"""
Protocol Intelligence Graph — LangGraph pipeline.

    START → Explorer → Extractor → Reviewer → END

With cycles for complex query types:
  Extractor → Explorer (content insufficient via NEED_MORE)
  Reviewer → Explorer (critical completeness gaps)
"""
from __future__ import annotations
import time, logging
from langgraph.graph import StateGraph, START, END
from agents.state import ProtocolState, RuntimeContext

logger = logging.getLogger(__name__)


def route_after_extractor(state: dict) -> str:
    error = state.get("error", "")
    if error.startswith("NEED_MORE:"):
        from shared.registry import get_config
        config = get_config(state.get("query_type", ""))
        if config.allow_cycles:
            explore_count = sum(1 for s in state.get("steps", []) if s.get("agent") == "Explorer")
            if explore_count < config.max_cycles:
                logger.info(f"  Route: Extractor needs more -> Explorer (round {explore_count + 1})")
                return "explorer"
    if state.get("extracted_data"):
        return "reviewer"
    return END


def route_after_reviewer(state: dict) -> str:
    signals = state.get("signals", [])
    from shared.registry import get_config
    config = get_config(state.get("query_type", ""))
    if not config.allow_cycles:
        return END
    actionable = [s for s in signals
                  if s.get("severity") == "critical"
                  and s.get("signal_type") in ("completeness", "cross_reference")]
    explore_count = sum(1 for s in state.get("steps", []) if s.get("agent") == "Explorer")
    if actionable and explore_count < config.max_cycles:
        return "set_reviewer_error"
    return END


def set_reviewer_error(state: dict) -> dict:
    """Convert critical reviewer signals into NEED_MORE error for Explorer cycle."""
    signals = state.get("signals", [])
    actionable = [s for s in signals
                  if s.get("severity") == "critical"
                  and s.get("signal_type") in ("completeness", "cross_reference")]
    if actionable:
        return {"error": f"NEED_MORE:{actionable[0].get('description', '')[:200]}"}
    return {}


def build_graph():
    from agents.explorer import explorer_node
    from agents.extractor_node import extractor_node
    from agents.reviewer import reviewer_node

    g = StateGraph(ProtocolState)
    g.add_node("explorer", explorer_node)
    g.add_node("extractor", extractor_node)
    g.add_node("reviewer", reviewer_node)
    g.add_node("set_reviewer_error", set_reviewer_error)

    g.add_edge(START, "explorer")
    g.add_edge("explorer", "extractor")
    g.add_conditional_edges("extractor", route_after_extractor,
                            {"explorer": "explorer", "reviewer": "reviewer", END: END})
    g.add_conditional_edges("reviewer", route_after_reviewer,
                            {"set_reviewer_error": "set_reviewer_error", END: END})
    g.add_edge("set_reviewer_error", "explorer")

    return g.compile()


protocol_graph = build_graph()


def run_query(query, query_type, retriever, pdf_path, json_data,
              store=None, debug_log=None, event_bus=None):
    t0 = time.time()
    logger.info(f"Graph: {query_type} -- '{query[:60]}'")

    initial = {
        "query": query, "query_type": query_type, "pdf_path": pdf_path,
        "sections_content": {}, "tables_content": {},
        "sections_read": [], "extracted_data": {}, "validation": {},
        "signals": [], "steps": [], "error": "",
    }

    runtime = RuntimeContext(
        retriever=retriever, store=store,
        json_data=json_data, event_bus=event_bus,
    )
    config = {"configurable": {"runtime": runtime}}

    try:
        final = protocol_graph.invoke(initial, config=config)
        steps = final.get("steps", [])
        total = sum(s.get("turns", 0) for s in steps)
        logger.info(f"Graph done: {total} turns, {time.time()-t0:.1f}s")
        return {
            "data": final.get("extracted_data", {}),
            "validation": final.get("validation", {}),
            "signals": final.get("signals", []),
            "steps": steps, "total_turns": total,
            "error": final.get("error", ""),
        }
    except Exception as e:
        logger.error(f"Graph failed: {e}", exc_info=True)
        return {"data": {}, "validation": {}, "signals": [],
                "steps": [], "total_turns": 0, "error": str(e)}
