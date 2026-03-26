"""
Router Node — Query classification and multi-type detection.

NEW node (old code had no routing — query_type was always passed in).

Handles:
  1. Auto-detect query type from natural language
  2. Detect multi-type queries (e.g. "safety and endpoints")
  3. Route to Planner for complex queries, directly to Explorer for simple ones
"""
from __future__ import annotations

import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from protocol_engine.config import FAST_MODEL, OPENAI_API_KEY, get_langfuse_handler
from protocol_engine.models.enums import QueryType, EdgeSignal, NodeName
from protocol_engine.models.state import get_runtime

logger = logging.getLogger(__name__)

# Detection keywords per query type (from old registry.py)
DETECTION_KEYWORDS: dict[str, list[str]] = {
    "endpoints": ["endpoint", "objective", "efficacy", "primary", "secondary", "exploratory"],
    "eligibility": ["eligib", "inclusion", "exclusion", "enroll", "criteria"],
    "safety": ["safety", "adverse", "ae", "sae", "aesi", "monitoring", "stopping"],
    "deviation": ["deviation", "violation", "protocol deviation", "rule"],
    "soa": ["schedule", "activities", "soa", "soe", "visit"],
    "study_design": ["design", "phase", "randomiz", "blind", "stratif", "arm"],
    "risk": ["risk", "gap", "signal", "monitor"],
    "ambiguity": ["ambig", "vague", "unclear", "undefined", "subjective"],
    "consistency": ["consisten", "mismatch", "contradict", "compare"],
    "intervention": ["intervention", "drug", "dose", "dosing", "vaccine", "placebo",
                     "comparator", "formulation", "concomitant", "prohibited"],
    "statistical": ["statistic", "sample size", "power", "interim", "multiplicity",
                    "analysis population", "itt", "per-protocol"],
    "kri": ["kri", "key risk", "risk indicator", "quality tolerance",
            "qtl", "rbqm", "smart", "signal detection"],
}


def router_node(state: dict, config: RunnableConfig) -> dict:
    """Classify the query and decide routing.

    If query_type is already set (user specified), respect it.
    Otherwise, auto-detect from query text.
    """
    runtime = get_runtime(config)
    bus = runtime.event_bus
    query = state.get("query", "")
    query_type = state.get("query_type", "")

    if bus:
        bus.emit(NodeName.ROUTER, "starting", "Classifying query...")

    # If already classified, just pass through
    if query_type and query_type != "general":
        logger.info(f"Router: pre-classified as '{query_type}'")
        needs_planner = _is_complex_query(query, query_type)
        if bus:
            bus.emit(NodeName.ROUTER, "done", f"Type: {query_type}")
        return {
            "query_type": query_type,
            "edge_signal": EdgeSignal.CONTINUE,
            "steps": [{"agent": NodeName.ROUTER, "turns": 0,
                       "tool_calls": 0, "tools_used": ["passthrough"]}],
        }

    # Auto-detect via keyword matching first (fast, free)
    detected = _keyword_detect(query)
    if detected:
        logger.info(f"Router: keyword-detected as '{detected}'")
        if bus:
            bus.emit(NodeName.ROUTER, "done", f"Type: {detected}")
        return {
            "query_type": detected,
            "edge_signal": EdgeSignal.CONTINUE,
            "steps": [{"agent": NodeName.ROUTER, "turns": 0,
                       "tool_calls": 0, "tools_used": ["keyword_detect"]}],
        }

    # LLM classification for ambiguous queries
    detected = _llm_classify(query)
    logger.info(f"Router: LLM-classified as '{detected}'")
    if bus:
        bus.emit(NodeName.ROUTER, "done", f"Type: {detected}")

    return {
        "query_type": detected,
        "edge_signal": EdgeSignal.CONTINUE,
        "steps": [{"agent": NodeName.ROUTER, "turns": 1,
                   "tool_calls": 0, "tools_used": ["llm_classify"]}],
    }


def _keyword_detect(query: str) -> str:
    """Fast keyword-based query type detection."""
    q = query.lower()
    scores: dict[str, int] = {}
    for qtype, keywords in DETECTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in q)
        if score > 0:
            scores[qtype] = score

    if not scores:
        return ""

    best = max(scores, key=scores.get)
    # If multiple types have similar scores, might be a multi-type query
    if len(scores) > 1:
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if sorted_scores[0][1] == sorted_scores[1][1]:
            return ""  # ambiguous — let LLM decide

    return best


def _llm_classify(query: str) -> str:
    """LLM-based query classification using the fast model."""
    valid_types = [qt.value for qt in QueryType]
    lf = get_langfuse_handler()
    callbacks = [lf] if lf else []

    try:
        llm = ChatOpenAI(
            model=FAST_MODEL, api_key=OPENAI_API_KEY,
            temperature=0, callbacks=callbacks,
        )
        response = llm.invoke([
            SystemMessage(content=(
                "Classify this clinical protocol query into exactly ONE type.\n"
                f"Valid types: {valid_types}\n"
                "Return ONLY the type string, nothing else."
            )),
            HumanMessage(content=query),
        ])
        result = response.content.strip().lower().replace('"', '').replace("'", "")
        if result in valid_types:
            return result
    except Exception as e:
        logger.warning(f"LLM classification failed: {e}")

    return QueryType.GENERAL.value


def _is_complex_query(query: str, query_type: str) -> bool:
    """Determine if a query needs the Planner (decomposition)."""
    # Multi-type indicators
    if " and " in query.lower() and any(
        kw in query.lower() for qtype, kws in DETECTION_KEYWORDS.items()
        if qtype != query_type for kw in kws
    ):
        return True
    # Long queries tend to be complex
    if len(query.split()) > 30:
        return True
    return False
