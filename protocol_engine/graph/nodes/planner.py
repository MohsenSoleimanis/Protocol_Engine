"""
Planner Node — Query decomposition for complex queries.

NEW node (old code had no planning — single query went straight to Explorer).

For simple queries: passes through (no decomposition).
For complex queries: breaks into sub-tasks with ordering.

Skipped when query_type is already specific and query is simple.
"""
from __future__ import annotations

import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from protocol_engine.config import FAST_MODEL, OPENAI_API_KEY, get_langfuse_handler
from protocol_engine.models.enums import EdgeSignal, NodeName
from protocol_engine.models.state import get_runtime

logger = logging.getLogger(__name__)


def planner_node(state: dict, config: RunnableConfig) -> dict:
    """Decompose complex queries into sub-tasks, or pass through for simple ones."""
    runtime = get_runtime(config)
    bus = runtime.event_bus
    query = state.get("query", "")
    query_type = state.get("query_type", "general")

    if bus:
        bus.emit(NodeName.PLANNER, "starting", "Planning extraction strategy...")

    # Simple queries don't need decomposition
    word_count = len(query.split())
    if word_count <= 30 and query_type != "general":
        logger.info(f"Planner: simple query, passing through")
        sub_tasks = [{"query": query, "query_type": query_type,
                      "description": f"Extract {query_type}", "completed": False}]
        if bus:
            bus.emit(NodeName.PLANNER, "done", "Simple query — no decomposition needed")
        return {
            "sub_tasks": sub_tasks,
            "current_task_index": 0,
            "edge_signal": EdgeSignal.CONTINUE,
            "steps": [{"agent": NodeName.PLANNER, "turns": 0,
                       "tool_calls": 0, "tools_used": ["passthrough"]}],
        }

    # Complex query — decompose with LLM
    lf = get_langfuse_handler()
    callbacks = [lf] if lf else []
    try:
        llm = ChatOpenAI(
            model=FAST_MODEL, api_key=OPENAI_API_KEY,
            temperature=0, callbacks=callbacks,
        )
        response = llm.invoke([
            SystemMessage(content=(
                "You decompose complex clinical protocol queries into ordered sub-tasks.\n"
                "Return a JSON array of sub-tasks, each with:\n"
                '  {"query": "specific sub-query", "query_type": "type", "description": "what this extracts"}\n'
                "Valid query_types: study_design, endpoints, eligibility, intervention, soa, "
                "safety, statistical, deviation, kri, risk, ambiguity, consistency, general\n"
                "Return ONLY the JSON array."
            )),
            HumanMessage(content=f"Decompose this query:\n{query}"),
        ])
        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()
        tasks = json.loads(raw)
        sub_tasks = [
            {"query": t["query"], "query_type": t.get("query_type", query_type),
             "description": t.get("description", ""), "completed": False}
            for t in tasks
        ]
        logger.info(f"Planner: decomposed into {len(sub_tasks)} sub-tasks")
    except Exception as e:
        logger.warning(f"Planner decomposition failed: {e}. Using single task.")
        sub_tasks = [{"query": query, "query_type": query_type,
                      "description": f"Extract {query_type}", "completed": False}]

    if bus:
        bus.emit(NodeName.PLANNER, "done", f"{len(sub_tasks)} sub-tasks planned")

    return {
        "sub_tasks": sub_tasks,
        "current_task_index": 0,
        "edge_signal": EdgeSignal.CONTINUE,
        "steps": [{"agent": NodeName.PLANNER, "turns": 1,
                   "tool_calls": 0, "tools_used": ["decompose"]}],
    }
