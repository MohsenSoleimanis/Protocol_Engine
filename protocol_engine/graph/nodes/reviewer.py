"""
Reviewer — Semantic-only checks (complements deterministic validator).

Only checks what the validator CANNOT:
  - Hallucination (plausible but not in source)
  - Semantic accuracy (right text, wrong interpretation)
  - Clinical plausibility (500mg/kg dose?)
  - Cross-section contradiction

Receives validator results. Does NOT re-check source grounding or numbers.
Decides: DONE or NEED_MORE (cycle back to Explorer).
"""
from __future__ import annotations

import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from protocol_engine.config import LLM_MODEL, OPENAI_API_KEY, MAX_CYCLES, get_langfuse_handler
from protocol_engine.models.enums import EdgeSignal
from protocol_engine.models.state import get_runtime
from protocol_engine.tools.knowledge import lookup_knowledge
from protocol_engine.prompts import render as render_prompt

logger = logging.getLogger(__name__)

MAX_TURNS = 8

VALID_SIGNAL_TYPES = {"hallucination", "completeness", "consistency", "plausibility"}


def reviewer_node(state: dict, config: RunnableConfig) -> dict:
    runtime = get_runtime(config)
    bus = runtime.event_bus
    qt = state.get("query_type", "general")
    extracted = state.get("extracted_data", {})
    validation = state.get("validation", {})
    context = state.get("assembled_context", "")

    if not extracted:
        return {"signals": [], "edge_signal": EdgeSignal.DONE,
                "steps": [{"agent": "reviewer", "turns": 0}]}

    if bus:
        bus.emit("reviewer", "starting", "Reviewing...")

    # Build review tools
    signals = []
    context_shown = [False]

    @tool
    def get_extraction() -> str:
        """View the complete extracted data."""
        return json.dumps(extracted, indent=2, default=str)

    @tool
    def get_validation() -> str:
        """View deterministic validation results."""
        if not validation:
            return "No validation results."
        s = (f"Verified: {validation.get('verified', 0)}/{validation.get('total', 0)}, "
             f"Failed: {validation.get('failed', 0)}")
        fails = [f"  {d['item']}: {d['verdict']}" for d in validation.get("details", [])
                 if d.get("verdict", "").startswith("FAILED")]
        if fails:
            s += "\n" + "\n".join(fails[:10])
        xf = validation.get("cross_field", [])
        if xf:
            s += "\nCross-field issues: " + "; ".join(xf)
        return s

    @tool
    def get_context() -> str:
        """View the gathered protocol content. Call ONCE."""
        if context_shown[0]:
            return "[Already shown.]"
        context_shown[0] = True
        return context or "No content available."

    @tool
    def flag(signal_type: str, severity: str, title: str, description: str) -> str:
        """Flag a verified semantic issue."""
        if signal_type not in VALID_SIGNAL_TYPES:
            return f"Invalid type '{signal_type}'. Use: {VALID_SIGNAL_TYPES}"
        if severity not in ("critical", "major", "minor"):
            return f"Invalid severity. Use: critical, major, minor"
        signals.append({"signal_type": signal_type, "severity": severity,
                        "title": title, "description": description})
        return f"Recorded: [{severity}] {title}"

    @tool
    def done(summary: str) -> str:
        """Submit completed review."""
        return f"Review done: {summary}. {len(signals)} signals."

    tools = [get_extraction, get_validation, get_context, flag, done, lookup_knowledge]

    lf = get_langfuse_handler()
    cbs = [lf] if lf else []
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY,
                     temperature=0.1, callbacks=cbs).bind_tools(tools)

    system_prompt = render_prompt("reviewer", "system")
    user_prompt = render_prompt("reviewer", "user",
                                query_type=qt,
                                verified=validation.get('verified', 0),
                                total=validation.get('total', 0),
                                failed=validation.get('failed', 0))

    msgs = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    tmap = {t.name: t for t in tools}
    turns = 0

    for turn in range(1, MAX_TURNS + 1):
        turns = turn
        resp = llm.invoke(msgs)
        msgs.append(resp)
        if not resp.tool_calls:
            break
        for tc in resp.tool_calls:
            fn = tmap.get(tc["name"])
            try:
                result = fn.invoke(tc["args"]) if fn else f"Unknown: {tc['name']}"
            except Exception as e:
                result = f"Error: {e}"
            msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # Decide: cycle or done
    critical = [s for s in signals if s.get("severity") == "critical"]
    cycle_count = state.get("cycle_count", 0)

    if critical and cycle_count < MAX_CYCLES:
        edge_signal = EdgeSignal.NEED_MORE
    else:
        edge_signal = EdgeSignal.DONE

    if bus:
        bus.emit("reviewer", "done", f"{len(signals)} signals")

    return {
        "signals": signals,
        "edge_signal": edge_signal,
        "steps": [{"agent": "reviewer", "turns": turns, "signals": len(signals)}],
    }
