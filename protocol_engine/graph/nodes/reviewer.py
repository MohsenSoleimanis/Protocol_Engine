"""
Reviewer Node — Semantic-only review (complements deterministic validator).

Key fixes from old code:
  1. Reviewer receives validator output — no duplicate work
  2. ONLY checks what validator CANNOT: hallucination, semantic accuracy,
     cross-section contradiction, clinical plausibility
  3. get_extraction() returns ALL data (old code ignored the parameter)
  4. get_gathered_content() has no char truncation producing invalid state
  5. Signals use proper append reducer (old code dropped all but last)
"""
from __future__ import annotations

import json
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from protocol_engine.config import LLM_MODEL, OPENAI_API_KEY, MAX_REVIEWER_TURNS, get_langfuse_handler
from protocol_engine.models.enums import EdgeSignal, NodeName
from protocol_engine.models.state import get_runtime

logger = logging.getLogger(__name__)


SYSTEM = """You are a SKEPTICAL clinical protocol reviewer.

The deterministic validator has ALREADY checked:
  - Source grounding (exact text match)
  - Numerical accuracy (numbers match source)
  - Section references (section IDs exist)
  - Schema completeness (required fields present)

YOUR job is ONLY to check things the validator CANNOT:
  1. HALLUCINATION: Is any extracted information plausible but NOT in the source?
  2. SEMANTIC ACCURACY: Is the meaning correct? (right text but wrong interpretation?)
  3. CROSS-SECTION CONTRADICTION: Do different parts of the extraction contradict?
  4. CLINICAL PLAUSIBILITY: Are clinical values reasonable? (e.g. 500mg/kg dose?)

Do NOT re-check source grounding or numbers — the validator already did that.

WORKFLOW:
  1. get_extraction() — see the extracted JSON
  2. get_validation_results() — see what the validator already found
  3. get_gathered_content() — see the source content (call ONCE)
  4. flag_signal() for verified semantic issues only
  5. submit_review()

SIGNAL TYPES:
  hallucination (critical): Data not supported by gathered content
  completeness (critical): Important items in content but missing from extraction
  consistency (major): Contradictions within the extraction
  cross_reference (critical): Unresolved references in content

USE critical ONLY when extraction is WRONG or INCOMPLETE.
DO NOT flag items from OTHER extraction types or internal schema fields."""


def reviewer_node(state: dict, config: RunnableConfig) -> dict:
    """Reviewer node — semantic-only checks, receives validator results."""
    runtime = get_runtime(config)
    bus = runtime.event_bus
    qt = state.get("query_type", "general")
    extractions = state.get("extracted_data", {})
    validation = state.get("validation", {})
    context = state.get("assembled_context", "")

    if not extractions:
        return {
            "signals": [],
            "edge_signal": EdgeSignal.DONE,
            "steps": [{"agent": NodeName.REVIEWER, "turns": 0,
                       "tool_calls": 0, "tools_used": []}],
        }

    if bus:
        bus.emit(NodeName.REVIEWER, "starting", "Reviewing extraction...")

    # Build tools
    signals = []
    content_shown = [False]

    @tool
    def get_extraction() -> str:
        """View the complete extracted data."""
        return json.dumps(extractions, indent=2, default=str)

    @tool
    def get_validation_results() -> str:
        """View deterministic validation results (already computed)."""
        if not validation:
            return "No validation results available."
        summary = (
            f"Verified: {validation.get('verified', 0)}/{validation.get('total', 0)}, "
            f"Flagged: {validation.get('flagged', 0)}, "
            f"Failed: {validation.get('failed', 0)}"
        )
        failed_details = [
            f"  {d.get('item', '')}: {d.get('verdict', '')} - {d.get('checks', {})}"
            for d in validation.get("details", [])
            if d.get("verdict", "").startswith("FAILED")
        ]
        if failed_details:
            summary += "\n\nFailed items:\n" + "\n".join(failed_details[:10])
        return summary

    @tool
    def get_gathered_content() -> str:
        """View the gathered protocol content. Call ONCE."""
        if content_shown[0]:
            return "[Already shown — use the content from before.]"
        content_shown[0] = True
        if not context:
            return "No gathered content available."
        return context

    @tool
    def flag_signal(signal_type: str, severity: str, title: str, description: str) -> str:
        """Flag a verified semantic issue. Only for things the validator cannot check."""
        signals.append({
            "signal_type": signal_type,
            "severity": severity,
            "title": title,
            "description": description,
        })
        logger.info(f"Reviewer signal: [{severity}] {signal_type}: {title}")
        return f"Recorded: [{severity}] {title}"

    @tool
    def submit_review(summary: str) -> str:
        """Submit completed review."""
        return f"Review done: {summary}. {len(signals)} signals."

    tools = [get_extraction, get_validation_results, get_gathered_content,
             flag_signal, submit_review]

    lf = get_langfuse_handler()
    callbacks = [lf] if lf else []
    llm = ChatOpenAI(
        model=LLM_MODEL, api_key=OPENAI_API_KEY,
        temperature=0.1, callbacks=callbacks,
    ).bind_tools(tools)

    msgs = [
        SystemMessage(content=SYSTEM),
        HumanMessage(content=(
            f"Review the {qt} extraction. "
            f"The validator already checked {validation.get('total', 0)} items: "
            f"{validation.get('verified', 0)} verified, "
            f"{validation.get('failed', 0)} failed. "
            f"Focus on semantic issues the validator cannot catch."
        )),
    ]
    tmap = {t.name: t for t in tools}
    turns = 0
    tclog = []

    for turn in range(1, MAX_REVIEWER_TURNS + 1):
        turns = turn
        resp = llm.invoke(msgs)
        msgs.append(resp)
        if resp.tool_calls:
            for tc in resp.tool_calls:
                tclog.append(tc["name"])
                logger.info(f"Reviewer turn {turn}: {tc['name']}")
                if bus:
                    desc = {
                        "get_extraction": "Reading extraction...",
                        "get_validation_results": "Reading validation...",
                        "get_gathered_content": "Reading source content...",
                        "flag_signal": f"Flagging: {tc['args'].get('title', '')}",
                        "submit_review": "Submitting review...",
                    }.get(tc["name"], tc["name"])
                    bus.emit_tool(NodeName.REVIEWER, tc["name"], desc)
                fn = tmap.get(tc["name"])
                try:
                    result = fn.invoke(tc["args"]) if fn else f"Unknown: {tc['name']}"
                except Exception as e:
                    result = f"Error: {e}"
                msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        else:
            logger.info(f"Reviewer done: {turn} turns, {len(signals)} signals")
            break

    # Determine edge signal
    critical_signals = [s for s in signals if s.get("severity") == "critical"]
    cycle_count = state.get("cycle_count", 0)

    if critical_signals and cycle_count < 2:
        edge_signal = EdgeSignal.NEED_MORE_CONTENT
        edge_detail = "; ".join(s["title"] for s in critical_signals[:3])
    else:
        edge_signal = EdgeSignal.DONE
        edge_detail = ""

    if bus:
        bus.emit(NodeName.REVIEWER, "done", f"{len(signals)} signals")

    return {
        "signals": signals,
        "edge_signal": edge_signal,
        "edge_detail": edge_detail,
        "steps": [{"agent": NodeName.REVIEWER, "turns": turns,
                   "tool_calls": len(tclog), "tools_used": tclog}],
    }
