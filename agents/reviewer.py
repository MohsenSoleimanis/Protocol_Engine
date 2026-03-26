"""
Reviewer Agent — Read-only verification of extraction.

Compares extracted data against gathered content.
Does NOT search — that's Explorer's job.
If gaps found, flags them → graph cycles to Explorer.
"""
from __future__ import annotations
import json, logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from config import LLM_MODEL, OPENAI_API_KEY, get_langfuse_handler
from agents.state import get_runtime

logger = logging.getLogger(__name__)


SYSTEM = """You are a SKEPTICAL clinical protocol reviewer. You verify extracted
data against the gathered protocol content.

You are READ-ONLY. You do NOT search for new content — that's the Explorer's job.
If content is missing, flag it and the system will fetch it automatically.

WORKFLOW:
  1. get_extraction() — see the extracted JSON
  2. get_gathered_content() — see the content that was gathered (call ONCE)
  3. Compare: does every extracted field have support in the gathered content?
  4. flag_signal() for verified issues:
     - Data in extraction that is NOT in gathered content → hallucination
     - Data in gathered content that is NOT in extraction → missed item
     - Contradictory values between extraction and source → inconsistency
  5. submit_review()

SIGNAL TYPES:
  ambiguity (critical): Vague text that could be interpreted multiple ways
  completeness (critical): Items in gathered content that extraction missed
  consistency (major): Contradictory values between sections
  cross_reference (critical): Unresolved "see Section X" in gathered content

USE critical SEVERITY when the extraction is WRONG or INCOMPLETE.
The system will automatically try to fix critical issues.

DO NOT FLAG:
  - Items from OTHER extraction types
  - Internal schema fields (gaps, insufficient_data)"""


def build_tools(extractions, sections_content, tables_content):
    """Build Reviewer tools — read-only, no search."""
    signals = []
    content_shown = [False]

    @tool
    def get_extraction(extraction_type: str) -> str:
        """View the extracted data."""
        if not extractions: return "No extraction data."
        return json.dumps(extractions, indent=2)[:15000]

    @tool
    def get_gathered_content() -> str:
        """View ALL gathered content (sections + tables). Call ONCE."""
        if content_shown[0]:
            return "[Already shown.]"
        content_shown[0] = True
        if not sections_content and not tables_content:
            return "No content was gathered."

        parts = []
        char_count = 0
        for sid, data in sections_content.items():
            text = data.get("text", "")
            if char_count + len(text) > 40000:
                remaining = len(sections_content) + len(tables_content) - len(parts)
                parts.append(f"[... {remaining} more items truncated]")
                break
            parts.append(f"§{sid} ({len(text)} chars):\n{text}")
            char_count += len(text)
        for tid, data in tables_content.items():
            text = data.get("text", "")
            if char_count + len(text) > 40000:
                break
            parts.append(f"TABLE {tid} ({len(text)} chars):\n{text}")
            char_count += len(text)

        return "\n\n".join(parts)

    @tool
    def flag_signal(signal_type: str, severity: str, title: str, description: str) -> str:
        """Record a verified finding. Use 'critical' severity for issues
        that need the system to fetch more content and re-extract."""
        signals.append({"signal_type": signal_type, "severity": severity,
                        "title": title, "description": description})
        logger.info(f"  Signal: [{severity}] {signal_type}: {title}")
        return f"Recorded: [{severity}] {title}"

    @tool
    def submit_review(summary: str) -> str:
        """Submit completed review."""
        return f"Review done: {summary}. {len(signals)} signals."

    return [get_extraction, get_gathered_content, flag_signal, submit_review], signals


def reviewer_node(state: dict, config: RunnableConfig) -> dict:
    """Reviewer node — read-only verification, flags issues for cycle recovery."""
    runtime = get_runtime(config)
    bus = runtime.event_bus
    qt = state.get("query_type", "general")
    extractions = state.get("extracted_data", {})
    sections = state.get("sections_content", {})
    tables = state.get("tables_content", {})

    if not extractions:
        return {"signals": [], "steps": [{"agent": "Reviewer", "turns": 0,
                                           "tool_calls": 0, "tools_used": []}]}

    if bus: bus.emit("reviewer", "starting", "Reviewing extraction...")

    tools, signals = build_tools(extractions, sections, tables)

    lf = get_langfuse_handler()
    callbacks = [lf] if lf else []
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY,
                     temperature=0.1, callbacks=callbacks).bind_tools(tools)

    msgs = [SystemMessage(content=SYSTEM),
            HumanMessage(content=f"Review the {qt} extraction. Verify every field is grounded in gathered content.")]
    tmap = {t.name: t for t in tools}
    turns, tclog = 0, []

    for turn in range(1, 10):
        turns = turn
        resp = llm.invoke(msgs)
        msgs.append(resp)
        if resp.tool_calls:
            for tc in resp.tool_calls:
                tclog.append(tc["name"])
                logger.info(f"  Reviewer turn {turn}: {tc['name']}")
                if bus:
                    desc = {"get_extraction": "Reading extraction...",
                            "get_gathered_content": "Reading gathered content...",
                            "flag_signal": f"Flagging: {tc['args'].get('title','')}",
                            "submit_review": "Submitting review..."
                            }.get(tc["name"], tc["name"])
                    bus.emit_tool("reviewer", tc["name"], desc)
                fn = tmap.get(tc["name"])
                try:
                    result = fn.invoke(tc["args"]) if fn else f"Unknown: {tc['name']}"
                except Exception as e:
                    result = f"Error: {e}"
                msgs.append(ToolMessage(content=str(result)[:15000], tool_call_id=tc["id"]))
        else:
            logger.info(f"  Reviewer done: {turn} turns, {len(signals)} signals")
            break

    if bus:
        bus.emit("reviewer", "done", f"{len(signals)} signals")

    return {
        "signals": signals,
        "steps": [{"agent": "Reviewer", "turns": turns,
                   "tool_calls": len(tclog), "tools_used": tclog}],
    }
