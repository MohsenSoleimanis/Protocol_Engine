"""
Explorer — Retrieve content + assemble context.

One node does two things:
  1. Gather relevant sections (schema-based parallel retrieval OR agentic search)
  2. Score and assemble into a token-budgeted context string

No separate Router/Planner/ContextAssembler needed.
"""
from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from protocol_engine.config import (
    LLM_MODEL, OPENAI_API_KEY, CONTEXT_BUDGET_TOKENS,
    HIGH_RELEVANCE_THRESHOLD, MEDIUM_RELEVANCE_THRESHOLD,
    get_langfuse_handler,
)
from protocol_engine.models.enums import EdgeSignal
from protocol_engine.models.state import get_runtime
from protocol_engine.tools.search import make_search_tool
from protocol_engine.tools.read_section import make_read_section_tool
from protocol_engine.tools.vision import make_vision_tool
from protocol_engine.prompts import load_prompt, render as render_prompt

logger = logging.getLogger(__name__)

MAX_TURNS = 8

_XREF = [
    re.compile(r'(?:see|refer to)\s+(?:Section|§)\s*(\d+(?:\.\d+)*)', re.I),
    re.compile(r'(?:see|refer to)\s+Appendix\s+([A-Z])', re.I),
]


def explorer_node(state: dict, config: RunnableConfig) -> dict:
    runtime = get_runtime(config)
    bus = runtime.event_bus
    qt = state.get("query_type", "general")
    query = state.get("query", "")
    edge_signal = state.get("edge_signal", "")

    if bus:
        bus.emit("explorer", "starting", f"Gathering {qt} content...")

    # On cycle re-entry or SoA: use agentic mode
    if edge_signal == EdgeSignal.NEED_MORE or qt == "soa":
        secs, tbls, ctx, steps = _agentic(runtime, state, qt, query)
    else:
        # First run: fast parallel retrieval
        secs, tbls, ctx, steps = _schema_retrieve(runtime, qt, query)
        if not secs and not tbls:
            # Fallback to agentic if schema retrieval found nothing
            secs, tbls, ctx, steps = _agentic(runtime, state, qt, query)

    if bus:
        bus.emit("explorer", "done", f"{len(secs)} sections, ~{len(ctx)//4} tokens")

    return {
        "sections_content": secs,
        "tables_content": tbls,
        "assembled_context": ctx,
        "edge_signal": EdgeSignal.DONE,
        "steps": [steps],
    }


# ── Schema-based retrieval (fast, no LLM) ───────────────────────────────────

def _schema_retrieve(runtime, qt: str, query: str):
    retriever = runtime.retriever
    store = runtime.store
    json_data = runtime.json_data
    if not retriever:
        return {}, {}, "", {"agent": "explorer", "mode": "schema", "turns": 0}

    # Load retrieval queries from externalized prompt config
    explorer_config = load_prompt("explorer")
    retrieval_queries = explorer_config.get("retrieval_queries", {})
    queries = list(retrieval_queries.get(qt, []))
    if query:
        queries.insert(0, query)

    # Parallel retrieval
    all_results = []
    with ThreadPoolExecutor(max_workers=min(len(queries), 8)) as pool:
        futs = {pool.submit(retriever.retrieve, q): q for q in queries}
        for f in as_completed(futs):
            try:
                all_results.extend(f.result())
            except Exception:
                pass

    # Deduplicate into sections/tables
    secs, tbls = {}, {}
    for node in all_results:
        meta = node.metadata
        text = node.text or ""
        if not text.strip():
            continue
        try:
            pages = json.loads(meta.get("pages", "[]"))
        except (json.JSONDecodeError, TypeError):
            pages = []
        ntype = meta.get("type", "section")
        sid = meta.get("section_id", meta.get("table_id", ""))
        if not sid:
            continue
        target = tbls if ntype == "table" else secs
        if sid not in target:
            target[sid] = {"text": text, "pages": pages, "chars": len(text)}

    # Cross-references
    if store:
        all_text = "\n".join(d["text"] for d in secs.values())
        for pat in _XREF:
            for m in pat.finditer(all_text):
                ref = m.group(1)
                if ref not in secs:
                    data = store.get_section(ref)
                    if data and data.get("content", "").strip():
                        secs[ref] = {"text": data["content"], "pages": data.get("pages", []),
                                     "chars": len(data["content"])}

    # Page-overlap tables
    gathered_pages = set()
    for d in secs.values():
        gathered_pages.update(d.get("pages", []))
    for tbl in json_data.get("tables", []):
        tid = tbl.get("id", "")
        if tid in tbls:
            continue
        if any(p in gathered_pages for p in tbl.get("page_range", [])):
            hdrs = tbl.get("column_headers", [])
            rows = tbl.get("rows", [])
            parts = [f"Table: {tbl.get('caption', '')}"]
            if hdrs:
                parts.append(" | ".join(str(h) for h in hdrs))
            for row in rows[:100]:
                parts.append(" | ".join(str(c) for c in row))
            text = "\n".join(parts)
            tbls[tid] = {"text": text, "pages": tbl.get("page_range", []), "chars": len(text)}

    ctx = _assemble_context(secs, tbls, "", qt)
    step = {"agent": "explorer", "mode": "schema", "turns": 0,
            "sections": len(secs), "tables": len(tbls)}
    return secs, tbls, ctx, step


# ── Agentic retrieval (LLM with tools) ──────────────────────────────────────

def _agentic(runtime, state: dict, qt: str, query: str):
    # Preserve existing content on re-entry (FIX C1)
    secs = dict(state.get("sections_content", {}))
    tbls = dict(state.get("tables_content", {}))

    gathered: dict[str, dict] = {}
    gathered.update({k: {**v, "type": "section"} for k, v in secs.items()})
    gathered.update({k: {**v, "type": "table"} for k, v in tbls.items()})
    budget = [sum(d.get("chars", 0) for d in gathered.values())]

    tools = [
        make_search_tool(runtime.retriever, gathered, budget),
        make_read_section_tool(runtime.store, gathered, budget),
        make_vision_tool(state.get("pdf_path", ""), gathered, budget),
    ]

    lf = get_langfuse_handler()
    cbs = [lf] if lf else []
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY,
                     temperature=0.1, callbacks=cbs).bind_tools(tools)

    goals = load_prompt("explorer").get("goals", {})
    goal = query or goals.get(qt, f"Find all {qt} content in this protocol.")
    msgs = [
        SystemMessage(content="You are a clinical protocol navigator. Use search() to find content, "
                      "read_section() for specific sections, vision_extract() for complex tables. "
                      "Stop when you have enough."),
        HumanMessage(content=goal),
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
                result = fn.invoke(tc["args"]) if fn else f"Unknown tool: {tc['name']}"
            except Exception as e:
                result = f"Error: {e}"
            msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

    # Split gathered into secs/tbls
    new_secs, new_tbls = {}, {}
    for sid, data in gathered.items():
        if data.get("type") == "table":
            new_tbls[sid] = {"text": data["text"], "pages": data.get("pages", []), "chars": data["chars"]}
        else:
            new_secs[sid] = {"text": data["text"], "pages": data.get("pages", []), "chars": data["chars"]}

    ctx = _assemble_context(new_secs, new_tbls, query, qt)
    step = {"agent": "explorer", "mode": "agentic", "turns": turns,
            "sections": len(new_secs), "tables": len(new_tbls)}
    return new_secs, new_tbls, ctx, step


# ── Context assembly (inline, not a separate node) ──────────────────────────

def _assemble_context(secs: dict, tbls: dict, query: str, qt: str) -> str:
    """Score sections by relevance, budget by tokens, return assembled context."""
    items = []
    for sid, d in secs.items():
        score = _relevance(d["text"], query, qt)
        items.append((sid, "section", d, score))
    for tid, d in tbls.items():
        score = _relevance(d["text"], query, qt) + 0.1  # tables get slight boost
        items.append((tid, "table", d, min(1.0, score)))

    items.sort(key=lambda x: x[3], reverse=True)

    parts = []
    tokens = 0
    budget = CONTEXT_BUDGET_TOKENS

    for item_id, item_type, data, score in items:
        text = data["text"]
        est = len(text) // 4

        if score >= HIGH_RELEVANCE_THRESHOLD:
            if tokens + est <= budget:
                label = f"TABLE: {item_id}" if item_type == "table" else f"§{item_id}"
                parts.append(f"[{label}]\n{text}")
                tokens += est
        elif score >= MEDIUM_RELEVANCE_THRESHOLD:
            trunc = text[:2000]
            est_t = len(trunc) // 4
            if tokens + est_t <= budget:
                label = f"TABLE: {item_id}" if item_type == "table" else f"§{item_id}"
                parts.append(f"[{label} (truncated)]\n{trunc}")
                tokens += est_t

    return "\n\n---\n\n".join(parts)


def _relevance(text: str, query: str, qt: str) -> float:
    tl = text.lower()
    ql = query.lower() if query else ""
    qwords = set(ql.split()) if ql else set()
    twords = set(tl.split())
    overlap = len(qwords & twords) / max(len(qwords), 1) if qwords else 0.3
    kws = {
        "endpoints": ["endpoint", "objective", "efficacy"],
        "eligibility": ["inclusion", "exclusion", "criteria"],
        "safety": ["safety", "adverse", "monitoring"],
        "soa": ["schedule", "activities", "visit"],
        "study_design": ["design", "randomiz", "blind"],
        "intervention": ["dose", "drug", "vaccine"],
        "statistical": ["statistic", "sample", "power"],
    }
    boost = sum(0.1 for kw in kws.get(qt, []) if kw in tl)
    return min(1.0, overlap * 0.5 + boost + 0.2)
