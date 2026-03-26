"""
Explorer Node — Content gathering via retrieval + agentic search.

Key fixes from old code:
  1. Consistent naming: always "Explorer" (not "Gatherer" sometimes)
  2. Agent generates its own queries (not only registry-hardcoded)
  3. Cross-reference following preserved
  4. Both schema-based (fast) and agentic (flexible) modes
  5. vision_extract tool for SoA tables
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
    LLM_MODEL, OPENAI_API_KEY, MAX_EXPLORER_TURNS,
    VISION_CALL_LIMIT, get_langfuse_handler,
)
from protocol_engine.models.enums import EdgeSignal, NodeName, QueryType
from protocol_engine.models.state import get_runtime
from protocol_engine.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Query type configs (from old registry.py — preserved)
QUERY_CONFIGS: dict[str, dict] = {
    "endpoints": {
        "goal": "Find ALL endpoint content: primary, secondary, exploratory objectives, case definitions, thresholds, timing.",
        "queries": [
            "objectives and endpoints primary secondary exploratory",
            "estimand endpoint definition population intercurrent events",
            "efficacy endpoint analysis assessment timepoint",
        ],
    },
    "eligibility": {
        "goal": "Find ALL eligibility criteria: inclusion and exclusion. Follow cross-references to appendices.",
        "queries": [
            "inclusion criteria participants eligible enrollment",
            "exclusion criteria medical conditions disqualification",
            "lifestyle considerations contraception restrictions",
        ],
    },
    "safety": {
        "goal": "Find ALL safety content: AESIs, monitoring rules, stopping rules, AE collection windows, SAE reporting.",
        "queries": [
            "adverse events serious adverse events reporting monitoring",
            "adverse events special interest AESI definition criteria",
            "stopping rules discontinuation safety monitoring committee",
            "AE collection window solicited unsolicited diary visit",
        ],
    },
    "deviation": {
        "goal": "Find eligibility criteria AND Schedule of Activities. Both needed for deviation rules.",
        "queries": [],
    },
    "soa": {
        "goal": "Find ALL Schedule of Activities tables. Process them in PAGE ORDER. Use vision_extract for each table.",
        "queries": [],
    },
    "study_design": {
        "goal": "Find study design: phase, randomization, blinding, stratification, sample size, dosing, arms, intervention details.",
        "queries": [
            "overall design randomization blinding stratification parallel",
            "sample size determination enrollment participants screened",
            "study intervention dose route administration treatment arms placebo",
            "interim analysis statistical primary analysis timing",
        ],
    },
    "risk": {
        "goal": "Find safety monitoring, endpoint definitions, schedule of activities.",
        "queries": [
            "safety monitoring adverse events stopping rules",
            "endpoint definition efficacy assessment criteria",
            "schedule activities visit procedures assessments",
        ],
    },
    "ambiguity": {
        "goal": "Find eligibility criteria, safety definitions. Look for undefined terms.",
        "queries": [],
    },
    "consistency": {
        "goal": "Find synopsis, endpoint definitions, AND statistical methods. Compare across sections.",
        "queries": [
            "synopsis protocol summary objectives endpoints",
            "statistical analysis primary secondary endpoint",
            "sample size determination power efficacy",
        ],
    },
    "intervention": {
        "goal": "Find study intervention details: drug/vaccine name, dose, route, formulation, comparator, storage.",
        "queries": [
            "investigational product dose formulation route administration",
            "preparation handling storage accountability",
            "concomitant therapy prohibited permitted medications",
            "dose modification discontinuation treatment compliance",
        ],
    },
    "statistical": {
        "goal": "Find statistical design: sample size, power, analysis populations, interim analyses.",
        "queries": [
            "sample size determination power calculation assumptions",
            "analysis populations intent-to-treat per-protocol safety",
            "interim analysis data monitoring committee multiplicity",
            "missing data handling sensitivity analysis",
        ],
    },
    "kri": {
        "goal": "Derive Key Risk Indicators from the protocol.",
        "queries": [
            "screening failure enrollment inclusion exclusion criteria",
            "adverse event reporting rate collection monitoring safety",
            "visit schedule compliance assessment window procedures",
            "protocol deviation violation eligibility criteria",
        ],
    },
}

# Cross-reference patterns
_XREF_PATTERNS = [
    re.compile(r'(?:see|See|refer to)\s+(?:Section|section|§)\s*(\d+(?:\.\d+)*)', re.IGNORECASE),
    re.compile(r'(?:see|See|refer to)\s+(?:Appendix|appendix)\s+([A-Z](?:\.\d+)?|\d+(?:\.\d+)*)', re.IGNORECASE),
    re.compile(r'(?:Appendix|APPENDIX)\s+([A-Z])\b'),
]


def explorer_node(state: dict, config: RunnableConfig) -> dict:
    """Explorer node — gathers content via retrieval or agentic search."""
    runtime = get_runtime(config)
    bus = runtime.event_bus
    qt = state.get("query_type", "general")
    edge_signal = state.get("edge_signal", "")
    edge_detail = state.get("edge_detail", "")

    # Cycle re-entry or SoA → agentic mode
    if edge_signal == EdgeSignal.NEED_MORE_CONTENT or qt == "soa":
        if bus:
            bus.emit(NodeName.EXPLORER, "starting",
                     f"Re-exploring: {edge_detail[:60]}..." if edge_detail else f"Exploring {qt}...")
        return _agentic_gather(runtime, state, qt, edge_detail)

    # First run → schema-based parallel retrieval (fast, no LLM)
    if bus:
        bus.emit(NodeName.EXPLORER, "starting", f"Gathering {qt} content...")
    secs, tbls, sread = _schema_gather(runtime, qt, state.get("query", ""))
    total = sum(d["chars"] for d in secs.values()) + sum(d["chars"] for d in tbls.values())

    if total > 0:
        if bus:
            bus.emit(NodeName.EXPLORER, "done",
                     f"Gathered {len(secs)} sections + {len(tbls)} tables ({total} chars)")
        return {
            "sections_content": secs,
            "tables_content": tbls,
            "sections_read": sread,
            "edge_signal": EdgeSignal.CONTINUE,
            "edge_detail": "",
            "steps": [{"agent": NodeName.EXPLORER, "turns": 0,
                       "tool_calls": 0, "tools_used": ["schema_gather"],
                       "content_chars": total}],
        }

    # Fallback to agentic
    logger.info(f"Schema gather found nothing for {qt}, falling back to agentic")
    if bus:
        bus.emit(NodeName.EXPLORER, "starting", f"Searching for {qt}...")
    return _agentic_gather(runtime, state, qt, "")


def _schema_gather(runtime, query_type: str, user_query: str):
    """Schema-based parallel retrieval — no LLM calls."""
    retriever = runtime.retriever
    store = runtime.store
    json_data = runtime.json_data

    if not retriever:
        return {}, {}, []

    qconfig = QUERY_CONFIGS.get(query_type, {})
    queries = []
    if qconfig.get("goal"):
        queries.append(qconfig["goal"])
    queries.extend(qconfig.get("queries", []))
    if user_query and user_query not in queries:
        queries.append(user_query)

    # Deduplicate
    seen = set()
    unique = []
    for q in queries:
        key = q.lower().strip()[:80]
        if key not in seen:
            seen.add(key)
            unique.append(q)

    if not unique:
        return {}, {}, []

    logger.info(f"Schema gather: {len(unique)} queries for {query_type}")

    # Parallel retrieval
    t0 = time.time()
    all_results = []
    with ThreadPoolExecutor(max_workers=min(len(unique), 8)) as pool:
        futures = {pool.submit(_retrieve_one, retriever, q): q for q in unique}
        for future in as_completed(futures):
            all_results.extend(future.result())

    logger.info(f"Parallel retrieval: {len(all_results)} results in {time.time() - t0:.1f}s")

    # Deduplicate into sections/tables
    sections_content = {}
    tables_content = {}
    sections_read = []

    for node in all_results:
        meta = node.metadata
        ntype = meta.get("type", "section")
        text = node.text or ""
        if not text.strip():
            continue
        pages = json.loads(meta.get("pages", "[]"))

        if ntype == "table":
            tid = meta.get("table_id", "")
            if tid and tid not in tables_content:
                tables_content[tid] = {"text": text, "pages": pages, "chars": len(text)}
                sections_read.append(tid)
        else:
            sid = meta.get("section_id", "")
            if sid and sid not in sections_content:
                sections_content[sid] = {"text": text, "pages": pages, "chars": len(text)}
                sections_read.append(sid)

    # Follow cross-references
    if store:
        all_text = "\n".join(d["text"] for d in sections_content.values())
        for pattern in _XREF_PATTERNS:
            for match in pattern.finditer(all_text):
                ref_sid = match.group(1)
                if ref_sid in sections_content:
                    continue
                data = store.get_section(ref_sid)
                if data and data.get("content", "").strip():
                    text = data["content"]
                    sections_content[ref_sid] = {
                        "text": text, "pages": data.get("pages", []), "chars": len(text),
                    }
                    sections_read.append(ref_sid)
                    logger.info(f"Cross-ref: §{ref_sid}")

    # Page-overlap tables
    gathered_pages = set()
    for d in sections_content.values():
        gathered_pages.update(d.get("pages", []))

    for tbl in json_data.get("tables", []):
        tid = tbl.get("id", "")
        if tid in tables_content:
            continue
        tp = tbl.get("page_range", [])
        if any(p in gathered_pages for p in tp):
            hdrs = tbl.get("column_headers", [])
            rows = tbl.get("rows", [])
            parts = [f"Table: {tbl.get('caption', '')}"]
            if hdrs:
                parts.append(" | ".join(str(h) for h in hdrs))
            for row in rows[:80]:
                parts.append(" | ".join(str(c) for c in row))
            text = "\n".join(parts)
            tables_content[tid] = {"text": text, "pages": tp, "chars": len(text)}
            sections_read.append(tid)

    total = sum(d["chars"] for d in sections_content.values()) + sum(d["chars"] for d in tables_content.values())
    logger.info(f"Schema gather done: {len(sections_content)} sections, {len(tables_content)} tables, {total} chars")
    return sections_content, tables_content, sections_read


def _agentic_gather(runtime, state: dict, query_type: str, detail: str) -> dict:
    """Agentic Explorer — LLM-driven search with tools."""
    bus = runtime.event_bus
    tool_registry = ToolRegistry(
        retriever=runtime.retriever, store=runtime.store,
        json_data=runtime.json_data, pdf_path=state.get("pdf_path", ""),
    )
    tools = tool_registry.get_tools_for_node(NodeName.EXPLORER)
    existing = set(state.get("sections_read", []))

    qconfig = QUERY_CONFIGS.get(query_type, {})
    base_goal = qconfig.get("goal", "") or f"Find: {state.get('query', '')}"

    if detail:
        goal = f"The Extractor needs more content: {detail}\n\nAlso: {base_goal}"
    else:
        goal = base_goal

    lf = get_langfuse_handler()
    callbacks = [lf] if lf else []
    llm = ChatOpenAI(
        model=LLM_MODEL, api_key=OPENAI_API_KEY,
        temperature=0.1, callbacks=callbacks,
    ).bind_tools(tools)

    system = (
        "You are a clinical protocol navigator. "
        "search() finds and gathers content by semantic query. "
        "read_section() reads a specific section by ID (e.g. '5.1'). "
        "vision_extract() extracts complex tables from page images. "
        "Stop when you have enough content for extraction."
    )
    msgs = [SystemMessage(content=system), HumanMessage(content=goal)]
    tmap = {t.name: t for t in tools}
    turns = 0
    tclog = []

    for turn in range(1, MAX_EXPLORER_TURNS + 1):
        turns = turn
        resp = llm.invoke(msgs)
        msgs.append(resp)
        if resp.tool_calls:
            for tc in resp.tool_calls:
                tclog.append(tc["name"])
                logger.info(f"Explorer turn {turn}: {tc['name']}({json.dumps(tc['args'])[:60]})")
                if bus:
                    bus.emit_tool(NodeName.EXPLORER, tc["name"],
                                  f"Searching: '{tc['args'].get('query', '')[:40]}'" if tc["name"] == "search"
                                  else f"Reading §{tc['args'].get('section_id', '')}" if tc["name"] == "read_section"
                                  else f"Vision pages {tc['args'].get('pages', [])}")
                fn = tmap.get(tc["name"])
                try:
                    result = fn.invoke(tc["args"]) if fn else f"Unknown: {tc['name']}"
                except Exception as e:
                    result = f"Error: {e}"
                msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
        else:
            logger.info(f"Explorer done: {turn} turns, {len(tclog)} calls")
            if bus:
                bus.emit(NodeName.EXPLORER, "done", f"Completed in {turn} turns")
            break

    # Collect gathered content from tools
    gathered = []
    for t in tools:
        if hasattr(t, "gathered"):
            gathered.extend(t.gathered)

    secs, tbls = {}, {}
    new_ids = []
    for item in gathered:
        entry = {"text": item["text"], "pages": item["pages"], "chars": item["chars"]}
        if item["type"] in ("section", "pages"):
            secs[item["id"]] = entry
        else:
            tbls[item["id"]] = entry
        new_ids.append(item["id"])

    total_chars = sum(i["chars"] for i in gathered)
    return {
        "sections_content": secs,
        "tables_content": tbls,
        "sections_read": new_ids,
        "edge_signal": EdgeSignal.CONTINUE,
        "edge_detail": "",
        "steps": [{"agent": NodeName.EXPLORER, "turns": turns,
                   "tool_calls": len(tclog), "tools_used": tclog,
                   "content_chars": total_chars}],
    }


def _retrieve_one(retriever, query: str) -> list:
    """Single retrieval call for ThreadPoolExecutor."""
    try:
        return retriever.retrieve(query)
    except Exception as e:
        logger.warning(f"Retrieval failed for '{query[:40]}': {e}")
        return []
