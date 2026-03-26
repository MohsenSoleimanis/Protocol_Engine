"""
Explorer Node — Content gathering for extraction.

First run: reads retrieval queries from the registry (ICH M11 vocabulary),
runs ALL queries in parallel, deduplicates, follows cross-references,
and gathers page-overlap tables. No LLM calls.

Cycle re-entry (NEED_MORE) and SoA: falls back to agentic Explorer
with search, read_section, and vision_extract tools.
"""
from __future__ import annotations
import json, logging, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from config import LLM_MODEL, OPENAI_API_KEY, get_langfuse_handler
from agents.state import get_runtime

logger = logging.getLogger(__name__)

# Cross-reference patterns
_XREF_PATTERNS = [
    re.compile(r'(?:see|See|refer to)\s+(?:Section|section|§)\s*(\d+(?:\.\d+)*)', re.IGNORECASE),
    re.compile(r'(?:see|See|refer to)\s+(?:Appendix|appendix)\s+([A-Z](?:\.\d+)?|\d+(?:\.\d+)*)', re.IGNORECASE),
    re.compile(r'(?:Appendix|APPENDIX)\s+([A-Z])\b'),
]


def _retrieve_one(retriever, query: str) -> list:
    """Single retrieval call — used by ThreadPoolExecutor."""
    try:
        return retriever.retrieve(query)
    except Exception as e:
        logger.warning(f"  Retrieval failed for '{query[:40]}': {e}")
        return []


def _parse_cross_references(text: str) -> dict:
    refs = {"sections": set(), "appendices": set()}
    for pattern in _XREF_PATTERNS:
        for match in pattern.finditer(text):
            val = match.group(1)
            full = match.group(0).lower()
            if "appendix" in full:
                refs["appendices"].add(val)
            else:
                refs["sections"].add(val)
    return {k: list(v) for k, v in refs.items()}


def _schema_gather(runtime, query_type: str):
    """Schema-based parallel retrieval.

    1. Build queries from schema field descriptions
    2. Add registry goal as primary query
    3. Run ALL in parallel
    4. Deduplicate into sections_content / tables_content
    5. Follow cross-references
    6. Gather page-overlap tables

    Returns: (sections_content, tables_content, sections_read)
    """
    from shared.registry import get_config
    config = get_config(query_type)
    retriever = runtime.retriever
    store = runtime.store
    json_data = runtime.json_data

    if not retriever:
        return {}, {}, []

    # Build query set: goal + retrieval_queries from registry
    # These use ICH M11 standard vocabulary that appears in ALL protocols.
    # NOT schema field descriptions (those contain CDISC codes for extraction output).
    queries = []
    if config.goal:
        queries.append(config.goal)
    queries.extend(config.retrieval_queries)

    # Deduplicate identical queries
    seen_queries = set()
    unique_queries = []
    for q in queries:
        key = q.lower().strip()[:80]
        if key not in seen_queries:
            seen_queries.add(key)
            unique_queries.append(q)

    logger.info(f"  Schema gather: {len(unique_queries)} queries for {query_type}")

    if not unique_queries:
        return {}, {}, []
    # Parallel retrieval
    t0 = time.time()


    all_results = []
    with ThreadPoolExecutor(max_workers=min(len(unique_queries), 8)) as pool:
        futures = {pool.submit(_retrieve_one, retriever, q): q for q in unique_queries}
        for future in as_completed(futures):
            nodes = future.result()
            all_results.extend(nodes)

    elapsed = time.time() - t0
    logger.info(f"  Parallel retrieval: {len(all_results)} results in {elapsed:.1f}s")

    # Deduplicate into sections_content / tables_content
    # Use retriever document text directly — it comes from _build_documents()
    # which reads from store.get_section() (includes tables on section pages)
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

    # Follow section cross-references (NOT appendices — too large/noisy)
    if store:
        all_text = "\n".join(d["text"] for d in sections_content.values())
        xrefs = _parse_cross_references(all_text)

        for ref_sid in xrefs.get("sections", []):
            if ref_sid in sections_content:
                continue
            data = store.get_section(ref_sid)
            if data and data.get("content", "").strip():
                text = data["content"]
                sections_content[ref_sid] = {"text": text, "pages": data.get("pages", []),
                                             "chars": len(text)}
                sections_read.append(ref_sid)
                logger.info(f"  Cross-ref: §{ref_sid}")

    # Same-caption table fragments + page-overlap tables
    gathered_pages = set()
    for d in sections_content.values():
        gathered_pages.update(d.get("pages", []))

    gathered_captions = set()
    for tid in tables_content:
        for tbl in json_data.get("tables", []):
            if tbl.get("id") == tid:
                cap = tbl.get("caption", "").strip()
                if cap:
                    gathered_captions.add(cap)
                break

    for tbl in json_data.get("tables", []):
        tid = tbl.get("id", "")
        if tid in tables_content:
            continue
        cap = tbl.get("caption", "").strip()
        tp = tbl.get("page_range", [])
        # Include if: same caption as gathered table, or on gathered pages
        if cap in gathered_captions or any(p in gathered_pages for p in tp):
            hdrs = tbl.get("column_headers", [])
            rows = tbl.get("rows", [])
            parts = [f"Table: {cap}"]
            if hdrs: parts.append(" | ".join(str(h) for h in hdrs))
            for row in rows[:80]: parts.append(" | ".join(str(c) for c in row))
            text = "\n".join(parts)
            tables_content[tid] = {"text": text, "pages": tp, "chars": len(text)}
            sections_read.append(tid)

    total = sum(d["chars"] for d in sections_content.values()) + sum(d["chars"] for d in tables_content.values())
    logger.info(f"  Schema gather done: {len(sections_content)} sections, "
                f"{len(tables_content)} tables, {total} chars")
    return sections_content, tables_content, sections_read


def _agentic_gather(runtime, state, query_type: str):
    """Agentic Explorer for SoA (vision) and cycle re-entries."""
    retriever = runtime.retriever
    store = runtime.store
    bus = runtime.event_bus
    existing = set(state.get("sections_read", []))
    gathered = []
    gathered_chars = [0]
    vision_cache = {}
    vision_calls = [0]

    from shared.registry import get_config
    budget = 100000 if get_config(query_type).allow_cycles else 60000

    @tool
    def search(query: str) -> str:
        """Search protocol. Content gathered automatically, returns summary."""
        if gathered_chars[0] >= budget: return "Budget reached."
        results = retriever.retrieve(query)
        if not results: return f"No results for '{query}'"
        summaries = []
        for node in results:
            meta = node.metadata
            sid = meta.get("section_id", meta.get("table_id", ""))
            if sid in existing: continue
            existing.add(sid)
            text = node.text or ""
            pages = json.loads(meta.get("pages", "[]"))
            gathered.append({"type": meta.get("type", "section"), "id": sid,
                            "text": text, "pages": pages, "chars": len(text)})
            gathered_chars[0] += len(text)
            summaries.append(f"  §{sid}: {meta.get('title','')} ({len(text)} chars)")
        return f"Gathered {len(summaries)} sections:\n" + "\n".join(summaries) if summaries else "All already gathered."

    @tool
    def read_section(section_id: str) -> str:
        """Read a specific section by ID."""
        if section_id in existing: return f"[Already have §{section_id}]"
        existing.add(section_id)
        if store:
            data = store.get_section(section_id)
            if data and data.get("content", "").strip():
                text = data["content"]
                pages = data.get("pages", [])
                gathered.append({"type": "section", "id": section_id,
                                "text": text, "pages": pages, "chars": len(text)})
                gathered_chars[0] += len(text)
                return f"Read §{section_id}: {data.get('title','')} ({len(text)} chars)"
        return f"Section {section_id} not found"

    @tool
    def vision_extract(pages: list[int]) -> str:
        """GPT-4o vision for complex SoA tables."""
        key = tuple(sorted(pages))
        if key in vision_cache: return vision_cache[key]
        if vision_calls[0] >= 6: return "Vision limit reached."
        from extraction.vision_table import extract_table_with_vision
        result = extract_table_with_vision(state["pdf_path"], pages)
        vision_calls[0] += 1
        if result:
            gathered.append({"type": "vision", "id": f"vision_{pages}", "text": result,
                            "pages": pages, "chars": len(result)})
            gathered_chars[0] += len(result)
            vision_cache[key] = result
        return result or "Vision returned no content"

    error = state.get("error", "")
    base_goal = get_config(query_type).goal or f"Find: {state.get('query', '')}"
    if error.startswith("NEED_MORE:"):
        extra = error.replace("NEED_MORE:", "").strip()
        goal = f"The Extractor needs: {extra}\n\nAlso: {base_goal}"
        if bus: bus.emit("explorer", "re-entry", f"Fetching: {extra[:80]}...")
    else:
        goal = base_goal
        if bus: bus.emit("explorer", "starting", f"Exploring for {query_type}...")

    tools = [search, read_section, vision_extract]
    lf = get_langfuse_handler()
    callbacks = [lf] if lf else []
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY,
                     temperature=0.1, callbacks=callbacks).bind_tools(tools)
    system = ("You are a clinical protocol navigator. search() finds and gathers content. "
              "read_section() for cross-references. vision_extract() for SoA tables. "
              "Stop when you have enough.")
    msgs = [SystemMessage(content=system), HumanMessage(content=goal)]
    tmap = {t.name: t for t in tools}
    turns, tclog = 0, []
    for turn in range(1, 10):
        turns = turn
        resp = llm.invoke(msgs)
        msgs.append(resp)
        if resp.tool_calls:
            for tc in resp.tool_calls:
                tclog.append(tc["name"])
                logger.info(f"  Explorer turn {turn}: {tc['name']}({json.dumps(tc['args'])[:60]})")
                if bus: bus.emit_tool("explorer", tc["name"],
                    f"Searching: '{tc['args'].get('query','')[:40]}'" if tc["name"] == "search"
                    else f"Reading §{tc['args'].get('section_id','')}" if tc["name"] == "read_section"
                    else f"Vision pages {tc['args'].get('pages',[])}")
                fn = tmap.get(tc["name"])
                try: result = fn.invoke(tc["args"]) if fn else f"Unknown: {tc['name']}"
                except Exception as e: result = f"Error: {e}"
                msgs.append(ToolMessage(content=str(result)[:10000], tool_call_id=tc["id"]))
        else:
            logger.info(f"  Explorer done: {turn} turns, {len(tclog)} calls")
            if bus: bus.emit("explorer", "done", f"Found {len(gathered)} blocks")
            break

    secs, tbls = {}, {}
    new_ids = []
    for item in gathered:
        entry = {"text": item["text"], "pages": item["pages"], "chars": item["chars"]}
        if item["type"] in ("section", "pages"): secs[item["id"]] = entry
        else: tbls[item["id"]] = entry
        new_ids.append(item["id"])

    step = {"agent": "Explorer", "turns": turns, "tool_calls": len(tclog),
            "tools_used": tclog, "content_chars": sum(i["chars"] for i in gathered)}
    return secs, tbls, new_ids, [step]


def explorer_node(state: dict, config: RunnableConfig) -> dict:
    """Gatherer node — schema-based retrieval, agentic for SoA/cycles."""
    runtime = get_runtime(config)
    bus = runtime.event_bus
    qt = state.get("query_type", "general")
    error = state.get("error", "")

    # Cycle re-entry or SoA: agentic
    if error.startswith("NEED_MORE:") or qt == "soa":
        if qt == "soa" and bus: bus.emit("explorer", "starting", "Exploring SoA (vision)...")
        secs, tbls, sread, steps = _agentic_gather(runtime, state, qt)
        return {"sections_content": secs, "tables_content": tbls,
                "sections_read": sread, "error": "", "steps": steps}

    # Schema-based parallel retrieval
    if bus: bus.emit("explorer", "starting", f"Gathering {qt} content...")
    secs, tbls, sread = _schema_gather(runtime, qt)
    total = sum(d["chars"] for d in secs.values()) + sum(d["chars"] for d in tbls.values())

    if total > 0:
        if bus: bus.emit("explorer", "done",
                         f"Gathered {len(secs)} sections + {len(tbls)} tables ({total} chars)")
        return {"sections_content": secs, "tables_content": tbls,
                "sections_read": sread, "error": "",
                "steps": [{"agent": "Gatherer", "turns": 0, "tool_calls": 0,
                    "tools_used": ["schema_gather"], "content_chars": total}]}

    # Fallback: agentic
    logger.info(f"  Schema gather found nothing for {qt}, falling back to agentic")
    if bus: bus.emit("explorer", "starting", f"Searching for {qt}...")
    secs, tbls, sread, steps = _agentic_gather(runtime, state, qt)
    return {"sections_content": secs, "tables_content": tbls,
            "sections_read": sread, "error": "", "steps": steps}
