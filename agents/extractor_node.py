"""
Extractor Agent — Structures content into Pydantic schemas.
LangGraph node. ReAct LLM decides: extract → validate → request_more or submit.

Reads content from state["sections_content"] + state["tables_content"].
Content comes from the retriever's indexed documents via the Gatherer/Explorer.
"""
from __future__ import annotations
import json, logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from config import LLM_MODEL, OPENAI_API_KEY, get_langfuse_handler
from extraction.schemas import SCHEMA_MAP
from extraction.extractor import BASE_SYSTEM_PROMPT, KNOWLEDGE_APPENDICES
from extraction.validator import validate
from agents.state import get_runtime

logger = logging.getLogger(__name__)


def build_tools(query_type, content):
    extracted = {}
    val_result = {}
    need_more = {"flag": False, "what": ""}

    @tool
    def extract(schema_type: str) -> str:
        """Extract structured data from protocol content into the schema."""
        nonlocal extracted
        from extraction.extractor import extract as run_extract
        obj, raw, info = run_extract(context=content, observations="",
                                      query_type=schema_type, user_query=f"Extract {schema_type}")
        if obj and hasattr(obj, "model_dump"): extracted.update(obj.model_dump())
        elif isinstance(obj, dict): extracted.update(obj)
        else: return "Extraction failed"
        return json.dumps(extracted, indent=2)[:10000]

    @tool
    def validate_extraction() -> str:
        """Validate: check source grounding, numbers, completeness."""
        nonlocal val_result
        if not extracted: return "Nothing to validate. Call extract() first."
        val_result.update(validate(extracted, content))
        s = f"Verified: {val_result['verified']}/{val_result['total']}, Flagged: {val_result['flagged']}"
        issues = [f"  {i.get('id','?')}: {i.get('verdict')} - {i.get('reason','')}"
                  for i in val_result.get("details", []) if i.get("verdict") != "VERIFIED"]
        if issues: s += "\n" + "\n".join(issues[:10])
        return s

    @tool
    def request_more_content(what_is_missing: str) -> str:
        """Request more content from Explorer. Use when content is insufficient."""
        need_more["flag"] = True
        need_more["what"] = what_is_missing
        logger.info(f"  Extractor requests more: {what_is_missing[:60]}")
        return f"Noted: need '{what_is_missing}'. Graph will route back to Explorer."

    @tool
    def submit(summary: str) -> str:
        """Submit completed extraction."""
        return f"Submitted: {summary}"

    return [extract, validate_extraction, request_more_content, submit], extracted, val_result, need_more


def extractor_node(state: dict, config: RunnableConfig) -> dict:
    """Extractor node — builds content from sections_content + tables_content."""
    runtime = get_runtime(config)
    bus = runtime.event_bus
    qt = state.get("query_type", "general")
    
    sections = state.get("sections_content", {})
    tables = state.get("tables_content", {})
    sections_read = state.get("sections_read", [])
    
    if not sections and not tables:
        return {"error": "No content from Gatherer", "extracted_data": {}, "validation": {}}
    
    from shared.registry import get_config
    reg_config = get_config(qt)
    content_limit = 80000 if reg_config.allow_cycles else 50000
    
    content_parts = []
    char_count = 0
    seen = set()
    
    for item_id in sections_read:
        if item_id in seen:
            continue
        seen.add(item_id)
        if item_id in sections:
            text = sections[item_id].get("text", "")
        elif item_id in tables:
            text = tables[item_id].get("text", "")
        else:
            continue
        if char_count + len(text) > content_limit:
            break
        label = f"TABLE: {item_id}" if item_id in tables else f"§{item_id}"
        content_parts.append(f"[{label}]\n{text}")
        char_count += len(text)
    
    for tid, data in tables.items():
        if tid in seen:
            continue
        text = data.get("text", "")
        if char_count + len(text) > content_limit:
            break
        content_parts.append(f"[TABLE: {tid}]\n{text}")
        char_count += len(text)
    
    content = "\n\n".join(content_parts)
    
    if bus: bus.emit("extractor", "starting", f"Extracting {qt} from {char_count} chars...")

    tools, extracted, val_result, need_more = build_tools(qt, content)
    schema_cls = SCHEMA_MAP.get(qt)
    schema_fields = list(schema_cls.model_fields.keys()) if schema_cls else []
    schema_name = schema_cls.__name__ if schema_cls else ""
    appendix = KNOWLEDGE_APPENDICES.get(schema_name, "")

    system = f"""You are a clinical protocol extractor.
{BASE_SYSTEM_PROMPT}
EXTRACTION TYPE: {qt}
SCHEMA FIELDS: {schema_fields}
{"DOMAIN KNOWLEDGE:" + chr(10) + appendix if appendix else ""}

WORKFLOW:
1. Call extract(schema_type="{qt}") — this uses structured output to enforce the schema
2. Call validate_extraction() to check accuracy
3. If content is insufficient (missing sections, cross-references not resolved), 
   call request_more_content() describing what you need — the Explorer will fetch it
4. Call submit() when extraction is complete and validated"""

    lf = get_langfuse_handler()
    callbacks = [lf] if lf else []
    llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY, temperature=0.1, callbacks=callbacks).bind_tools(tools)
    
    section_ids = [sid for sid in seen if sid in sections]
    table_ids = [tid for tid in seen if tid in tables]
    content_summary = (f"Gathered content: {len(section_ids)} sections ({', '.join(section_ids[:10])}), "
                       f"{len(table_ids)} tables ({', '.join(table_ids[:5])}). "
                       f"Total: {char_count} chars.")

    msgs = [SystemMessage(content=system),
            HumanMessage(content=f"Extract {qt}.\n\n{content_summary}")]
    tmap = {t.name: t for t in tools}
    turns, tclog = 0, []
    for turn in range(1, 8):
        turns = turn
        resp = llm.invoke(msgs)
        msgs.append(resp)
        if resp.tool_calls:
            for tc in resp.tool_calls:
                tclog.append(tc["name"])
                logger.info(f"  Extractor turn {turn}: {tc['name']}")
                if bus:
                    desc = {"extract": "Running schema extraction...",
                            "validate_extraction": "Validating grounding...",
                            "request_more_content": "Requesting more content...",
                            "submit": "Submitting extraction..."}.get(tc["name"], tc["name"])
                    bus.emit_tool("extractor", tc["name"], desc)
                fn = tmap.get(tc["name"])
                try: result = fn.invoke(tc["args"]) if fn else f"Unknown: {tc['name']}"
                except Exception as e: result = f"Error: {e}"
                msgs.append(ToolMessage(content=str(result)[:15000], tool_call_id=tc["id"]))
        else:
            logger.info(f"  Extractor done: {turn} turns")
            break

    error = ""
    if need_more["flag"]:
        reason = need_more["what"].lower()
        if any(kw in reason for kw in ["validation", "verified", "failed", "flagged"]):
            logger.info(f"  Extractor: ignoring cycle request (validation issue)")
        else:
            error = f"NEED_MORE:{need_more['what']}"

    final_validation = val_result if val_result else validate(extracted, content)
    logger.info(f"  Extractor result: {len(extracted)} keys, "
                f"validation: {final_validation.get('verified', '?')}/{final_validation.get('total', '?')}")
    if bus:
        v = final_validation
        bus.emit("extractor", "done",
                 f"Extracted {len(extracted)} fields, {v.get('verified', 0)}/{v.get('total', 0)} verified")
    
    return {
        "extracted_data": extracted,
        "validation": final_validation,
        "error": error,
        "steps": [{"agent": "Extractor", "turns": turns,
                   "tool_calls": len(tclog), "tools_used": tclog}],
    }
