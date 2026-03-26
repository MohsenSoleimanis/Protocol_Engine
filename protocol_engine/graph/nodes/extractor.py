"""
Extractor Node — LLM-based structured extraction into Pydantic schemas.

Key fixes from old code:
  1. Extractor LLM sees ACTUAL content (old code only showed a summary)
  2. Single LLM call with content + schema (old code had extract() tool calling
     a SEPARATE LLM, so the deciding LLM never saw the content)
  3. Knowledge appendices loaded from JSON files (not hardcoded dict)
  4. No truncation of tool results to 10k chars
  5. request_more_content uses typed EdgeSignal (not "NEED_MORE:" string)
  6. No keyword filter blocking request_more_content
"""
from __future__ import annotations

import json
import logging
import time

from pydantic import BaseModel, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from protocol_engine.config import (
    LLM_MODEL, OPENAI_API_KEY, MAX_TOKENS,
    KNOWLEDGE_DIR, get_openai_client, get_langfuse_handler,
)
from protocol_engine.models.enums import EdgeSignal, NodeName
from protocol_engine.models.state import get_runtime
from protocol_engine.validation.validator import validate

logger = logging.getLogger(__name__)


BASE_SYSTEM_PROMPT = """You are a clinical protocol extraction specialist.

The context may contain [Page N] markers, [Table: ...] blocks, or raw protocol text.
Use these markers for grounding: copy page numbers into your grounding fields.

EXTRACTION RULES:

1. COMPLETENESS: Extract EVERY item you find in the provided content.
   If inclusion has 8 criteria, extract all 8. Do NOT summarize or merge.

2. PRESERVE DETAIL: Include ALL sub-bullets, sub-criteria, and definitions.

3. PRESERVE THRESHOLDS: Include ALL clinical numbers verbatim.
   WRONG: "fever threshold" -> RIGHT: "fever (>= 38C)"

4. GROUNDING: EVERY extracted item MUST have grounding:
   - section_id: the section number from the context
   - page: integer page number from [Page N] markers (NEVER 0 or -1)
   - exact_source_text: verbatim quote 10-80 chars from the source
   - confidence: 0.9 for verbatim, 0.7 for paraphrased, 0.5 if unsure

5. CRITICAL — "NOT IN CONTEXT" ≠ "NOT IN PROTOCOL":
   You are seeing a SUBSET of the protocol, not the entire document.
   If you cannot find information about a topic, add it to the 'gaps' field.
   Do NOT claim the protocol LACKS something unless the text explicitly says so.
"""


def _load_appendix(schema_name: str) -> str:
    """Load domain knowledge appendix for a schema from JSON files."""
    appendix_map = {
        "DeviationRuleSet": "cdisc",
        "KRIExtraction": "sdtm_mappings",
    }
    domain = appendix_map.get(schema_name)
    if not domain:
        return ""
    path = KNOWLEDGE_DIR / f"{domain}.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text())
        return json.dumps(data, indent=2)[:3000]
    except Exception:
        return ""


def extractor_node(state: dict, config: RunnableConfig) -> dict:
    """Extractor node — structured extraction with content VISIBLE to the LLM.

    Critical difference from old code: the LLM that decides what to extract
    IS the same LLM that sees the content. No indirection through tools.
    """
    runtime = get_runtime(config)
    bus = runtime.event_bus
    qt = state.get("query_type", "general")
    assembled_context = state.get("assembled_context", "")

    # Fallback: if context assembler didn't run, build from raw sections
    if not assembled_context:
        sections = state.get("sections_content", {})
        tables = state.get("tables_content", {})
        if not sections and not tables:
            return {
                "extracted_data": {},
                "validation": {},
                "edge_signal": EdgeSignal.DONE,
                "error": "No content available for extraction",
                "steps": [{"agent": NodeName.EXTRACTOR, "turns": 0,
                           "tool_calls": 0, "tools_used": []}],
            }
        # Build context from raw content
        parts = []
        for sid, data in sections.items():
            parts.append(f"[§{sid}]\n{data.get('text', '')}")
        for tid, data in tables.items():
            parts.append(f"[TABLE: {tid}]\n{data.get('text', '')}")
        assembled_context = "\n\n---\n\n".join(parts)

    if bus:
        bus.emit(NodeName.EXTRACTOR, "starting",
                 f"Extracting {qt} from ~{len(assembled_context) // 4} tokens...")

    # Get schema class
    from protocol_engine.models.schemas import SCHEMA_MAP
    schema_class = SCHEMA_MAP.get(qt, SCHEMA_MAP.get("general"))
    if not schema_class:
        return {
            "extracted_data": {},
            "validation": {},
            "edge_signal": EdgeSignal.ERROR_FATAL,
            "error": f"No schema for query type: {qt}",
            "steps": [{"agent": NodeName.EXTRACTOR, "turns": 0,
                       "tool_calls": 0, "tools_used": []}],
        }

    schema_name = schema_class.__name__
    appendix = _load_appendix(schema_name)

    # Build the extraction prompt — LLM sees REAL content
    system = f"""{BASE_SYSTEM_PROMPT}

EXTRACT: {qt} information. Fill every field in the schema.
{f'DOMAIN KNOWLEDGE:{chr(10)}{appendix}' if appendix else ''}"""

    user_msg = f"""Extract all {qt} information from the following protocol content.

PROTOCOL CONTENT:
{assembled_context}"""

    logger.info(
        f"Extractor: {qt}, ~{(len(system) + len(user_msg)) // 4} tokens input"
    )

    t0 = time.time()
    result, raw_text, info = _extract_with_structured_output(
        system, user_msg, schema_class,
    )
    elapsed = time.time() - t0

    if result is None:
        logger.warning(f"Extraction failed: {info.get('error', 'unknown')}")
        if bus:
            bus.emit(NodeName.EXTRACTOR, "done", "Extraction failed")
        return {
            "extracted_data": {},
            "validation": {},
            "edge_signal": EdgeSignal.ERROR_RETRY,
            "error": f"Extraction failed: {info.get('error', '')}",
            "steps": [{"agent": NodeName.EXTRACTOR, "turns": 1,
                       "tool_calls": 0, "tools_used": [info.get("method", "")]}],
        }

    # Convert to dict
    extracted = result.model_dump() if isinstance(result, BaseModel) else result

    # Run deterministic validation
    val_result = validate(extracted, assembled_context)

    logger.info(
        f"Extractor done: {len(extracted)} keys, "
        f"{val_result.get('verified', 0)}/{val_result.get('total', 0)} verified, "
        f"{elapsed:.1f}s"
    )
    if bus:
        bus.emit(NodeName.EXTRACTOR, "done",
                 f"{val_result.get('verified', 0)}/{val_result.get('total', 0)} verified")

    return {
        "extracted_data": extracted,
        "validation": val_result,
        "extraction_history": [{"query_type": qt, "method": info.get("method", ""),
                                "elapsed": round(elapsed, 1)}],
        "edge_signal": EdgeSignal.CONTINUE,
        "steps": [{"agent": NodeName.EXTRACTOR, "turns": 1,
                   "tool_calls": 0, "tools_used": [info.get("method", "")]}],
    }


def _extract_with_structured_output(
    system: str, user_msg: str, schema_class: type,
) -> tuple:
    """Extract using LangChain structured output, with JSON mode fallback.

    Returns: (parsed_result, raw_text, info_dict)
    """
    from protocol_engine.utils import parse_llm_json

    lf = get_langfuse_handler()
    callbacks = [lf] if lf else []

    # Strategy 1: Structured output (function calling)
    try:
        llm = ChatOpenAI(
            model=LLM_MODEL, api_key=OPENAI_API_KEY,
            temperature=0.1, max_tokens=MAX_TOKENS, callbacks=callbacks,
        )
        structured_llm = llm.with_structured_output(schema_class, include_raw=True)
        response = structured_llm.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ])

        if isinstance(response, dict):
            result = response.get("parsed")
            raw_msg = response.get("raw")
            raw_content = raw_msg.content if raw_msg and hasattr(raw_msg, "content") else ""
            if result and isinstance(result, BaseModel):
                return result, result.model_dump_json(), {"valid": True, "method": "structured_output"}
            # Try salvage from raw
            if raw_content:
                parsed = parse_llm_json(raw_content)
                if parsed:
                    try:
                        result = schema_class.model_validate(parsed)
                        return result, result.model_dump_json(), {"valid": True, "method": "raw_salvage"}
                    except ValidationError:
                        pass
        elif isinstance(response, BaseModel):
            return response, response.model_dump_json(), {"valid": True, "method": "structured_output"}

    except Exception as e:
        logger.warning(f"Structured output failed: {e}")

    # Strategy 2: JSON mode fallback
    try:
        client = get_openai_client()
        schema_fields = list(schema_class.model_fields.keys())
        json_system = system + f"\n\nReturn ONLY valid JSON with fields: {schema_fields}."

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": json_system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw_text = response.choices[0].message.content or ""
        parsed = parse_llm_json(raw_text)
        if parsed:
            try:
                result = schema_class.model_validate(parsed)
                return result, raw_text, {"valid": True, "method": "json_mode_fallback"}
            except ValidationError:
                return parsed, raw_text, {"valid": False, "method": "json_mode_fallback"}

    except Exception as e:
        logger.error(f"Both extraction strategies failed: {e}")

    return None, "", {"error": "both strategies failed", "method": "both_failed"}
