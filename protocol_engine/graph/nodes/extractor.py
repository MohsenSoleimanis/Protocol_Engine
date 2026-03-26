"""
Extractor — Structured extraction + deterministic validation.

The LLM sees ACTUAL content (not a summary). One call extracts,
then deterministic validation runs inline. Cross-field checks included.
"""
from __future__ import annotations

import json
import logging
import time

from pydantic import BaseModel, ValidationError
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from protocol_engine.config import (
    LLM_MODEL, OPENAI_API_KEY, MAX_TOKENS,
    KNOWLEDGE_DIR, get_openai_client, get_langfuse_handler,
)
from protocol_engine.models.enums import EdgeSignal
from protocol_engine.models.state import get_runtime
from protocol_engine.validation.validator import validate

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a clinical protocol extraction specialist.

RULES:
1. COMPLETENESS: Extract EVERY item. Do not summarize or merge.
2. PRESERVE THRESHOLDS: Include ALL clinical numbers verbatim.
   WRONG: "fever threshold" → RIGHT: "fever (>= 38C)"
3. GROUNDING: Every item needs section_id, page (from [Page N] markers),
   exact_source_text (10-80 char verbatim quote), confidence (0.9/0.7/0.5).
4. "NOT IN CONTEXT" ≠ "NOT IN PROTOCOL": You see a SUBSET.
   If you can't find something, add to 'gaps' field. Don't claim it's missing.
"""


def extractor_node(state: dict, config: RunnableConfig) -> dict:
    runtime = get_runtime(config)
    bus = runtime.event_bus
    qt = state.get("query_type", "general")
    context = state.get("assembled_context", "")

    if not context:
        logger.warning("Extractor: no assembled context")
        return {
            "extracted_data": {},
            "validation": {},
            "edge_signal": EdgeSignal.DONE,
            "steps": [{"agent": "extractor", "error": "no context"}],
        }

    if bus:
        bus.emit("extractor", "starting", f"Extracting {qt}...")

    # Get schema
    from protocol_engine.models.schemas import SCHEMA_MAP
    schema_class = SCHEMA_MAP.get(qt, SCHEMA_MAP.get("general"))

    # Load domain knowledge appendix if applicable
    appendix = _load_appendix(schema_class.__name__ if schema_class else "")

    system = f"{SYSTEM_PROMPT}\nEXTRACT: {qt} information. Fill every field.\n{appendix}"
    user_msg = f"Extract all {qt} information.\n\nPROTOCOL CONTENT:\n{context}"

    t0 = time.time()
    result, raw, info = _extract(system, user_msg, schema_class)
    elapsed = time.time() - t0

    if result is None:
        logger.warning(f"Extraction failed: {info}")
        if bus:
            bus.emit("extractor", "done", "Extraction failed")
        return {
            "extracted_data": {},
            "validation": {},
            "edge_signal": EdgeSignal.DONE,
            "steps": [{"agent": "extractor", "error": str(info), "elapsed": round(elapsed, 1)}],
        }

    extracted = result.model_dump() if isinstance(result, BaseModel) else result

    # Deterministic validation (source text, not full context — FIX C6)
    val = validate(extracted, context)

    # Cross-field consistency (was in Reconciler, now inline)
    _cross_field_check(extracted, val)

    logger.info(f"Extractor: {val.get('verified',0)}/{val.get('total',0)} verified, {elapsed:.1f}s")
    if bus:
        bus.emit("extractor", "done", f"{val.get('verified',0)}/{val.get('total',0)} verified")

    return {
        "extracted_data": extracted,
        "validation": val,
        "edge_signal": EdgeSignal.DONE,
        "steps": [{"agent": "extractor", "method": info.get("method", ""),
                   "elapsed": round(elapsed, 1),
                   "verified": val.get("verified", 0), "total": val.get("total", 0)}],
    }


def _extract(system: str, user_msg: str, schema_class: type) -> tuple:
    """Structured output with JSON-mode fallback. Returns (result, raw, info)."""
    lf = get_langfuse_handler()
    cbs = [lf] if lf else []

    # Strategy 1: LangChain structured output
    try:
        llm = ChatOpenAI(model=LLM_MODEL, api_key=OPENAI_API_KEY,
                         temperature=0.1, max_tokens=MAX_TOKENS, callbacks=cbs)
        structured = llm.with_structured_output(schema_class, include_raw=True)
        resp = structured.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ])
        if isinstance(resp, dict):
            parsed = resp.get("parsed")
            if parsed and isinstance(parsed, BaseModel):
                return parsed, parsed.model_dump_json(), {"method": "structured_output"}
            # Try salvage from raw
            raw_msg = resp.get("raw")
            if raw_msg and hasattr(raw_msg, "content") and raw_msg.content:
                return _try_parse(raw_msg.content, schema_class, "raw_salvage")
        elif isinstance(resp, BaseModel):
            return resp, resp.model_dump_json(), {"method": "structured_output"}
    except Exception as e:
        logger.warning(f"Structured output failed: {e}")

    # Strategy 2: JSON mode fallback
    try:
        client = get_openai_client()
        fields = list(schema_class.model_fields.keys())
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system + f"\nReturn ONLY JSON with fields: {fields}"},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=MAX_TOKENS, temperature=0.1,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
        return _try_parse(raw, schema_class, "json_mode")
    except Exception as e:
        logger.error(f"Both strategies failed: {e}")
        return None, "", {"error": str(e)}


def _try_parse(raw: str, schema_class: type, method: str) -> tuple:
    """Try to parse raw JSON into schema."""
    import re
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    for attempt in [cleaned, raw]:
        try:
            data = json.loads(attempt)
            result = schema_class.model_validate(data)
            return result, raw, {"method": method}
        except (json.JSONDecodeError, ValidationError):
            pass

    # Last resort: find outermost braces
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            data = json.loads(raw[start:end])
            result = schema_class.model_validate(data)
            return result, raw, {"method": f"{method}_brace"}
        except (json.JSONDecodeError, ValidationError):
            pass

    return None, raw, {"error": "parse_failed", "method": method}


def _cross_field_check(extracted: dict, validation: dict):
    """Inline cross-field consistency checks (was Reconciler node)."""
    details = validation.setdefault("cross_field", [])

    # Endpoint count match
    endpoints = extracted.get("endpoints", [])
    total = extracted.get("total", {})
    if isinstance(endpoints, list) and isinstance(total, dict):
        actual = sum(1 for e in endpoints if isinstance(e, dict) and e.get("category") == "Primary")
        declared = total.get("primary", 0)
        if declared > 0 and actual != declared:
            details.append(f"Declared {declared} primary endpoints but extracted {actual}")

    # Arms count match
    arms = extracted.get("arms", [])
    n_arms = extracted.get("number_of_arms", 0)
    if isinstance(arms, list) and n_arms > 0 and len(arms) != n_arms:
        details.append(f"Declared {n_arms} arms but extracted {len(arms)}")


def _load_appendix(schema_name: str) -> str:
    mapping = {"DeviationRuleSet": "cdisc", "KRIExtraction": "sdtm_mappings"}
    domain = mapping.get(schema_name)
    if not domain:
        return ""
    path = KNOWLEDGE_DIR / f"{domain}.json"
    if not path.exists():
        return ""
    try:
        return json.dumps(json.loads(path.read_text()), indent=2)[:3000]
    except Exception:
        return ""
