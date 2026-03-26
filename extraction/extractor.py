"""
Extractor — Schema-typed structured extraction.

Strategy:
  1. Primary: with_structured_output(method="function_calling") — framework-native
  2. Fallback: JSON mode (response_format=json_object) + Pydantic model_validate

Both paths return a validated Pydantic object. The fallback exists because
complex nested schemas with defaults can cause function calling to return None.
"""
from __future__ import annotations
import json
import time
import logging
from pydantic import BaseModel, ValidationError

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from config import get_openai_client, LLM_MODEL, OPENAI_API_KEY, MAX_TOKENS
from extraction.schemas import SCHEMA_MAP
from shared.json_parser import parse_llm_json

logger = logging.getLogger(__name__)


# Domain knowledge appendices — injected when the schema needs them
KNOWLEDGE_APPENDICES = {
    "DeviationRuleSet": """
CDISC SDTM DOMAIN REFERENCE:
DM (Demographics): AGE, SEX, RACE, ETHNIC, COUNTRY, SITEID, RFSTDTC, DTHFL
VS (Vital Signs): VSTESTCD -> TEMP, SYSBP/DIABP, HR, RESP, WEIGHT, HEIGHT, BMI
LB (Laboratory): LBTESTCD -> ALT, AST, CREAT, HGB, WBC, PLT, GLUC, BILI, PREG
MH (Medical History): MHTERM, MHDECOD (MedDRA), MHONGO
CM (Concomitant Meds): CMTRT, CMDECOD (WHODrug), CMCAT, CMONGO
AE (Adverse Events): AETERM, AEDECOD, AESEV, AESER
EX (Exposure): EXTRT, EXDOSE, EXROUTE, EXSTDTC/EXENDTC
DS (Disposition): DSDECOD (COMPLETED/SCREEN FAILURE/ADVERSE EVENT)
IE (Inclusion/Exclusion): IETESTCD, IESTRESC (Y/N)
SC (Subject Characteristics): SCTESTCD
SV (Subject Visits): VISITNUM, VISITDY, SVSTDTC (actual visit date)
TV (Trial Visits): VISITNUM, VISITDY (planned visit day)
DV (Protocol Deviations): DVTERM, DVCAT, DVDECOD

DEVIATION CATEGORIES (per SDTM DV.DVCAT):
  eligibility: enrollment of ineligible subject
  dosing: wrong dose, missed dose, wrong route
  visit_compliance: visit outside window, missed visit
  prohibited_medication: use of disallowed concomitant medication
  safety_reporting: late SAE report, missed AESI notification
  consent: consent issues
  assessment: missed protocol-required assessment

AUTOMATION LEVELS:
  FULL = computable from SDTM data alone (DM.AGE >= 18)
  PARTIAL = needs data + clinical interpretation
  MANUAL = purely subjective, no SDTM variable exists
""",
    "KRIExtraction": """
KEY RISK INDICATOR FRAMEWORK (per ICH E6(R3) §5.0 and TransCelerate RBQM):

Common KRIs derivable from protocols:
  ENROLLMENT: Screening Failure Rate (DS), Enrollment Rate, Randomization Imbalance
  SAFETY: AE Reporting Rate (AE), SAE Reporting Timeliness (AE.AESTDTC vs report date),
          AESI Incidence Rate, Dose Modification Rate (EX)
  DATA QUALITY: Query Rate, Data Entry Lag, Missing Data Rate
  VISIT COMPLIANCE: Visit Window Compliance (SV vs TV), Early Discontinuation Rate (DS)
  PROTOCOL DEVIATION: Overall Deviation Rate (DV), Eligibility Deviation Rate (IE),
          Dosing Deviation Rate (EX), Prohibited Med Violations (CM)
  CONSENT: Consent Before Procedures Rate, Re-consent Compliance

Each KRI needs:
  1. Protocol source: which section defines the requirement
  2. SDTM data source: which domain/variables to compute it
  3. Metric: rate, count, proportion, or time
  4. QTL threshold: what level triggers a signal (per ICH E6(R3) §5.0.3)
  5. Signal direction: above (too high), below (underreporting), or both
""",
}


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
   If you cannot find information about a topic, add it to the 'gaps' field as:
   "Not found in provided content — may exist in other protocol sections."
   Do NOT claim the protocol LACKS something unless the text explicitly says so.
   Do NOT generate findings about missing procedures, missing safety elements, 
   or missing timeframes based on absence from your context window.
"""


def extract(
    context: str,
    observations: str,
    query_type: str,
    user_query: str = "",
    debug_log=None,
) -> tuple[BaseModel | dict | None, str, dict]:
    """
    Run typed extraction.
    
    Primary: LangChain with_structured_output (function calling).
    Fallback: OpenAI JSON mode + Pydantic model_validate.
    
    Returns: (parsed_result, raw_response_text, info_dict)
    """
    schema_class = SCHEMA_MAP.get(query_type, SCHEMA_MAP["general"])
    schema_name = schema_class.__name__
    appendix = KNOWLEDGE_APPENDICES.get(schema_name, "")

    system = f"""{BASE_SYSTEM_PROMPT}

EXTRACT: {query_type} information. Fill every field in the schema.
{appendix}"""

    user_msg = f"""{f'Query: {user_query}' if user_query else f'Extract all {query_type} information.'}

{f'AGENT OBSERVATIONS: {observations}' if observations else ''}

PROTOCOL CONTEXT:
{context}"""

    input_chars = len(system) + len(user_msg)
    logger.info(
        f"LLM CALL [Extractor] model={LLM_MODEL} "
        f"type={query_type} input={input_chars} chars "
        f"(~{input_chars // 4} tokens)"
    )
    if debug_log:
        debug_log.extraction_input(system, user_msg, query_type)

    t0 = time.time()

    # ── Strategy 1: LangChain structured output (function calling) ──
    try:
        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0.1,
            max_tokens=MAX_TOKENS,
        )
        structured_llm = llm.with_structured_output(schema_class, include_raw=True)

        response = structured_llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user_msg),
        ])

        elapsed = time.time() - t0

        # include_raw=True returns {"raw": AIMessage, "parsed": Pydantic | None, "parsing_error": Exception | None}
        result = None
        raw_content = ""
        if isinstance(response, dict):
            result = response.get("parsed")
            raw_msg = response.get("raw")
            parsing_error = response.get("parsing_error")
            raw_content = raw_msg.content if raw_msg and hasattr(raw_msg, "content") else ""
            if parsing_error:
                logger.warning(f"Structured output parsing error: {parsing_error}")
        elif isinstance(response, BaseModel):
            result = response

        if result and isinstance(result, BaseModel):
            raw_text = result.model_dump_json()
            logger.info(f"LLM DONE [Extractor/structured] {elapsed:.1f}s output={len(raw_text)} chars")
            if debug_log:
                debug_log.extraction_output(raw_text, elapsed, 0, 0)
                debug_log.extraction_parsed(result.model_dump(), True)
            return result, raw_text, {"valid": True, "method": "structured_output"}

        # Structured output returned None — try to salvage from raw content
        if raw_content:
            logger.warning(f"Structured output returned None. Trying to parse raw content ({len(raw_content)} chars).")
            parsed = parse_llm_json(raw_content)
            if parsed:
                try:
                    result = schema_class.model_validate(parsed)
                    raw_text = result.model_dump_json()
                    logger.info(f"LLM DONE [Extractor/raw_salvage] {elapsed:.1f}s output={len(raw_text)} chars")
                    if debug_log:
                        debug_log.extraction_output(raw_text, elapsed, 0, 0)
                        debug_log.extraction_parsed(result.model_dump(), True)
                    return result, raw_text, {"valid": True, "method": "raw_salvage"}
                except ValidationError:
                    logger.warning("Raw salvage validation failed. Falling back to JSON mode.")

        logger.warning(f"Structured output failed completely after {elapsed:.1f}s. Falling back to JSON mode.")

    except Exception as e:
        elapsed = time.time() - t0
        logger.warning(f"Structured output exception after {elapsed:.1f}s: {e}. Falling back to JSON mode.")

    # ── Strategy 2: Fallback to JSON mode + Pydantic validation ─────
    try:
        t0 = time.time()
        client = get_openai_client()

        schema_fields = list(schema_class.model_fields.keys())
        json_system = system + f"\n\nReturn ONLY valid JSON matching this schema with fields: {schema_fields}. No markdown, no explanation."

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
        elapsed = time.time() - t0
        raw_text = response.choices[0].message.content or ""

        tokens_in = response.usage.prompt_tokens if response.usage else 0
        tokens_out = response.usage.completion_tokens if response.usage else 0
        logger.info(
            f"LLM DONE [Extractor/json_mode] {elapsed:.1f}s output={len(raw_text)} chars "
            f"tokens_in={tokens_in} tokens_out={tokens_out}"
        )
        if debug_log:
            debug_log.extraction_output(raw_text, elapsed, tokens_in, tokens_out)

        parsed = parse_llm_json(raw_text)
        if parsed is None:
            logger.error("JSON mode: failed to parse response")
            return None, raw_text, {"error": "JSON parse failed", "method": "json_mode"}

        try:
            result = schema_class.model_validate(parsed)
            if debug_log:
                debug_log.extraction_parsed(parsed, True)
            return result, raw_text, {"valid": True, "method": "json_mode_fallback"}
        except ValidationError as ve:
            logger.warning(f"JSON mode validation failed: {ve.errors()[:2]}")
            if debug_log:
                debug_log.extraction_parsed(parsed, False, str(ve.errors()[:3]))
            # Return raw dict as best effort — UI can still render it
            return parsed, raw_text, {"valid": False, "method": "json_mode_fallback", "errors": str(ve.errors()[:3])}

    except Exception as e:
        logger.error(f"Both extraction strategies failed: {e}")
        return None, "", {"error": str(e), "method": "both_failed"}
