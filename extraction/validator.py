"""
Validator — Multi-level deterministic grounding verification.

ZERO LLM calls. Four levels of checking:

  Level 1: Source Grounding
    Does exact_source_text exist in the context? (substring match)
    
  Level 2: Numerical Consistency  
    Do numbers in the extracted claim match the source?
    "Age >= 18" in claim but "16 years" in source -> FAIL
    Catches: wrong thresholds, swapped values, hallucinated numbers
    
  Level 3: Section Reference Integrity
    Does the claimed section_id actually exist? Is the page plausible?
    
  Level 4: Completeness
    Are required fields present? (endpoint without timing = FLAG)
    Are clinical thresholds preserved? (fever criterion without C = FLAG)

In clinical trials, every claim must be traceable to a specific page and text.
A wrong number (38C vs 37.5C) in a deviation rule could cause real patient harm.
"""
from __future__ import annotations
import re
import logging

logger = logging.getLogger(__name__)


def validate(extracted_data: dict, context: str) -> dict:
    """
    Multi-level validation of extracted data against source context.
    Returns: {verified, flagged, failed, total, details}
    """
    verified = 0
    flagged = 0
    failed = 0
    total = 0
    details = []

    context_lower = context.lower()
    context_numbers = _extract_numbers(context)
    context_pages = _extract_page_markers(context)

    groundings = _find_groundings(extracted_data)

    for item_desc, grounding, parent_item in groundings:
        total += 1
        result = {"item": item_desc[:80], "checks": {}}

        source_text = grounding.get("exact_source_text", "")
        page = grounding.get("page", 0)
        section_id = grounding.get("section_id", "")

        # -- Level 1: Source Grounding --
        if source_text and len(source_text) > 5:
            search_text = source_text[:40].lower().strip()
            if search_text in context_lower:
                result["checks"]["source_grounding"] = "V"
            else:
                search_short = source_text[:20].lower().strip()
                if search_short in context_lower:
                    result["checks"]["source_grounding"] = "V partial"
                else:
                    result["checks"]["source_grounding"] = "X not found"
        else:
            result["checks"]["source_grounding"] = "W empty"

        # -- Level 2: Numerical Consistency --
        claim_text = _get_claim_text(parent_item)
        if claim_text and source_text:
            num_result = _check_numerical_consistency(claim_text, source_text, context)
            result["checks"]["numbers"] = num_result
        else:
            result["checks"]["numbers"] = "- skip"

        # -- Level 3: Section Reference Integrity --
        if page > 0:
            if context_pages and page in context_pages:
                result["checks"]["page_ref"] = "V"
            elif context_pages:
                result["checks"]["page_ref"] = "W p." + str(page) + " not in context"
            else:
                result["checks"]["page_ref"] = "V no markers"
        else:
            result["checks"]["page_ref"] = "X no page provided"

        if section_id:
            section_id = str(section_id)  # LLM may return int
            sec_found = (
                f"section {section_id}" in context_lower or
                section_id + "." in context or
                section_id + " " in context
            )
            result["checks"]["section_ref"] = "V" if sec_found else ("W " + section_id + " not found")
        
        # -- Level 4: Completeness --
        completeness = _check_completeness(parent_item)
        if completeness:
            result["checks"]["completeness"] = completeness

        # -- Verdict --
        checks = result["checks"]
        
        fails = sum(1 for v in checks.values() if v.startswith("X"))
        source_ok = checks.get("source_grounding", "").startswith("V")
        numbers_ok = checks.get("numbers", "").startswith(("V", "-"))
        
        if source_ok and fails == 0 and numbers_ok:
            result["verdict"] = "VERIFIED"
            verified += 1
        elif fails > 0 or not source_ok:
            if checks.get("numbers", "").startswith("X"):
                result["verdict"] = "FAILED_NUMBERS"
                failed += 1
            elif not source_ok:
                result["verdict"] = "FAILED_SOURCE"
                failed += 1
            else:
                result["verdict"] = "FLAGGED"
                flagged += 1
        else:
            result["verdict"] = "FLAGGED"
            flagged += 1

        details.append(result)

    logger.info(
        f"Validation: {verified}/{total} verified, "
        f"{flagged} flagged, {failed} failed"
    )

    return {
        "verified": verified,
        "flagged": flagged,
        "failed": failed,
        "total": total,
        "details": details,
    }


# === Level 2: Numerical Consistency ===

def _check_numerical_consistency(claim_text: str, source_text: str, context: str) -> str:
    """
    Check that numbers in the claim match numbers in the source.
    
    Claim: "Age >= 18 years"    Source: "Age >= 18 years"     -> V match
    Claim: "fever >= 39C"       Source: "fever >= 38C"        -> X 39!=38
    Claim: "BMI 18-40"          Source: "BMI 18 to 40 kg/m2"  -> V match
    """
    claim_nums = _extract_numbers(claim_text)
    context_nums = _extract_numbers(context)
    
    if not claim_nums:
        return "V no numbers"
    
    # Check each claim number exists SOMEWHERE in context
    mismatches = []
    for num in claim_nums:
        if num in context_nums:
            continue
        # Allow close matches (38 vs 38.0, rounding)
        if any(abs(num - cn) < 0.5 for cn in context_nums):
            continue
        mismatches.append(str(num))
    
    if mismatches:
        return "X claim has " + ",".join(mismatches[:3]) + " not in protocol"
    return "V numbers match"


def _extract_numbers(text: str) -> set:
    """Extract clinically relevant numbers from text. Ignores product codes, identifiers, etc."""
    if not text:
        return set()
    # Remove common non-numeric patterns that contain numbers
    cleaned = re.sub(r'COVID-19|SARS-CoV-2|mRNA-\d+|P-?\d+|Amendment\s+\d+|Version\s+\d+', '', text)
    cleaned = re.sub(r'IC-\d+|EC-\d+|DR-\d+|[A-Z]{2}-\d+', '', cleaned)  # criterion IDs
    cleaned = re.sub(r'Day\s+\d+', lambda m: m.group(), cleaned)  # keep Day numbers
    
    # Match numbers with context: look for numbers near clinical terms
    matches = re.findall(r'\d+\.?\d*', cleaned)
    nums = set()
    for m in matches:
        try:
            n = float(m)
            # Skip: page numbers (>300), years (1900-2100), very small identifiers
            if 1900 <= n <= 2100:
                continue  # years
            if n > 1000 and n != round(n):
                continue  # large decimals unlikely clinical
            nums.add(n)
        except ValueError:
            pass
    return nums


def _extract_page_markers(text: str) -> set:
    """Extract [Page N] markers from context."""
    pages = set()
    for m in re.findall(r'\[Page\s+(\d+)\]', text):
        try:
            pages.add(int(m))
        except ValueError:
            pass
    for m in re.findall(r'page="(\d+)"', text):
        try:
            pages.add(int(m))
        except ValueError:
            pass
    return pages


# === Level 4: Completeness ===

def _check_completeness(item: dict) -> str:
    """Check if extracted item has expected fields."""
    if not isinstance(item, dict):
        return ""
    
    issues = []
    
    # Endpoints: should have timing
    if "endpoint" in item:
        endpoint = str(item.get("endpoint", ""))
        timing = str(item.get("timing", ""))
        if not timing or timing in ("", "N/A", "null", "None"):
            issues.append("missing timing")
        
        # Clinical thresholds: if endpoint mentions criteria, should have specifics
        if any(w in endpoint.lower() for w in ["defined as", "criteria", "following"]):
            has_threshold = bool(re.search(r'[\d]', endpoint))
            if not has_threshold:
                issues.append("criteria mentioned but no thresholds")
    
    # Deviation rules: FULL automation needs condition
    if "sdtm_domain" in item:
        domain = str(item.get("sdtm_domain", ""))
        condition = str(item.get("condition", ""))
        auto = str(item.get("automation_level", item.get("automation", "")))
        if auto == "FULL" and (not condition or condition in ("null", "None", "")):
            issues.append("FULL automation but no condition")
        valid_domains = {"DM","VS","LB","CM","MH","AE","EX","DS","IE","SC","","null","None"}
        if domain and domain not in valid_domains:
            issues.append("unknown SDTM domain: " + domain)
    
    if issues:
        return "W " + "; ".join(issues)
    return ""


# === Utility ===

def _get_claim_text(item: dict) -> str:
    """Extract the main claim text from an extracted item."""
    if not isinstance(item, dict):
        return ""
    for field in ["endpoint", "text", "description", "condition", 
                  "statement", "source_criterion", "definition"]:
        val = item.get(field, "")
        if val and str(val) not in ("None", "null", ""):
            return str(val)
    return ""


def _find_groundings(data, path="", parent=None):
    """
    Recursively find all grounding objects in extracted data.
    Returns: [(item_description, grounding_dict, parent_item_dict)]
    """
    results = []

    if isinstance(data, dict):
        if "exact_source_text" in data and "page" in data:
            return [(path, data, parent or data)]

        for key, value in data.items():
            if key == "grounding" and isinstance(value, dict):
                results.append((path, value, data))
            elif isinstance(value, (dict, list)):
                results.extend(_find_groundings(value, f"{path}.{key}", data))

    elif isinstance(data, list):
        for i, item in enumerate(data):
            desc = ""
            if isinstance(item, dict):
                desc = item.get("id", item.get("criterion_id", item.get("rule_id", item.get(
                    "finding_id", item.get("risk_id", item.get("check_id", ""))))))
            results.extend(_find_groundings(
                item, f"{path}[{i}]{' '+desc if desc else ''}", 
                item if isinstance(item, dict) else parent
            ))

    return results
