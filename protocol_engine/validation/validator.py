"""
Validator — Multi-level deterministic grounding verification.

ZERO LLM calls. Four levels:
  Level 1: Source Grounding — exact_source_text in context?
  Level 2: Numerical Consistency — numbers in claim match source?
  Level 3: Section Reference — section_id exists? page plausible?
  Level 4: Completeness — required fields present?

Key fixes from old code:
  1. Numerical check uses field-specific source text, NOT entire context
     (fixes "39 subjects" matching "fever >= 39C")
  2. Page marker extraction handles more formats
  3. Number extraction ignores more non-clinical patterns
"""
from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)


def validate(extracted_data: dict, context: str) -> dict:
    """Multi-level validation. Returns {verified, flagged, failed, total, details}."""
    verified = flagged = failed = total = 0
    details = []

    context_lower = context.lower()
    context_numbers = _extract_numbers(context)
    context_pages = _extract_page_markers(context)

    for item_desc, grounding, parent_item in _find_groundings(extracted_data):
        total += 1
        result = {"item": item_desc[:80], "checks": {}}

        source_text = grounding.get("exact_source_text", "")
        page = grounding.get("page", 0)
        section_id = grounding.get("section_id", "")

        # Level 1: Source Grounding
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

        # Level 2: Numerical Consistency
        # FIX: Check against SOURCE TEXT, not entire context
        claim_text = _get_claim_text(parent_item)
        if claim_text and source_text:
            num_result = _check_numerical_consistency(claim_text, source_text)
            result["checks"]["numbers"] = num_result
        else:
            result["checks"]["numbers"] = "- skip"

        # Level 3: Section Reference
        if page > 0:
            if context_pages and page in context_pages:
                result["checks"]["page_ref"] = "V"
            elif context_pages:
                result["checks"]["page_ref"] = f"W p.{page} not in context"
            else:
                result["checks"]["page_ref"] = "V no markers"
        else:
            result["checks"]["page_ref"] = "X no page provided"

        if section_id:
            section_id = str(section_id)
            sec_found = (
                f"section {section_id}" in context_lower
                or f"§{section_id}" in context
                or section_id + "." in context
                or section_id + " " in context
            )
            result["checks"]["section_ref"] = "V" if sec_found else f"W {section_id} not found"

        # Level 4: Completeness
        completeness = _check_completeness(parent_item)
        if completeness:
            result["checks"]["completeness"] = completeness

        # Verdict
        checks = result["checks"]
        fails = sum(1 for v in checks.values() if v.startswith("X"))
        source_ok = checks.get("source_grounding", "").startswith("V")
        numbers_ok = checks.get("numbers", "").startswith(("V", "-"))

        if source_ok and fails == 0 and numbers_ok:
            result["verdict"] = "VERIFIED"
            verified += 1
        elif checks.get("numbers", "").startswith("X"):
            result["verdict"] = "FAILED_NUMBERS"
            failed += 1
        elif not source_ok:
            result["verdict"] = "FAILED_SOURCE"
            failed += 1
        else:
            result["verdict"] = "FLAGGED"
            flagged += 1

        details.append(result)

    logger.info(f"Validation: {verified}/{total} verified, {flagged} flagged, {failed} failed")
    return {
        "verified": verified,
        "flagged": flagged,
        "failed": failed,
        "total": total,
        "details": details,
    }


def _check_numerical_consistency(claim_text: str, source_text: str) -> str:
    """Check numbers in claim match numbers in SOURCE TEXT (not full context).

    This is the critical fix: old code checked against the entire context,
    causing "39 subjects" to match "fever >= 39C".
    """
    claim_nums = _extract_numbers(claim_text)
    source_nums = _extract_numbers(source_text)

    if not claim_nums:
        return "V no numbers"

    mismatches = []
    for num in claim_nums:
        if num in source_nums:
            continue
        if any(abs(num - sn) < 0.5 for sn in source_nums):
            continue
        mismatches.append(str(num))

    if mismatches:
        return "X claim has " + ",".join(mismatches[:3]) + " not in source"
    return "V numbers match"


def _extract_numbers(text: str) -> set:
    """Extract clinically relevant numbers, ignoring IDs and codes."""
    if not text:
        return set()
    cleaned = re.sub(r'COVID-19|SARS-CoV-2|mRNA-\d+|P-?\d+|Amendment\s+\d+|Version\s+\d+', '', text)
    cleaned = re.sub(r'IC-\d+|EC-\d+|DR-\d+|[A-Z]{2}-\d+', '', cleaned)
    matches = re.findall(r'\d+\.?\d*', cleaned)
    nums = set()
    for m in matches:
        try:
            n = float(m)
            if 1900 <= n <= 2100:
                continue
            if n > 1000 and n != round(n):
                continue
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


def _check_completeness(item: dict) -> str:
    """Check if extracted item has expected fields."""
    if not isinstance(item, dict):
        return ""
    issues = []

    if "endpoint" in item:
        timing = str(item.get("timing", ""))
        if not timing or timing in ("", "N/A", "null", "None"):
            issues.append("missing timing")
        endpoint = str(item.get("endpoint", ""))
        if any(w in endpoint.lower() for w in ["defined as", "criteria", "following"]):
            if not re.search(r'\d', endpoint):
                issues.append("criteria mentioned but no thresholds")

    if "sdtm_domain" in item:
        domain = str(item.get("sdtm_domain", ""))
        condition = str(item.get("condition", ""))
        auto = str(item.get("automation_level", item.get("automation", "")))
        if auto == "FULL" and (not condition or condition in ("null", "None", "")):
            issues.append("FULL automation but no condition")
        valid_domains = {"DM", "VS", "LB", "CM", "MH", "AE", "EX", "DS", "IE", "SC", "SV", "TV", "DV", "FA", "", "null", "None"}
        if domain and domain not in valid_domains:
            issues.append("unknown SDTM domain: " + domain)

    return ("W " + "; ".join(issues)) if issues else ""


def _get_claim_text(item: dict) -> str:
    """Extract main claim text from an extracted item."""
    if not isinstance(item, dict):
        return ""
    for field in ["endpoint", "text", "description", "condition",
                  "statement", "source_criterion", "definition"]:
        val = item.get(field, "")
        if val and str(val) not in ("None", "null", ""):
            return str(val)
    return ""


def _find_groundings(data, path="", parent=None):
    """Recursively find all grounding objects in extracted data."""
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
                for id_key in ("id", "criterion_id", "rule_id", "finding_id", "risk_id", "check_id", "kri_id"):
                    if id_key in item:
                        desc = str(item[id_key])
                        break
            results.extend(_find_groundings(
                item, f"{path}[{i}]{' ' + desc if desc else ''}",
                item if isinstance(item, dict) else parent,
            ))
    return results
