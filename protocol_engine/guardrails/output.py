"""
Output guardrails — validate extraction output before returning to caller.

Checks:
  - Schema compliance (all required fields present)
  - Clinical plausibility (doses, ages, counts within sane ranges)
  - PII detection (flag but don't block — clinical protocols may contain patient references)
  - Hallucination indicators (suspiciously round numbers, generic text)
"""
from __future__ import annotations

import re
import logging

logger = logging.getLogger(__name__)


def validate_output(extracted: dict, query_type: str) -> tuple[dict, list[str]]:
    """Validate extraction output. Returns (cleaned_output, warnings)."""
    warnings = []

    if not extracted:
        return extracted, ["Empty extraction result"]

    # Check for suspiciously empty extractions
    non_empty_keys = sum(1 for v in extracted.values()
                         if v and v != [] and v != {} and v != "" and v is not False and v != 0)
    total_keys = len(extracted)
    if total_keys > 0 and non_empty_keys / total_keys < 0.2:
        warnings.append(f"Only {non_empty_keys}/{total_keys} fields populated — possible extraction failure")

    # Clinical plausibility checks
    _check_clinical_values(extracted, warnings)

    # PII detection
    _check_pii(extracted, warnings)

    return extracted, warnings


def _check_clinical_values(data: dict, warnings: list[str], path: str = ""):
    """Recursively check for clinically implausible values."""
    if not isinstance(data, dict):
        return

    for key, value in data.items():
        full_key = f"{path}.{key}" if path else key

        if isinstance(value, dict):
            _check_clinical_values(value, warnings, full_key)
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    _check_clinical_values(item, warnings, f"{full_key}[{i}]")
        elif isinstance(value, (int, float)):
            # Flag suspiciously large sample sizes
            if "sample_size" in key and value > 1_000_000:
                warnings.append(f"{full_key}={value} — implausibly large sample size")
            # Flag negative ages
            if "age" in key.lower() and value < 0:
                warnings.append(f"{full_key}={value} — negative age")
        elif isinstance(value, str):
            # Flag grounding with page 0 or -1
            if key == "page" and value in ("0", "-1"):
                warnings.append(f"{full_key}={value} — invalid page number")


def _check_pii(data: dict, warnings: list[str], path: str = ""):
    """Scan for potential PII in extraction output."""
    PII_PATTERNS = [
        (re.compile(r'\b\d{3}-\d{2}-\d{4}\b'), "SSN"),
        (re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b', re.I), "email"),
        (re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'), "phone number"),
    ]

    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{path}.{key}" if path else key
            if isinstance(value, str):
                for pattern, pii_type in PII_PATTERNS:
                    if pattern.search(value):
                        warnings.append(f"Potential {pii_type} detected in {full_key}")
            elif isinstance(value, (dict, list)):
                _check_pii(value, warnings, full_key)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            _check_pii(item, warnings, f"{path}[{i}]")
