"""
Scoring functions for extraction evaluation.

Per the report: evaluate across 5 pillars — logic/accuracy, performance,
resilience, governance, and user experience.

For clinical protocol extraction, the key metrics are:
  1. Extraction completeness (did we get everything?)
  2. Grounding accuracy (can we trace every claim to source?)
  3. Numerical fidelity (do numbers match?)
  4. Retrieval recall (did we find all relevant sections?)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def score_extraction(
    extracted: dict,
    validation: dict,
    expected: dict | None = None,
) -> dict:
    """Score an extraction result.

    Args:
        extracted: The extraction output
        validation: Deterministic validation results
        expected: Optional expected values for regression testing

    Returns:
        {scores: {metric: float}, details: [...]}
    """
    scores = {}
    details = []

    # 1. Grounding rate (from validator)
    total = validation.get("total", 0)
    verified = validation.get("verified", 0)
    scores["grounding_rate"] = verified / max(total, 1)
    details.append(f"Grounding: {verified}/{total} verified")

    # 2. Failure rate
    failed = validation.get("failed", 0)
    scores["failure_rate"] = failed / max(total, 1)
    details.append(f"Failures: {failed}/{total}")

    # 3. Completeness (non-empty fields)
    if extracted:
        non_empty = sum(1 for v in extracted.values()
                       if v and v != [] and v != {} and v != "")
        total_fields = len(extracted)
        scores["field_completeness"] = non_empty / max(total_fields, 1)
        details.append(f"Fields: {non_empty}/{total_fields} populated")

    # 4. Cross-field issues
    xf = validation.get("cross_field", [])
    scores["cross_field_issues"] = len(xf)
    if xf:
        details.append(f"Cross-field issues: {xf}")

    # 5. Expected value checks (regression testing)
    if expected:
        passes, checks = _check_expected(extracted, expected)
        scores["expected_pass_rate"] = passes / max(checks, 1)
        details.append(f"Expected checks: {passes}/{checks} passed")

    # Overall score (weighted average)
    scores["overall"] = (
        scores.get("grounding_rate", 0) * 0.4 +
        (1 - scores.get("failure_rate", 0)) * 0.3 +
        scores.get("field_completeness", 0) * 0.2 +
        (1 if scores.get("cross_field_issues", 0) == 0 else 0) * 0.1
    )

    return {"scores": scores, "details": details}


def _check_expected(extracted: dict, expected: dict) -> tuple[int, int]:
    """Check extraction against expected values. Returns (passes, total_checks)."""
    passes = 0
    checks = 0

    # Count checks (e.g. inclusion_count_min: 5)
    for key, value in expected.items():
        checks += 1
        if key.endswith("_count_min"):
            field = key.replace("_count_min", "")
            items = extracted.get(field, [])
            if isinstance(items, list) and len(items) >= value:
                passes += 1
        elif key == "must_contain":
            text = json.dumps(extracted).lower()
            all_found = all(term.lower() in text for term in value)
            if all_found:
                passes += 1

    return passes, checks


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, help="Path to extraction result JSON")
    args = parser.parse_args()

    data = json.loads(Path(args.result).read_text())
    result = score_extraction(
        data.get("data", {}),
        data.get("validation", {}),
    )
    print(json.dumps(result, indent=2))
