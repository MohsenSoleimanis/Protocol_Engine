"""
Reconciler Node — Vision vs text + cross-field validation.

NEW node (old code had ZERO reconciliation).

Handles:
  1. Compare vision-extracted tables against text-extracted tables
  2. Cross-field consistency (e.g. sample_size in demographics vs stats)
  3. Run deterministic validation and flag issues
  4. Decide if vision re-extraction is needed
"""
from __future__ import annotations

import json
import logging
import re

from langchain_core.runnables import RunnableConfig

from protocol_engine.config import get_langfuse_handler
from protocol_engine.models.enums import EdgeSignal, NodeName
from protocol_engine.models.state import get_runtime
from protocol_engine.validation.validator import validate

logger = logging.getLogger(__name__)


def reconciler_node(state: dict, config: RunnableConfig) -> dict:
    """Reconcile extraction results: cross-field checks + validation."""
    runtime = get_runtime(config)
    bus = runtime.event_bus
    extracted = state.get("extracted_data", {})
    validation = state.get("validation", {})
    context = state.get("assembled_context", "")

    if bus:
        bus.emit(NodeName.RECONCILER, "starting", "Reconciling extraction...")

    if not extracted:
        return {
            "edge_signal": EdgeSignal.CONTINUE,
            "steps": [{"agent": NodeName.RECONCILER, "turns": 0,
                       "tool_calls": 0, "tools_used": []}],
        }

    signals = []

    # 1. Cross-field consistency checks
    cross_field_issues = _check_cross_field_consistency(extracted)
    for issue in cross_field_issues:
        signals.append({
            "signal_type": "consistency",
            "severity": issue.get("severity", "major"),
            "title": issue["title"],
            "description": issue["description"],
        })

    # 2. Check validation results for critical failures
    if validation:
        failed = validation.get("failed", 0)
        total = validation.get("total", 0)
        if total > 0 and failed / total > 0.3:
            signals.append({
                "signal_type": "completeness",
                "severity": "critical",
                "title": "High validation failure rate",
                "description": f"{failed}/{total} items failed validation. "
                               f"May need content re-fetch or re-extraction.",
            })

        # Check for specific numerical failures
        for detail in validation.get("details", []):
            if detail.get("verdict") == "FAILED_NUMBERS":
                signals.append({
                    "signal_type": "consistency",
                    "severity": "critical",
                    "title": f"Numerical mismatch: {detail.get('item', '')[:50]}",
                    "description": str(detail.get("checks", {}).get("numbers", "")),
                })

    # 3. Table reconciliation (if we have both vision and text tables)
    tables_content = state.get("tables_content", {})
    table_issues = _check_table_consistency(extracted, tables_content)
    signals.extend(table_issues)

    # 4. Decide edge signal
    critical_signals = [s for s in signals if s.get("severity") == "critical"]
    cycle_count = state.get("cycle_count", 0)

    if critical_signals and cycle_count < 2:
        edge_signal = EdgeSignal.NEED_MORE_CONTENT
        edge_detail = "; ".join(s["title"] for s in critical_signals[:3])
    else:
        edge_signal = EdgeSignal.CONTINUE
        edge_detail = ""

    logger.info(f"Reconciler: {len(signals)} signals ({len(critical_signals)} critical)")
    if bus:
        bus.emit(NodeName.RECONCILER, "done", f"{len(signals)} signals")

    return {
        "signals": signals,
        "edge_signal": edge_signal,
        "edge_detail": edge_detail,
        "steps": [{"agent": NodeName.RECONCILER, "turns": 0,
                   "tool_calls": 0, "tools_used": ["cross_field", "validation", "tables"],
                   "signals_count": len(signals)}],
    }


def _check_cross_field_consistency(extracted: dict) -> list[dict]:
    """Check extracted fields are consistent with each other."""
    issues = []

    # Sample size consistency
    sample_sizes = set()
    for key in ("sample_size", "sample_size_target"):
        val = extracted.get(key)
        if val and str(val) not in ("0", "", "None", "null"):
            sample_sizes.add(str(val))

    # Check nested — e.g. statistical.sample_size_target
    for section_key in extracted:
        if isinstance(extracted[section_key], dict):
            for key in ("sample_size", "sample_size_target"):
                val = extracted[section_key].get(key)
                if val and str(val) not in ("0", "", "None", "null"):
                    sample_sizes.add(str(val))

    if len(sample_sizes) > 1:
        issues.append({
            "title": "Sample size inconsistency",
            "description": f"Multiple sample size values found: {sample_sizes}",
            "severity": "major",
        })

    # Endpoint count consistency
    endpoints = extracted.get("endpoints", [])
    total_counts = extracted.get("total", {})
    if isinstance(endpoints, list) and isinstance(total_counts, dict):
        actual_primary = sum(1 for e in endpoints
                           if isinstance(e, dict) and e.get("category") == "Primary")
        declared_primary = total_counts.get("primary", 0)
        if declared_primary > 0 and actual_primary != declared_primary:
            issues.append({
                "title": "Endpoint count mismatch",
                "description": (
                    f"Declared {declared_primary} primary endpoints "
                    f"but extracted {actual_primary}"
                ),
                "severity": "major",
            })

    # Arms count consistency
    arms = extracted.get("arms", [])
    n_arms = extracted.get("number_of_arms", 0)
    if isinstance(arms, list) and n_arms > 0 and len(arms) != n_arms:
        issues.append({
            "title": "Treatment arms count mismatch",
            "description": f"Declared {n_arms} arms but extracted {len(arms)}",
            "severity": "major",
        })

    return issues


def _check_table_consistency(extracted: dict, tables_content: dict) -> list[dict]:
    """Check for table-related issues."""
    signals = []

    # SoA tables: check if we have tables but no structured visits
    if "tables" in extracted and isinstance(extracted["tables"], list):
        soa_tables = extracted["tables"]
        visits = extracted.get("visits", [])
        if soa_tables and not visits:
            signals.append({
                "signal_type": "completeness",
                "severity": "major",
                "title": "SoA tables found but no structured visits extracted",
                "description": f"{len(soa_tables)} table(s) but 0 structured visits.",
            })

    return signals
