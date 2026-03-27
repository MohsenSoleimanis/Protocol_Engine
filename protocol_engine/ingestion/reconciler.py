"""
Vision vs Text Table Reconciler — NEW module.

Old code had ZERO reconciliation — vision results were blindly trusted.

This module compares vision-extracted tables against text-extracted tables
and produces a high-confidence merged result.

Strategy:
  - Trust text extraction for numbers (vision hallucinates digits)
  - Trust vision for structure/layout (text misses borderless tables)
  - Flag conflicts for human review when >10% cells differ
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class CellConflict:
    """A conflict between vision and text extraction for a single cell."""
    row: int
    col: int
    text_value: str
    vision_value: str
    resolution: str = ""  # "text", "vision", or "unresolved"
    reason: str = ""


@dataclass
class ReconciliationResult:
    """Result of reconciling vision vs text table extraction."""
    merged_headers: list[str] = field(default_factory=list)
    merged_rows: list[list[str]] = field(default_factory=list)
    conflicts: list[CellConflict] = field(default_factory=list)
    total_cells: int = 0
    conflict_count: int = 0
    confidence: float = 1.0
    source: str = "text"  # which source was primarily used


def reconcile_tables(
    text_headers: list[str],
    text_rows: list[list[str]],
    vision_headers: list[str],
    vision_rows: list[list[str]],
) -> ReconciliationResult:
    """Reconcile vision vs text table extraction.

    Returns a merged table with conflicts flagged.
    """
    result = ReconciliationResult()

    # If one source is empty, use the other
    if not text_rows and not vision_rows:
        return result
    if not text_rows:
        result.merged_headers = vision_headers
        result.merged_rows = vision_rows
        result.source = "vision"
        result.confidence = 0.75  # vision-only is lower confidence
        return result
    if not vision_rows:
        result.merged_headers = text_headers
        result.merged_rows = text_rows
        result.source = "text"
        result.confidence = 0.90
        return result

    # Align columns
    col_mapping = _align_columns(text_headers, vision_headers)
    n_text_cols = len(text_headers)
    n_vision_cols = len(vision_headers)

    # Use text headers as base (more reliable for naming)
    result.merged_headers = list(text_headers)
    result.source = "merged"

    # Compare rows
    n_rows = min(len(text_rows), len(vision_rows))
    total_cells = 0
    conflicts = []

    for r_idx in range(n_rows):
        text_row = text_rows[r_idx] if r_idx < len(text_rows) else [""] * n_text_cols
        vision_row = vision_rows[r_idx] if r_idx < len(vision_rows) else [""] * n_vision_cols
        merged_row = list(text_row)  # start with text

        for t_col, v_col in col_mapping.items():
            if t_col >= len(text_row) or v_col >= len(vision_row):
                continue

            text_val = text_row[t_col].strip()
            vision_val = vision_row[v_col].strip()
            total_cells += 1

            if _values_match(text_val, vision_val):
                continue  # agreement — keep text value

            # Conflict — resolve
            conflict = CellConflict(
                row=r_idx, col=t_col,
                text_value=text_val, vision_value=vision_val,
            )

            if _is_numeric(text_val) or _is_numeric(vision_val):
                # Trust text for numbers
                conflict.resolution = "text"
                conflict.reason = "numeric — text extraction more reliable for digits"
            elif not text_val and vision_val:
                # Text missed it, vision found it — trust vision
                merged_row[t_col] = vision_val
                conflict.resolution = "vision"
                conflict.reason = "text extraction missed this cell"
            elif text_val and not vision_val:
                # Vision missed it — keep text
                conflict.resolution = "text"
                conflict.reason = "vision extraction missed this cell"
            else:
                # Both have different non-empty values
                # Use the longer/more detailed one
                if len(vision_val) > len(text_val) * 1.5:
                    merged_row[t_col] = vision_val
                    conflict.resolution = "vision"
                    conflict.reason = "vision has more detail"
                else:
                    conflict.resolution = "text"
                    conflict.reason = "text value preferred (default)"

            conflicts.append(conflict)

        result.merged_rows.append(merged_row)

    # Add remaining rows from whichever source has more
    if len(text_rows) > n_rows:
        result.merged_rows.extend(text_rows[n_rows:])
    elif len(vision_rows) > n_rows:
        for row in vision_rows[n_rows:]:
            # Remap vision columns to text column order
            mapped_row = [""] * n_text_cols
            for t_col, v_col in col_mapping.items():
                if v_col < len(row) and t_col < n_text_cols:
                    mapped_row[t_col] = row[v_col]
            result.merged_rows.append(mapped_row)

    result.total_cells = total_cells
    result.conflict_count = len(conflicts)
    result.conflicts = conflicts
    result.confidence = 1.0 - (len(conflicts) / max(total_cells, 1))

    if result.confidence < 0.9:
        logger.warning(
            f"Table reconciliation: {len(conflicts)}/{total_cells} cells conflict "
            f"(confidence: {result.confidence:.2f})"
        )

    return result


def _align_columns(
    text_headers: list[str], vision_headers: list[str],
) -> dict[int, int]:
    """Map text column indices to vision column indices by header similarity."""
    mapping: dict[int, int] = {}

    for t_idx, t_header in enumerate(text_headers):
        best_score = 0.0
        best_v_idx = t_idx  # default: same position

        for v_idx, v_header in enumerate(vision_headers):
            score = SequenceMatcher(
                None, t_header.lower().strip(), v_header.lower().strip()
            ).ratio()
            if score > best_score:
                best_score = score
                best_v_idx = v_idx

        if best_score >= 0.5:
            mapping[t_idx] = best_v_idx

    return mapping


def _values_match(a: str, b: str) -> bool:
    """Check if two cell values are equivalent."""
    a_norm = a.lower().strip()
    b_norm = b.lower().strip()

    if a_norm == b_norm:
        return True

    # Checkmark equivalents
    checks = {"x", "✓", "✔", "✕", "✗", "yes", "√"}
    if a_norm in checks and b_norm in checks:
        return True

    # Empty equivalents
    empties = {"", "-", "—", "n/a", "na", "none"}
    if a_norm in empties and b_norm in empties:
        return True

    # Fuzzy match for longer text
    if len(a_norm) > 3 and len(b_norm) > 3:
        ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
        if ratio > 0.85:
            return True

    return False


def _is_numeric(text: str) -> bool:
    """Check if text contains meaningful numbers."""
    cleaned = re.sub(r'[%$€£,\s]', '', text)
    try:
        float(cleaned)
        return True
    except ValueError:
        pass
    return bool(re.search(r'\d+\.?\d*', text))
