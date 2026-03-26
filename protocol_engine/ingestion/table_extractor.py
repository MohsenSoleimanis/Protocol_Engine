"""
Table Extractor — Multi-strategy extraction (ruled + borderless + sparse).

Key fixes from old code:
  1. Added "text" strategy for borderless/sparse tables (old code only used "lines")
  2. Cell bboxes use actual pdfplumber cell positions (old code fabricated uniform grids)
  3. Explicit table settings per page as fallback
  4. Continuation detection improved
"""
from __future__ import annotations

import re
import pdfplumber

from ingestion.src.models import (
    ExtractedTable, TableCell, RowMetadata, RowType,
    ColumnGroup, Source, ExtractionMethod,
)

# ── Table extraction strategies ──────────────────────────────────────────────

# Strategy 1: Ruled lines (original, works for bordered tables)
RULED_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_tolerance": 4,
    "join_tolerance": 4,
    "edge_min_length": 10,
    "min_words_vertical": 1,
    "min_words_horizontal": 1,
}

# Strategy 2: Text-based (NEW — for borderless/sparse tables)
TEXT_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "text",
    "snap_tolerance": 5,
    "join_tolerance": 5,
    "min_words_vertical": 2,
    "min_words_horizontal": 1,
}

# Strategy 3: Explicit (mixed — lines for horizontal, text for vertical)
EXPLICIT_SETTINGS = {
    "vertical_strategy": "text",
    "horizontal_strategy": "lines",
    "snap_tolerance": 4,
    "join_tolerance": 4,
    "edge_min_length": 10,
    "min_words_vertical": 2,
    "min_words_horizontal": 1,
}


def extract_tables_from_page(
    pdf_path: str,
    page_idx: int,
    page_num: int,
    strategies: list[str] | None = None,
) -> list[ExtractedTable]:
    """Extract tables using multiple strategies, picking the best result.

    Strategies tried in order:
      1. "lines" — ruled tables (fast, high confidence)
      2. "text" — borderless tables (catches sparse/borderless)
      3. "explicit" — mixed strategy

    Returns the result from the strategy that finds the most data.
    """
    if strategies is None:
        strategies = ["lines", "text"]

    settings_map = {
        "lines": RULED_SETTINGS,
        "text": TEXT_SETTINGS,
        "explicit": EXPLICIT_SETTINGS,
    }

    best_tables: list[ExtractedTable] = []
    best_cell_count = 0
    best_strategy = "none"

    with pdfplumber.open(pdf_path) as pdf:
        if page_idx >= len(pdf.pages):
            return []
        page = pdf.pages[page_idx]

        for strategy_name in strategies:
            settings = settings_map.get(strategy_name, RULED_SETTINGS)
            try:
                found = page.find_tables(settings)
            except Exception:
                continue

            tables = []
            total_cells = 0

            for t_idx, table in enumerate(found):
                raw_rows = table.extract()
                if not raw_rows or len(raw_rows) < 2:
                    continue

                column_count = max(len(row) for row in raw_rows) if raw_rows else 0
                cell_bboxes = _get_cell_bboxes(table)

                cells: list[TableCell] = []
                rows_data: list[list[str]] = []

                for r_idx, row in enumerate(raw_rows):
                    row_cells: list[str] = []
                    for c_idx, cell_text in enumerate(row):
                        text = (cell_text or "").strip()
                        text = re.sub(r'\s+', ' ', text)

                        cell_bbox = cell_bboxes.get((r_idx, c_idx))
                        source = Source(
                            page=page_num,
                            bbox=cell_bbox,
                            extraction_method=ExtractionMethod.PDFPLUMBER,
                            confidence=0.92 if cell_bbox else 0.75,
                        )
                        if text:
                            source.raw_text_hash = Source.hash_text(text)

                        footnote_refs = _detect_footnote_refs(text)
                        cells.append(TableCell(
                            row=r_idx, col=c_idx, text=text,
                            footnote_refs=footnote_refs, source=source,
                        ))
                        row_cells.append(text)

                    while len(row_cells) < column_count:
                        row_cells.append("")
                    rows_data.append(row_cells)

                total_cells += len(cells)
                row_metadata = _classify_rows(rows_data, page_num)
                column_groups = _detect_column_groups(rows_data, row_metadata)
                headers = []
                for rm in row_metadata:
                    if rm.row_type == RowType.HEADER:
                        headers = rows_data[rm.row_index]
                        break

                all_text = "|".join(c.text for c in cells if c.text)
                source_hash = Source.hash_text(all_text) if all_text else None

                tables.append(ExtractedTable(
                    id=f"table_p{page_num}_{t_idx + 1}",
                    page_range=[page_num],
                    column_groups=column_groups,
                    column_headers=headers,
                    column_count=column_count,
                    row_metadata=row_metadata,
                    rows=rows_data,
                    cells=cells,
                    extraction_method=ExtractionMethod.PDFPLUMBER,
                    confidence=0.92 if strategy_name == "lines" else 0.80,
                    source_hash=source_hash,
                ))

            # Keep the strategy that found the most content
            if total_cells > best_cell_count:
                best_tables = tables
                best_cell_count = total_cells
                best_strategy = strategy_name

    return best_tables


def _get_cell_bboxes(
    table,
) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    """Extract ACTUAL cell bounding boxes from pdfplumber.

    FIX: Old code fabricated uniform grid bboxes. This uses pdfplumber's
    actual cell positions when available.
    """
    bboxes: dict[tuple[int, int], tuple[float, float, float, float]] = {}

    # pdfplumber's table.cells gives us (x0, y0, x1, y1) for each cell
    # but we need to map them to (row, col) indices
    if hasattr(table, 'cells') and table.cells:
        rows = table.extract()
        if not rows:
            return bboxes

        n_rows = len(rows)
        n_cols = max(len(row) for row in rows) if rows else 0

        # pdfplumber cells are ordered left-to-right, top-to-bottom
        # Sort by y then x to map to row/col
        sorted_cells = sorted(table.cells, key=lambda c: (round(c[1], 1), c[0]))

        # Group by y-proximity for row detection
        row_groups: list[list[tuple]] = []
        current_group: list[tuple] = []
        prev_y = None

        for cell in sorted_cells:
            y = cell[1]
            if prev_y is not None and abs(y - prev_y) > 5:
                if current_group:
                    row_groups.append(current_group)
                current_group = []
            current_group.append(cell)
            prev_y = y

        if current_group:
            row_groups.append(current_group)

        for r_idx, group in enumerate(row_groups):
            if r_idx >= n_rows:
                break
            # Sort cells in this row by x
            group.sort(key=lambda c: c[0])
            for c_idx, cell in enumerate(group):
                if c_idx >= n_cols:
                    break
                bboxes[(r_idx, c_idx)] = (
                    round(cell[0], 2),
                    round(cell[1], 2),
                    round(cell[2], 2),
                    round(cell[3], 2),
                )

    return bboxes


def _detect_footnote_refs(text: str) -> list[str]:
    """Detect footnote markers in cell text."""
    refs: list[str] = []
    if not text:
        return refs
    m = re.findall(r'\s+([a-z*\u2020\u2021\u00a7\u00b6#])\s*$', text)
    refs.extend(m)
    m2 = re.search(r'\s+([a-z](?:\s*,\s*[a-z])*)\s*$', text)
    if m2 and len(m2.group(1)) <= 5:
        parts = [p.strip() for p in m2.group(1).split(",")]
        if all(len(p) == 1 for p in parts):
            refs = parts
    return refs


def _classify_rows(rows: list[list[str]], page_num: int) -> list[RowMetadata]:
    """Classify each row as header, data, group_header, window, or separator."""
    metadata: list[RowMetadata] = []
    if not rows:
        return metadata

    for r_idx, row in enumerate(rows):
        non_empty = sum(1 for c in row if c.strip())
        total = len(row)
        row_type = RowType.DATA

        if r_idx == 0:
            row_type = RowType.HEADER
        elif r_idx == 1 and total > 0:
            has_window = any(
                re.search(r'[\u00b1\u2264\u2265<>]\s*\d', c) or c.strip().upper() == "NA"
                for c in row if c.strip()
            )
            if has_window and non_empty > total * 0.3:
                row_type = RowType.WINDOW

        if non_empty <= 2 and total > 3 and row[0].strip():
            rest_empty = all(not c.strip() for c in row[1:])
            if rest_empty and r_idx > 0:
                row_type = RowType.GROUP_HEADER

        if non_empty == 0:
            row_type = RowType.SEPARATOR

        metadata.append(RowMetadata(
            row_index=r_idx, row_type=row_type, source_page=page_num,
        ))

    current_group = None
    for rm in metadata:
        if rm.row_type == RowType.GROUP_HEADER:
            current_group = rows[rm.row_index][0].strip()
        elif rm.row_type == RowType.DATA and current_group:
            rm.group_label = current_group

    return metadata


def _detect_column_groups(
    rows: list[list[str]], row_metadata: list[RowMetadata],
) -> list[ColumnGroup]:
    """Detect multi-level column headers (column groups)."""
    groups: list[ColumnGroup] = []
    if len(rows) < 2:
        return groups
    first_row = rows[0]
    n_cols = len(first_row)
    if n_cols < 3:
        return groups

    current_label = None
    current_cols: list[int] = []
    for c_idx, cell in enumerate(first_row):
        text = cell.strip()
        if text == current_label:
            current_cols.append(c_idx)
        else:
            if current_label and len(current_cols) > 1:
                groups.append(ColumnGroup(label=current_label, column_indices=current_cols))
            current_label = text
            current_cols = [c_idx]
    if current_label and len(current_cols) > 1:
        groups.append(ColumnGroup(label=current_label, column_indices=current_cols))
    return groups
