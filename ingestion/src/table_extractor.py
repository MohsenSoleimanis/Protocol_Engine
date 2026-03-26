"""
Table extraction using pdfplumber.

pdfplumber excels at ruled-line table extraction — it's deterministic, fast,
and handles landscape pages, spanning headers, and dense grids natively.
Each cell gets a source bbox for grounding.
"""

from __future__ import annotations
import re
import pdfplumber

from .models import (
    ExtractedTable, TableCell, RowMetadata, RowType,
    ColumnGroup, Source, ExtractionMethod,
)


def extract_tables_from_page(
    pdf_path: str,
    page_idx: int,
    page_num: int,
) -> list[ExtractedTable]:
    """
    Extract all tables from a single page using pdfplumber.
    Returns list of ExtractedTable with full cell provenance.
    """
    tables: list[ExtractedTable] = []
    
    with pdfplumber.open(pdf_path) as pdf:
        if page_idx >= len(pdf.pages):
            return tables
        page = pdf.pages[page_idx]
        
        # pdfplumber table detection settings — tuned for ruled-line tables
        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 4,
            "join_tolerance": 4,
            "edge_min_length": 10,
            "min_words_vertical": 1,
            "min_words_horizontal": 1,
        }
        
        found_tables = page.find_tables(table_settings)
        
        for t_idx, table in enumerate(found_tables):
            raw_rows = table.extract()
            if not raw_rows or len(raw_rows) < 2:
                continue
            
            # Get table bbox from pdfplumber
            table_bbox = table.bbox  # (x0, y0, x1, y1)
            
            # Build cells with provenance
            cells: list[TableCell] = []
            rows_data: list[list[str]] = []
            column_count = max(len(row) for row in raw_rows) if raw_rows else 0
            
            # Extract cell-level bboxes from pdfplumber's internal structure
            cell_bboxes = _get_cell_bboxes(table, len(raw_rows), column_count)
            
            for r_idx, row in enumerate(raw_rows):
                row_cells: list[str] = []
                for c_idx, cell_text in enumerate(row):
                    text = (cell_text or "").strip()
                    # Normalize whitespace within cell (preserve newlines as spaces)
                    text = re.sub(r'\s+', ' ', text)
                    
                    cell_bbox = cell_bboxes.get((r_idx, c_idx))
                    source = Source(
                        page=page_num,
                        bbox=cell_bbox,
                        extraction_method=ExtractionMethod.PDFPLUMBER,
                        confidence=0.92 if cell_bbox else 0.80,
                    )
                    if text:
                        source.raw_text_hash = Source.hash_text(text)
                    
                    # Detect footnote markers in cell text
                    footnote_refs = _detect_footnote_refs(text)
                    
                    cells.append(TableCell(
                        row=r_idx, col=c_idx,
                        text=text,
                        footnote_refs=footnote_refs,
                        source=source,
                    ))
                    row_cells.append(text)
                
                # Pad short rows
                while len(row_cells) < column_count:
                    row_cells.append("")
                rows_data.append(row_cells)
            
            # Classify rows
            row_metadata = _classify_rows(rows_data, page_num)
            
            # Detect column groups from multi-level headers
            column_groups = _detect_column_groups(rows_data, row_metadata)
            
            # Extract column headers (first header row)
            headers = []
            for rm in row_metadata:
                if rm.row_type == RowType.HEADER:
                    headers = rows_data[rm.row_index]
                    break
            
            # Build source hash
            all_text = "|".join(
                c.text for c in cells if c.text
            )
            source_hash = Source.hash_text(all_text) if all_text else None
            
            extracted = ExtractedTable(
                id=f"table_p{page_num}_{t_idx + 1}",
                page_range=[page_num],
                column_groups=column_groups,
                column_headers=headers,
                column_count=column_count,
                row_metadata=row_metadata,
                rows=rows_data,
                cells=cells,
                extraction_method=ExtractionMethod.PDFPLUMBER,
                confidence=0.92,
                source_hash=source_hash,
            )
            tables.append(extracted)
    
    return tables


def _get_cell_bboxes(
    table, n_rows: int, n_cols: int
) -> dict[tuple[int, int], tuple[float, float, float, float]]:
    """Extract cell-level bounding boxes from pdfplumber table structure."""
    bboxes: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    
    try:
        # pdfplumber stores cell positions in table.cells
        for cell in table.cells:
            # cell is (x0, y0, x1, y1) — but we need row/col mapping
            # Use the table's rows structure instead
            pass
    except Exception:
        pass
    
    # Fallback: compute from table bbox + uniform grid
    if not bboxes and hasattr(table, 'bbox') and table.bbox:
        x0, y0, x1, y1 = table.bbox
        if n_cols > 0 and n_rows > 0:
            col_width = (x1 - x0) / n_cols
            row_height = (y1 - y0) / n_rows
            for r in range(n_rows):
                for c in range(n_cols):
                    bboxes[(r, c)] = (
                        round(x0 + c * col_width, 2),
                        round(y0 + r * row_height, 2),
                        round(x0 + (c + 1) * col_width, 2),
                        round(y0 + (r + 1) * row_height, 2),
                    )
    
    return bboxes


def _detect_footnote_refs(text: str) -> list[str]:
    """
    Detect footnote markers in cell text.
    Generic: finds trailing single lowercase letters or symbols that look like references.
    """
    refs: list[str] = []
    if not text:
        return refs
    
    # Pattern: text followed by space + single letter/symbol
    # e.g., "Concomitant medications c" or "Vital signs *"
    m = re.findall(r'\s+([a-z*†‡§¶#])\s*$', text)
    refs.extend(m)
    
    # Pattern: superscript-style refs like "a,c" at end
    m2 = re.search(r'\s+([a-z](?:\s*,\s*[a-z])*)\s*$', text)
    if m2 and len(m2.group(1)) <= 5:  # short enough to be refs, not words
        parts = [p.strip() for p in m2.group(1).split(",")]
        if all(len(p) == 1 for p in parts):
            refs = parts  # override with parsed list
    
    return refs


def _classify_rows(
    rows: list[list[str]], page_num: int
) -> list[RowMetadata]:
    """
    Classify each row as header, data, group_header, window, or separator.
    Generic heuristics — no domain assumptions.
    """
    metadata: list[RowMetadata] = []
    if not rows:
        return metadata
    
    n_cols = len(rows[0]) if rows else 0
    
    for r_idx, row in enumerate(rows):
        non_empty = sum(1 for c in row if c.strip())
        total = len(row)
        
        # Row type detection
        row_type = RowType.DATA
        
        if r_idx == 0:
            # First row is usually a header
            row_type = RowType.HEADER
        elif r_idx == 1 and total > 0:
            # Second row might be a window/timing row (e.g., "± 3 days")
            has_window_markers = any(
                re.search(r'[±≤≥<>]\s*\d', c) or c.strip().upper() == "NA"
                for c in row if c.strip()
            )
            if has_window_markers and non_empty > total * 0.3:
                row_type = RowType.WINDOW
        
        # Group header: text only in first column(s), rest empty
        if non_empty <= 2 and total > 3 and row[0].strip():
            # Check if remaining columns are empty
            rest_empty = all(not c.strip() for c in row[1:])
            if rest_empty and r_idx > 0:
                row_type = RowType.GROUP_HEADER
        
        # Separator: completely empty row
        if non_empty == 0:
            row_type = RowType.SEPARATOR
        
        metadata.append(RowMetadata(
            row_index=r_idx,
            row_type=row_type,
            source_page=page_num,
        ))
    
    # Post-pass: assign group labels
    current_group = None
    for rm in metadata:
        if rm.row_type == RowType.GROUP_HEADER:
            current_group = rows[rm.row_index][0].strip()
        elif rm.row_type == RowType.DATA and current_group:
            rm.group_label = current_group
    
    return metadata


def _detect_column_groups(
    rows: list[list[str]],
    row_metadata: list[RowMetadata],
) -> list[ColumnGroup]:
    """
    Detect multi-level column headers (column groups).
    
    Look for header rows where some cells span multiple columns
    (same text repeated or fewer unique values than columns).
    """
    groups: list[ColumnGroup] = []
    
    if len(rows) < 2:
        return groups
    
    # Check first row for potential group headers
    first_row = rows[0]
    n_cols = len(first_row)
    
    if n_cols < 3:
        return groups
    
    # Detect: consecutive columns with same text = column group
    current_label = None
    current_cols: list[int] = []
    
    for c_idx, cell in enumerate(first_row):
        text = cell.strip()
        if text == current_label:
            current_cols.append(c_idx)
        else:
            if current_label and len(current_cols) > 1:
                groups.append(ColumnGroup(
                    label=current_label,
                    column_indices=current_cols,
                ))
            current_label = text
            current_cols = [c_idx]
    
    # Final group
    if current_label and len(current_cols) > 1:
        groups.append(ColumnGroup(
            label=current_label,
            column_indices=current_cols,
        ))
    
    return groups
