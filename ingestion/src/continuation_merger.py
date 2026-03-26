"""
Continuation Table Merger.

Detects tables that span multiple pages and merges them into single tables.

Merge algorithm:
1. Table extends to bottom of page (no non-table content after it)
2. Next page starts with same caption OR starts directly with table
3. Column count matches within ±1
4. If header row is repeated, skip it during merge
5. Footnotes from ALL pages collected into merged table
6. Confidence scored based on match quality
"""

from __future__ import annotations
import re
from difflib import SequenceMatcher

from .models import (
    ExtractedTable, TableCell, RowMetadata, RowType,
    Source, ExtractionMethod, TableFootnote,
)


def merge_continuation_tables(
    tables_by_page: dict[int, list[ExtractedTable]],
    page_heights: dict[int, float],
) -> tuple[list[ExtractedTable], int]:
    """
    Detect and merge continuation tables across pages.
    
    Returns:
        merged_tables: all tables (merged + non-continuation)
        merge_count: number of merges performed
    """
    # Flatten and sort by page
    all_tables: list[ExtractedTable] = []
    for page_num in sorted(tables_by_page.keys()):
        all_tables.extend(tables_by_page[page_num])
    
    if not all_tables:
        return [], 0
    
    merged: list[ExtractedTable] = []
    skip_indices: set[int] = set()
    merge_count = 0
    
    for i, table in enumerate(all_tables):
        if i in skip_indices:
            continue
        
        # Try to find continuation tables
        current = table
        j = i + 1
        
        while j < len(all_tables):
            next_table = all_tables[j]
            
            # Must be on the very next page
            current_last_page = max(current.page_range) if current.page_range else 0
            next_first_page = min(next_table.page_range) if next_table.page_range else 0
            
            if next_first_page != current_last_page + 1:
                break
            
            # Check if this is a continuation
            match_score = _continuation_match_score(current, next_table)
            
            if match_score >= 0.6:
                current = _merge_two_tables(current, next_table, match_score)
                skip_indices.add(j)
                merge_count += 1
                j += 1
            else:
                break
        
        merged.append(current)
    
    return merged, merge_count


def _continuation_match_score(
    table_a: ExtractedTable,
    table_b: ExtractedTable,
) -> float:
    """
    Score how likely table_b is a continuation of table_a.
    
    Factors:
    - Column count match: 0.4 weight
    - Caption similarity: 0.3 weight
    - Header row similarity: 0.3 weight
    """
    score = 0.0
    
    # Column count match
    if table_a.column_count > 0 and table_b.column_count > 0:
        diff = abs(table_a.column_count - table_b.column_count)
        if diff == 0:
            score += 0.4
        elif diff == 1:
            score += 0.2  # slight mismatch OK
    
    # Caption similarity
    if table_a.caption and table_b.caption:
        ratio = SequenceMatcher(
            None, table_a.caption.lower(), table_b.caption.lower()
        ).ratio()
        score += 0.3 * ratio
    elif not table_b.caption:
        # No caption on next page — could be continuation
        score += 0.15
    
    # Header row similarity
    if table_a.column_headers and table_b.rows:
        first_row_b = table_b.rows[0] if table_b.rows else []
        if first_row_b:
            # Check if first row of table_b matches headers of table_a
            header_match = _row_similarity(table_a.column_headers, first_row_b)
            score += 0.3 * header_match
    
    return score


def _row_similarity(row_a: list[str], row_b: list[str]) -> float:
    """Compare two rows for text similarity."""
    if not row_a or not row_b:
        return 0.0
    
    min_len = min(len(row_a), len(row_b))
    if min_len == 0:
        return 0.0
    
    matches = 0
    for i in range(min_len):
        a = row_a[i].strip().lower()
        b = row_b[i].strip().lower()
        if a == b:
            matches += 1
        elif a and b and SequenceMatcher(None, a, b).ratio() > 0.8:
            matches += 0.5
    
    return matches / min_len


def _merge_two_tables(
    table_a: ExtractedTable,
    table_b: ExtractedTable,
    match_score: float,
) -> ExtractedTable:
    """Merge table_b into table_a."""
    
    # Determine if first row of table_b is a repeated header
    skip_first_row = False
    if table_a.column_headers and table_b.rows:
        first_row_b = table_b.rows[0]
        header_sim = _row_similarity(table_a.column_headers, first_row_b)
        if header_sim > 0.7:
            skip_first_row = True
    
    # Merge rows
    new_rows = list(table_a.rows)
    start_idx = 1 if skip_first_row else 0
    
    for row in table_b.rows[start_idx:]:
        # Pad to match column count
        padded = list(row)
        while len(padded) < table_a.column_count:
            padded.append("")
        new_rows.append(padded[:table_a.column_count])
    
    # Merge cells (reindex rows)
    new_cells = list(table_a.cells)
    row_offset = len(table_a.rows)
    
    for cell in table_b.cells:
        actual_row = cell.row - (1 if skip_first_row else 0)
        if skip_first_row and cell.row == 0:
            continue  # skip header cells
        new_cells.append(TableCell(
            row=actual_row + row_offset - (1 if skip_first_row else 0),
            col=cell.col,
            text=cell.text,
            footnote_refs=cell.footnote_refs,
            source=cell.source,
        ))
    
    # Merge row metadata
    new_row_meta = list(table_a.row_metadata)
    for rm in table_b.row_metadata:
        if skip_first_row and rm.row_index == 0:
            continue
        new_row_meta.append(RowMetadata(
            row_index=rm.row_index + row_offset - (1 if skip_first_row else 0),
            row_type=rm.row_type,
            source_page=rm.source_page,
            group_label=rm.group_label,
        ))
    
    # Merge footnotes
    merged_footnotes = dict(table_a.footnotes)
    merged_footnotes.update(table_b.footnotes)
    
    merged_fn_sources = list(table_a.footnote_sources)
    merged_fn_sources.extend(table_b.footnote_sources)
    
    # Merge page ranges
    all_pages = sorted(set(table_a.page_range + table_b.page_range))
    
    return ExtractedTable(
        id=table_a.id,
        caption=table_a.caption or table_b.caption,
        caption_source=table_a.caption_source or table_b.caption_source,
        section_id=table_a.section_id,
        page_range=all_pages,
        is_continuation_merged=True,
        column_groups=table_a.column_groups,
        column_headers=table_a.column_headers,
        column_count=table_a.column_count,
        row_metadata=new_row_meta,
        rows=new_rows,
        cells=new_cells,
        footnotes=merged_footnotes,
        footnote_sources=merged_fn_sources,
        extraction_method=ExtractionMethod.MERGED,
        confidence=min(table_a.confidence, table_b.confidence) * match_score,
        source_hash=Source.hash_text(
            "|".join(c.text for c in new_cells if c.text)
        ),
    )
