"""
Generic Document Structure Discovery Engine.

Discovers ALL document structure from the PDF itself:
- Section hierarchy (from font sizes + numbering patterns)
- Table captions (from proximity + font weight above tables)
- Footnote blocks (from font size drop below tables/regions)
- Abbreviation definitions (from inline "full text (ABBR)" patterns)
- Cross-references (from "see Section X", "Table Y" patterns)
- Structured lists (from numbered/lettered/bulleted patterns)
- Figures (from image regions + nearby captions)

NO hardcoded domain patterns. Everything is discovered generically.
"""

from __future__ import annotations
import re
from collections import Counter, defaultdict

from .models import (
    TextSpan, Section, ContentBlock, ContentBlockType, ListItem,
    CrossReference, InlineFormat, Figure, Source, ExtractionMethod,
    ExtractedTable, TableFootnote, ReferenceEntry, PageResult,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SECTION HIERARCHY DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

# Generic numbering schemes to try
SECTION_PATTERNS = [
    # "1.2.3.4" dotted numeric (most common in clinical protocols)
    # Also handles trailing dot: "5.2. Title" (Moderna format)
    re.compile(r'^(\d+(?:\.\d+)+)\.?\s+(.+)$'),    # "1.1 Title" or "1.1. Title"
    re.compile(r'^(\d+)\.?\s+([A-Z][A-Z ].{3,})$'), # top-level: "1 INTRODUCTION" or "1. INTRODUCTION"
    # "A.1.2" letter-dotted (appendices) — require dot to avoid "A word..." false positives
    re.compile(r'^(Appendix\s+[A-Z])\s+(.+)$', re.IGNORECASE),
    re.compile(r'^([A-Z]\.\d+(?:\.\d+)*)\.?\s+(.+)$'),
    # "IV.b" roman
    re.compile(r'^([IVXLC]+(?:\.[a-z0-9]+)+)\s+(.+)$'),
]

SECTION_SCHEME_MAP = {
    0: "numeric_dotted",
    1: "numeric_dotted",
    2: "appendix",
    3: "letter_dotted",
    4: "roman",
}


def discover_section_hierarchy(
    page_spans: dict[int, list[TextSpan]],
    body_font_size: float,
) -> tuple[list[Section], str]:
    """
    Discover sections from font analysis + numbering patterns.
    
    Combines adjacent bold/large spans on the same line before matching,
    since PDF section headings often split "5.1" and "Inclusion Criteria"
    into separate spans.
    """
    # Phase 1: Find heading LINES (combine same-y bold/large spans)
    heading_lines: list[tuple[str, list[TextSpan], int]] = []  # (combined_text, spans, page)
    
    for page_num, spans in sorted(page_spans.items()):
        # Filter to bold/large spans
        heading_spans: list[TextSpan] = []
        for span in spans:
            if span.font_size >= body_font_size + 0.5 or span.is_bold:
                if 1 <= len(span.text.strip()) < 200:
                    heading_spans.append(span)
        
        if not heading_spans:
            continue
        
        # Group by y-position (same line = within 3pt)
        heading_spans.sort(key=lambda s: (round(s.y0, 1), s.x0))
        current_line: list[TextSpan] = [heading_spans[0]]
        
        for span in heading_spans[1:]:
            if abs(span.y0 - current_line[0].y0) < 3:
                current_line.append(span)
            else:
                combined = " ".join(s.text.strip() for s in current_line if s.text.strip())
                if len(combined) > 2:
                    heading_lines.append((combined, list(current_line), page_num))
                current_line = [span]
        
        if current_line:
            combined = " ".join(s.text.strip() for s in current_line if s.text.strip())
            if len(combined) > 2:
                heading_lines.append((combined, list(current_line), page_num))
    
    # Phase 2: Detect which numbering pattern matches most candidates
    best_pattern = None
    best_matches = 0
    best_scheme = "unknown"
    
    for i, pat in enumerate(SECTION_PATTERNS):
        matches = sum(1 for text, _, _ in heading_lines if pat.match(text.strip()))
        if matches > best_matches:
            best_matches = matches
            best_pattern = pat
            best_scheme = SECTION_SCHEME_MAP.get(i, "unknown")
    
    if not best_pattern or best_matches < 2:
        return [], best_scheme
    
    # Phase 3: Build section tree from ALL matching patterns
    sections: list[Section] = []
    section_by_number: dict[str, Section] = {}
    seen_pages_numbers: set[tuple[int, str]] = set()
    
    for pat_idx, pat in enumerate(SECTION_PATTERNS):
        for text, spans_list, page_num in heading_lines:
            m = pat.match(text.strip())
            if not m:
                continue
            
            number = m.group(1)
            title = m.group(2).strip()
            
            if len(title) < 2:
                continue
            
            key = (page_num, number)
            if key in seen_pages_numbers:
                continue
            seen_pages_numbers.add(key)
            
            scheme = SECTION_SCHEME_MAP.get(pat_idx, "unknown")
            if scheme in ("numeric_dotted", "letter_dotted"):
                level = number.count('.')
            elif scheme == "appendix":
                level = 0
            else:
                level = 0
            
            parent_id = None
            if '.' in number:
                parent_number = number.rsplit('.', 1)[0]
                if parent_number in section_by_number:
                    parent_id = section_by_number[parent_number].id
            
            first_span = spans_list[0]
            last_span = spans_list[-1]
            sec_id = f"sec_{number.replace('.', '_').replace(' ', '_')}"
            section = Section(
                id=sec_id,
                number=number,
                title=title,
                level=level,
                parent_id=parent_id,
                page_range=[page_num],
                source=Source(
                    page=page_num,
                    bbox=(first_span.x0, first_span.y0, last_span.x1, last_span.y1),
                    extraction_method=ExtractionMethod.PYMUPDF,
                    confidence=0.90,
                    raw_text_hash=Source.hash_text(text.strip()),
                ),
            )
            sections.append(section)
            section_by_number[number] = section
    
    # Phase 4: Deduplicate — if same number appears on multiple pages,
    # keep the one from the body (higher page), discard TOC entries.
    # TOC entries typically have trailing dots: "Synopsis ..."
    by_number: dict[str, list[Section]] = {}
    for sec in sections:
        by_number.setdefault(sec.number, []).append(sec)
    
    deduped: list[Section] = []
    for number, entries in by_number.items():
        if len(entries) == 1:
            deduped.append(entries[0])
        else:
            # Prefer entry without trailing dots (body section)
            body_entries = [e for e in entries if '...' not in e.title]
            if body_entries:
                # Pick highest page number (body, not TOC)
                best = max(body_entries, key=lambda e: e.page_range[0] if e.page_range else 0)
            else:
                best = max(entries, key=lambda e: e.page_range[0] if e.page_range else 0)
            deduped.append(best)
    
    # Clean titles: strip trailing dots
    for sec in deduped:
        sec.title = re.sub(r'\s*\.{3,}\s*\d*$', '', sec.title).strip()
    
    # Re-sort by page, then number
    deduped.sort(key=lambda s: (s.page_range[0] if s.page_range else 0, s.number))
    
    return deduped, best_scheme


def detect_body_font_size(all_spans: list[TextSpan]) -> float:
    """Detect the most common (body) font size from all spans."""
    sizes = [round(s.font_size, 1) for s in all_spans if s.font_size > 4]
    if not sizes:
        return 10.0
    counter = Counter(sizes)
    return counter.most_common(1)[0][0]


def assign_blocks_to_sections(
    sections: list[Section],
    page_results: list[PageResult],
) -> int:
    """Assign each content block to its owning section based on position.

    When two sections share a page (e.g., §5.2 ends mid-page, §5.3 starts
    mid-page), content blocks are split by the heading's y-position.
    Without this, the store assigns entire pages to sections and loses
    content that continues from a previous section onto a shared page.

    Returns: number of blocks assigned.
    """
    if not sections:
        return 0

    # Clear any existing assignments
    for sec in sections:
        sec.content_blocks = []

    # Build section heading positions: (page, y0, section)
    # y0 = top of heading bbox (PyMuPDF: y increases downward)
    sec_starts: list[tuple[int, float, Section]] = []
    for sec in sections:
        if sec.source and sec.source.page is not None:
            y0 = sec.source.bbox[1] if sec.source.bbox else 0.0
            sec_starts.append((sec.source.page, y0, sec))
    sec_starts.sort(key=lambda x: (x[0], x[1]))

    if not sec_starts:
        return 0

    # Index: which section headings are on each page
    headings_on_page: dict[int, list[tuple[float, Section]]] = defaultdict(list)
    for page, y0, sec in sec_starts:
        headings_on_page[page].append((y0, sec))

    # Process pages in order, tracking "current section" (carry-over)
    current_section: Section | None = None
    total_assigned = 0

    for page_result in sorted(page_results, key=lambda p: p.page_num):
        pn = page_result.page_num
        page_headings = sorted(headings_on_page.get(pn, []), key=lambda x: x[0])

        for block in page_result.content_blocks:
            # Get block's y-position
            block_y = 0.0
            if block.source and block.source.bbox:
                block_y = block.source.bbox[1]

            # Find the owning section:
            # The last heading on this page whose y0 <= block_y + tolerance
            # (block starts at or below the heading → belongs to that section)
            owner = current_section
            for heading_y, sec in page_headings:
                if heading_y <= block_y + 5:  # 5pt tolerance for same-line
                    owner = sec
                else:
                    break  # headings are sorted, rest are below this block

            if owner:
                owner.content_blocks.append(block)
                total_assigned += 1

        # Update current_section to the last heading on this page
        if page_headings:
            current_section = page_headings[-1][1]

    # Update page_range for each section based on actual block positions
    for sec in sections:
        if sec.content_blocks:
            block_pages = set()
            for b in sec.content_blocks:
                if b.source and b.source.page is not None:
                    block_pages.add(b.source.page)
            if block_pages:
                sec.page_range = sorted(block_pages)

    return total_assigned


# ═══════════════════════════════════════════════════════════════════════════════
# 2. TABLE CAPTION DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def discover_table_captions(
    page_spans: dict[int, list[TextSpan]],
    tables_by_page: dict[int, list[ExtractedTable]],
) -> dict[str, tuple[str, Source]]:
    """
    Find captions near the top of each table.
    
    Generic detection: bold/larger text within ~30pt above a table's top edge.
    Discovers the caption pattern from the document (e.g., "Table N" prefix).
    
    Returns: {table_id: (caption_text, source)}
    """
    captions: dict[str, tuple[str, Source]] = {}
    
    for page_num, tables in tables_by_page.items():
        spans = page_spans.get(page_num, [])
        
        for table in tables:
            if not table.rows:
                continue
            
            # Get table top edge from first cell bbox
            table_top = None
            for cell in table.cells:
                if cell.source and cell.source.bbox:
                    y = cell.source.bbox[1]
                    if table_top is None or y < table_top:
                        table_top = y
            
            if table_top is None:
                continue
            
            # Find bold/larger spans within 40pt above table
            caption_candidates: list[TextSpan] = []
            for span in spans:
                if span.y1 < table_top and span.y1 > table_top - 40:
                    if span.is_bold or span.font_size > 9:
                        caption_candidates.append(span)
            
            if caption_candidates:
                # Sort by y position, combine into caption text
                caption_candidates.sort(key=lambda s: (s.y0, s.x0))
                caption_text = " ".join(s.text for s in caption_candidates).strip()
                
                if len(caption_text) > 5:
                    first = caption_candidates[0]
                    last = caption_candidates[-1]
                    source = Source(
                        page=page_num,
                        bbox=(first.x0, first.y0, last.x1, last.y1),
                        extraction_method=ExtractionMethod.PYMUPDF,
                        confidence=0.85,
                        raw_text_hash=Source.hash_text(caption_text),
                    )
                    captions[table.id] = (caption_text, source)
    
    return captions


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FOOTNOTE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def discover_footnotes_for_table(
    spans: list[TextSpan],
    table: ExtractedTable,
    body_font_size: float,
    page_num: int,
) -> dict[str, TableFootnote]:
    """
    Find footnotes below a table.
    
    Three detection strategies:
    1. Small superscript marker (a, b, c) followed by body-sized text
    2. Lines starting with marker patterns: "a. text" or "a text"  
    3. Abbreviation lines: "AE = adverse event; ..."
    """
    footnotes: dict[str, TableFootnote] = {}
    
    # Find table bottom
    table_bottom = 0
    for cell in table.cells:
        if cell.source and cell.source.bbox:
            y = cell.source.bbox[3]
            if y > table_bottom:
                table_bottom = y
    
    if table_bottom == 0:
        return footnotes
    
    # Collect ALL spans below the table bottom (with small tolerance)
    below_spans: list[TextSpan] = []
    for span in spans:
        if span.y0 > table_bottom - 5:
            below_spans.append(span)
    
    if not below_spans:
        return footnotes
    
    # Sort into lines
    below_spans.sort(key=lambda s: (round(s.y0, 1), s.x0))
    lines: list[tuple[list[TextSpan], str]] = []
    current_line: list[TextSpan] = []
    prev_y = None
    
    for span in below_spans:
        if prev_y is not None and abs(span.y0 - prev_y) > 5:
            if current_line:
                text = " ".join(s.text for s in current_line)
                lines.append((current_line, text))
            current_line = []
        current_line.append(span)
        prev_y = span.y0
    if current_line:
        text = " ".join(s.text for s in current_line)
        lines.append((current_line, text))
    
    # Strategy 1: Small marker span (superscript) + body text
    # Pattern: first span is small (< body - 2pt) and single char, rest is body-sized
    for line_spans, line_text in lines:
        if len(line_spans) < 2:
            continue
        first = line_spans[0]
        if (first.font_size < body_font_size - 1.5
                and len(first.text.strip()) == 1
                and first.text.strip() in 'abcdefghijklmnopqrstuvwxyz*†‡§¶#'):
            marker = first.text.strip().lower()
            text = " ".join(s.text for s in line_spans[1:]).strip()
            if text and len(text) > 5:
                source = Source(
                    page=page_num,
                    bbox=(first.x0, first.y0, line_spans[-1].x1, line_spans[-1].y1),
                    extraction_method=ExtractionMethod.PYMUPDF,
                    confidence=0.85,
                    raw_text_hash=Source.hash_text(line_text),
                )
                footnotes[marker] = TableFootnote(
                    marker=marker, text=text, source=source,
                )
    
    # Strategy 2: Lines starting with "a." or "a)" or "1." pattern
    fn_pattern = re.compile(
        r'^([a-z*†‡§¶#]|\d+)[.\s:)]\s*(.+)$', re.IGNORECASE
    )
    for line_spans, line_text in lines:
        stripped = line_text.strip()
        m = fn_pattern.match(stripped)
        if m:
            marker = m.group(1).lower()
            text = m.group(2).strip()
            if marker not in footnotes and text and len(text) > 5:
                first_span = line_spans[0]
                last_span = line_spans[-1]
                source = Source(
                    page=page_num,
                    bbox=(first_span.x0, first_span.y0, last_span.x1, last_span.y1),
                    extraction_method=ExtractionMethod.PYMUPDF,
                    confidence=0.80,
                    raw_text_hash=Source.hash_text(stripped),
                )
                footnotes[marker] = TableFootnote(
                    marker=marker, text=text, source=source,
                )
    
    return footnotes


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ABBREVIATION DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

# Pattern: "full text (ABBR)" where ABBR is 2-8 uppercase letters
ABBR_INLINE_RE = re.compile(
    r'([A-Z][a-z]+(?:\s+[a-z]+)*(?:\s+[A-Z][a-z]+)*)\s+\(([A-Z]{2,8}s?)\)'
)


def discover_abbreviations(
    page_spans: dict[int, list[TextSpan]],
    tables: list[ExtractedTable],
) -> dict[str, str]:
    """
    Discover abbreviation definitions from:
    1. Inline definitions: "medically attended adverse events (MAAEs)"
    2. Abbreviation tables: two-column tables with short uppercase + descriptive text
    
    Returns: {ABBR: "full definition"}
    """
    abbrevs: dict[str, str] = {}
    
    # Method 1: Inline definitions in body text
    for page_num, spans in page_spans.items():
        full_text = " ".join(s.text for s in sorted(spans, key=lambda s: (s.y0, s.x0)))
        for m in ABBR_INLINE_RE.finditer(full_text):
            full_form = m.group(1).strip()
            abbr = m.group(2).strip().rstrip('s')  # remove plural 's'
            if len(abbr) >= 2 and len(full_form) > len(abbr):
                abbrevs[abbr] = full_form
    
    # Method 2: Abbreviation tables (2-column: short uppercase | descriptive)
    for table in tables:
        if table.column_count != 2:
            continue
        abbr_like_rows = 0
        for row in table.rows:
            if len(row) >= 2:
                col0 = row[0].strip()
                col1 = row[1].strip()
                if (col0.isupper() and 2 <= len(col0) <= 10
                        and len(col1) > len(col0)):
                    abbr_like_rows += 1
        
        # If >50% of rows look like abbreviation definitions
        if len(table.rows) > 3 and abbr_like_rows > len(table.rows) * 0.5:
            for row in table.rows:
                if len(row) >= 2:
                    abbr = row[0].strip()
                    defn = row[1].strip()
                    if abbr and defn and abbr.isupper():
                        abbrevs[abbr] = defn
    
    return abbrevs


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CROSS-REFERENCE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

# Generic cross-reference patterns (discovered, not hardcoded)
XREF_PATTERNS = [
    re.compile(r'(?:see\s+|See\s+|refer\s+to\s+)?Section\s+([\d.]+)', re.IGNORECASE),
    re.compile(r'(?:see\s+|See\s+)?Table\s+(\d+)', re.IGNORECASE),
    re.compile(r'(?:see\s+|See\s+)?Figure\s+(\d+)', re.IGNORECASE),
    re.compile(r'(?:see\s+|See\s+)?Appendix\s+([A-Z])', re.IGNORECASE),
]


def discover_cross_references(text: str) -> list[CrossReference]:
    """Find all cross-references in a text string."""
    refs: list[CrossReference] = []
    
    type_map = {
        0: "section",
        1: "table",
        2: "figure",
        3: "appendix",
    }
    
    for i, pattern in enumerate(XREF_PATTERNS):
        for m in pattern.finditer(text):
            target_type = type_map[i]
            target_id = m.group(1)
            
            # Build canonical target_id
            if target_type == "section":
                target_id = f"Section {target_id}"
            elif target_type == "table":
                target_id = f"Table {target_id}"
            elif target_type == "figure":
                target_id = f"Figure {target_id}"
            elif target_type == "appendix":
                target_id = f"Appendix {target_id}"
            
            refs.append(CrossReference(
                text=m.group(0),
                target_type=target_type,
                target_id=target_id,
                char_start=m.start(),
                char_end=m.end(),
            ))
    
    return refs


def build_reference_index(
    sections: list[Section],
    tables: list[ExtractedTable],
    figures: list[Figure],
) -> dict[str, ReferenceEntry]:
    """Build a lookup index for all referenceable entities."""
    index: dict[str, ReferenceEntry] = {}
    
    for sec in sections:
        key = f"Section {sec.number}"
        index[key] = ReferenceEntry(
            target_type="section",
            target_id=key,
            title=sec.title,
            page=sec.page_range[0] if sec.page_range else None,
        )
    
    for table in tables:
        # Try to extract table number from caption
        if table.caption:
            m = re.match(r'Table\s+(\d+)', table.caption, re.IGNORECASE)
            if m:
                key = f"Table {m.group(1)}"
                index[key] = ReferenceEntry(
                    target_type="table",
                    target_id=key,
                    title=table.caption,
                    page=table.page_range[0] if table.page_range else None,
                )
    
    for fig in figures:
        if fig.caption:
            m = re.match(r'Figure\s+(\d+)', fig.caption, re.IGNORECASE)
            if m:
                key = f"Figure {m.group(1)}"
                index[key] = ReferenceEntry(
                    target_type="figure",
                    target_id=key,
                    title=fig.caption,
                    page=fig.page,
                )
    
    return index


# ═══════════════════════════════════════════════════════════════════════════════
# 6. STRUCTURED LIST DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

LIST_MARKER_RE = re.compile(
    r'^(\d+[.)]\s+|[a-z][.)]\s+|[•\-–—]\s+|[ivxlc]+[.)]\s+)(.*)',
    re.IGNORECASE
)


def discover_lists_in_spans(
    spans: list[TextSpan],
    body_font_size: float,
    page_num: int,
) -> list[ContentBlock]:
    """
    Detect structured lists in body text.
    
    Groups consecutive spans that match list marker patterns.
    """
    # Sort spans into lines
    sorted_spans = sorted(spans, key=lambda s: (round(s.y0, 1), s.x0))
    
    lines: list[tuple[str, float, TextSpan]] = []  # (text, x0, first_span)
    current_text_parts: list[str] = []
    current_x0 = 0.0
    current_span = None
    prev_y = None
    
    for span in sorted_spans:
        if prev_y is not None and abs(span.y0 - prev_y) > 3:
            if current_text_parts:
                lines.append((" ".join(current_text_parts), current_x0, current_span))
            current_text_parts = [span.text]
            current_x0 = span.x0
            current_span = span
        else:
            current_text_parts.append(span.text)
            if current_span is None:
                current_x0 = span.x0
                current_span = span
        prev_y = span.y0
    
    if current_text_parts:
        lines.append((" ".join(current_text_parts), current_x0, current_span))
    
    # Find consecutive list items
    list_blocks: list[ContentBlock] = []
    current_list_items: list[ListItem] = []
    list_start_span = None
    list_end_span = None
    base_indent = None
    
    for text, x0, first_span in lines:
        m = LIST_MARKER_RE.match(text.strip())
        if m:
            marker = m.group(1).strip()
            content = m.group(2).strip()
            
            # Determine indent level
            level = 0
            if base_indent is None:
                base_indent = x0
            elif x0 > base_indent + 15:
                level = 1
            elif x0 > base_indent + 30:
                level = 2
            
            item = ListItem(
                marker=marker,
                text=content,
                level=level,
                source=Source(
                    page=page_num,
                    bbox=(first_span.x0, first_span.y0, first_span.x1, first_span.y1),
                    extraction_method=ExtractionMethod.PYMUPDF,
                    confidence=0.85,
                ),
            )
            
            if level > 0 and current_list_items:
                # Add as sub-item to last top-level item
                current_list_items[-1].sub_items.append(item)
            else:
                current_list_items.append(item)
            
            if list_start_span is None:
                list_start_span = first_span
            list_end_span = first_span
        else:
            # End of list
            if len(current_list_items) >= 2:
                list_blocks.append(ContentBlock(
                    type=ContentBlockType.LIST,
                    list_items=current_list_items,
                    source=Source(
                        page=page_num,
                        bbox=(
                            list_start_span.x0 if list_start_span else 0,
                            list_start_span.y0 if list_start_span else 0,
                            list_end_span.x1 if list_end_span else 0,
                            list_end_span.y1 if list_end_span else 0,
                        ),
                        extraction_method=ExtractionMethod.PYMUPDF,
                        confidence=0.80,
                    ),
                ))
            current_list_items = []
            list_start_span = None
            list_end_span = None
            base_indent = None
    
    # Final flush
    if len(current_list_items) >= 2:
        list_blocks.append(ContentBlock(
            type=ContentBlockType.LIST,
            list_items=current_list_items,
            source=Source(
                page=page_num,
                bbox=(
                    list_start_span.x0 if list_start_span else 0,
                    list_start_span.y0 if list_start_span else 0,
                    list_end_span.x1 if list_end_span else 0,
                    list_end_span.y1 if list_end_span else 0,
                ),
                extraction_method=ExtractionMethod.PYMUPDF,
                confidence=0.80,
            ),
        ))
    
    return list_blocks


# ═══════════════════════════════════════════════════════════════════════════════
# 7. INLINE FORMATTING DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_inline_formats(
    spans: list[TextSpan],
    body_font_size: float,
) -> tuple[str, list[InlineFormat]]:
    """
    Reconstruct text from spans while tracking bold/italic/super/sub ranges.
    
    Returns (full_text, list of InlineFormat marking formatting ranges).
    """
    if not spans:
        return "", []
    
    sorted_spans = sorted(spans, key=lambda s: (round(s.y0, 1), s.x0))
    
    text_parts: list[str] = []
    formats: list[InlineFormat] = []
    pos = 0
    prev_y = None
    
    for span in sorted_spans:
        # Add newline for new visual line
        if prev_y is not None and abs(span.y0 - prev_y) > 3:
            text_parts.append(" ")
            pos += 1
        elif text_parts:
            text_parts.append(" ")
            pos += 1
        
        start = pos
        
        # Handle superscript notation
        if span.is_superscript:
            display_text = f"^{{{span.text}}}"
        elif span.is_subscript:
            display_text = f"_{{{span.text}}}"
        else:
            display_text = span.text
        
        text_parts.append(display_text)
        pos += len(display_text)
        
        # Track non-body formatting
        if span.is_bold or span.is_italic or span.is_superscript or span.is_subscript:
            formats.append(InlineFormat(
                start=start,
                end=pos,
                bold=span.is_bold,
                italic=span.is_italic,
                superscript=span.is_superscript,
                subscript=span.is_subscript,
                font_size=span.font_size if span.font_size != body_font_size else None,
            ))
        
        prev_y = span.y0
    
    return "".join(text_parts), formats
