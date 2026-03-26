"""
Header/footer detection and filtering.

Generic approach: scans first N pages for text that repeats in the same
vertical zone. No hardcoded patterns — discovers them from the document.
"""

from __future__ import annotations
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import fitz

from .models import TextSpan


@dataclass
class PageMeta:
    page_num_display: str = ""
    page_number: int | None = None
    total_pages: int | None = None
    is_landscape: bool = False


# Multiple page number patterns (generic discovery)
PAGE_NUM_PATTERNS = [
    re.compile(r"(\d+)\s+of\s+(\d+)", re.IGNORECASE),        # "19 of 130"
    re.compile(r"Page\s+(\d+)\s+of\s+(\d+)", re.IGNORECASE),  # "Page 19 of 130"
    re.compile(r"^(\d+)$"),                                     # standalone number
    re.compile(r"-\s*(\d+)\s*-"),                               # "- 19 -"
]


def detect_header_footer_patterns(
    doc: fitz.Document,
    sample_pages: int = 30,
    min_occurrence_ratio: float = 0.4,
) -> tuple[dict[str, float], str]:
    """
    Scan pages to find repeating text in header/footer zones.
    
    Returns:
        patterns: dict of {normalized_text: avg_y_position}
        page_num_format: detected format string (e.g., "{n} of {total}")
    """
    n_pages = min(sample_pages, len(doc))
    text_positions: dict[str, list[float]] = defaultdict(list)
    text_page_count: Counter = Counter()
    page_num_format = ""
    
    for i in range(n_pages):
        page = doc[i]
        ph = page.rect.height
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        seen_texts: set[str] = set()

        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text or len(text) < 3:
                        continue
                    y = span["bbox"][1]
                    
                    # Only consider header zone (top 10%) or footer zone (bottom 10%)
                    in_header_zone = y < ph * 0.10
                    in_footer_zone = y > ph * 0.88
                    if not (in_header_zone or in_footer_zone):
                        continue
                    
                    # Normalize: strip page numbers
                    normalized = text
                    for pat in PAGE_NUM_PATTERNS:
                        normalized = pat.sub("", normalized).strip()
                    if not normalized or len(normalized) < 3:
                        # This was just a page number — detect format
                        for pat in PAGE_NUM_PATTERNS:
                            m = pat.search(text)
                            if m:
                                if len(m.groups()) >= 2:
                                    page_num_format = "{n} of {total}"
                                else:
                                    page_num_format = "{n}"
                        continue
                    
                    if normalized not in seen_texts:
                        text_page_count[normalized] += 1
                        seen_texts.add(normalized)
                    text_positions[normalized].append(y)

    min_count = max(3, int(n_pages * min_occurrence_ratio))
    patterns: dict[str, float] = {}
    for text, count in text_page_count.items():
        if count >= min_count:
            avg_y = sum(text_positions[text]) / len(text_positions[text])
            patterns[text] = avg_y

    return patterns, page_num_format


def extract_page_meta(page: fitz.Page, page_idx: int) -> PageMeta:
    """Extract page metadata: page number, orientation."""
    meta = PageMeta()
    meta.is_landscape = page.rect.width > page.rect.height
    ph = page.rect.height

    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text:
                    continue
                y = span["bbox"][1]
                # Only look in footer zone for page numbers
                if y < ph * 0.85:
                    continue
                for pat in PAGE_NUM_PATTERNS:
                    m = pat.search(text)
                    if m:
                        meta.page_num_display = m.group(0)
                        meta.page_number = int(m.group(1))
                        if len(m.groups()) >= 2:
                            meta.total_pages = int(m.group(2))
                        break
    return meta


def filter_header_footer_spans(
    spans: list[TextSpan],
    page_height: float,
    header_footer_patterns: dict[str, float],
    margin_top: float = 60,
    margin_bottom: float = 55,
) -> tuple[list[TextSpan], list[TextSpan], PageMeta]:
    """
    Remove header/footer/page-number spans.
    Returns (filtered_spans, removed_spans, page_meta).
    """
    filtered: list[TextSpan] = []
    removed: list[TextSpan] = []
    meta = PageMeta()

    for span in spans:
        should_remove = False

        # Normalize span text
        normalized = span.text.strip()
        for pat in PAGE_NUM_PATTERNS:
            normalized = pat.sub("", normalized).strip()

        # Check against discovered patterns
        if normalized in header_footer_patterns:
            should_remove = True

        # Check for page number patterns
        for pat in PAGE_NUM_PATTERNS:
            m = pat.search(span.text)
            if m and span.y0 > page_height - margin_bottom:
                should_remove = True
                meta.page_num_display = m.group(0)
                meta.page_number = int(m.group(1))
                if len(m.groups()) >= 2:
                    meta.total_pages = int(m.group(2))
                break

        # Position-based: top margin + matching pattern
        if span.y0 < margin_top and normalized in header_footer_patterns:
            should_remove = True

        if should_remove:
            removed.append(span)
        else:
            filtered.append(span)

    return filtered, removed, meta
