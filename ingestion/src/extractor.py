"""
PyMuPDF-based raw extraction with enhanced font metadata.

Detects superscripts, subscripts, bold, italic, and preserves
exact Unicode characters (checkmarks, arrows, special dashes).
"""

from __future__ import annotations
import fitz  # PyMuPDF
from .models import TextSpan, DrawingLine


def extract_text_spans(page: fitz.Page, page_num: int) -> list[TextSpan]:
    """
    Extract every text span with coordinates and full font metadata.
    
    Superscript detection: smaller font + higher baseline than adjacent spans.
    Subscript detection: smaller font + lower baseline than adjacent spans.
    """
    raw_spans: list[dict] = []
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]

    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            line_spans = line.get("spans", [])
            # Collect baseline info for super/subscript detection
            if not line_spans:
                continue
            
            # Dominant font size in this line (mode)
            sizes = [s.get("size", 0) for s in line_spans if s.get("size", 0) > 0]
            dominant_size = max(set(sizes), key=sizes.count) if sizes else 0
            # Baseline y (origin y) of the line
            line_origin_y = line.get("bbox", [0, 0, 0, 0])[3]  # bottom of line bbox
            
            for span in line_spans:
                text = span["text"]
                # Preserve whitespace-only spans for layout but skip truly empty
                if not text or text.isspace():
                    continue
                text = text.strip()
                if not text:
                    continue
                    
                bbox = span["bbox"]
                font_name = span.get("font", "")
                font_size = span.get("size", 0.0)
                flags = span.get("flags", 0)
                
                is_bold = bool(
                    "bold" in font_name.lower()
                    or "Bold" in font_name
                    or (flags & 16)  # bit 4 = bold
                )
                is_italic = bool(
                    "italic" in font_name.lower()
                    or "oblique" in font_name.lower()
                    or (flags & 2)   # bit 1 = italic
                )
                
                # Superscript: significantly smaller + positioned higher
                is_superscript = False
                is_subscript = False
                if dominant_size > 0 and font_size > 0:
                    size_ratio = font_size / dominant_size
                    if size_ratio < 0.8:  # notably smaller
                        span_top = bbox[1]
                        span_bottom = bbox[3]
                        line_top = line.get("bbox", [0, 0, 0, 0])[1]
                        line_bottom = line.get("bbox", [0, 0, 0, 0])[3]
                        line_mid = (line_top + line_bottom) / 2
                        
                        span_mid = (span_top + span_bottom) / 2
                        if span_mid < line_mid:
                            is_superscript = True
                        elif span_bottom > line_mid + (line_bottom - line_mid) * 0.3:
                            is_subscript = True

                raw_spans.append({
                    "text": text,
                    "bbox": bbox,
                    "font_name": font_name,
                    "font_size": font_size,
                    "is_bold": is_bold,
                    "is_italic": is_italic,
                    "is_superscript": is_superscript,
                    "is_subscript": is_subscript,
                    "color": span.get("color", 0),
                })

    # Build TextSpan objects
    spans: list[TextSpan] = []
    for s in raw_spans:
        bbox = s["bbox"]
        spans.append(TextSpan(
            text=s["text"],
            x0=round(bbox[0], 2),
            y0=round(bbox[1], 2),
            x1=round(bbox[2], 2),
            y1=round(bbox[3], 2),
            font_name=s["font_name"],
            font_size=round(s["font_size"], 2),
            is_bold=s["is_bold"],
            is_italic=s["is_italic"],
            is_superscript=s["is_superscript"],
            is_subscript=s["is_subscript"],
            color=s["color"],
            page_num=page_num,
        ))
    return spans


def extract_drawing_lines(page: fitz.Page, page_num: int) -> list[DrawingLine]:
    """Extract ruling lines from PDF vector drawings."""
    lines: list[DrawingLine] = []

    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            kind = item[0]
            if kind == "l":
                p1, p2 = item[1], item[2]
                line = _make_line(p1.x, p1.y, p2.x, p2.y, page_num)
                if line:
                    lines.append(line)
            elif kind == "re":
                rect = item[1]
                edges = [
                    (rect.x0, rect.y0, rect.x1, rect.y0),
                    (rect.x0, rect.y1, rect.x1, rect.y1),
                    (rect.x0, rect.y0, rect.x0, rect.y1),
                    (rect.x1, rect.y0, rect.x1, rect.y1),
                ]
                for x0, y0, x1, y1 in edges:
                    line = _make_line(x0, y0, x1, y1, page_num)
                    if line:
                        lines.append(line)

    return _deduplicate_lines(lines)


def extract_images_info(page: fitz.Page, page_num: int) -> list[dict]:
    """Detect image regions on a page (bbox only, not image content)."""
    images = []
    for img in page.get_images(full=True):
        xref = img[0]
        rects = page.get_image_rects(xref)
        for rect in rects:
            if rect.width > 50 and rect.height > 50:  # skip tiny decorative images
                images.append({
                    "page": page_num,
                    "bbox": (round(rect.x0, 2), round(rect.y0, 2),
                             round(rect.x1, 2), round(rect.y1, 2)),
                    "width": round(rect.width, 2),
                    "height": round(rect.height, 2),
                })
    return images


def _make_line(x0: float, y0: float, x1: float, y1: float, page_num: int) -> DrawingLine | None:
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    length = max(dx, dy)
    if length < 10:
        return None
    if dy < 2:
        orientation = "horizontal"
    elif dx < 2:
        orientation = "vertical"
    else:
        return None
    return DrawingLine(
        x0=round(min(x0, x1), 2), y0=round(min(y0, y1), 2),
        x1=round(max(x0, x1), 2), y1=round(max(y0, y1), 2),
        orientation=orientation, page_num=page_num,
    )


def _deduplicate_lines(lines: list[DrawingLine], tolerance: float = 2.0) -> list[DrawingLine]:
    unique: list[DrawingLine] = []
    for line in lines:
        is_dup = False
        for existing in unique:
            if (existing.orientation == line.orientation
                    and abs(existing.x0 - line.x0) < tolerance
                    and abs(existing.y0 - line.y0) < tolerance
                    and abs(existing.x1 - line.x1) < tolerance
                    and abs(existing.y1 - line.y1) < tolerance):
                is_dup = True
                break
        if not is_dup:
            unique.append(line)
    return unique


def reconstruct_line_text(spans: list[TextSpan]) -> str:
    """
    Reconstruct text from spans preserving superscripts with ^{} notation.
    Adjacent spans on the same line are joined with appropriate spacing.
    """
    if not spans:
        return ""
    
    # Sort by y then x
    sorted_spans = sorted(spans, key=lambda s: (round(s.y0, 1), s.x0))
    
    parts: list[str] = []
    prev_x1 = None
    prev_y = None
    
    for span in sorted_spans:
        # New line?
        if prev_y is not None and abs(span.y0 - prev_y) > 3:
            parts.append("\n")
            prev_x1 = None
        
        # Gap between spans?
        if prev_x1 is not None and span.x0 - prev_x1 > 3:
            parts.append(" ")
        
        # Format superscript/subscript
        if span.is_superscript:
            parts.append(f"^{{{span.text}}}")
        elif span.is_subscript:
            parts.append(f"_{{{span.text}}}")
        else:
            parts.append(span.text)
        
        prev_x1 = span.x1
        prev_y = span.y0
    
    return "".join(parts)
