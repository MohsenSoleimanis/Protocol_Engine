"""
Contextual Chunker — Anthropic's contextual retrieval method.

Key improvements from old code:
  1. Contextual: prepends document-level context to each chunk before embedding
     (Anthropic research: 67% fewer retrieval failures with hybrid + reranking)
  2. Token-based sizes (1024 tokens, 128 overlap) not char-based (8192 chars, 0 overlap)
  3. Tables are NEVER split across chunks (each table = one chunk)
  4. Lists are NEVER split across chunks (each list = one chunk)
  5. Section boundaries respected — no splitting mid-section if section < 2048 tokens
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Approximate: 1 token ≈ 4 chars for English text
CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    """A contextual chunk ready for embedding."""
    chunk_id: str
    text: str                    # The chunk text (with contextual prefix)
    raw_text: str                # Original text without context prefix
    context_prefix: str          # The prepended context
    section_id: str = ""
    section_title: str = ""
    chunk_type: str = "text"     # "text", "table", "list"
    page_range: list[int] = field(default_factory=list)
    token_estimate: int = 0
    metadata: dict = field(default_factory=dict)


def create_chunks(
    sections: list[dict],
    tables: list[dict],
    document_summary: str = "",
    chunk_size_tokens: int = 1024,
    chunk_overlap_tokens: int = 128,
) -> list[Chunk]:
    """Create contextual chunks from parsed document sections and tables.

    Args:
        sections: List of section dicts with keys: section_id, title, content, pages
        tables: List of table dicts with keys: id, caption, column_headers, rows, page_range
        document_summary: Brief document description for context prefix
        chunk_size_tokens: Target chunk size in tokens
        chunk_overlap_tokens: Overlap between adjacent chunks in tokens

    Returns:
        List of Chunk objects ready for embedding
    """
    chunks: list[Chunk] = []
    chunk_idx = 0
    chunk_size_chars = chunk_size_tokens * CHARS_PER_TOKEN
    overlap_chars = chunk_overlap_tokens * CHARS_PER_TOKEN

    # Process sections
    for section in sections:
        sid = section.get("section_id", "")
        title = section.get("title", "")
        content = section.get("content", "")
        pages = section.get("pages", [])

        if not content.strip():
            continue

        # Build context prefix for this section
        context = _build_section_context(sid, title, document_summary)

        # If section fits in one chunk, don't split
        if len(content) <= chunk_size_chars * 2:
            chunk_idx += 1
            chunk_text = f"{context}\n\n{content}"
            chunks.append(Chunk(
                chunk_id=f"chunk_{chunk_idx}",
                text=chunk_text,
                raw_text=content,
                context_prefix=context,
                section_id=sid,
                section_title=title,
                chunk_type="text",
                page_range=pages,
                token_estimate=len(chunk_text) // CHARS_PER_TOKEN,
                metadata={"type": "section", "section_id": sid, "title": title},
            ))
        else:
            # Split large sections at paragraph boundaries
            paragraphs = _split_at_paragraphs(content)
            current_text = ""

            for para in paragraphs:
                if len(current_text) + len(para) > chunk_size_chars and current_text:
                    # Emit current chunk
                    chunk_idx += 1
                    chunk_text = f"{context}\n\n{current_text}"
                    chunks.append(Chunk(
                        chunk_id=f"chunk_{chunk_idx}",
                        text=chunk_text,
                        raw_text=current_text,
                        context_prefix=context,
                        section_id=sid,
                        section_title=title,
                        chunk_type="text",
                        page_range=pages,
                        token_estimate=len(chunk_text) // CHARS_PER_TOKEN,
                        metadata={"type": "section", "section_id": sid, "title": title},
                    ))
                    # Overlap: keep the last part of current text
                    if overlap_chars > 0 and len(current_text) > overlap_chars:
                        current_text = current_text[-overlap_chars:]
                    else:
                        current_text = ""

                current_text += ("\n\n" if current_text else "") + para

            # Emit final chunk
            if current_text.strip():
                chunk_idx += 1
                chunk_text = f"{context}\n\n{current_text}"
                chunks.append(Chunk(
                    chunk_id=f"chunk_{chunk_idx}",
                    text=chunk_text,
                    raw_text=current_text,
                    context_prefix=context,
                    section_id=sid,
                    section_title=title,
                    chunk_type="text",
                    page_range=pages,
                    token_estimate=len(chunk_text) // CHARS_PER_TOKEN,
                    metadata={"type": "section", "section_id": sid, "title": title},
                ))

    # Process tables — each table is ONE chunk (never split)
    for table in tables:
        tid = table.get("id", "")
        caption = table.get("caption", "")
        headers = table.get("column_headers", [])
        rows = table.get("rows", [])
        pages = table.get("page_range", [])

        parts = [f"Table: {caption}"]
        if headers:
            parts.append(" | ".join(str(h) for h in headers))
        for row in rows[:100]:  # safety limit
            parts.append(" | ".join(str(c) for c in row))
        table_text = "\n".join(parts)

        if not table_text.strip():
            continue

        context = f"This is a table from the protocol document."
        if caption:
            context += f" Table caption: {caption}."

        chunk_idx += 1
        chunk_text = f"{context}\n\n{table_text}"
        chunks.append(Chunk(
            chunk_id=f"chunk_{chunk_idx}",
            text=chunk_text,
            raw_text=table_text,
            context_prefix=context,
            chunk_type="table",
            page_range=pages,
            token_estimate=len(chunk_text) // CHARS_PER_TOKEN,
            metadata={"type": "table", "table_id": tid, "title": caption[:100]},
        ))

    logger.info(
        f"Created {len(chunks)} contextual chunks "
        f"({sum(c.token_estimate for c in chunks)} estimated tokens)"
    )
    return chunks


def _build_section_context(section_id: str, title: str, doc_summary: str) -> str:
    """Build a context prefix for a section chunk.

    This is the core of Anthropic's contextual retrieval technique:
    prepend document-level context to each chunk before embedding.
    """
    parts = []
    if doc_summary:
        parts.append(f"Document: {doc_summary}")
    if section_id:
        parts.append(f"Section §{section_id}: {title}")
    elif title:
        parts.append(f"Section: {title}")
    return ". ".join(parts)


def _split_at_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs, preserving list blocks together."""
    raw_paragraphs = text.split("\n\n")
    paragraphs = []
    current_list = []

    for para in raw_paragraphs:
        stripped = para.strip()
        if not stripped:
            continue

        # Check if this looks like a list item
        is_list_item = bool(
            stripped.startswith(("- ", "• ", "– ", "— "))
            or (len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)")
            or (len(stripped) > 2 and stripped[0].isalpha() and stripped[1] in ".)")
        )

        if is_list_item:
            current_list.append(stripped)
        else:
            # Flush accumulated list items as one block
            if current_list:
                paragraphs.append("\n".join(current_list))
                current_list = []
            paragraphs.append(stripped)

    if current_list:
        paragraphs.append("\n".join(current_list))

    return paragraphs
