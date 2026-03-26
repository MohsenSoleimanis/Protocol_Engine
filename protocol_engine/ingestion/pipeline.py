"""
Ingestion Pipeline Orchestrator — Coordinates PDF parsing → chunking → indexing.

This wraps the existing ingestion/run.py pipeline and adds:
  1. Multi-strategy table extraction
  2. Contextual chunking
  3. Vision reconciliation trigger

Usage:
    from protocol_engine.ingestion.pipeline import ingest_protocol

    result = ingest_protocol("protocol.pdf")
    # result contains: json_data, chunks, store
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def ingest_protocol(
    pdf_path: str,
    output_dir: str | None = None,
    with_llm: bool = True,
    with_contextual_chunks: bool = True,
) -> dict:
    """Full ingestion pipeline: PDF → structured JSON → contextual chunks.

    This delegates to the existing ingestion pipeline for PDF parsing,
    then adds contextual chunking on top.

    Args:
        pdf_path: Path to the protocol PDF
        output_dir: Directory for output files (default: output/)
        with_llm: Enable LLM repair for low-quality tables
        with_contextual_chunks: Enable contextual chunking

    Returns:
        {
            "json_data": dict,        # The structured JSON from parsing
            "chunks": list[Chunk],    # Contextual chunks for embedding
            "json_path": str,         # Path to structured JSON file
            "stats": dict,            # Pipeline statistics
        }
    """
    pdf_path = Path(pdf_path)
    if output_dir is None:
        output_dir = str(pdf_path.parent / "output")
    output_prefix = f"{output_dir}/{pdf_path.stem}"

    t0 = time.time()
    logger.info(f"Ingesting: {pdf_path.name}")

    # Step 1: Run the existing PDF parsing pipeline
    json_path = Path(f"{output_prefix}_structured.json")
    if json_path.exists():
        logger.info(f"Found existing parsed data at {json_path}")
        json_data = json.loads(json_path.read_text())
    else:
        logger.info("Running PDF parsing pipeline...")
        json_data = _run_pdf_pipeline(str(pdf_path), output_prefix, with_llm)

    # Step 2: Build sections list for chunking
    sections = _extract_sections_from_json(json_data)
    tables = json_data.get("tables", [])

    # Step 3: Create contextual chunks
    chunks = []
    if with_contextual_chunks:
        from protocol_engine.ingestion.chunker import create_chunks
        from protocol_engine.config import CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS

        doc_summary = _build_document_summary(json_data)
        chunks = create_chunks(
            sections=sections,
            tables=tables,
            document_summary=doc_summary,
            chunk_size_tokens=CHUNK_SIZE_TOKENS,
            chunk_overlap_tokens=CHUNK_OVERLAP_TOKENS,
        )

    elapsed = time.time() - t0
    stats = {
        "elapsed": round(elapsed, 1),
        "sections": len(sections),
        "tables": len(tables),
        "chunks": len(chunks),
        "total_pages": json_data.get("total_pages", 0),
    }

    logger.info(
        f"Ingestion done: {stats['sections']} sections, {stats['tables']} tables, "
        f"{stats['chunks']} chunks, {elapsed:.1f}s"
    )

    return {
        "json_data": json_data,
        "chunks": chunks,
        "json_path": str(json_path),
        "stats": stats,
    }


def _run_pdf_pipeline(pdf_path: str, output_prefix: str, with_llm: bool) -> dict:
    """Run the PDF parsing pipeline via the legacy bridge."""
    from protocol_engine.ingestion._legacy import run_legacy_pipeline
    return run_legacy_pipeline(
        pdf_path=pdf_path,
        output_prefix=output_prefix,
        with_llm=with_llm,
    )


def _extract_sections_from_json(json_data: dict) -> list[dict]:
    """Extract section list from parsed JSON for chunking."""
    sections = []
    for section in json_data.get("sections", []):
        sid = section.get("number", section.get("id", ""))
        title = section.get("title", "")
        pages = section.get("page_range", [])

        # Build content from content_blocks
        content_parts = []
        for block in section.get("content_blocks", []):
            block_type = block.get("type", "paragraph")
            if block_type == "paragraph":
                text = block.get("text", "")
                if text:
                    content_parts.append(text)
            elif block_type == "list":
                for item in block.get("list_items", []):
                    marker = item.get("marker", "-")
                    text = item.get("text", "")
                    content_parts.append(f"{marker} {text}")
                    for sub in item.get("sub_items", []):
                        sub_marker = sub.get("marker", "-")
                        content_parts.append(f"  {sub_marker} {sub.get('text', '')}")

        content = "\n\n".join(content_parts)
        if content.strip():
            sections.append({
                "section_id": sid,
                "title": title,
                "content": content,
                "pages": pages,
            })

    return sections


def _build_document_summary(json_data: dict) -> str:
    """Build a brief document summary for contextual chunk prefixes."""
    filename = json_data.get("filename", "")
    total_pages = json_data.get("total_pages", 0)
    n_sections = len(json_data.get("sections", []))
    n_tables = len(json_data.get("tables", []))

    parts = []
    if filename:
        parts.append(f"Clinical protocol: {filename}")
    if total_pages:
        parts.append(f"{total_pages} pages")
    if n_sections:
        parts.append(f"{n_sections} sections")
    if n_tables:
        parts.append(f"{n_tables} tables")

    return ", ".join(parts) if parts else "Clinical protocol document"
