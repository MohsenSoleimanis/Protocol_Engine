"""
Legacy bridge — placeholder for old ingestion pipeline.

The old ingestion/ folder has been removed. This returns empty data
and logs a warning. Run the new pipeline via protocol_engine.ingestion.pipeline.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def run_legacy_pipeline(
    pdf_path: str,
    output_prefix: str,
    with_llm: bool = True,
) -> dict:
    """Placeholder — old pipeline removed. Returns empty data."""
    logger.warning("Legacy ingestion pipeline removed. Use protocol_engine.ingestion.pipeline instead.")
    return {"sections": [], "tables": [], "total_pages": 0}
