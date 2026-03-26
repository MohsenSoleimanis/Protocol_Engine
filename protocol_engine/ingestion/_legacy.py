"""
Legacy bridge — imports the old ingestion pipeline if available.

This isolates the old code dependency so it doesn't pollute the new package.
If the old pipeline isn't available, returns empty data.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def run_legacy_pipeline(
    pdf_path: str,
    output_prefix: str,
    with_llm: bool = True,
) -> dict:
    """Run the old ingestion/run.py pipeline and return parsed data.

    Returns a DocumentResult-like dict, or empty data if unavailable.
    """
    try:
        # Add old ingestion directory to path
        repo_root = Path(__file__).parent.parent.parent
        ingestion_dir = repo_root / "ingestion"

        if ingestion_dir.exists() and str(ingestion_dir) not in sys.path:
            sys.path.insert(0, str(ingestion_dir))

        from ingestion.run import process_document

        result = process_document(
            pdf_path=pdf_path,
            output_prefix=output_prefix,
            with_llm=with_llm,
        )

        # Load the generated JSON file
        json_path = Path(f"{output_prefix}_structured.json")
        if json_path.exists():
            return json.loads(json_path.read_text())

        # Fallback: convert result model to dict
        if hasattr(result, "model_dump"):
            return result.model_dump(mode="json", exclude_none=True)

        return {}

    except ImportError as e:
        logger.warning(f"Legacy ingestion pipeline not available: {e}")
        return {"sections": [], "tables": [], "total_pages": 0}
    except Exception as e:
        logger.error(f"Legacy pipeline failed: {e}")
        return {"sections": [], "tables": [], "total_pages": 0}
