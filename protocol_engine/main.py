"""
Protocol Engine — Entry point.

Usage:
    from protocol_engine.main import run_query, initialize

    store, retriever, json_data = initialize("protocol.pdf")
    result = run_query("Extract all eligibility criteria", "eligibility",
                       retriever=retriever, store=store, json_data=json_data,
                       pdf_path="protocol.pdf")
"""
from __future__ import annotations

import time
import logging

from protocol_engine.config import OPENAI_API_KEY
from protocol_engine.models.state import ProtocolState, RuntimeContext
from protocol_engine.models.enums import EdgeSignal

logger = logging.getLogger(__name__)


def initialize(pdf_path: str, json_data: dict | None = None):
    """Initialize store + retriever for a protocol PDF."""
    if json_data is None:
        import json
        from pathlib import Path
        json_path = Path(pdf_path).parent / "output" / (Path(pdf_path).stem + ".json")
        if json_path.exists():
            json_data = json.loads(json_path.read_text())
            logger.info(f"Loaded parsed data from {json_path}")
        else:
            logger.warning(f"No parsed data at {json_path}. Run ingestion first.")
            json_data = {"sections": [], "tables": []}

    from protocol_engine.store import ProtocolStore
    store = ProtocolStore(json_data)

    from protocol_engine.retrieval.engine import build_retriever
    retriever = build_retriever(store, json_data, api_key=OPENAI_API_KEY)

    return store, retriever, json_data


def run_query(
    query: str, query_type: str,
    retriever, store, json_data: dict, pdf_path: str,
    event_bus=None,
) -> dict:
    """Run extraction through the 3-node graph."""
    from protocol_engine.graph.builder import protocol_graph
    from protocol_engine.guardrails.input import sanitize_input

    t0 = time.time()

    # Input guardrails
    query, query_type, input_warnings = sanitize_input(query, query_type)
    if input_warnings:
        logger.warning(f"Input guardrails: {input_warnings}")

    logger.info(f"Query: {query_type} — '{query[:60]}'")

    initial: ProtocolState = {
        "query": query,
        "query_type": query_type,
        "pdf_path": pdf_path,
        "sections_content": {},
        "tables_content": {},
        "assembled_context": "",
        "extracted_data": {},
        "validation": {},
        "signals": [],
        "edge_signal": "",
        "cycle_count": 0,
        "steps": [],
    }

    runtime = RuntimeContext(
        retriever=retriever, store=store,
        json_data=json_data, event_bus=event_bus,
    )

    try:
        final = protocol_graph.invoke(initial, config={"configurable": {"runtime": runtime}})
        elapsed = time.time() - t0
        steps = final.get("steps", [])
        logger.info(f"Done: {elapsed:.1f}s, {len(final.get('signals', []))} signals")
        return {
            "data": final.get("extracted_data", {}),
            "validation": final.get("validation", {}),
            "signals": final.get("signals", []),
            "steps": steps,
        }
    except Exception as e:
        logger.error(f"Graph failed: {e}", exc_info=True)
        return {"data": {}, "validation": {}, "signals": [], "steps": [], "error": str(e)}
