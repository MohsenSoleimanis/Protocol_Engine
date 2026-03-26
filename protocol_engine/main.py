"""
Protocol Engine — Main entry point.

Wires together: ingestion → retrieval → graph pipeline.

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
    """Initialize the system for a protocol PDF.

    Args:
        pdf_path: Path to the protocol PDF
        json_data: Pre-parsed JSON data (from ingestion pipeline).
                   If None, runs ingestion first.

    Returns:
        (store, retriever, json_data) tuple
    """
    # If no pre-parsed data, try to load from output directory
    if json_data is None:
        from pathlib import Path
        import json
        output_dir = Path(pdf_path).parent / "output"
        json_path = output_dir / (Path(pdf_path).stem + ".json")
        if json_path.exists():
            json_data = json.loads(json_path.read_text())
            logger.info(f"Loaded pre-parsed data from {json_path}")
        else:
            logger.warning(f"No pre-parsed data found at {json_path}. Run ingestion first.")
            json_data = {"sections": [], "tables": []}

    # Build store
    from knowledge_base.protocol_store import ProtocolStore
    store = ProtocolStore(json_data)

    # Build retriever
    from protocol_engine.retrieval.engine import build_retriever
    retriever = build_retriever(store, json_data, api_key=OPENAI_API_KEY)

    return store, retriever, json_data


def run_query(
    query: str,
    query_type: str,
    retriever,
    store,
    json_data: dict,
    pdf_path: str,
    event_bus=None,
    debug_log=None,
) -> dict:
    """Run a protocol extraction query through the 7-node graph.

    Args:
        query: Natural language query
        query_type: Query type (or "general" for auto-detection)
        retriever: Hybrid retriever instance
        store: ProtocolStore instance
        json_data: Parsed protocol JSON
        pdf_path: Path to the protocol PDF
        event_bus: Optional event bus for UI updates
        debug_log: Optional debug logger

    Returns:
        {data, validation, signals, steps, total_turns, error}
    """
    from protocol_engine.graph.builder import protocol_graph

    t0 = time.time()
    logger.info(f"Query: {query_type} -- '{query[:60]}'")

    initial: ProtocolState = {
        "query": query,
        "query_type": query_type,
        "pdf_path": pdf_path,
        # Planner
        "sub_tasks": [],
        "current_task_index": 0,
        # Explorer
        "sections_content": {},
        "tables_content": {},
        "sections_read": [],
        # Context Assembler
        "assembled_context": "",
        "context_tokens_used": 0,
        "context_sections_included": 0,
        "context_relevance_scores": {},
        # Extractor
        "extracted_data": {},
        "extraction_history": [],
        # Validation
        "validation": {},
        # Reviewer
        "signals": [],
        # Control
        "edge_signal": "",
        "edge_detail": "",
        "cycle_count": 0,
        "steps": [],
        "error": "",
    }

    runtime = RuntimeContext(
        retriever=retriever,
        store=store,
        json_data=json_data,
        event_bus=event_bus,
    )
    config = {"configurable": {"runtime": runtime}}

    try:
        final = protocol_graph.invoke(initial, config=config)
        steps = final.get("steps", [])
        total_turns = sum(s.get("turns", 0) for s in steps)
        elapsed = time.time() - t0

        logger.info(f"Done: {total_turns} turns, {elapsed:.1f}s, "
                    f"{len(final.get('signals', []))} signals")

        return {
            "data": final.get("extracted_data", {}),
            "validation": final.get("validation", {}),
            "signals": final.get("signals", []),
            "steps": steps,
            "total_turns": total_turns,
            "error": final.get("error", ""),
        }

    except Exception as e:
        logger.error(f"Graph failed: {e}", exc_info=True)
        return {
            "data": {},
            "validation": {},
            "signals": [],
            "steps": [],
            "total_turns": 0,
            "error": str(e),
        }
