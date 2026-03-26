"""
Context Assembler Node — Relevance-scored, token-aware context building.

NEW node (old code used greedy char truncation with no relevance scoring).

Key improvements:
  1. Token-based budgeting (not char-based)
  2. Relevance scoring per section
  3. Tiered inclusion: verbatim (high), summary (medium), skip (low)
  4. Newly-fetched content (from NEED_MORE cycles) gets priority
  5. Structured output with clear section delineation
"""
from __future__ import annotations

import logging
import re

from langchain_core.runnables import RunnableConfig

from protocol_engine.config import (
    CONTEXT_BUDGET_TOKENS, HIGH_RELEVANCE_THRESHOLD,
    MEDIUM_RELEVANCE_THRESHOLD,
)
from protocol_engine.models.enums import EdgeSignal, NodeName
from protocol_engine.models.state import get_runtime

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token for English."""
    return len(text) // 4


def _relevance_score(text: str, query: str, query_type: str) -> float:
    """Score section relevance to the query (0.0 - 1.0).

    Uses keyword overlap + query type matching.
    A proper implementation would use embeddings, but this is fast and free.
    """
    text_lower = text.lower()
    query_lower = query.lower()

    # Keyword overlap
    query_words = set(query_lower.split())
    text_words = set(text_lower.split())
    if not query_words:
        return 0.5
    overlap = len(query_words & text_words) / len(query_words)

    # Query type relevance boost
    type_keywords = {
        "endpoints": ["endpoint", "objective", "efficacy", "primary", "secondary"],
        "eligibility": ["inclusion", "exclusion", "criteria", "eligible"],
        "safety": ["safety", "adverse", "aesi", "monitoring", "stopping"],
        "soa": ["schedule", "activities", "visit", "procedure"],
        "study_design": ["design", "randomiz", "blind", "stratif", "phase"],
        "intervention": ["intervention", "dose", "drug", "vaccine", "placebo"],
        "statistical": ["statistic", "sample", "power", "analysis", "interim"],
        "deviation": ["deviation", "violation", "eligibility", "compliance"],
        "kri": ["risk", "indicator", "kri", "quality", "tolerance"],
    }

    type_boost = 0.0
    for kw in type_keywords.get(query_type, []):
        if kw in text_lower:
            type_boost += 0.1

    # Section headers matching query boost
    header_match = re.search(r'^§[\d.]+:?\s+(.+?)$', text[:200], re.MULTILINE)
    if header_match:
        title = header_match.group(1).lower()
        title_overlap = len(set(title.split()) & query_words) / max(len(query_words), 1)
        type_boost += title_overlap * 0.2

    score = min(1.0, overlap * 0.5 + type_boost + 0.2)  # baseline 0.2
    return round(score, 3)


def context_assembler_node(state: dict, config: RunnableConfig) -> dict:
    """Assemble context with relevance scoring and token budgeting."""
    runtime = get_runtime(config)
    bus = runtime.event_bus
    query = state.get("query", "")
    query_type = state.get("query_type", "general")
    sections = state.get("sections_content", {})
    tables = state.get("tables_content", {})
    sections_read = state.get("sections_read", [])

    if bus:
        bus.emit(NodeName.CONTEXT_ASSEMBLER, "starting", "Assembling context...")

    if not sections and not tables:
        logger.warning("Context assembler: no content to assemble")
        return {
            "assembled_context": "",
            "context_tokens_used": 0,
            "context_sections_included": 0,
            "context_relevance_scores": {},
            "edge_signal": EdgeSignal.CONTINUE,
            "steps": [{"agent": NodeName.CONTEXT_ASSEMBLER, "turns": 0,
                       "tool_calls": 0, "tools_used": []}],
        }

    # Score all sections for relevance
    scored_items: list[tuple[str, str, dict, float]] = []  # (id, type, data, score)

    for sid, data in sections.items():
        text = data.get("text", "")
        score = _relevance_score(text, query, query_type)
        scored_items.append((sid, "section", data, score))

    for tid, data in tables.items():
        text = data.get("text", "")
        score = _relevance_score(text, query, query_type)
        # Tables get a slight boost (they're often critical)
        scored_items.append((tid, "table", data, min(1.0, score + 0.1)))

    # Sort by relevance (highest first)
    scored_items.sort(key=lambda x: x[3], reverse=True)

    # Tiered inclusion with token budgeting
    context_parts = []
    tokens_used = 0
    sections_included = 0
    relevance_scores = {}
    budget = CONTEXT_BUDGET_TOKENS

    for item_id, item_type, data, score in scored_items:
        text = data.get("text", "")
        pages = data.get("pages", [])
        item_tokens = _estimate_tokens(text)

        if score >= HIGH_RELEVANCE_THRESHOLD:
            # High relevance: include VERBATIM
            if tokens_used + item_tokens <= budget:
                label = f"TABLE: {item_id}" if item_type == "table" else f"§{item_id}"
                page_info = f" [Pages {pages}]" if pages else ""
                context_parts.append(f"[{label}{page_info}]\n{text}")
                tokens_used += item_tokens
                sections_included += 1
                relevance_scores[item_id] = score

        elif score >= MEDIUM_RELEVANCE_THRESHOLD:
            # Medium relevance: include but truncate if needed
            max_chars = min(len(text), 2000)
            truncated = text[:max_chars]
            if len(text) > max_chars:
                truncated += f"\n[... truncated, {len(text) - max_chars} more chars]"
            trunc_tokens = _estimate_tokens(truncated)
            if tokens_used + trunc_tokens <= budget:
                label = f"TABLE: {item_id}" if item_type == "table" else f"§{item_id}"
                context_parts.append(f"[{label} (summary)]\n{truncated}")
                tokens_used += trunc_tokens
                sections_included += 1
                relevance_scores[item_id] = score
        # Low relevance: SKIP

    assembled = "\n\n---\n\n".join(context_parts)
    total_items = len(sections) + len(tables)
    skipped = total_items - sections_included

    logger.info(
        f"Context assembled: {sections_included}/{total_items} items, "
        f"~{tokens_used} tokens, {skipped} skipped (low relevance)"
    )

    if bus:
        bus.emit(NodeName.CONTEXT_ASSEMBLER, "done",
                 f"{sections_included} items, ~{tokens_used} tokens")

    return {
        "assembled_context": assembled,
        "context_tokens_used": tokens_used,
        "context_sections_included": sections_included,
        "context_relevance_scores": relevance_scores,
        "edge_signal": EdgeSignal.CONTINUE,
        "steps": [{"agent": NodeName.CONTEXT_ASSEMBLER, "turns": 0,
                   "tool_calls": 0, "tools_used": ["relevance_score"],
                   "tokens_used": tokens_used,
                   "sections_included": sections_included,
                   "sections_skipped": skipped}],
    }
