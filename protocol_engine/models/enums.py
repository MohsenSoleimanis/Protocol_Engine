"""
Protocol Engine — Typed enums replacing all string-based routing.

EdgeSignal replaces "NEED_MORE:" string parsing.
QueryType replaces raw string matching.
NodeName replaces hardcoded node name strings.
"""
from __future__ import annotations
from enum import Enum


class QueryType(str, Enum):
    """All supported query types — maps 1:1 to extraction schemas."""
    STUDY_DESIGN = "study_design"
    ENDPOINTS = "endpoints"
    ELIGIBILITY = "eligibility"
    INTERVENTION = "intervention"
    SOA = "soa"
    SAFETY = "safety"
    STATISTICAL = "statistical"
    DEVIATION = "deviation"
    KRI = "kri"
    RISK = "risk"
    AMBIGUITY = "ambiguity"
    CONSISTENCY = "consistency"
    GENERAL = "general"


class EdgeSignal(str, Enum):
    """Typed routing signals between graph nodes.

    Replaces the old string-based 'NEED_MORE:' pattern.
    Each signal maps to exactly one edge in the graph.
    """
    CONTINUE = "continue"             # Move to the next node in sequence
    NEED_MORE_CONTENT = "need_more"   # Reviewer/Reconciler → Explorer
    NEED_REEXTRACT = "reextract"      # Reviewer → Extractor (same content, retry)
    NEED_VISION = "vision"            # Reconciler → Vision extraction
    PLAN_NEXT = "plan_next"           # Back to Planner for next sub-task
    DONE = "done"                     # → END
    ERROR_RETRY = "retry"             # Transient error → retry same node
    ERROR_FATAL = "fatal"             # → END with error


class NodeName(str, Enum):
    """All graph node names — avoids typos like 'Gatherer' vs 'Explorer'."""
    ROUTER = "router"
    PLANNER = "planner"
    EXPLORER = "explorer"
    CONTEXT_ASSEMBLER = "context_assembler"
    EXTRACTOR = "extractor"
    RECONCILER = "reconciler"
    REVIEWER = "reviewer"
