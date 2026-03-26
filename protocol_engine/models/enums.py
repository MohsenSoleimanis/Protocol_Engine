"""Minimal enums — only what's actually used."""
from __future__ import annotations
from enum import Enum


class QueryType(str, Enum):
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
    DONE = "done"
    NEED_MORE = "need_more"
