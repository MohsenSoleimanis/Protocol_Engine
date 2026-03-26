"""
Query Type Registry — single source of truth.

Every module reads from this registry instead of maintaining its own
hardcoded dicts. Adding a new query type = one new entry here.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class QueryTypeConfig:
    """Configuration for a single query type."""
    name: str
    schema_name: str                      # key in SCHEMA_MAP
    goal: str                             # Explorer's primary search goal
    domains: list[str]                    # metadata filter for retrieval
    retrieval_queries: list[str] = field(default_factory=list)  # additional focused queries using protocol vocabulary
    allow_cycles: bool = False            # can Explorer<->Extractor cycle?
    max_cycles: int = 1
    detection_keywords: list[str] = field(default_factory=list)
    appendix_key: str = ""                # key in KNOWLEDGE_APPENDICES (if any)


REGISTRY: dict[str, QueryTypeConfig] = {
    "endpoints": QueryTypeConfig(
        name="endpoints",
        schema_name="endpoints",
        goal="Find ALL endpoint content: primary, secondary, exploratory objectives, case definitions, thresholds, timing.",
        domains=["endpoints", "statistical"],
        retrieval_queries=[
            "objectives and endpoints primary secondary exploratory",
            "estimand endpoint definition population intercurrent events",
            "efficacy endpoint analysis assessment timepoint",
        ],
        allow_cycles=True,
        detection_keywords=["endpoint", "objective", "efficacy", "primary", "secondary", "exploratory"],
    ),
    "eligibility": QueryTypeConfig(
        name="eligibility",
        schema_name="eligibility",
        goal="Find ALL eligibility criteria: inclusion and exclusion. Follow cross-references to appendices.",
        domains=["eligibility"],
        retrieval_queries=[
            "inclusion criteria participants eligible enrollment",
            "exclusion criteria medical conditions disqualification",
            "lifestyle considerations contraception restrictions",
        ],
        allow_cycles=True,
        max_cycles=1,
        detection_keywords=["eligib", "inclusion", "exclusion", "enroll", "criteria"],
    ),
    "safety": QueryTypeConfig(
        name="safety",
        schema_name="safety",
        goal="Find ALL safety content: AESIs, monitoring rules, stopping rules, AE collection windows, SAE reporting.",
        domains=["safety", "soa", "intervention"],
        retrieval_queries=[
            "adverse events serious adverse events reporting monitoring",
            "adverse events special interest AESI definition criteria",
            "stopping rules discontinuation safety monitoring committee",
            "AE collection window solicited unsolicited diary visit",
        ],
        allow_cycles=True,
        max_cycles=1,
        detection_keywords=["safety", "adverse", "ae", "sae", "aesi", "monitoring", "stopping"],
    ),
    "deviation": QueryTypeConfig(
        name="deviation",
        schema_name="deviation",
        goal="Find eligibility criteria AND Schedule of Activities. Both needed for deviation rules.",
        domains=["eligibility", "soa"],
        allow_cycles=True,
        max_cycles=1,
        detection_keywords=["deviation", "violation", "protocol deviation", "rule"],
        appendix_key="DeviationRuleSet",
    ),
    "soa": QueryTypeConfig(
        name="soa",
        schema_name="soa",
        goal="Find ALL Schedule of Activities tables. Process them in PAGE ORDER "
             "(lowest page first = vaccination/treatment phase). "
             "Use vision_extract for each table. Extract each table separately.",
        domains=["soa", "administrative"],
        allow_cycles=True,
        max_cycles=1,
        detection_keywords=["schedule", "activities", "soa", "soe", "visit"],
    ),
    "study_design": QueryTypeConfig(
        name="study_design",
        schema_name="study_design",
        goal="Find study design: phase, randomization, blinding, stratification, sample size, dosing, arms, intervention details, interim analyses.",
        domains=["study_design", "intervention", "statistical"],
        retrieval_queries=[
            "overall design randomization blinding stratification parallel",
            "sample size determination enrollment participants screened",
            "study intervention dose route administration treatment arms placebo",
            "interim analysis statistical primary analysis timing",
        ],
        allow_cycles=True,
        detection_keywords=["design", "phase", "randomiz", "blind", "stratif", "arm"],
    ),
    "risk": QueryTypeConfig(
        name="risk",
        schema_name="risk",
        goal="Find safety monitoring, endpoint definitions, schedule of activities.",
        domains=["safety", "eligibility", "soa"],
        retrieval_queries=[
            "safety monitoring adverse events stopping rules",
            "endpoint definition efficacy assessment criteria",
            "schedule activities visit procedures assessments",
        ],
        allow_cycles=True,
        max_cycles=1,
        detection_keywords=["risk", "gap", "signal", "monitor"],
    ),
    "ambiguity": QueryTypeConfig(
        name="ambiguity",
        schema_name="ambiguity",
        goal="Find eligibility criteria, safety definitions. Look for undefined terms.",
        domains=["eligibility", "safety"],
        allow_cycles=True,
        detection_keywords=["ambig", "vague", "unclear", "undefined", "subjective"],
    ),
    "consistency": QueryTypeConfig(
        name="consistency",
        schema_name="consistency",
        goal="Find synopsis, endpoint definitions, AND statistical methods. Compare across sections.",
        domains=["endpoints", "statistical", "overview"],
        retrieval_queries=[
            "synopsis protocol summary objectives endpoints",
            "statistical analysis primary secondary endpoint",
            "sample size determination power efficacy",
        ],
        allow_cycles=True,
        max_cycles=1,
        detection_keywords=["consisten", "mismatch", "contradict", "compare"],
    ),
    "intervention": QueryTypeConfig(
        name="intervention",
        schema_name="intervention",
        goal="Find study intervention details: drug/vaccine name, dose, route, formulation, "
             "comparator, storage, dose modifications, prohibited medications.",
        domains=["intervention"],
        retrieval_queries=[
            "investigational product dose formulation route administration",
            "preparation handling storage accountability",
            "concomitant therapy prohibited permitted medications",
            "dose modification discontinuation treatment compliance",
        ],
        allow_cycles=True,
        detection_keywords=["intervention", "drug", "dose", "dosing", "vaccine", "placebo",
                           "comparator", "formulation", "concomitant", "prohibited"],
    ),
    "statistical": QueryTypeConfig(
        name="statistical",
        schema_name="statistical",
        goal="Find statistical design: sample size, power, analysis populations, "
             "interim analyses, multiplicity, missing data strategy.",
        domains=["statistical"],
        retrieval_queries=[
            "sample size determination power calculation assumptions",
            "analysis populations intent-to-treat per-protocol safety",
            "interim analysis data monitoring committee multiplicity",
            "missing data handling sensitivity analysis",
        ],
        allow_cycles=True,
        detection_keywords=["statistic", "sample size", "power", "interim", "multiplicity",
                           "analysis population", "itt", "per-protocol"],
    ),
    "kri": QueryTypeConfig(
        name="kri",
        schema_name="kri",
        goal="Derive Key Risk Indicators from the protocol: screening failure rate, "
             "AE reporting rate, visit compliance, protocol deviation rate. "
             "Map each to SDTM data sources and Quality Tolerance Limits.",
        domains=["eligibility", "safety", "soa", "study_design"],
        retrieval_queries=[
            "screening failure enrollment inclusion exclusion criteria",
            "adverse event reporting rate collection monitoring safety",
            "visit schedule compliance assessment window procedures",
            "protocol deviation violation eligibility criteria",
        ],
        allow_cycles=True,
        max_cycles=1,
        detection_keywords=["kri", "key risk", "risk indicator", "quality tolerance",
                           "qtl", "rbqm", "smart", "signal detection"],
    ),
    "general": QueryTypeConfig(
        name="general",
        schema_name="general",
        goal="",
        domains=[],
        allow_cycles=False,
        detection_keywords=[],
    ),
}


def get_config(query_type: str) -> QueryTypeConfig:
    """Get config for a query type, falling back to 'general'."""
    return REGISTRY.get(query_type, REGISTRY["general"])
