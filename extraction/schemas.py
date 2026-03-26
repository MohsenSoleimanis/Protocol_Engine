"""
Extraction Schemas — Pydantic models for structured LLM output.

Grounded in international standards:
  - ICH M11 CeSHarP: Protocol section structure and required content
  - ICH E6(R3): GCP, RBQM, quality tolerance limits
  - ICH E8(R1): Quality by design, critical to quality factors
  - ICH E9(R1): Estimand framework for endpoints
  - ICH E2A: Safety reporting timelines
  - CDISC SDTM v2.0: Domain/variable definitions
  - CDISC CT: Controlled terminology (NCI C-codes where applicable)
  - CDISC USDM v4.0: Study definition model
  - TransCelerate RBQM: KRI framework

Used with LangChain with_structured_output(). Class docstrings and Field
descriptions become function/tool metadata sent to the LLM.
"""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════
# GROUNDING — Cross-cutting provenance model
# ═══════════════════════════════════════════════════════════════════

class Grounding(BaseModel):
    """Source provenance per ICH E6(R3) §6.0 source data verification.
    Every extracted claim must be traceable to a specific protocol page and section."""
    section_id: str = Field(
        default="",
        description="ICH M11 section number, e.g. '3.1', '5.1.1', '8.3.4'. "
        "Use the exact section numbering from the protocol."
    )
    page: int = Field(
        default=-1,
        description="Page number from [Page N] markers in the context. Must be > 0."
    )
    exact_source_text: str = Field(
        default="",
        description="Verbatim quote from the protocol, 10-80 characters. Copy exactly as written."
    )
    confidence: float = Field(
        default=0.5,
        description="0.9 = verbatim match found in text, 0.7 = paraphrased but numbers/terms match, "
        "0.5 = inferred from surrounding context."
    )
    is_inferred: bool = Field(
        default=False,
        description="True if derived/inferred rather than directly stated."
    )


# ═══════════════════════════════════════════════════════════════════
# 1. STUDY DESIGN — ICH M11 §4 (Overall Design and Plan of the Study)
# ═══════════════════════════════════════════════════════════════════

class StudyDesignExtraction(BaseModel):
    """Overall study design per ICH M11 §4. Includes phase, randomization, blinding,
    stratification, arms, and population. Statistical design is separate (§9)."""
    design_type: str = Field(
        description="Per CDISC CT C66731 (Trial Type): describe the overall design, "
        "e.g. 'Randomized, stratified, double-blind, placebo-controlled, parallel-group'."
    )
    phase: Literal[
        "Phase I", "Phase I/II", "Phase II", "Phase II/III",
        "Phase III", "Phase III/IV", "Phase IV", "Not Applicable"
    ] = Field(
        description="CDISC CT C49666 (Study Phase). Use the exact phase from the protocol."
    )
    randomization_ratio: str = Field(
        default="",
        description="Per ICH E9 §2.3: randomization ratio, e.g. '1:1', '2:1:1'."
    )
    blinding: Literal[
        "Open Label", "Single Blind", "Double Blind",
        "Triple Blind", "Observer Blind", "Quadruple Blind"
    ] = Field(
        description="CDISC CT C49660 (Blinding Schema). "
        "Observer Blind = participants/site blinded, sponsor unblinded."
    )
    stratification_factors: list[str] = Field(
        default_factory=list,
        description="Per ICH E9 §2.3: all randomization stratification factors, "
        "e.g. 'Age group (18-64 vs ≥65 years)', 'At risk for severe disease (yes/no)'."
    )
    target_population: str = Field(
        default="",
        description="Per ICH M11 §5: population description including age range and key characteristics."
    )
    sample_size: str = Field(
        default="",
        description="Per ICH E9 §3.5: planned enrollment number with brief rationale."
    )
    study_duration: str = Field(
        default="",
        description="Total planned study duration for each participant."
    )
    dosing_schedule: str = Field(
        default="",
        description="Per ICH M11 §6.1: dose, route, interval, number of doses for all arms."
    )
    number_of_arms: int = Field(
        default=0,
        description="Number of treatment arms per SDTM TA (Trial Arms) domain."
    )
    arm_descriptions: list[str] = Field(
        default_factory=list,
        description="Per SDTM TA.ARM: description of each treatment arm."
    )
    primary_analysis_timing: str = Field(
        default="",
        description="Per ICH E9 §4: when the primary analysis occurs."
    )
    interim_analyses: list[str] = Field(
        default_factory=list,
        description="Per ICH E9 §4.5: planned interim analyses with timing."
    )
    grounding: Grounding = Field(default_factory=Grounding, description="Source provenance.")
    insufficient_data: bool = False
    gaps: list[str] = Field(default_factory=list, description="Missing information.")


# ═══════════════════════════════════════════════════════════════════
# 2. OBJECTIVES & ENDPOINTS — ICH M11 §3 + ICH E9(R1) Estimand
# ═══════════════════════════════════════════════════════════════════

class Endpoint(BaseModel):
    """A study endpoint per ICH M11 §3 and ICH E9(R1) estimand framework.
    Each endpoint must specify the variable, metric, aggregation, and timepoint."""
    id: str = Field(
        description="Endpoint ID: P1, P2 for primary; S1, S2 for secondary; E1, E2 for exploratory."
    )
    category: Literal[
        "Primary", "Primary Safety", "Secondary", "Exploratory"
    ] = Field(
        description="CDISC CT C98772/C98724: Primary, Secondary, or Exploratory per ICH M11 §3."
    )
    objective: str = Field(
        description="The study objective per ICH M11 §3: 'each objective should be classifiable "
        "as primary, secondary, or exploratory'."
    )
    endpoint: str = Field(
        default="",
        description="Full endpoint definition per ICH E9(R1): include the variable (what is measured), "
        "the analysis metric (how it is summarized), the method of aggregation, "
        "and the timepoint. Preserve ALL clinical thresholds."
    )
    population: str = Field(
        default="",
        description="CDISC CT C71106: analysis population, e.g. 'Intent-to-Treat (ITT)', "
        "'Modified ITT (mITT)', 'Per-Protocol (PP)', 'Safety Population'."
    )
    timing: str = Field(
        default="",
        description="Timepoint for endpoint assessment, e.g. 'Starting 14 days after second dose'."
    )
    grounding: Grounding = Field(default_factory=Grounding, description="Source provenance.")

class EndpointCounts(BaseModel):
    """Counts of endpoints per CDISC CT category."""
    primary: int = Field(default=0, description="Number of primary endpoints.")
    secondary: int = Field(default=0, description="Number of secondary endpoints.")
    exploratory: int = Field(default=0, description="Number of exploratory endpoints.")

class EndpointExtraction(BaseModel):
    """All study objectives and endpoints per ICH M11 §3.
    Each objective links to one or more endpoints. Extract ALL categories."""
    endpoints: list[Endpoint] = Field(
        default_factory=list,
        description="ALL endpoints: primary, secondary, AND exploratory. Do not stop at primary."
    )
    total: EndpointCounts = Field(
        default_factory=EndpointCounts,
        description="Counts per CDISC CT category."
    )
    insufficient_data: bool = Field(default=False)
    gaps: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# 3. ELIGIBILITY — ICH M11 §5 + SDTM IE/TI domains
# ═══════════════════════════════════════════════════════════════════

class Criterion(BaseModel):
    """A single inclusion or exclusion criterion per ICH M11 §5.1.
    Maps to SDTM TI (Trial Inclusion/Exclusion) domain."""
    id: str = Field(
        description="Criterion ID preserving protocol numbering per SDTM TI.IETESTCD, "
        "e.g. 'IC-1', 'IC-2', 'EC-1'. Use IC- for inclusion, EC- for exclusion."
    )
    category: str = Field(
        default="",
        description="Criterion category per SDTM IE.IECAT: demographic, medical_history, "
        "laboratory, reproductive, lifestyle, consent, vaccination_history, allergy, psychiatric."
    )
    text: str = Field(
        description="FULL criterion text per ICH M11 §5.1: complete and unambiguous. "
        "Include ALL sub-bullets (a), (b), (c), thresholds, and qualifiers."
    )
    automation_level: Literal["FULL", "PARTIAL", "MANUAL"] = Field(
        default="MANUAL",
        description="Per CDISC CORE concept: FULL = computable from SDTM data alone "
        "(e.g. DM.AGE >= 18). PARTIAL = needs data + clinical interpretation. "
        "MANUAL = purely subjective, no SDTM variable exists."
    )
    logic: str = Field(
        default="",
        description="SDTM variable logic for FULL/PARTIAL criteria. "
        "Syntax: DOMAIN.VARIABLE OPERATOR VALUE, "
        "e.g. 'DM.AGE >= 18', 'LB.LBSTRESC WHERE LBTESTCD = PREG = NEGATIVE'."
    )
    exceptions: list[str] = Field(
        default_factory=list,
        description="Explicit exceptions per ICH M11: protocol-stated qualifiers to the criterion."
    )
    cross_references: list[str] = Field(
        default_factory=list,
        description="References to other protocol sections, e.g. 'Appendix 11.3', 'Section 8.3.6'."
    )
    grounding: Grounding = Field(default_factory=Grounding, description="Source provenance.")

class EligibilitySummary(BaseModel):
    """Counts of criteria by automation level."""
    total: int = Field(default=0, description="Total criteria.")
    full_auto: int = Field(default=0, description="FULL automation criteria.")
    partial: int = Field(default=0, description="PARTIAL automation criteria.")
    manual: int = Field(default=0, description="MANUAL criteria.")

class EligibilityExtraction(BaseModel):
    """All eligibility criteria per ICH M11 §5.1. Maps to SDTM TI domain.
    Extract EVERY criterion — do not merge or skip any."""
    inclusion: list[Criterion] = Field(
        default_factory=list,
        description="All inclusion criteria. Per SDTM TI: IECAT = 'INCLUSION'."
    )
    exclusion: list[Criterion] = Field(
        default_factory=list,
        description="All exclusion criteria. Per SDTM TI: IECAT = 'EXCLUSION'."
    )
    summary: EligibilitySummary = Field(
        default_factory=EligibilitySummary,
        description="Counts by automation level."
    )
    insufficient_data: bool = Field(default=False)
    gaps: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# 4. INTERVENTION — ICH M11 §6.1 (Study Intervention)
# ═══════════════════════════════════════════════════════════════════

class StudyArm(BaseModel):
    """A treatment arm per SDTM TA (Trial Arms) domain."""
    arm_name: str = Field(description="Arm name per SDTM TA.ARM, e.g. 'mRNA-1273 100 µg'.")
    arm_type: Literal[
        "Experimental", "Active Comparator", "Placebo Comparator",
        "Sham Comparator", "No Intervention", "Other"
    ] = Field(
        description="CDISC CT C66767 (Arm Type)."
    )
    intervention_name: str = Field(default="", description="Drug/biologic name per SDTM EX.EXTRT.")
    dose: str = Field(default="", description="Dose amount and units, e.g. '100 µg', '0.5 mL'.")
    route: str = Field(
        default="",
        description="CDISC CT C66729 (Route of Administration): "
        "INTRAMUSCULAR, INTRAVENOUS, ORAL, SUBCUTANEOUS, TOPICAL, etc."
    )
    frequency: str = Field(default="", description="Dosing frequency, e.g. 'Day 1 and Day 29', 'Once daily'.")
    grounding: Grounding = Field(default_factory=Grounding, description="Source provenance.")

class InterventionExtraction(BaseModel):
    """Study intervention details per ICH M11 §6.1.
    Maps to SDTM EX (Exposure) domain."""
    arms: list[StudyArm] = Field(
        default_factory=list,
        description="All treatment arms including comparator/placebo."
    )
    formulation: str = Field(default="", description="Drug formulation and presentation.")
    storage: str = Field(default="", description="Storage requirements, e.g. '-20°C, protect from light'.")
    dose_modifications: list[str] = Field(
        default_factory=list,
        description="Dose modification rules per ICH M11 §6.1.3."
    )
    concomitant_medications: str = Field(
        default="",
        description="Prohibited/permitted concomitant meds per ICH M11 §6.2. "
        "Maps to SDTM CM domain for deviation detection."
    )
    grounding: Grounding = Field(default_factory=Grounding, description="Source provenance.")
    insufficient_data: bool = False
    gaps: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# 5. SCHEDULE OF ACTIVITIES — ICH M11 §6.4
# ═══════════════════════════════════════════════════════════════════

class SoAVisit(BaseModel):
    """A visit/timepoint per SDTM TV (Trial Visits) domain."""
    visit_name: str = Field(description="Visit name per SDTM TV.VISIT, e.g. 'Screening', 'Day 1'.")
    day: str = Field(default="", description="Study day per SDTM TV.VISITDY, e.g. 'Day -28 to -1', 'Day 29'.")
    window: str = Field(default="", description="Visit window per SDTM TV, e.g. '±3 days'.")
    visit_type: Literal["site_visit", "phone_call", "remote", "unscheduled"] = Field(
        default="site_visit", description="Visit type classification."
    )
    procedures: list[str] = Field(
        default_factory=list,
        description="All procedures at this visit per ICH M11 §6.4."
    )
    notes: str = Field(default="", description="Visit-specific notes or conditions.")
    grounding: Grounding = Field(default_factory=Grounding, description="Source provenance.")

class SoATable(BaseModel):
    """A single SoA table (protocols often have multiple: vaccination phase, surveillance, booster)."""
    title: str = Field(default="", description="Table title, e.g. 'Table 16: Vaccination Phase Day 1 to Day 29'.")
    study_part: str = Field(default="", description="Which study part this covers, e.g. 'Part A', 'Part C Booster'.")
    matrix_markdown: str = Field(
        default="",
        description="This table as markdown pipe table: procedures as rows, visits as columns, X for performed."
    )
    page_range: list[int] = Field(default_factory=list, description="PDF pages this table spans.")

class SoAExtraction(BaseModel):
    """Schedule of Activities per ICH M11 §6.4. Master table of visits × procedures.
    Protocols often have MULTIPLE SoA tables (vaccination phase, surveillance, booster, etc.).
    Extract each as a separate table entry."""
    tables: list[SoATable] = Field(
        default_factory=list,
        description="Each SoA table separately. Most protocols have 2-6 tables for different study phases."
    )
    # Keep matrix_markdown for backward compatibility with UI
    matrix_markdown: str = Field(
        default="",
        description="Combined SoA as markdown. If multiple tables, concatenate with headers."
    )
    visits: list[SoAVisit] = Field(default_factory=list, description="Structured visit data.")
    study_parts: list[str] = Field(default_factory=list, description="Study parts, e.g. 'Part A: Blinded'.")
    total_visits: int = Field(default=0, description="Total visits across all parts.")
    study_duration: str = Field(default="", description="Overall study duration.")
    footnotes: list[str] = Field(
        default_factory=list,
        description="ALL SoA footnotes — these contain critical conditional logic for deviation detection."
    )
    insufficient_data: bool = False
    gaps: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# 6. SAFETY MONITORING — ICH M11 §9 + ICH E2A
# ═══════════════════════════════════════════════════════════════════

class SafetyRule(BaseModel):
    """A safety reporting rule per ICH E2A and ICH E6(R3) §6.6."""
    id: str = Field(description="Rule ID, e.g. 'SR-1'.")
    description: str = Field(description="What this rule requires.")
    trigger: str = Field(
        default="",
        description="Event triggering this rule per ICH E2A, e.g. 'Any SAE', 'Grade ≥3 allergic reaction'."
    )
    action: str = Field(
        default="",
        description="Required action per ICH E2A: e.g. 'Report to sponsor within 24 hours', "
        "'Report to IRB within 5 days', 'Complete MedWatch 3500A'."
    )
    timeframe: str = Field(
        default="",
        description="Per ICH E2A: '24 hours for SAEs', '15 calendar days for ICSRs to regulatory', "
        "'7 days for fatal/life-threatening'."
    )
    grounding: Grounding = Field(default_factory=Grounding, description="Source provenance.")

class CollectionWindow(BaseModel):
    """AE collection window per ICH M11 §9.3 and SDTM AE domain."""
    name: str = Field(
        description="Window name per protocol and SDTM AE.AEPRESP: "
        "'Solicited Local ARs', 'Solicited Systemic ARs', 'Unsolicited AEs', 'SAEs'."
    )
    start_day: int = Field(default=0, description="Start day relative to dosing (Day 1 = first dose).")
    end_day: int = Field(default=0, description="End day relative to dosing.")
    population: str = Field(default="", description="Which participants: 'All', 'Part A only'.")
    detail: str = Field(default="", description="Collection method: 'eDiary', 'site visit', 'phone call'.")

class AESI(BaseModel):
    """Adverse Event of Special Interest per ICH E2A §III.B and protocol-specific definitions."""
    name: str = Field(
        description="AESI name using MedDRA preferred term, "
        "e.g. 'Myocarditis/Pericarditis', 'Anaphylaxis', 'Bell's Palsy'."
    )
    definition: str = Field(
        default="",
        description="Clinical definition and diagnostic criteria. "
        "For vaccines, reference Brighton Collaboration case definitions where applicable."
    )
    reporting_timeframe: str = Field(
        default="",
        description="Reporting window per ICH E2A, e.g. 'Throughout study and 2 years post-vaccination'."
    )
    grounding: Grounding = Field(default_factory=Grounding, description="Source provenance.")

class StoppingRule(BaseModel):
    """Study pause/stopping rule per ICH E9 §4.5 and DSMB charter."""
    description: str = Field(description="Full stopping rule text including specific threshold.")
    trigger_threshold: str = Field(
        default="",
        description="Statistical or clinical threshold, e.g. '≥3 confirmed myocarditis cases in vaccine group'."
    )
    decision_body: str = Field(
        default="",
        description="Per ICH E6(R3): who decides — 'DSMB', 'DMC', 'IDMC', 'Sponsor Safety Team'."
    )

class SafetyExtraction(BaseModel):
    """Safety monitoring extraction per ICH M11 §9, ICH E2A, and ICH E6(R3)."""
    monitoring_rules: list[SafetyRule] = Field(
        default_factory=list, description="All safety reporting/monitoring rules."
    )
    collection_windows: list[CollectionWindow] = Field(
        default_factory=list, description="AE collection timeframes."
    )
    aesis: list[AESI] = Field(
        default_factory=list, description="Adverse Events of Special Interest."
    )
    stopping_rules: list[StoppingRule] = Field(
        default_factory=list, description="Study pause/stopping rules."
    )
    insufficient_data: bool = False
    gaps: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# 7. STATISTICAL DESIGN — ICH M11 §10 + ICH E9
# ═══════════════════════════════════════════════════════════════════

class StatisticalDesignExtraction(BaseModel):
    """Statistical design per ICH M11 §10 and ICH E9.
    Sample size, analysis populations, interim analyses, multiplicity."""
    sample_size_target: int = Field(
        default=0,
        description="Per ICH E9 §3.5: total planned enrollment number."
    )
    sample_size_rationale: str = Field(
        default="",
        description="Per ICH E9 §3.5: power calculation basis, assumptions, clinically meaningful difference."
    )
    power: str = Field(default="", description="Statistical power, e.g. '90% power at one-sided alpha 0.025'.")
    primary_analysis_method: str = Field(
        default="",
        description="Primary statistical method, e.g. 'modified intention-to-treat with stratified log-rank test'."
    )
    analysis_populations: list[str] = Field(
        default_factory=list,
        description="CDISC CT C71106: ITT, mITT, Per-Protocol, Safety Population. "
        "Include definitions of each."
    )
    interim_analyses: list[str] = Field(
        default_factory=list,
        description="Per ICH E9 §4.5: timing, boundaries (O'Brien-Fleming, Lan-DeMets), spending functions."
    )
    multiplicity_adjustment: str = Field(
        default="",
        description="Per ICH E9 §2.2.5: method for controlling type I error across endpoints/analyses."
    )
    missing_data_handling: str = Field(
        default="",
        description="Per ICH E9(R1): strategy for intercurrent events and missing data."
    )
    grounding: Grounding = Field(default_factory=Grounding, description="Source provenance.")
    insufficient_data: bool = False
    gaps: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# 8. DEVIATION RULES — SDTM DV domain + CluePoints SMART
# Expanded: eligibility + dosing + visit compliance + prohibited meds
# ═══════════════════════════════════════════════════════════════════

class eCRFField(BaseModel):
    """An SDTM variable mapped to a criterion per SDTM v2.0."""
    variable: str = Field(
        description="SDTM variable path: DOMAIN.VARIABLE, "
        "e.g. 'DM.BRTHDTC', 'LB.LBSTRESC WHERE LBTESTCD = PREG'."
    )
    description: str = Field(description="SDTM variable label, e.g. 'Date/Time of Birth'.")

class ValidationRule(BaseModel):
    """A computable validation rule for CluePoints SMART engine or CDISC CORE."""
    rule_id: str = Field(description="Rule ID, e.g. 'VR-01a', 'DR-03b'.")
    rule_type: Literal["Hard", "Soft"] = Field(
        description="Hard = blocks enrollment, generates SCREEN FAILURE in SDTM DS.DSDECOD. "
        "Soft = generates query/KRI signal per ICH E6(R3) RBQM. "
        "Maps to CDISC CORE severity: Hard = 'Error', Soft = 'Warning'."
    )
    deviation_category: Literal[
        "eligibility", "dosing", "visit_compliance",
        "prohibited_medication", "safety_reporting", "consent", "assessment"
    ] = Field(
        default="eligibility",
        description="Per SDTM DV.DVCAT: category of protocol deviation."
    )
    logic: str = Field(
        description="SDTM-based computable logic, "
        "e.g. 'AGE = floor((RFSTDTC - BRTHDTC) / 365.25); AGE >= 18'."
    )
    check: str = Field(description="Human-readable check description.")
    action: str = Field(
        description="Per SDTM DS/DV: 'SCREEN FAILURE', 'MAJOR PROTOCOL DEVIATION', "
        "'QUERY', 'DOSING HOLD', 'STUDY DISCONTINUATION'."
    )

class CriterionValidation(BaseModel):
    """A protocol requirement mapped to eCRF fields and validation rules.
    Per SDTM DV (Protocol Deviations) domain."""
    criterion_id: str = Field(description="ID preserving protocol numbering, e.g. 'IC-01', 'DR-01'.")
    part: str = Field(default="All Parts", description="Study part applicability.")
    title: str = Field(description="Short title, e.g. 'Age ≥ 18 at consent'.")
    verbatim: str = Field(description="Exact protocol text — copy verbatim.")
    fields: list[eCRFField] = Field(
        default_factory=list,
        description="All SDTM variables needed to check this requirement."
    )
    sdtm_domains: list[str] = Field(
        default_factory=list,
        description="SDTM domains involved. Standard domain codes: "
        "DM, VS, LB, MH, CM, AE, EX, DS, IE, SC, DV, SV, TV, FA."
    )
    rules: list[ValidationRule] = Field(
        default_factory=list, description="Hard and Soft validation rules."
    )
    grounding: Grounding = Field(default_factory=Grounding, description="Source provenance.")

class DeviationRuleSet(BaseModel):
    """Protocol deviation rules per SDTM DV domain and CluePoints SMART engine.
    Covers eligibility violations, dosing deviations, visit window compliance,
    prohibited medications, and safety reporting deviations."""
    criteria: list[CriterionValidation] = Field(
        default_factory=list,
        description="All requirements with validation rules — eligibility, dosing, visit compliance, and more."
    )
    total_rules: int = Field(default=0, description="Total validation rules.")
    hard_rules: int = Field(default=0, description="Hard rules (block enrollment).")
    soft_rules: int = Field(default=0, description="Soft rules (generate queries/KRI signals).")


# ═══════════════════════════════════════════════════════════════════
# 9. KEY RISK INDICATORS — ICH E6(R3) RBQM + TransCelerate KRI
# ═══════════════════════════════════════════════════════════════════

class KeyRiskIndicator(BaseModel):
    """A KRI derived from the protocol per ICH E6(R3) §5.0 RBQM and
    TransCelerate RBQM framework. Feeds CluePoints SMART statistical signal detection."""
    kri_id: str = Field(description="KRI identifier, e.g. 'KRI-01', 'KRI-02'.")
    name: str = Field(
        description="KRI name per TransCelerate taxonomy: "
        "'Screening Failure Rate', 'AE Reporting Rate', 'Protocol Deviation Rate', "
        "'Visit Window Compliance', 'Informed Consent Timeliness', "
        "'SAE Reporting Timeliness', 'Query Response Time', 'Data Entry Timeliness'."
    )
    category: Literal[
        "enrollment", "safety_reporting", "data_quality",
        "visit_compliance", "protocol_deviation", "consent", "endpoint_assessment"
    ] = Field(
        description="KRI category per TransCelerate RBQM and ICH E6(R3) §5.0."
    )
    protocol_source: str = Field(
        description="Which protocol section defines this requirement, e.g. 'Section 5.1 Eligibility'."
    )
    sdtm_data_source: str = Field(
        description="SDTM domain and variables to compute this KRI, "
        "e.g. 'DS.DSDECOD = SCREEN FAILURE / total screened', "
        "'AE domain: count per subject per site', "
        "'SV vs TV: actual vs planned visit dates'."
    )
    metric: Literal["rate", "count", "proportion", "time_to_event", "mean", "median"] = Field(
        description="How to compute the KRI statistic."
    )
    threshold: str = Field(
        description="Quality Tolerance Limit per ICH E6(R3) §5.0.3. "
        "If the protocol specifies a threshold, use it exactly. "
        "If not, write 'SUGGESTED: [value]' to indicate this is a recommendation, not from the protocol."
    )
    threshold_source: Literal["protocol_specified", "industry_standard", "suggested"] = Field(
        default="suggested",
        description="protocol_specified = threshold is stated in the protocol text, "
        "industry_standard = common industry threshold (cite source), "
        "suggested = you are recommending a threshold (not from the protocol)."
    )
    signal_direction: Literal["above", "below", "both"] = Field(
        description="Which direction indicates a risk signal."
    )
    rationale: str = Field(
        description="Why this KRI matters for this specific protocol. "
        "Reference the specific protocol requirement this KRI monitors."
    )
    grounding: Grounding = Field(default_factory=Grounding, description="Source provenance.")

class KRIExtraction(BaseModel):
    """Key Risk Indicators per ICH E6(R3) RBQM framework.
    Each KRI maps a protocol requirement to a computable site-level metric.
    IMPORTANT: Only derive KRIs from actual protocol requirements you can cite.
    For QTL thresholds, distinguish between protocol-specified values and suggestions."""
    indicators: list[KeyRiskIndicator] = Field(
        default_factory=list,
        description="All KRIs derivable from this protocol."
    )
    total: int = Field(default=0, description="Total KRIs identified.")
    critical_count: int = Field(default=0, description="KRIs flagged as critical to quality.")


# ═══════════════════════════════════════════════════════════════════
# 10. PROTOCOL QUALITY — ICH E8(R1) Quality by Design
# (was: RiskAssessment — renamed to avoid confusion with KRI risk)
# ═══════════════════════════════════════════════════════════════════

class Risk(BaseModel):
    """A protocol quality risk per ICH E8(R1) quality by design framework.
    IMPORTANT: Only flag issues where you can point to SPECIFIC protocol text.
    Do NOT flag 'missing' items unless the protocol explicitly should contain them
    and you have searched the provided content thoroughly."""
    risk_id: str = Field(description="Risk ID, e.g. 'R-1'.")
    claim_type: Literal["present_problem", "absent_requirement"] = Field(
        default="present_problem",
        description="present_problem = the protocol text IS problematic (vague, inconsistent). "
        "You MUST quote the specific text. "
        "absent_requirement = something required is missing. "
        "Only use this if you searched the content and it's genuinely not there. "
        "Mark 'NOT_IN_CONTEXT' in text_flagged if you're unsure."
    )
    category: Literal[
        "vague_language", "inconsistency", "operability_concern",
        "missing_timeframe", "missing_procedure", "safety_gap"
    ] = Field(
        description="vague_language = undefined term (quote it), "
        "inconsistency = conflicting text in two locations (cite both), "
        "operability_concern = difficult to execute at site (explain why), "
        "missing_timeframe = timing not specified for a requirement, "
        "missing_procedure = SoA gap, safety_gap = missing safety element. "
        "For the last three: ONLY flag if you searched and confirmed absence."
    )
    severity: Literal["critical", "major", "minor"] = Field(
        description="critical = affects patient safety, major = affects data integrity, minor = cosmetic."
    )
    description: str = Field(description="What the risk is and why it matters.")
    text_flagged: str = Field(
        default="",
        description="For present_problem: the EXACT problematic protocol text (verbatim quote). "
        "For absent_requirement: describe what's missing and note 'NOT_IN_CONTEXT' if the "
        "content provided to you may not include all relevant sections."
    )
    recommendation: str = Field(default="", description="Suggested fix.")
    grounding: Grounding = Field(default_factory=Grounding, description="Source provenance.")

class RiskAssessment(BaseModel):
    """Protocol quality risks per ICH E8(R1). 
    FOCUS on issues you can PROVE from the text: vague language, inconsistencies, 
    operability concerns. Be cautious with 'missing' claims — the protocol may 
    contain the information in a section you haven't seen."""
    risks: list[Risk] = Field(default_factory=list, description="Verified protocol quality risks.")
    overall_level: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        default="MEDIUM", description="Overall protocol quality risk level."
    )


# ═══════════════════════════════════════════════════════════════════
# 11. AMBIGUITY ANALYSIS — ICH M11 principle: "complete, unambiguous"
# ═══════════════════════════════════════════════════════════════════

class AmbiguityFinding(BaseModel):
    """An ambiguous term per ICH M11 §1 principle that protocols must be
    'complete, unambiguous, well organised'. These block computable deviation rules."""
    finding_id: str = Field(description="Finding ID, e.g. 'A-1'.")
    category: Literal["deontic", "quantifier", "temporal", "subjective", "undefined_term"] = Field(
        description="deontic = investigator discretion (must/should/may per ICH E6), "
        "quantifier = undefined amount ('significant', 'adequate'), "
        "temporal = unclear timing ('recently', 'prior to'), "
        "subjective = opinion-based ('clinically significant'), "
        "undefined_term = used but not defined in protocol."
    )
    severity: Literal["critical", "major", "minor"] = Field(
        description="Per ICH E6(R3) QTL impact: critical = cannot create any deviation rule, "
        "major = partially computable, minor = edge case only."
    )
    text_flagged: str = Field(description="The exact ambiguous protocol text.")
    why_ambiguous: str = Field(description="Why two reasonable people could disagree about meaning.")
    suggested_clarification: str = Field(
        description="Specific suggestion with concrete thresholds or criteria to resolve the ambiguity."
    )
    impact_on_deviation_detection: str = Field(
        description="Impact on RBQM/SMART: can this become a CDISC CORE rule? "
        "FULL (computable) / PARTIAL (needs interpretation) / NONE (manual only)."
    )
    grounding: Grounding = Field(default_factory=Grounding, description="Source provenance.")

class AmbiguityAnalysis(BaseModel):
    """Ambiguity analysis per ICH M11 principle of 'complete, unambiguous' protocols.
    Identifies terms that block automated deviation detection."""
    findings: list[AmbiguityFinding] = Field(
        default_factory=list, description="All ambiguity findings."
    )
    total: int = Field(default=0, description="Total findings.")
    critical_count: int = Field(default=0, description="Critical-severity count.")


# ═══════════════════════════════════════════════════════════════════
# 12. CONSISTENCY CHECK — ICH M11 cross-section verification
# ═══════════════════════════════════════════════════════════════════

class SourceReference(BaseModel):
    """A specific protocol location for cross-reference comparison."""
    location: str = Field(description="ICH M11 section reference, e.g. 'Synopsis §1.1 p.12'.")
    text: str = Field(description="Relevant text at this location.")

class ConsistencyResult(BaseModel):
    """A comparison between two protocol sections that should agree per ICH M11."""
    check_id: str = Field(description="Check ID, e.g. 'CC-1'.")
    check_type: Literal[
        "synopsis_vs_detail", "endpoint_vs_stats", "soa_vs_procedures",
        "inclusion_vs_soa", "dosing_vs_soa", "safety_vs_soa", "other"
    ] = Field(
        description="Type of ICH M11 cross-reference check."
    )
    status: Literal["match", "flag", "mismatch"] = Field(
        description="match = sections consistent, flag = possibly inconsistent, "
        "mismatch = definitely contradictory."
    )
    description: str = Field(description="What was compared and what was found.")
    source_a: SourceReference = Field(default_factory=SourceReference)
    source_b: SourceReference = Field(default_factory=SourceReference)
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(default="HIGH")

class ConsistencyCheck(BaseModel):
    """Cross-section consistency checks per ICH M11 structure.
    Verifies synopsis matches body, endpoints match statistical plan, SoA matches procedures."""
    checks: list[ConsistencyResult] = Field(
        default_factory=list, description="All consistency checks."
    )
    pass_count: int = Field(default=0, description="Matching checks.")
    flag_count: int = Field(default=0, description="Flagged or mismatched checks.")


# ═══════════════════════════════════════════════════════════════════
# 13. GENERAL EXTRACTION — Free-form questions
# ═══════════════════════════════════════════════════════════════════

class GroundedClaim(BaseModel):
    """A single factual claim with source provenance."""
    statement: str = Field(description="The factual claim.")
    grounding: Grounding = Field(default_factory=Grounding, description="Source provenance.")

class GeneralExtraction(BaseModel):
    """General-purpose extraction for free-form protocol questions."""
    answer: str = Field(description="Comprehensive answer to the question.")
    claims: list[GroundedClaim] = Field(default_factory=list, description="Individual claims with grounding.")
    insufficient_data: bool = False
    gaps: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════
# SCHEMA MAP — query_type string → Pydantic class
# ═══════════════════════════════════════════════════════════════════

SCHEMA_MAP = {
    # Protocol extraction (ICH M11 sections)
    "study_design": StudyDesignExtraction,       # §4
    "endpoints": EndpointExtraction,             # §3
    "eligibility": EligibilityExtraction,        # §5
    "intervention": InterventionExtraction,      # §6.1  NEW
    "soa": SoAExtraction,                        # §6.4
    "safety": SafetyExtraction,                  # §9
    "statistical": StatisticalDesignExtraction,  # §10   NEW

    # RBQM/CluePoints analysis
    "deviation": DeviationRuleSet,
    "kri": KRIExtraction,                        # NEW — core CluePoints
    "risk": RiskAssessment,                      # Protocol quality
    "ambiguity": AmbiguityAnalysis,
    "consistency": ConsistencyCheck,

    # General
    "general": GeneralExtraction,
}
