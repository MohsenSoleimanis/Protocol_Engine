"""
Domain classification — heuristic fallback for ManifestBuilder.

Used by: manifest_builder.py (as fallback when LLM enrichment fails).
The LlamaIndex retriever does NOT use domain filtering — semantic search
handles relevance naturally.
"""

DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "endpoints":    ["endpoint", "objective", "efficacy"],
    "eligibility":  ["eligib", "inclusion", "exclusion", "population"],
    "soa":          ["schedule", "activities", "event", "visit", "appendix 1"],
    "safety":       ["safety", "adverse", "aesi", "monitoring", "stopping"],
    "statistical":  ["statistic", "analysis", "sample size", "power"],
    "study_design": ["design", "rationale", "phase", "randomiz", "blind"],
    "intervention": ["intervention", "treatment", "dosage", "dose", "drug", "vaccine"],
    "overview":     ["synopsis", "summary", "overview", "introduction"],
    "regulatory":   ["regulatory", "ethical", "irb", "consent", "compliance"],
    "administrative": ["abbrevi", "definition", "reference", "appendix"],
}


def classify_domain(title: str) -> str:
    """Classify a section title into a domain."""
    t = title.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(kw in t for kw in keywords):
            return domain
    return "administrative"
