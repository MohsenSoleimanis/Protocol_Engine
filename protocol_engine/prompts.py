"""
Prompt loader — loads externalized prompts from YAML files.

Per the report: "Prompts must be maintained in a centralized, version-controlled
registry... decoupled from the application runtime."
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


@lru_cache(maxsize=16)
def load_prompt(name: str) -> dict:
    """Load a prompt template by name. Returns dict with 'system', 'user', etc."""
    path = PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        logger.warning(f"Prompt file not found: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def render(name: str, section: str = "system", **kwargs) -> str:
    """Load and render a prompt template with variables.

    Usage:
        render("extractor", "system", query_type="eligibility", appendix="...")
        render("reviewer", "user", query_type="safety", verified=5, total=8, failed=1)
    """
    data = load_prompt(name)
    template = data.get(section, "")
    if not template:
        return ""
    try:
        return template.format(**kwargs)
    except KeyError as e:
        logger.warning(f"Missing variable {e} in prompt '{name}.{section}'")
        # Return template with unfilled variables rather than crash
        return template
