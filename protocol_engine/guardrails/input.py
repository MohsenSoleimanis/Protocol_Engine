"""
Input guardrails — sanitize and validate all inputs before they reach the LLM.

Prevents:
  - Prompt injection attacks
  - Excessively long inputs that waste tokens
  - Malformed query types
"""
from __future__ import annotations

import re
import logging

from protocol_engine.models.enums import QueryType

logger = logging.getLogger(__name__)

# Known prompt injection patterns
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.I),
    re.compile(r"you\s+are\s+now\s+a", re.I),
    re.compile(r"system\s*:\s*", re.I),
    re.compile(r"<\s*/?\s*system\s*>", re.I),
    re.compile(r"\[INST\]", re.I),
]

MAX_QUERY_LENGTH = 2000
MAX_CONTEXT_LENGTH = 500_000


def sanitize_input(query: str, query_type: str, context: str = "") -> tuple[str, str, list[str]]:
    """Sanitize inputs. Returns (clean_query, clean_type, warnings).

    Raises ValueError for dangerous inputs rather than silently modifying.
    """
    warnings = []

    # Validate query type
    valid_types = {qt.value for qt in QueryType}
    if query_type not in valid_types:
        warnings.append(f"Unknown query_type '{query_type}', defaulting to 'general'")
        query_type = "general"

    # Check query length
    if len(query) > MAX_QUERY_LENGTH:
        warnings.append(f"Query truncated from {len(query)} to {MAX_QUERY_LENGTH} chars")
        query = query[:MAX_QUERY_LENGTH]

    # Check for injection patterns
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(query):
            raise ValueError(f"Potential prompt injection detected in query: {pattern.pattern}")

    # Check context length
    if context and len(context) > MAX_CONTEXT_LENGTH:
        warnings.append(f"Context exceeds {MAX_CONTEXT_LENGTH} chars — may be truncated downstream")

    # Strip control characters
    query = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', query)

    return query, query_type, warnings
