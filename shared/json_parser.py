"""
Shared JSON parser — single implementation for all LLM JSON parsing.

Handles: markdown fences, leading text, trailing garbage.
Used by: extractor.py, manifest_builder.py, vision_table.py.
"""
import json
import re


def parse_llm_json(text: str) -> dict | None:
    """Parse JSON from LLM response, handling common LLM output patterns."""
    if not text or not isinstance(text, str):
        return None

    cleaned = text.strip()

    # Strip markdown code fences
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[-1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    cleaned = cleaned.strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try regex for fenced JSON
    match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Try finding outermost braces
    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end])
        except (json.JSONDecodeError, ValueError):
            pass

    return None
