"""
Guardrails — deterministic safety layer between LLM and output.

Per the report: "Guardrail frameworks operate as a deterministic intermediate
layer between the probabilistic foundational model and the user or system interfaces."

For clinical protocol extraction:
  - Input: sanitize queries, block injection attempts
  - Output: validate clinical values, redact PII, check schema compliance
"""
from protocol_engine.guardrails.input import sanitize_input
from protocol_engine.guardrails.output import validate_output
