# Evaluation Framework

Per the report: "Evaluation must target system-level behavior over time,
assessing planning quality, tool selection accuracy, contextual groundedness,
and resilience."

## Scoring Dimensions

1. **Extraction Completeness** — Did we extract all items from the source?
2. **Grounding Accuracy** — Are source_text quotes actually in the context?
3. **Numerical Fidelity** — Do numbers match between extraction and source?
4. **Clinical Plausibility** — Are values in sane clinical ranges?
5. **Retrieval Recall** — Did we find all relevant sections?

## Running Evals

```bash
python -m evals.score --pdf output/protocol.json --query-type eligibility
```

## Adding Test Cases

Add JSON test cases to `evals/cases/`. Each file should contain:
```json
{
  "query": "Extract all eligibility criteria",
  "query_type": "eligibility",
  "expected": {
    "inclusion_count_min": 5,
    "exclusion_count_min": 3,
    "must_contain": ["age", "consent"]
  }
}
```
