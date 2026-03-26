# Protocol Engine — Self-Audit (Honest)

Comprehensive audit of all 37 files in `protocol_engine/` by severity.

---

## CRITICAL (7) — Will cause data loss or silent failures

### C1. Explorer erases prior content on cycle re-entry
**File:** `graph/nodes/explorer.py:354`
```python
secs, tbls = {}, {}  # Empty dicts — but state has _merge_dicts reducer
```
On NEED_MORE_CONTENT cycles, returning empty dicts through `_merge_dicts` reducer means previously gathered sections/tables are **erased** (empty dict merges over existing). Explorer must start from existing state:
```python
secs = dict(state.get("sections_content", {}))
tbls = dict(state.get("tables_content", {}))
```

### C2. Missing return in `route_after_reviewer()` — implicit fallthrough
**File:** `graph/edges.py:84-93`
```python
if edge_signal == EdgeSignal.NEED_MORE_CONTENT:
    if cycle_count < MAX_CYCLES:
        return "increment_cycle"
    logger.info("max cycles reached → END")
    # NO RETURN — falls through to NEED_REEXTRACT check
```
Missing `return END` after max-cycles log. Currently falls through to the next `if` block which checks `NEED_REEXTRACT`. If signal is `NEED_MORE_CONTENT`, the REEXTRACT check won't match and it falls to `return END` at the bottom — correct by accident, but fragile.

### C3. Tool state attached to function objects — fragile
**File:** `tools/registry.py:115-118`
```python
search.gathered = gathered
read_section.gathered = gathered
vision_extract.gathered = gathered
```
LangChain's `@tool` decorator wraps functions. The `gathered` attribute is attached to the inner function, but the explorer node accesses it via `t.gathered` on the tool object. This may not survive LangChain's tool wrapping/serialization. If it breaks, **zero content is collected** during agentic exploration.

### C4. Reconciler column alignment always succeeds (useless condition)
**File:** `ingestion/reconciler.py:183`
```python
if best_score >= 0.5 or best_v_idx < len(vision_headers):
```
`best_v_idx` is always `< len(vision_headers)` by construction (it's an enumeration index), so the OR makes the threshold check useless. Low-scoring column pairs are always mapped, producing garbage merges.
**Fix:** `if best_score >= 0.5:`

### C5. Extractor returns DONE on empty content (should be ERROR)
**File:** `graph/nodes/extractor.py:101`
```python
"edge_signal": EdgeSignal.DONE,
"error": "No content available for extraction",
```
Returns DONE (success signal) with an error message. Downstream sees "done" and stops. Should be `EdgeSignal.ERROR_FATAL`.

### C6. Validation overwrites on each cycle (no history)
**File:** `graph/nodes/extractor.py:185` + `models/state.py:100`
```python
"validation": val_result,  # Overwrites — no reducer on this field
```
State field `validation: dict` has no reducer, so each extraction attempt replaces the previous validation. On retry cycles, old validation results are lost.

### C7. Vision table column mismatch
**File:** `tools/vision.py:165-167`
```python
while len(values) < len(columns) - 1:
    values.append("")
cells = [proc] + values[:len(columns) - 1]
```
Assumes first column is always "procedure". If vision returns a table where the first column isn't procedure, data gets misaligned. Also silently drops values if row has more entries than columns.

---

## HIGH (10) — Wrong behavior or dead features

### H1. `needs_planner` variable computed but never used
**File:** `graph/nodes/router.py:63`
```python
needs_planner = _is_complex_query(query, query_type)  # assigned, never read
```
The function `_is_complex_query()` (lines 150-161) is also dead code.

### H2. Reviewer and Reconciler hardcode `cycle_count < 2` instead of using MAX_CYCLES
**Files:** `graph/nodes/reviewer.py:192`, `graph/nodes/reconciler.py:90`
```python
if critical_signals and cycle_count < 2:  # Should be MAX_CYCLES
```
Config `MAX_CYCLES` is imported by edges.py but these nodes ignore it.

### H3. NEED_VISION edge signal never triggered
**File:** `models/enums.py:38` + `graph/edges.py:78`
EdgeSignal.NEED_VISION exists, edges.py has a handler for it, but no node ever sets it. Incomplete feature.

### H4. PLAN_NEXT edge signal never triggered
**File:** `models/enums.py:39`
EdgeSignal.PLAN_NEXT exists but no node ever sets it. Planner sub-task tracking is also dead — `current_task_index` and `sub_tasks[].completed` are never updated.

### H5. Explorer: no dedup for repeated tool calls
**File:** `graph/nodes/explorer.py:323-346`
Agentic loop runs up to MAX_EXPLORER_TURNS (8) but doesn't detect if the LLM is making the same search query repeatedly. Budget exhausted without progress.

### H6. Reviewer doesn't validate signal_type parameter
**File:** `graph/nodes/reviewer.py:121-130`
`flag_signal(signal_type, severity, ...)` accepts any string. LLM can pass invalid types like `"important"` and they get stored in state without validation.

### H7. Reconciler imports validate() but never calls it
**File:** `graph/nodes/reconciler.py:23`
```python
from protocol_engine.validation.validator import validate  # imported, never called
```
Docstring says "Run deterministic validation" but code only reads existing validation from state.

### H8. No signal deduplication across nodes
Both Reconciler and Reviewer append signals to state via `operator.add` reducer. Same issue flagged by both nodes appears twice in output. No dedup.

### H9. Footnote refs overwritten in table_extractor
**File:** `ingestion/table_extractor.py:248`
```python
refs.extend(m)  # line 243 — appends single-char refs
# ...
refs = parts     # line 248 — REPLACES entire list
```
If both patterns match, only the second is kept.

### H10. Chunk size doesn't account for context prefix
**File:** `ingestion/chunker.py:77`
```python
if len(content) <= chunk_size_chars * 2:  # doesn't include context
```
Context prefix is prepended on line 79 but not included in the size check. Chunks exceed target by the context size (~50-200 tokens).

---

## MEDIUM (12) — Suboptimal but functional

### M1. Token estimation is rough (÷4)
**File:** `graph/nodes/context_assembler.py:30-32`
`len(text) // 4` is a 15-30% inaccurate approximation for medical text. Over-estimates, causing under-utilization of context budget.

### M2. Hardcoded medium-relevance truncation (2000 chars)
**File:** `graph/nodes/context_assembler.py:146`

### M3. Explorer character budget (120000) disconnected from token config
**File:** `tools/registry.py:53` — hardcoded `budget = 120000` chars, but config uses `CONTEXT_BUDGET_TOKENS`.

### M4. Silent reranker fallback with no degradation warning
**File:** `retrieval/engine.py:151-160` — Falls back to old ms-marco-MiniLM silently, then to None silently.

### M5. `json.loads()` without try/except in explorer tools
**File:** `tools/registry.py:71` — `pages = json.loads(meta.get("pages", "[]"))` can crash on malformed metadata.

### M6. Page marker extraction only handles 2 formats
**File:** `validation/validator.py:167-180` — Misses `Page 5:`, `p. 5`, `(page 5)` formats.

### M7. Section ID matching too loose
**File:** `validation/validator.py:75-83` — `section_id + "."` matches substrings (section "5" matches "15.3").

### M8. Empty string in valid SDTM domains set
**File:** `validation/validator.py:204` — `""` in valid_domains means undefined domains pass silently.

### M9. Increment cycle nodes use string names, not NodeName enum
**File:** `graph/builder.py:57-58` — `"increment_cycle"` and `"increment_cycle_extractor"` are bare strings.

### M10. Error field never cleared on success
**File:** All nodes — `error` field in state persists across nodes. Successful nodes don't clear it.

### M11. Late imports inside functions
**File:** `graph/nodes/extractor.py:119, 201` — `from protocol_engine.models.schemas import SCHEMA_MAP` inside function body, executes on every call.

### M12. Table rows silently truncated to 100 in chunker
**File:** `ingestion/chunker.py:150` — `for row in rows[:100]` silently drops data.

---

## LOW (7) — Cosmetic or minor

### L1. Unused import: `VISION_CALL_LIMIT` in explorer.py:25
### L2. Unused import: `hashlib` in store.py:14
### L3. Unused import: `json` in chunker.py:14
### L4. Non-deterministic retrieval order (ThreadPoolExecutor + as_completed)
**File:** `graph/nodes/explorer.py:211-214`
### L5. Inconsistent `insufficient_data` field style in schemas
**File:** `models/schemas.py` — Some use `Field(default=False)`, others `= False`.
### L6. No recursion depth limit in `_find_groundings`
**File:** `validation/validator.py:223`
### L7. Markdown separator malformed in vision output
**File:** `tools/vision.py:159-160` — `"|---|---|"` should be `"| --- | --- |"`.

---

## Summary

| Severity | Count | Key Theme |
|----------|-------|-----------|
| CRITICAL | 7 | State loss, broken edges, fragile tool state |
| HIGH | 10 | Dead code, incomplete features, missing validation |
| MEDIUM | 12 | Hardcoded values, loose matching, silent fallbacks |
| LOW | 7 | Unused imports, cosmetic issues |
| **TOTAL** | **36** | |

## Top 5 Fixes (by impact)

1. **C1** — Explorer state merge on re-entry (causes data loss every cycle)
2. **C3** — Tool state attachment redesign (core mechanism for content gathering)
3. **C4** — Reconciler column alignment condition (makes reconciliation useless)
4. **C2** — Missing return in edge routing (accidental correctness)
5. **H2** — Use MAX_CYCLES config instead of hardcoded 2
