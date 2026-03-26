# Protocol Engine — Complete Code Audit

> **Date**: 2026-03-26
> **Scope**: Line-by-line review of all 21 modules (~5,100 lines)
> **Findings**: 30 P0/P1 bugs, 40+ P2 issues, 25+ code smells

---

## P0 — CRITICAL BUGS (Data Corruption / Silent Data Loss)

### 1. Cell bboxes are fabricated — every table's spatial data is wrong
**File:** `ingestion/src/table_extractor.py:134-164`

The `_get_cell_bboxes` function's `try` block is a no-op (`pass`). It **always** falls through to a "fallback" that assumes a **uniform grid** (equal-width columns, equal-height rows). This means:
- Every cell bbox is wrong for any real table
- `_exclude_table_spans` relies on these bboxes to remove text overlapping tables — wrong bboxes cause body text to leak into tables AND real text near tables to be swallowed
- The confidence score of 0.92 is a lie

### 2. Lists and paragraphs are double-emitted
**File:** `ingestion/__init__.py:190-194`, `ingestion/run.py:239-248`

```python
lists = discover_lists_in_spans(non_table_spans, ...)
content_blocks.extend(lists)
paragraphs = _group_into_paragraphs(non_table_spans, ...)  # SAME spans!
content_blocks.extend(paragraphs)
```

Every bulleted/numbered list item appears **twice** in the output — once as a list item and again as a paragraph. The spans consumed by list detection are never filtered before the paragraph pass.

### 3. Continuation merge row index is off-by-one
**File:** `ingestion/src/continuation_merger.py:174-186`

When `skip_first_row=True`, the offset subtracts 1 **twice**: once in `actual_row` and once in the final expression. All merged continuation table cells map to the wrong rows. Silent data corruption on every multi-page table.

### 4. Borderless/sparse tables are completely invisible
**File:** `ingestion/src/validator.py:28-39`, `ingestion/src/table_extractor.py:36-44`

The page verdict requires `>=3 horizontal + >=2 vertical ruling lines` to trigger pdfplumber. Tables using whitespace alignment, background shading, or partial borders are never detected. AND pdfplumber uses `"lines"` strategy only — no fallback to `"text"` strategy. **Entire categories of tables are silently dropped.**

### 5. Vision table extraction has ZERO reconciliation
**File:** `extraction/vision_table.py:66-201`

GPT-4o vision output is **never validated** against the text-parsed data that already exists. No row-count check, no column-count check, no cell-text comparison. Hallucinated rows, missing columns, or swapped values propagate silently. The multi-page merge assumes first two lines are always header + separator — if the model returns differently formatted output, data rows are dropped.

### 6. LLM table repair blindly trusts the response
**File:** `ingestion/src/llm_repair.py:118-151`

No validation after LLM repair: no column count check, no cell text subset check, no hallucination detection. Original cell bboxes are discarded (all become `bbox=None`). If the LLM returns fewer rows, data is silently lost.

### 7. Extractor LLM never sees the actual content
**File:** `agents/extractor_node.py:148`

The `HumanMessage` contains only a content *summary* (section names and sizes), NOT the actual text. The real content is only accessible through the `extract()` tool call. The LLM makes extraction decisions without seeing source material, cannot assess content sufficiency before calling extract.

### 8. Cycle count bug — "Gatherer" vs "Explorer" agent name mismatch
**File:** `agents/explorer.py:333` vs `agents/graph.py:24`

Schema-gather path records `"agent": "Gatherer"` but cycle routing counts `"agent": "Explorer"`. The first exploration pass is never counted, allowing `max_cycles + 1` explorer runs instead of `max_cycles`.

### 9. Block index is entirely dead code — `search()` always returns `[]`
**File:** `knowledge_base/protocol_store.py:50-53`

`_build_block_index` is commented out, so `self._blocks` is always empty. `search()` (line 239) always returns `[]`. `get_pages()` always falls through to raw text fallback. Core retrieval functionality is broken.

### 10. Numerical validation checks against entire context, not source
**File:** `extraction/validator.py:145-171`

A hallucinated number passes validation if it appears **anywhere** in the protocol. "39" passes for "fever >= 39C" if "39 subjects" exists in any section. This is not validation — it's a coincidence detector.

---

## P1 — SERIOUS DESIGN FLAWS

### 11. Header/footer filter removes body text by content match
**File:** `ingestion/src/header_footer_filter.py:155-158`

If a header pattern (company name, protocol number) also appears in body text — which is common — it gets stripped **regardless of y-position**. The position check only applies as an additional removal, not as a guard.

### 12. Top-level section pattern rejects title-case headings
**File:** `ingestion/src/structure_discovery.py:36`

`r'^(\d+)\.?\s+([A-Z][A-Z ].{3,})$'` requires two uppercase chars. "1 Introduction" fails. "5 Study Design" fails. Only ALL-CAPS titles like "1 INTRODUCTION" match. Many protocols use title case.

### 13. Indent level 2 is unreachable — elif chain bug
**File:** `ingestion/src/structure_discovery.py:674-679`

```python
elif x0 > base_indent + 15:
    level = 1
elif x0 > base_indent + 30:  # UNREACHABLE — first elif already matches
    level = 2
```

All nested lists flatten to max 2 levels.

### 14. List continuation lines break list detection
**File:** `ingestion/src/structure_discovery.py:666-723`

When a list item wraps to a second line (same indent, no marker), the `else` branch terminates the list. Multi-line list items get split: first line becomes list item, continuation becomes orphaned paragraph.

### 15. String-based control flow for cycle routing
**File:** `agents/graph.py:20`

`error.startswith("NEED_MORE:")` uses a serialized string as a control signal. Any error message starting with "NEED_MORE:" triggers false cycles. The `error` field is overloaded as both error channel and control signal — real errors and cycle signals can overwrite each other.

### 16. Global mutable state with no locking
**File:** `app.py:50-58`

`AppState` is a module-level singleton. `/api/upload` writes state while `/api/query` reads it. No locks. Concurrent requests corrupt state. SSE thread (line 391) is never cancelled on client disconnect.

### 17. Source grounding only checks first 40 characters
**File:** `extraction/validator.py:57-66`

LLM can fabricate a matching 40-char prefix then diverge. The "partial" match on 20 chars is even weaker. Not meaningful validation.

### 18. `__init__.py` missing `assign_blocks_to_sections`
**File:** `ingestion/__init__.py:223`

The API path (`IngestionPipeline`) produces sections with empty `content_blocks`. The CLI path (`run.py`) calls `assign_blocks_to_sections`. Behavioral divergence between entry points.

### 19. `run.py` and `__init__.py` are duplicated pipelines
The entire pipeline logic is duplicated (~400 lines each). Bug #2, #3, and #18 demonstrate how fixes in one path are missed in the other. Maintenance trap.

### 20. Three identical ReAct loops with no shared abstraction
**File:** `agents/explorer.py:275`, `agents/extractor_node.py:151`, `agents/reviewer.py:133`

Same pattern copy-pasted: `for turn in range(1, N): resp = llm.invoke(msgs); if resp.tool_calls: ... else: break`. Different magic numbers (9, 7, 9), same missing error handling, same missing timeout.

### 21. No LLM API error handling anywhere
**Files:** `explorer.py:277`, `extractor_node.py:153`, `reviewer.py:135`

`llm.invoke(msgs)` has no try/except in any agent. Rate limits, timeouts, or 500 errors crash the entire graph. No retry, no graceful degradation.

### 22. Case-sensitive severity matching
**File:** `agents/reviewer.py:90-96` vs `agents/graph.py:40`

Reviewer accepts any string for `severity`. Routing checks `s.get("severity") == "critical"` (lowercase exact). If the LLM returns "Critical", the cycle is never triggered.

### 23. Reviewer ignores its `extraction_type` parameter
**File:** `agents/reviewer.py:56-59`

The `get_extraction` tool accepts `extraction_type: str` but always dumps ALL extractions. Misleading to the LLM.

### 24. Section reference validation is meaninglessly loose
**File:** `extraction/validator.py:89-96`

Checks if `section_id + "."` appears **anywhere** in context. "3" passes if "Table 3." or "Phase 3." exists. Massive false positive rate.

### 25. Reviewer cannot see validation results
**File:** `agents/reviewer.py:111`

Reviewer sees extractions but NOT the validator's quality assessment. It re-derives everything from scratch, unable to leverage the extractor's own findings.

---

## P2 — MODERATE ISSUES

### Retrieval

| # | Issue | File:Line |
|---|---|---|
| 26 | `text-embedding-3-small` (cheapest model) — `text-embedding-3-large` is meaningfully better for clinical terminology | `llamaindex_retriever.py:86` |
| 27 | `chunk_overlap=0` — critical sentences at chunk boundaries are split | `llamaindex_retriever.py:94` |
| 28 | `num_queries=1` defeats `QueryFusionRetriever` — generates zero query variants | `llamaindex_retriever.py:109` |
| 29 | Tables indexed as pipe-separated text — destroys hierarchical header structure | `llamaindex_retriever.py:58-76` |
| 30 | Table-to-section mapping uses page overlap, not assignment — wrong sections get wrong tables | `protocol_store.py:111-118` |
| 31 | `_find_section_end_page` is O(n log n) per call, called for every section | `protocol_store.py:182-199` |
| 32 | Last section gets arbitrary +10 page extension | `protocol_store.py:198-199` |
| 33 | `get_table_content()` triggers vision API calls inline during retrieval | `protocol_store.py:393-418` |

### Agents

| # | Issue | File:Line |
|---|---|---|
| 34 | Page-overlap table gathering pulls all tables from section's page range — irrelevant data floods context | `explorer.py:165-180` |
| 35 | `rows[:80]` hardcoded table truncation, no warning logged | `explorer.py:177` |
| 36 | Tool results truncated to 10000 chars — can corrupt JSON mid-stream | `explorer.py:290` |
| 37 | Greedy content assembly breaks on first oversized item, leaving budget unused | `extractor_node.py:89-103` |
| 38 | `get_gathered_content` has 40000 char limit, no relevance prioritization | `reviewer.py:74` |
| 39 | `content_shown` flag prevents reviewer from re-reading content | `reviewer.py:53, 64-65` |
| 40 | Keyword filter suppresses legitimate content requests ("failed", "flagged") | `extractor_node.py:174-178` |
| 41 | `set_reviewer_error` only uses FIRST critical signal, drops others | `graph.py:55` |
| 42 | `RuntimeContext` uses `Any` types — zero type safety on critical dependencies | `state.py:45-48` |
| 43 | `get_runtime` silently returns empty context if config is wrong | `state.py:52` |
| 44 | `signals` list has no reducer — replacement instead of append | `state.py:36` |

### Ingestion

| # | Issue | File:Line |
|---|---|---|
| 45 | Footnote regex strips real words — "Treatment group a" loses "a" | `table_extractor.py:178-188` |
| 46 | Cross-reference patterns miss dotted table numbers (Table 5.1) | `structure_discovery.py:525` |
| 47 | Abbreviation regex misses hyphens ("dose-limiting toxicity"), mixed case ("mRNA") | `structure_discovery.py:466-468` |
| 48 | Caption detection too loose — `font_size > 9` matches body text | `structure_discovery.py:330-331` |
| 49 | Continuation merge confidence degrades multiplicatively — 5-page table reaches 0.22 | `continuation_merger.py:227` |
| 50 | Continuation merge fails for captionless tables | `continuation_merger.py:105-112` |
| 51 | Page number regex matches body text standalone numbers | `header_footer_filter.py:31` |
| 52 | Header/footer zones use inconsistent coordinate systems (% vs pixels) | `header_footer_filter.py:69-70 vs 138-139` |
| 53 | Markdown renderer ignores merged cells (`row_span`, `col_span`) | `output_renderer.py:166-228` |
| 54 | `_partial_overlaps` in table exclusion lets text leak | `__init__.py:329-332`, `run.py:428-430` |

### Schemas & Validation

| # | Issue | File:Line |
|---|---|---|
| 55 | `EndpointCounts`, `total_rules`, `hard_rules` are manually filled — will drift from list contents | `schemas.py:160-164, 532-543` |
| 56 | `SoAExtraction` has 3 representations of same data — guaranteed inconsistency | `schemas.py:325-346` |
| 57 | `Grounding.page` default (-1) vs prompt prohibition (0 or -1) vs validator threshold (>0) — inconsistent | `schemas.py:27-52` |
| 58 | `phase` Literal type rejects real protocol values ("Phase 1b/2a") | `schemas.py:65-67` |
| 59 | Year numbers (1900-2100) excluded from validation — study date errors invisible | `validator.py:174-197` |
| 60 | No cross-item validation — missing criteria undetected | `validator.py` (missing entirely) |

### App / Config / Infra

| # | Issue | File:Line |
|---|---|---|
| 61 | CORS wide open (`allow_origins=["*"]`) | `app.py:47` |
| 62 | No file size limit on PDF upload | `app.py:162-215` |
| 63 | No API key validation at startup — empty key causes cryptic failures | `config.py:15` |
| 64 | `get_openai_client()` creates new client per call — wastes connection pooling | `config.py:22-24` |
| 65 | Debug log grows unbounded in append mode | `debug_logger.py:237` |
| 66 | SSE keepalive every 0.5s = 1200 events per 10min timeout — excessive | `event_bus.py:71-84` |
| 67 | SSE result event can be lost in race condition | `event_bus.py:82-84` |
| 68 | `json_parser.py` "outermost braces" fallback matches first `{` to last `}` across unrelated JSON objects | `shared/json_parser.py:39-46` |
| 69 | No token/cost tracking across 25 potential LLM turns per query | All agents |
| 70 | No timeout on any LLM call | All agents |

---

## ARCHITECTURAL GAPS

### Missing from the LangGraph Design

| Missing Component | Impact | Current Workaround |
|---|---|---|
| **Router/Classifier Node** | Cannot auto-detect query type or route to specialized pipelines | `query_type` passed in externally, no auto-detection |
| **Planner/Decomposer Node** | Cannot break "extract endpoints AND eligibility" into sub-tasks | Each query type runs separately, user must submit multiple queries |
| **Parallel Extraction** | Cannot fan-out for multi-faceted queries | Strictly sequential pipeline |
| **Reflection Node** | No self-assessment of plan quality or extraction strategy | Only reviewer signals, no meta-reasoning |
| **Context Assembly Node** | No intelligent context ranking/filtering between Explorer and Extractor | Raw dump of everything gathered, greedy truncation |
| **Reconciliation Node** | No cross-checking of vision vs text-parsed tables | Two independent representations, never compared |
| **Error Recovery** | No distinction between transient (API timeout) and permanent (schema mismatch) errors | Single `except Exception` that stringifies everything |

### The Graph Should Look Like This

```
                    ┌─────────────┐
         START ────▶│   ROUTER    │ (classify query, detect type)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   PLANNER   │ (decompose, set strategy)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
              ┌────▶│  EXPLORER   │ (agentic retrieval)
              │     └──────┬──────┘
              │            │
              │     ┌──────▼──────┐
              │     │  CONTEXT    │ (rank, filter, tier, assemble)
              │     │  ASSEMBLER  │
              │     └──────┬──────┘
              │            │
              │     ┌──────▼──────┐
    NEED_MORE │     │  EXTRACTOR  │ (schema-driven LLM extraction)
              │     └──────┬──────┘
              │            │
              │     ┌──────▼──────┐
              │     │ RECONCILER  │ (vision vs text, cross-item checks)
              │     └──────┬──────┘
              │            │
              │     ┌──────▼──────┐
              └─────│  REVIEWER   │ (verify, flag, decide cycle)
                    └──────┬──────┘
                           │
                          END
```

**Key additions:**
1. **Router** — auto-detect query type, handle multi-type queries
2. **Planner** — decompose complex queries, set retrieval strategy
3. **Context Assembler** — relevance-scored, token-aware context construction (replaces greedy char-limit truncation)
4. **Reconciler** — cross-check vision vs text tables, validate cross-item consistency, compute derived counts

---

## HOW TO USE THIS DOCUMENT

This audit identifies **70+ concrete issues**. The recommended approach:

1. **Fix P0 bugs first** (issues 1-10) — These cause silent data corruption
2. **Refactor the duplicated pipeline** (issue 19) — Single source of truth for ingestion
3. **Add the missing nodes** (Router, Planner, Context Assembler, Reconciler)
4. **Extract the shared ReAct loop** (issue 20) — Single implementation with error handling, timeouts, token tracking
5. **Fix the tool architecture** — Proper registry, typed inputs/outputs, testable
6. **Upgrade retrieval** — Contextual chunks, better reranker, real query fusion

The `ARCHITECTURE_RND.md` document provides the research-backed target architecture. This document provides the specific bugs and issues to fix along the way.
