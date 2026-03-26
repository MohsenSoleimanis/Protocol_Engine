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

## DEEP DIVE: CONTEXT ENGINEERING — Where It Happens and What's Broken

There is **no context engineering** in this codebase. What exists is a series of ad-hoc char-limit truncations scattered across 4 files. Here's the complete data flow:

### The Current "Context Pipeline" (Broken)

```
Explorer (_schema_gather)          Explorer (_agentic_gather)
  │                                  │
  │ Retrieves sections/tables        │ LLM decides what to search
  │ NO relevance scoring             │ NO relevance scoring
  │ NO ranking                       │ Budget = 100k/60k chars (hardcoded)
  │                                  │
  ▼                                  ▼
state["sections_content"]    ◄── merge ──►   state["tables_content"]
  │                                            │
  │  Dict of {section_id: {text, pages, chars}}│
  │  NO ordering by relevance                  │
  │  NO scoring                                │
  │                                            │
  ▼────────────────────────────────────────────▼
                    │
         extractor_node.py:85-113
         "Context Assembly" (BROKEN)
                    │
  1. Iterate sections_read in ORDER THEY WERE FOUND (not relevance)
  2. For each: if char_count + len(text) > 80k → STOP (greedy break)
  3. Then iterate remaining tables, same greedy break at 80k
  4. Join everything with "\n\n"
  5. That's it. No ranking, no filtering, no summarization.
                    │
                    ▼
         content string (up to 80k chars)
                    │
         ┌──────────┴──────────┐
         │                     │
    NEVER SENT TO LLM    Passed to extract()
    (extractor_node:148)  tool via closure
    LLM only sees a       (extractor_node:33)
    summary like:
    "Gathered content:
     5 sections, 3 tables.
     Total: 45000 chars."
         │                     │
         ▼                     ▼
    LLM decides blind    extraction/extractor.py:141
    whether to extract   Finally sees the content
    or request_more      as user_msg to a DIFFERENT
                         LLM call (structured output)
```

### Specific Problems

**1. Zero relevance scoring** — `extractor_node.py:89-103`
Sections are iterated in `sections_read` order (insertion order from retrieval). A section with 0.95 relevance score and one with 0.3 are treated identically. If a low-relevance 30k-char section is read first, it eats the budget and the high-relevance section gets dropped.

**2. Greedy truncation wastes budget** — `extractor_node.py:99-100`
```python
if char_count + len(text) > content_limit:
    break  # stops on first item that doesn't fit
```
If item A is 60k chars and doesn't fit, the loop breaks. But items B (5k), C (3k), D (2k) would all fit. The greedy approach leaves ~70k chars of budget unused.

**3. Char-based, not token-based** — `extractor_node.py:83`
`content_limit = 80000` chars ≈ 20k tokens (rough). But model context windows are measured in tokens. 80k chars of ASCII text vs 80k chars of clinical abbreviations have very different token counts. No tiktoken or equivalent.

**4. The Extractor LLM is blind** — `extractor_node.py:143-148`
The ReAct agent only receives:
```
"Gathered content: 5 sections (3.1, 3.2, 5.1, 9.2, 11), 3 tables (tbl_1, tbl_2, tbl_3). Total: 45000 chars."
```
It NEVER sees the actual protocol text. It then calls `extract()` which fires a SEPARATE LLM call with the full content. The ReAct agent cannot:
- Judge if content is sufficient
- Decide which sections matter more
- Request specific sections
- Compare source text against extraction

**5. No context passed between cycles** — `state.py:27-28`
`sections_content` uses `_merge_dicts` (shallow merge). On cycle re-entry, the Explorer adds NEW sections to state. The Extractor rebuilds the full content from scratch including OLD + NEW sections. But there's no prioritization of new vs old — the new content (what was specifically requested) might be at the end and get truncated.

**6. Reviewer gets even less context** — `reviewer.py:70-85`
The reviewer has a SEPARATE 40k char limit (hardcoded), iterates sections in dict order (arbitrary), and can only call `get_gathered_content()` ONCE (`content_shown` flag). After that first call, if context scrolled out of the LLM's attention, there's no way to re-read it.

**7. No tiered assembly**
Every section is included verbatim or not at all. No option for:
- High-relevance: full text
- Medium-relevance: key paragraphs
- Low-relevance: one-line summary
This means a 20k-char section that's only marginally relevant takes the same space as one that's critical.

**8. No context for the Reviewer to leverage validation**
`extractor_node.py:189-191` stores `validation` in state. But `reviewer.py:111` reads `extracted_data` and never reads `validation`. The reviewer re-derives quality assessment from scratch, duplicating work the validator already did (see next section).

### What Should Exist

A dedicated **Context Assembly** node between Explorer and Extractor that:
1. Scores each section's relevance to the query + schema fields
2. Ranks by relevance score (not retrieval order)
3. Tiers: verbatim (>0.8), key paragraphs (0.5-0.8), summary (<0.5)
4. Assembles within a **token** budget, not char budget
5. Passes the assembled context directly to the Extractor LLM (not via a blind tool call)
6. On cycles, prioritizes newly-fetched content

---

## DEEP DIVE: BROKEN LINKS, EDGES, AND TOOLS

### Broken Graph Edges

**1. `explorer_node` returns `"agent": "Gatherer"` but cycle counting checks `"agent": "Explorer"`**
- `explorer.py:333`: `"agent": "Gatherer"` (schema-gather path)
- `explorer.py:304`: `"agent": "Explorer"` (agentic path)
- `graph.py:24`: `sum(1 for s in steps if s.get("agent") == "Explorer")`
- **Result**: First pass is never counted. System allows `max_cycles + 1` runs.

**2. `route_after_extractor` silently ends on empty extraction**
- `graph.py:28-30`: If `extracted_data` is empty AND no `NEED_MORE:` error → goes to `END`
- No edge to retry, no edge to Explorer, no error surfaced
- **Result**: If extraction fails silently (e.g., schema mismatch), the query returns empty with no explanation

**3. `set_reviewer_error` only passes first signal, edge is lossy**
- `graph.py:55`: `actionable[0].get('description', '')[:200]`
- If reviewer flags 3 critical issues, only the first one's description (truncated to 200 chars) reaches the Explorer
- The other 2 signals are lost in the edge

**4. `signals` state field has no reducer — replacement not append**
- `state.py:36`: `signals: list[dict]` — no `Annotated[list, operator.add]`
- When reviewer writes `{"signals": [...]}`, it REPLACES previous signals
- In current linear flow this is fine, but if cycles add another reviewer pass, the previous signals vanish

**5. Dead edge: `"pages"` type never produced**
- `explorer.py:300`: `if item["type"] in ("section", "pages"): secs[item["id"]] = entry`
- No code ever creates an item with `type="pages"`. This branch of the conditional is dead code.

### Broken Tools

**6. `get_extraction(extraction_type)` ignores its parameter**
- `reviewer.py:56-59`: Accepts `extraction_type: str` but always returns ALL extractions via `json.dumps(extractions)`
- The LLM thinks it can request specific types. It can't. Tool description misleads the LLM.

**7. `extract(schema_type)` uses LLM-provided argument unsafely**
- `extractor_node.py:33`: `run_extract(query_type=schema_type)` uses whatever the LLM passes
- No validation that `schema_type` is a valid key in `SCHEMA_MAP`
- If LLM passes `"endpoint"` instead of `"endpoints"`, extraction fails

**8. `request_more_content` is filtered by keyword, not intent**
- `extractor_node.py:174-178`: If the reason contains "failed", "flagged", "validation", or "verified", the request is suppressed
- "The eligibility section **failed** to include age ranges" → suppressed
- Legitimate content requests get blocked by keyword collision

**9. `vision_extract` pages parameter not deduplicated**
- `explorer.py:241`: `key = tuple(sorted(pages))` — `[1,1,2]` caches as `(1,1,2)` not `(1,2)`
- Duplicate pages waste vision API calls

**10. Tool result truncation corrupts data**
- `explorer.py:290`: `str(result)[:10000]` — if result is JSON, truncation at 10k chars produces invalid JSON
- `extractor_node.py:38`: `json.dumps(extracted)[:10000]` — same problem
- `extractor_node.py:168`: `str(result)[:15000]` — same pattern, different magic number
- `reviewer.py:59`: `json.dumps(extractions)[:15000]` — same
- **Result**: LLM receives truncated, invalid JSON and must parse it or hallucinate the rest

### Missing Edges

**11. No edge from empty extraction to retry**
If Extractor produces nothing (schema mismatch, API error), graph goes to END. Should route back to Explorer or surface an error.

**12. No edge for transient errors**
API timeouts, rate limits, and network errors all crash the current node. No retry edge, no fallback edge. The only catch is the top-level `except Exception` in `run_query`.

**13. No edge from Reviewer to Extractor**
Reviewer can only route back to Explorer (via `set_reviewer_error`). If the issue is extraction quality (not missing content), there's no way to re-run just the Extractor with the same content but different instructions.

---

## DEEP DIVE: VALIDATOR vs REVIEWER — Redundancy Analysis

### What Each Does

| Aspect | Validator (`extraction/validator.py`) | Reviewer (`agents/reviewer.py`) |
|---|---|---|
| **Type** | Deterministic code, zero LLM | LLM agent with tools |
| **Cost** | Free (CPU only) | $0.01-0.05 per run (LLM calls) |
| **Input** | `extracted_data` dict + `content` string | `extracted_data` + `sections_content` + `tables_content` |
| **When called** | Inside Extractor node (line 181) or by extract tool (line 45) | After Extractor, as separate graph node |
| **Output** | `{verified, flagged, failed, total, details}` | `signals` list of `{signal_type, severity, title, description}` |
| **Can trigger cycle** | No (stored in state, but never checked by router) | Yes (critical signals → `set_reviewer_error` → Explorer) |

### The Overlap

Both check for **the same categories of issues**:

| Check | Validator | Reviewer |
|---|---|---|
| Source grounding (is text in context?) | Yes — substring match on first 40 chars | Yes — LLM reads content and compares |
| Numerical accuracy | Yes — number set comparison (broken, checks entire context) | Yes — LLM reads numbers and compares |
| Section reference exists | Yes — string match `section_id + "."` in context | Yes — LLM checks cross-refs |
| Completeness | Yes — only 2 patterns (endpoint timing, FULL automation) | Yes — LLM checks if items in content are missing from extraction |
| Hallucination detection | Partially — source grounding only | Yes — LLM can spot fabricated content |
| Cross-item consistency | No | Yes — LLM can compare across extracted items |

### The Real Problem: Neither Uses the Other's Output

**Validator runs first** (inside Extractor, `extractor_node.py:181`), produces `validation` dict stored in `state["validation"]`.

**Reviewer runs second** but **never reads `state["validation"]`** (`reviewer.py:111` only reads `extracted_data`). The Reviewer:
1. Calls `get_extraction()` — sees the extracted JSON
2. Calls `get_gathered_content()` — sees raw content (40k char limit)
3. Re-derives all the same checks the Validator already computed
4. Has NO tool to see validation results

This means:
- If Validator found "FAILED_NUMBERS: claim has 39 not in protocol", the Reviewer doesn't know and wastes LLM tokens re-checking every number
- If Validator verified 15/17 items, the Reviewer has no way to focus on just the 2 problematic ones
- The Reviewer might DISAGREE with the Validator (LLM says "looks fine" for something Validator flagged), creating conflicting quality signals

### What Should Happen

**Option A: Remove Reviewer, enhance Validator**
- The Validator is free (no LLM cost) and deterministic
- Add cross-item validation (missing criteria count)
- Add full source text matching (not just 40 chars)
- Add number-in-source checking (not entire context)
- Use saved $ to add more retrieval/extraction budget
- **Downside**: Loses the LLM's ability to catch semantic hallucinations

**Option B: Keep both, make them complementary (recommended)**
- Validator: deterministic checks (grounding, numbers, references, completeness)
- Reviewer: ONLY checks what Validator can't — semantic hallucination, missed nuance, cross-section contradictions
- **Give the Reviewer access to validation results** via a `get_validation()` tool
- Reviewer's prompt should say: "The validator has already verified grounding and numbers. Focus on semantic accuracy and completeness."
- **Result**: Reviewer becomes a focused semantic checker instead of a redundant pattern matcher

**Option C: Merge into a Reconciler node**
- Single node that runs deterministic checks first, then LLM checks only for flagged/failed items
- Runs the Validator's 4 levels
- For any FAILED or FLAGGED items, asks the LLM to investigate specifically
- Also handles vision vs text table reconciliation
- Also computes derived counts (replacing manual `EndpointCounts`, `total_rules`, etc.)

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
