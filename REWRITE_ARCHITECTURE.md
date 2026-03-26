# Protocol Engine — Complete Rewrite Architecture

## Overview

A complete rewrite of the clinical protocol extraction system, fixing 70+ bugs found in the audit. This document is the blueprint.

---

## 1. New Directory Structure

```
protocol_engine/
├── main.py                     # Entry point
├── config.py                   # All configuration in one place
├── models/
│   ├── state.py                # LangGraph state (typed, with reducers)
│   ├── schemas.py              # All 13 Pydantic extraction schemas
│   └── enums.py                # QueryType, NodeName, EdgeSignal enums
├── graph/
│   ├── builder.py              # LangGraph graph construction
│   ├── nodes/
│   │   ├── router.py           # NEW: Query classification + decomposition
│   │   ├── planner.py          # NEW: Multi-step plan for complex queries
│   │   ├── explorer.py         # Retrieval + search (simplified)
│   │   ├── context_assembler.py# NEW: Relevance-scored context building
│   │   ├── extractor.py        # LLM extraction (sees ACTUAL content now)
│   │   ├── reconciler.py       # NEW: Vision vs text + cross-field checks
│   │   └── reviewer.py         # Semantic-only review (validator handles deterministic)
│   └── edges.py                # All conditional edge functions
├── ingestion/
│   ├── pipeline.py             # Orchestrates ingestion
│   ├── pdf_parser.py           # Docling-based PDF parsing
│   ├── table_extractor.py      # Multi-strategy table extraction
│   ├── vision_extractor.py     # Vision model for complex tables
│   ├── reconciler.py           # Vision vs text table reconciliation
│   ├── chunker.py              # Contextual chunking (Anthropic method)
│   └── indexer.py              # Vector + BM25 index building
├── retrieval/
│   ├── engine.py               # Hybrid retrieval (vector + BM25 + RRF)
│   ├── reranker.py             # Modern reranker (bge-reranker-v2-m3)
│   └── contextual.py           # Contextual retrieval prepend logic
├── tools/
│   ├── registry.py             # Tool registry (all tools in one place)
│   ├── search.py               # search_sections tool
│   ├── read_section.py         # read_full_section tool
│   ├── extract.py              # extract_to_schema tool
│   ├── knowledge_base.py       # CDISC/ICH/SDTM lookup tool (NOT hardcoded)
│   ├── vision.py               # vision_extract tool
│   └── validate.py             # validation tool (deterministic)
├── validation/
│   ├── validator.py            # 4-level deterministic validation
│   ├── numerical.py            # Numerical grounding checks
│   └── completeness.py         # Schema completeness checks
└── knowledge/
    ├── cdisc.json              # CDISC controlled terminology
    ├── ich_guidelines.json     # ICH E6/E8/E9 rules
    └── sdtm_mappings.json      # SDTM domain mappings
```

---

## 2. New LangGraph Design — 7 Nodes

### Current (broken): 3 nodes
```
Explorer → Extractor → Reviewer → END (or cycle back to Explorer)
```

### New: 7 nodes with typed routing
```
         ┌─────────┐
         │  Router  │  ← Classifies query, detects multi-type
         └────┬─────┘
              │
         ┌────▼─────┐
         │  Planner  │  ← Decomposes into sub-tasks (skip for simple queries)
         └────┬──────┘
              │
         ┌────▼──────┐
         │  Explorer  │  ← Retrieval + search tools
         └────┬──────┘
              │
    ┌─────────▼───────────┐
    │  Context Assembler   │  ← Relevance scoring, token budgeting
    └─────────┬───────────┘
              │
         ┌────▼──────┐
         │ Extractor  │  ← LLM extraction (sees REAL content)
         └────┬──────┘
              │
         ┌────▼───────┐
         │ Reconciler  │  ← Vision vs text, cross-field, deterministic validation
         └────┬───────┘
              │
         ┌────▼──────┐
         │  Reviewer  │  ← Semantic-only checks (hallucination, nuance)
         └────┬──────┘
              │
             END
```

### Edge Logic (typed, not string-based)

```python
class EdgeSignal(str, Enum):
    CONTINUE = "continue"           # Move to next node
    NEED_MORE_CONTENT = "need_more" # Reviewer/Reconciler → Explorer
    NEED_REEXTRACT = "reextract"    # Reviewer → Extractor (same content, new attempt)
    NEED_VISION = "vision"          # Reconciler → Vision extraction
    PLAN_NEXT = "plan_next"         # Back to Planner for next sub-task
    DONE = "done"                   # → END
    ERROR_RETRY = "retry"           # Transient error → retry same node
    ERROR_FATAL = "fatal"           # → END with error
```

### State Design (with proper reducers)

```python
from langgraph.graph import MessagesState
from typing import Annotated
import operator

class ProtocolState(MessagesState):
    # Query
    query: str
    query_type: QueryType
    sub_tasks: list[SubTask]           # From Planner
    current_task_index: int

    # Content (with append reducer)
    retrieved_sections: Annotated[list[Section], operator.add]
    context_budget_used: int           # Token count, not char count

    # Context (assembled)
    assembled_context: str             # From Context Assembler
    context_relevance_scores: dict[str, float]

    # Extraction
    extraction: dict                   # Current extraction result
    extraction_history: Annotated[list[dict], operator.add]  # All attempts

    # Validation (deterministic)
    validation_result: ValidationResult
    signals: Annotated[list[Signal], operator.add]  # WITH reducer!

    # Control
    cycle_count: int
    max_cycles: int
    edge_signal: EdgeSignal
```

---

## 3. New Ingestion Pipeline

### Problem: Current parser misses sparse tables, double-emits bullets, fabricates cell bboxes

### Solution: Docling + multi-strategy tables + vision reconciliation

```
PDF
 │
 ├─► Docling (primary parser)
 │    ├─ Structural elements (headings, paragraphs, lists)
 │    ├─ Table detection (ML-based, handles borderless)
 │    └─ Reading order preservation
 │
 ├─► pdfplumber (fallback for tables)
 │    ├─ Strategy 1: "lines" (ruled tables)
 │    ├─ Strategy 2: "text" (borderless/sparse tables)  ← CURRENTLY MISSING
 │    └─ Strategy 3: explicit settings per page
 │
 ├─► Vision Model (complex tables only)
 │    ├─ Triggered when: confidence < threshold OR sparse layout detected
 │    ├─ Returns structured markdown table
 │    └─ RECONCILED against text extraction (not blind trust)
 │
 └─► Reconciler
      ├─ Compare vision vs text: cell-by-cell alignment
      ├─ Flag conflicts (>10% cells differ)
      ├─ Merge strategy: text for numbers, vision for structure
      └─ Output: high-confidence merged table
```

### Contextual Chunking (Anthropic Method)

Instead of naive `SentenceSplitter(chunk_size=8192, overlap=0)`:

```python
def contextual_chunk(document, chunk):
    """Prepend document-level context to each chunk before embedding."""
    prompt = f"""Given this document:
    <document>{document.summary}</document>

    And this chunk:
    <chunk>{chunk.text}</chunk>

    Provide a short (2-3 sentence) context that situates this chunk
    within the overall document. Focus on: which section this is from,
    what protocol element it describes, and how it relates to adjacent sections."""

    context = llm.invoke(prompt)
    return f"{context}\n\n{chunk.text}"
```

This gives 67% fewer retrieval failures (Anthropic's published results).

### Chunking Strategy
- **Chunk size**: 1024 tokens (not 8192 chars)
- **Overlap**: 128 tokens (not 0)
- **Boundaries**: Respect section headers — never split mid-section if section < 2048 tokens
- **Tables**: Each table is ONE chunk (never split tables)
- **Lists**: Each list block is ONE chunk

---

## 4. New Retrieval

### Current problems:
- Queries hardcoded per type in registry.py
- Block index search() always returns [] (dead code)
- Reranker uses outdated ms-marco-MiniLM
- No adaptive retrieval

### New design:

```python
class RetrievalEngine:
    def search(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        # 1. Vector search (contextual embeddings)
        vector_results = self.vector_index.query(query, top_k=top_k)

        # 2. BM25 search
        bm25_results = self.bm25_index.query(query, top_k=top_k)

        # 3. Reciprocal Rank Fusion
        fused = self.rrf_merge(vector_results, bm25_results, k=60)

        # 4. Rerank with modern model
        reranked = self.reranker.rerank(query, fused, top_k=top_k)
        # Use: BAAI/bge-reranker-v2-m3 (NOT ms-marco-MiniLM)

        return reranked
```

### Agentic Retrieval
The Explorer node decides:
- **What** to search (generates queries, not hardcoded)
- **How many** results to fetch (adapts based on query complexity)
- **Whether** to do follow-up searches (based on initial results quality)
- **When** to stop (confidence threshold, not fixed cycle count)

---

## 5. New Context Engineering

### The Write-Select-Compress-Isolate Pattern

**Current**: Greedy char truncation at 80k, no relevance scoring, newly-fetched content has no priority.

**New**:

```python
class ContextAssembler:
    """Assembles context with relevance scoring and token budgeting."""

    def __init__(self, token_budget: int = 32000):
        self.token_budget = token_budget

    def assemble(self, sections: list[Section], query: str) -> AssembledContext:
        # 1. SCORE each section for relevance
        scored = []
        for section in sections:
            score = self.relevance_score(section, query)
            scored.append((section, score))

        # 2. SORT by relevance (highest first)
        scored.sort(key=lambda x: x[1], reverse=True)

        # 3. TIERED inclusion
        context_parts = []
        tokens_used = 0

        for section, score in scored:
            section_tokens = count_tokens(section.text)

            if score > 0.8:
                # High relevance: include VERBATIM
                if tokens_used + section_tokens <= self.token_budget:
                    context_parts.append(section.text)
                    tokens_used += section_tokens
            elif score > 0.5:
                # Medium relevance: COMPRESS to summary
                summary = self.summarize(section, max_tokens=200)
                summary_tokens = count_tokens(summary)
                if tokens_used + summary_tokens <= self.token_budget:
                    context_parts.append(f"[Summary of {section.title}]: {summary}")
                    tokens_used += summary_tokens
            # Low relevance: SKIP entirely

        # 4. STRUCTURE the context
        return AssembledContext(
            text="\n\n---\n\n".join(context_parts),
            tokens_used=tokens_used,
            sections_included=len(context_parts),
            sections_skipped=len(sections) - len(context_parts),
        )
```

### Key Difference: Extractor Sees Real Content

**Current (broken)**: Extractor LLM gets a summary ("5 sections, 45000 chars"), then calls `extract()` tool which sends content to a SEPARATE LLM call.

**New**: Extractor LLM gets the assembled context DIRECTLY in its prompt, plus the schema. One LLM call, not two. The LLM that decides what to extract IS the LLM that sees the content.

---

## 6. New Tool Registry

### Current problems:
- Tools defined inline with @tool decorators
- KNOWLEDGE_APPENDICES is a 500-line hardcoded dict
- get_extraction() ignores its parameter
- request_more_content() blocked by keyword filter
- Tool results truncated to 10k chars (produces invalid JSON)

### New design:

```python
# tools/registry.py
class ToolRegistry:
    """Central registry for all agent tools."""

    def __init__(self, retrieval_engine, knowledge_base, vision_model):
        self.tools = {
            "search": SearchTool(retrieval_engine),
            "read_section": ReadSectionTool(retrieval_engine),
            "lookup_cdisc": KnowledgeBaseTool("cdisc"),
            "lookup_ich": KnowledgeBaseTool("ich"),
            "lookup_sdtm": KnowledgeBaseTool("sdtm"),
            "vision_extract": VisionTool(vision_model),
            "get_validation": ValidationTool(),
        }

    def get_tools_for_node(self, node: str) -> list[Tool]:
        """Each node gets only the tools it needs."""
        if node == "explorer":
            return [self.tools["search"], self.tools["read_section"]]
        elif node == "extractor":
            return []  # Extractor doesn't need tools — it gets context directly
        elif node == "reconciler":
            return [self.tools["vision_extract"], self.tools["get_validation"]]
        elif node == "reviewer":
            return [self.tools["get_validation"], self.tools["lookup_cdisc"]]
```

### Knowledge Base as a Tool (not hardcoded)

```python
# tools/knowledge_base.py
class KnowledgeBaseTool:
    """Query CDISC/ICH/SDTM knowledge on demand."""

    def __init__(self, domain: str):
        self.data = json.load(open(f"knowledge/{domain}.json"))

    def lookup(self, term: str) -> str:
        """Fuzzy lookup of a term in the knowledge base."""
        matches = fuzzy_search(self.data, term, threshold=0.7)
        return json.dumps(matches[:5], indent=2)
```

---

## 7. Reconciliation (NEW — Currently Missing)

### Table Reconciliation (Vision vs Text)

```python
class TableReconciler:
    def reconcile(self, text_table: Table, vision_table: Table) -> Table:
        # 1. Align columns by header similarity
        column_mapping = self.align_columns(text_table.headers, vision_table.headers)

        # 2. Cell-by-cell comparison
        conflicts = []
        for row_idx in range(min(len(text_table.rows), len(vision_table.rows))):
            for col_idx in column_mapping:
                text_val = text_table.rows[row_idx][col_idx]
                vision_val = vision_table.rows[row_idx][column_mapping[col_idx]]

                if not self.values_match(text_val, vision_val):
                    conflicts.append(Conflict(row_idx, col_idx, text_val, vision_val))

        # 3. Resolution strategy
        merged = text_table.copy()
        for conflict in conflicts:
            if is_numeric(conflict.text_val):
                # Trust text extraction for numbers (vision hallucinates digits)
                merged.rows[conflict.row][conflict.col] = conflict.text_val
            else:
                # Trust vision for structure/layout
                merged.rows[conflict.row][conflict.col] = conflict.vision_val

        # 4. Flag if >10% cells conflict
        conflict_rate = len(conflicts) / total_cells
        merged.confidence = 1.0 - conflict_rate

        return merged
```

### Cross-Field Reconciliation

```python
class FieldReconciler:
    """Check extracted fields are consistent with each other."""

    def check(self, extraction: dict) -> list[Issue]:
        issues = []

        # Example: sample_size in demographics must match sample_size in stats
        if "demographics" in extraction and "statistical_analysis" in extraction:
            demo_n = extraction["demographics"].get("sample_size")
            stats_n = extraction["statistical_analysis"].get("sample_size")
            if demo_n and stats_n and demo_n != stats_n:
                issues.append(Issue(
                    "CROSS_FIELD_MISMATCH",
                    f"Demographics sample_size ({demo_n}) != Stats sample_size ({stats_n})"
                ))

        return issues
```

---

## 8. Validation vs Reviewer — Complementary, Not Duplicated

### Current: Both do the same 4 checks. Reviewer wastes LLM tokens.

### New:

| Check | Validator (deterministic, free) | Reviewer (LLM, costly) |
|-------|-------------------------------|----------------------|
| Source grounding | YES — exact string match | NO — already done |
| Numerical accuracy | YES — regex + comparison | NO — already done |
| Section references | YES — section ID lookup | NO — already done |
| Completeness | YES — schema field check | NO — already done |
| Hallucination detection | NO — can't do this | YES — is this plausible? |
| Semantic accuracy | NO — can't do this | YES — does this mean what the LLM thinks? |
| Cross-section contradiction | NO — can't do this | YES — do sections contradict each other? |
| Clinical plausibility | NO — can't do this | YES — is 500mg/kg a real dose? |

### Reviewer gets validation results as input:
```python
def reviewer_node(state):
    validation = state["validation_result"]

    prompt = f"""You are reviewing an extraction. The deterministic validator
    has already checked: source grounding, numbers, completeness.

    Validator results: {validation.summary}
    Validator flags: {validation.flags}

    YOUR job is ONLY to check things the validator CANNOT:
    1. Is any extracted information a hallucination? (plausible but not in source)
    2. Are there semantic errors? (correct text but wrong interpretation)
    3. Do any sections contradict each other?
    4. Are clinical values plausible? (doses, frequencies, durations)

    Do NOT re-check what the validator already verified."""
```

---

## 9. Domain Knowledge to Preserve

These are CORRECT in the current code and must be carried forward:

### 13 Extraction Schemas (schemas.py)
- StudyDesign, Objectives, Endpoints, Population, Interventions
- SafetyMonitoring, StatisticalAnalysis, DataMonitoring, RegulatoryCompliance
- ProtocolAmendments, VisitSchedule, Demographics, Biomarkers

### 13 Query Types (enums.py)
- study_design, objectives, endpoints, population, interventions
- safety_monitoring, statistical_analysis, data_monitoring, regulatory
- amendments, visit_schedule, demographics, biomarkers

### CDISC/ICH/SDTM Mappings
- Extract from current KNOWLEDGE_APPENDICES dict → JSON files
- SDTM domain mappings (DM, AE, VS, LB, etc.)
- ICH E6(R2) GCP requirements
- CDISC controlled terminology

### Validation Rules
- 4-level validation logic is sound, just needs bug fixes
- Numerical grounding approach is correct (just fix the substring bug)
- Source verification approach is correct

---

## 10. Implementation Phases

### Phase 1: Foundation (Week 1)
- [ ] New directory structure
- [ ] State model with typed enums and proper reducers
- [ ] Extract knowledge bases to JSON files
- [ ] Tool registry with proper tool classes
- [ ] Basic graph with 7 nodes (Router → Planner → Explorer → Context Assembler → Extractor → Reconciler → Reviewer)

### Phase 2: Ingestion Fix (Week 2)
- [ ] Docling integration for PDF parsing
- [ ] Multi-strategy table extraction (lines + text + explicit)
- [ ] Fix bullet point double-emission
- [ ] Fix table merge off-by-one
- [ ] Contextual chunking
- [ ] Vision + text table reconciliation

### Phase 3: Retrieval + Context (Week 3)
- [ ] Contextual retrieval (prepend context to chunks)
- [ ] Modern reranker (bge-reranker-v2-m3)
- [ ] Context Assembler node with relevance scoring
- [ ] Token-aware budgeting (not char-based)
- [ ] Agentic retrieval (Explorer generates its own queries)

### Phase 4: Extraction + Reconciliation (Week 4)
- [ ] Extractor sees REAL content (not summary)
- [ ] Reconciler node: vision vs text, cross-field
- [ ] Reviewer sees validator output, does semantic-only checks
- [ ] Typed edge signals (not string-based)
- [ ] Proper cycle counting

### Phase 5: Polish (Week 5)
- [ ] Model abstraction (LiteLLM)
- [ ] Error handling with typed retries
- [ ] End-to-end testing with real protocols
- [ ] Performance benchmarking

---

## 11. Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| PDF Parser | Docling (primary) + pdfplumber (tables) | Docling handles structure; pdfplumber handles precise table extraction |
| Chunking | 1024 tokens, 128 overlap, contextual | Anthropic research: 67% fewer retrieval failures |
| Embeddings | text-embedding-3-large | Best price/performance for domain text |
| Reranker | BAAI/bge-reranker-v2-m3 | Replaces outdated ms-marco-MiniLM |
| Graph framework | LangGraph with Command pattern | Already using LangGraph; Command is the 2025 best practice for dynamic routing |
| Extraction LLM | Claude Sonnet (primary) | Better at structured extraction than GPT-4o |
| Vision LLM | GPT-4o | Still best for table/figure vision |
| State management | TypedDict with Annotated reducers | Fixes current signal-dropping bug |
| Tool protocol | LangChain tools (MCP-compatible structure) | Easy migration to MCP later |
| Validation | Deterministic first, LLM second | Don't waste tokens on regex-checkable things |
