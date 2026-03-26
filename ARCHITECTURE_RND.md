# Protocol Engine — Architecture R&D Report

> **Date**: 2026-03-26
> **Scope**: Agentic architecture, retrieval, context engineering, tool abstraction
> **Verdict**: Keep LangGraph + LlamaIndex foundation. Refactor implementation in 5 phases.

---

## 1. Current State Assessment

### What Works (Keep)
| Component | Why It's Good |
|---|---|
| LangGraph pipeline (Explorer→Extractor→Reviewer) | Matches Anthropic's "Prompt Chain + Evaluator-Optimizer" pattern — the sweet spot for structured extraction |
| LlamaIndex hybrid retrieval (BM25 + Vector + RRF) | State-of-the-art hybrid search, proven for document retrieval |
| 4-level deterministic validation (zero LLM) | Catches hallucinations without cost — rare in production systems |
| Schema-driven extraction via Pydantic | Right abstraction — adding new analysis = new schema, not new prompts |
| Ingestion pipeline (PDF → structured JSON) | Solid multi-phase parser with fallback hierarchy |

### What's Wrong (Fix)

| Problem | Where | Impact |
|---|---|---|
| **Hardcoded knowledge** | `KNOWLEDGE_APPENDICES` dict in `extractor.py`, regex in `explorer.py`, keywords in `domain.py` | Brittle, not queryable, wastes context tokens |
| **No contextual retrieval** | `llamaindex_retriever.py` indexes raw chunks | 35-67% retrieval accuracy left on the table (Anthropic research) |
| **Naive context assembly** | Fixed 80k/50k char limits, no relevance ranking | Context rot — irrelevant sections dilute extraction quality |
| **Rigid retrieval queries** | Hardcoded in `registry.py` per query type | Not truly agentic — agent can't adapt retrieval strategy |
| **Inline tool definitions** | `@tool` decorators scattered in `explorer.py` | Not reusable, not testable, no standard protocol |
| **Tight OpenAI coupling** | `langchain_openai` everywhere | Can't use Claude (better at structured extraction) or local models |
| **Shallow agent graph** | 3 nodes, string-based routing (`NEED_MORE:`) | No planning, no query decomposition, no reflection |
| **No cross-query memory** | Each query starts from scratch | Repeated work for multi-query analysis sessions |

---

## 2. Research Findings

### 2.1 Agentic Architecture Patterns

**Anthropic's recommended hierarchy** (simplest → most complex):

```
1. Augmented LLM         — Single LLM + tools (use this first)
2. Prompt Chaining        — Fixed sequence, output feeds forward
3. Routing                — Classifier dispatches to handler
4. Parallelization        — Multiple LLM calls, results aggregated
5. Orchestrator-Workers   — Central LLM dispatches dynamically
6. Evaluator-Optimizer    — Generate + critique loop
7. Full Autonomous Agent  — Open-ended think-act-observe
```

**Our current system** = Pattern 2 (Prompt Chain) + Pattern 6 (Evaluator-Optimizer via Reviewer cycles). This is **correct positioning** — structured enough to be predictable, flexible enough to self-correct.

**What to add**:
- **Planner node** for complex queries (Pattern 5: Orchestrator-Workers)
- **Query decomposition** before retrieval
- **Typed state transitions** instead of string-based `NEED_MORE:`

**What NOT to add**:
- Full autonomous agents (Pattern 7) — too unpredictable for clinical data extraction
- CrewAI-style multi-agent chat — our pipeline is deterministic, not conversational

### 2.2 Retrieval Systems

#### Current: Solid Foundation
```
BM25 + Vector (text-embedding-3-small) → RRF fusion → Reranking (ms-marco-MiniLM)
```

#### Missing: Contextual Retrieval (Anthropic, Sep 2024)

Prepend LLM-generated context to each chunk **before embedding**:

```
Before: "Patients must be ≥18 years of age..."
After:  "[Section 5.1 Inclusion Criteria for Phase III trial of Drug X
         in moderate-to-severe Condition Y] Patients must be ≥18 years..."
```

**Impact** (from Anthropic's research):
| Technique | Retrieval Failure Reduction |
|---|---|
| Contextual Embeddings alone | 35% |
| + Contextual BM25 | 49% |
| + Reranking | **67%** |

This is the **single highest-impact improvement** available. Our ingestion pipeline already processes sections with structure discovery — adding a contextual prefix step is straightforward.

#### Missing: Better Reranking

`ms-marco-MiniLM-L-6-v2` is outdated. Modern alternatives:

| Reranker | Quality | Speed | Notes |
|---|---|---|---|
| `ms-marco-MiniLM-L-6-v2` (current) | Baseline | Fast | 2021 model |
| `bge-reranker-v2-m3` | +15-20% | Moderate | Multilingual, strong on domain text |
| `Cohere Rerank v3` | +20-25% | API call | Best quality, costs $0.002/query |
| `jina-reranker-v2` | +15% | Fast | Good balance |

#### Missing: Agentic Retrieval

Current: Registry hardcodes queries → parallel execution → done.
Better: LLM decides query strategy based on the actual question.

```python
# Current (rigid)
queries = registry.get_config("endpoints").retrieval_queries  # fixed list

# Better (agentic)
planner_output = llm("What specific sections would contain endpoint
                      definitions for this protocol? Generate 3-5
                      targeted search queries.")
```

#### Optional: HyDE (Hypothetical Document Embeddings)

Generate a hypothetical answer, embed it, retrieve similar real content. Bridges the vocabulary gap between user queries and protocol language. Low effort, medium impact.

### 2.3 Context Engineering

**Core principle** (Anthropic, 2025): *Context is RAM, not a dumping ground. A focused 300-token context often outperforms 100K tokens of unfocused content.*

#### Current Problems

1. **No relevance ranking of retrieved content** — Explorer gathers 15 sections, Extractor gets all 15 even if only 5 matter
2. **Fixed char limits** (80k/50k) — no token-awareness, no priority
3. **No compression** — long sections passed verbatim even when only a paragraph is relevant
4. **No cross-query state** — querying "endpoints" then "eligibility" reprocesses everything

#### Recommended Architecture

```
┌─────────────────────────────────────────────────────┐
│              CONTEXT ASSEMBLY PIPELINE                │
│                                                      │
│  1. Retrieve candidate sections (Explorer)           │
│  2. Score relevance to specific query (0-1)          │
│  3. Tier the content:                                │
│     - Tier 1 (score > 0.8): Full verbatim text      │
│     - Tier 2 (score 0.5-0.8): Key paragraphs only   │
│     - Tier 3 (score < 0.5): One-line summary         │
│  4. Assemble within token budget (not char budget)   │
│  5. Add structured markers for grounding             │
└─────────────────────────────────────────────────────┘
```

#### Memory System

| Type | Implementation | Priority |
|---|---|---|
| **Working memory** | `ProtocolState` dict (already exists) | Already done |
| **Session memory** | Cache extraction results across queries in same session | Phase 4 |
| **Document memory** | Persist learned document structure (TOC, key sections) after first query | Phase 4 |
| **Long-term memory** | Cross-document patterns (not needed for single-protocol analysis) | Not needed |

### 2.4 Tool Architecture

#### Current: Inline LangChain `@tool` Decorators

```python
# explorer.py — tools defined inline, coupled to explorer logic
@tool
def search(query: str) -> str:
    """Search the protocol..."""
    results = runtime.retriever.retrieve(query)
    ...
```

#### Recommended: Tool Registry Pattern

```python
# tools/registry.py — centralized, testable, reusable
class ToolRegistry:
    """Self-describing tools with typed inputs/outputs."""

    def register(self, name, fn, input_schema, output_schema, description): ...
    def get_tools(self, agent_role: str) -> list[Tool]: ...
    def execute(self, name: str, inputs: dict) -> ToolResult: ...

# tools/retrieval.py
class SearchTool(BaseTool):
    name = "search_protocol"
    description = "Search the protocol document using semantic + keyword search"
    input_schema = SearchInput  # Pydantic model
    output_schema = SearchOutput  # Pydantic model

    def execute(self, input: SearchInput) -> SearchOutput: ...

# tools/knowledge.py
class QueryKnowledgeBase(BaseTool):
    """Query domain knowledge (CDISC, ICH, SDTM) — replaces hardcoded KNOWLEDGE_APPENDICES"""
    name = "query_knowledge"
    ...
```

#### MCP Readiness

MCP (Model Context Protocol) is now the industry standard (adopted by Anthropic, OpenAI, Google, Microsoft). Current tools should be structured so they **can** be exposed as MCP tools later, even if we don't implement the full MCP server now.

Key principle: **Tools should be self-describing, independently testable, and decoupled from agent logic.**

### 2.5 Framework Validation

| Framework | Verdict | Reason |
|---|---|---|
| **LangGraph** | **Keep** | Best for stateful, precisely-defined pipelines. GA v1.0, 38M+ monthly downloads |
| **LlamaIndex** | **Keep** | Best-in-class for RAG. Use alongside LangGraph (recommended combo in 2026 guides) |
| **CrewAI** | Skip | Built for loosely-defined multi-agent chat, not structured extraction |
| **Pydantic AI** | Consider for new components | Type-safe, great structured outputs. Good if building new pipelines alongside PE |
| **Claude Agent SDK** | Watch | Only if switching to Anthropic-native stack |
| **AutoGen** | Skip | Over-engineered for our use case |

### 2.6 Model Abstraction

Current: Hardcoded to OpenAI via `langchain_openai`.

Recommended: **LiteLLM** or LangChain's model-agnostic interface.

| Task | Best Model (2026) | Why |
|---|---|---|
| Structured extraction | Claude Opus/Sonnet | Superior at following complex schemas |
| Vision (tables) | GPT-4o / Claude Sonnet | Both strong, need benchmarking |
| Reranking | Cross-encoder (local) | No API cost, fast |
| Contextual chunk prefixes | Claude Haiku / GPT-4o-mini | Cheap, just needs summarization |
| Planning/decomposition | Claude Sonnet / GPT-4o | Needs strong reasoning |

---

## 3. Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PROTOCOL ENGINE v2                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │   INGESTION   │────▶│  INDEXING     │────▶│  RETRIEVAL   │        │
│  │   (PDF Parse) │     │  (Contextual │     │  (Agentic    │        │
│  │              │     │   Chunks)    │     │   RAG)       │        │
│  └──────────────┘     └──────────────┘     └──────┬───────┘        │
│                                                    │                 │
│  ┌─────────────────────────────────────────────────┼───────────┐    │
│  │              LANGGRAPH PIPELINE                  │           │    │
│  │                                                  ▼           │    │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐  │    │
│  │  │ PLANNER  │──▶│ EXPLORER │──▶│EXTRACTOR │──▶│REVIEWER│  │    │
│  │  │ (new)    │   │          │   │          │   │        │  │    │
│  │  │ Decompose│   │ Agentic  │   │ Schema-  │   │ Verify │  │    │
│  │  │ query    │   │ retrieval│   │ driven   │   │ + flag │  │    │
│  │  └──────────┘   └────┬─────┘   └────┬─────┘   └───┬────┘  │    │
│  │                      │              │              │        │    │
│  │                      ▼              ▼              ▼        │    │
│  │              ┌──────────────────────────────────────┐       │    │
│  │              │         TOOL REGISTRY                │       │    │
│  │              │  search | read_section | vision      │       │    │
│  │              │  query_knowledge | extract | validate│       │    │
│  │              └──────────────────────────────────────┘       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    CONTEXT ENGINE (new)                       │   │
│  │  Relevance scoring → Tiered assembly → Token-aware budget    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   MODEL ABSTRACTION LAYER (new)              │   │
│  │  LiteLLM / LangChain → Claude, GPT-4o, local models         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Implementation Plan (5 Phases)

### Phase 1: Tool Registry + Knowledge as Tools
**Effort**: Medium | **Impact**: High | **Risk**: Low

**What changes:**
- Create `tools/` directory with `registry.py`, `base.py`
- Move `search`, `read_section`, `vision_extract` from inline `@tool` to proper tool classes
- Convert `KNOWLEDGE_APPENDICES` to a `QueryKnowledgeBase` tool that the agent calls on-demand
- Each tool: typed input/output (Pydantic), self-describing, independently testable
- Explorer/Extractor/Reviewer consume tools from registry, not inline definitions

**Why first:** Unblocks all other phases. Tools become composable building blocks.

**Files affected:**
- New: `tools/__init__.py`, `tools/base.py`, `tools/registry.py`, `tools/retrieval.py`, `tools/knowledge.py`, `tools/vision.py`
- Modified: `agents/explorer.py`, `agents/extractor_node.py`, `agents/reviewer.py`
- Deprecated: `KNOWLEDGE_APPENDICES` in `extraction/extractor.py`

---

### Phase 2: Contextual Retrieval + Better Reranking
**Effort**: Medium | **Impact**: Very High | **Risk**: Low

**What changes:**
- Add contextual prefix generation during indexing (cheap LLM call per chunk)
- Upgrade reranker from `ms-marco-MiniLM` to `bge-reranker-v2-m3`
- Implement parent-child retrieval (retrieve on small chunks, return parent section)
- Optional: Add HyDE for domain-specific queries

**Implementation:**
```python
# At index time (in llamaindex_retriever.py)
def _add_contextual_prefix(section_text: str, doc_summary: str) -> str:
    """Prepend LLM-generated context to chunk before embedding."""
    prefix = llm(f"Given this document about {doc_summary}, "
                 f"describe what this section covers in 1-2 sentences: "
                 f"{section_text[:500]}")
    return f"[{prefix}]\n\n{section_text}"
```

**Files affected:**
- Modified: `knowledge_base/llamaindex_retriever.py`
- Modified: `ingestion/__init__.py` (add contextual prefix step)

---

### Phase 3: Planner Node + Typed State Transitions
**Effort**: Medium | **Impact**: Medium-High | **Risk**: Medium

**What changes:**
- Add `PlannerNode` before Explorer that decomposes complex queries
- Replace string-based `NEED_MORE:` routing with typed enum transitions
- Add query decomposition for multi-part questions
- Implement proper state machine with `CycleSignal` enum

**New graph:**
```
START → Planner → Explorer → Extractor → Reviewer → END
                    ↑            │            │
                    └────────────┘            │
                    ↑                         │
                    └─────────────────────────┘
```

**State transitions (typed):**
```python
class CycleSignal(Enum):
    PROCEED = "proceed"
    NEED_MORE_CONTENT = "need_more_content"
    NEED_VERIFICATION = "need_verification"
    COMPLETE = "complete"

# Instead of: error.startswith("NEED_MORE:")
# Use: state.cycle_signal == CycleSignal.NEED_MORE_CONTENT
```

**Files affected:**
- New: `agents/planner.py`
- Modified: `agents/graph.py`, `agents/state.py`, `agents/explorer.py`, `agents/extractor_node.py`

---

### Phase 4: Context Engine + Session Memory
**Effort**: High | **Impact**: High | **Risk**: Medium

**What changes:**
- New `context/` module for intelligent context assembly
- Relevance scoring of retrieved sections (per-query, not just retrieval score)
- Tiered context: verbatim / key-paragraphs / summary based on relevance
- Token-aware budget (not char-based)
- Session memory: cache extraction results across queries in same upload

**Implementation:**
```python
class ContextEngine:
    def assemble(self,
                 sections: dict[str, str],
                 tables: dict[str, str],
                 query: str,
                 schema: type[BaseModel],
                 token_budget: int = 30_000) -> AssembledContext:
        """
        1. Score each section's relevance to query + schema fields
        2. Tier: verbatim (>0.8), key-paragraphs (0.5-0.8), summary (<0.5)
        3. Assemble within token budget, highest relevance first
        4. Add structural markers for grounding
        """
        ...
```

**Files affected:**
- New: `context/__init__.py`, `context/engine.py`, `context/scoring.py`, `context/compression.py`
- New: `memory/session.py`
- Modified: `agents/extractor_node.py` (use ContextEngine instead of manual assembly)

---

### Phase 5: Model Abstraction Layer
**Effort**: Low-Medium | **Impact**: Medium | **Risk**: Low

**What changes:**
- Replace `langchain_openai.ChatOpenAI` with model-agnostic interface
- Configure best model per task (extraction, vision, planning, reranking)
- Support Claude, GPT-4o, and local models via LiteLLM or LangChain's generic interface
- Environment-based model routing

**Config:**
```python
# config.py or .env
MODEL_EXTRACTION = "claude-sonnet-4-6"      # Best at structured output
MODEL_VISION = "gpt-4o"                      # Strong vision
MODEL_PLANNING = "claude-sonnet-4-6"         # Strong reasoning
MODEL_CHEAP = "claude-haiku-4-5-20251001"   # Contextual prefixes, repair
MODEL_RERANKER = "bge-reranker-v2-m3"        # Local, no API cost
```

**Files affected:**
- Modified: `config.py`
- New: `models/provider.py`
- Modified: All files importing `ChatOpenAI` directly

---

## 5. Priority Matrix

```
                        HIGH IMPACT
                            │
         Phase 2            │         Phase 4
    (Contextual Retrieval)  │    (Context Engine)
                            │
   ─────────────────────────┼─────────────────────────
                            │
         Phase 1            │         Phase 3
    (Tool Registry)         │    (Planner + Types)
                            │
                        LOW IMPACT

    LOW EFFORT ─────────────┼───────────── HIGH EFFORT
```

**Recommended order**: Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

Phase 1 is first because it unblocks clean implementation of everything else.
Phase 2 is second because it has the highest measurable impact on output quality.

---

## 6. What NOT to Do

1. **Don't rewrite from scratch** — The foundation (LangGraph + LlamaIndex + Pydantic schemas) is correct. Refactor iteratively.

2. **Don't switch to CrewAI or AutoGen** — They're built for loosely-defined multi-agent chat. Your pipeline is a precisely-defined state machine, which is LangGraph's strength.

3. **Don't add full autonomous agents** — Clinical data extraction needs predictability. Keep the structured pipeline with controlled cycles.

4. **Don't over-abstract** — MCP server is premature unless you need to share tools across systems. Structure tools to be MCP-compatible, but don't implement the protocol yet.

5. **Don't add long-term memory** — Single-protocol analysis doesn't benefit from cross-session persistence. Session memory (Phase 4) is sufficient.

---

## 7. Success Metrics

| Metric | Current (Estimated) | Target After Phases 1-4 |
|---|---|---|
| Retrieval recall (relevant sections found) | ~70% | >90% (contextual retrieval) |
| Extraction accuracy (verified fields) | ~85% | >93% (better context) |
| Context utilization (relevant tokens / total tokens) | ~50% | >80% (context engine) |
| Tool reusability (shared across agents) | 0% | 100% (tool registry) |
| Model flexibility (supported providers) | 1 (OpenAI) | 3+ (OpenAI, Anthropic, local) |
| Query decomposition | None | Auto-decompose complex queries |

---

## References

- [Building Effective AI Agents - Anthropic (2025)](https://resources.anthropic.com/building-effective-ai-agents)
- [Effective Context Engineering for AI Agents - Anthropic (2025)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Contextual Retrieval - Anthropic (Sep 2024)](https://www.anthropic.com/news/contextual-retrieval)
- [MCP Specification (Nov 2025)](https://modelcontextprotocol.io/specification/2025-11-25)
- [Agentic RAG Survey (arXiv, Jan 2025)](https://arxiv.org/abs/2501.09136)
- [LangGraph vs CrewAI vs OpenAI Agents SDK (2026)](https://particula.tech/blog/langgraph-vs-crewai-vs-openai-agents-sdk-2026)
- [2026 Agentic Coding Trends Report - Anthropic](https://resources.anthropic.com/hubfs/2026%20Agentic%20Coding%20Trends%20Report.pdf)
