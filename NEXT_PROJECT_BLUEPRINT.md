# Universal Document Intelligence — Architecture Blueprint

## What This Is

A production-grade system that extracts structured data from **any document format**
(PDF, Word, CSV, Excel, images, scanned docs) using a shared foundation + pluggable
domain agents. Built on LangGraph + MCP tools.

---

## Core Insight: Two Layers

```
┌─────────────────────────────────────────────────────────┐
│  DOMAIN LAYER (pluggable per use case)                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Clinical  │  │ Financial│  │ Legal    │  ...more    │
│  │ Protocol  │  │ Report   │  │ Contract │             │
│  │ Agent     │  │ Agent    │  │ Agent    │             │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘             │
│       │              │              │                    │
├───────┼──────────────┼──────────────┼────────────────────┤
│  FOUNDATION LAYER (shared across all domains)           │
│  ┌──────────────────────────────────────────────┐       │
│  │ Ingestion → Retrieval → Extraction → Review  │       │
│  │   (any format)  (hybrid)   (schema)   (eval) │       │
│  └──────────────────────────────────────────────┘       │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │
│  │ Parsers │ │ MCP Tools│ │Guardrails│ │ Eval/Obs  │  │
│  └─────────┘ └──────────┘ └──────────┘ └───────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Foundation layer** handles what EVERY document intelligence task needs.
**Domain layer** adds schemas, prompts, domain knowledge per use case.

---

## 1. Foundation: Universal Components

### 1a. Multi-Format Ingestion

The #1 lesson from research: **no single parser handles all formats**.
Use a router that picks the best parser per file type.

```
Incoming file
  │
  ├─ .pdf  → Docling (best tables, 97.9% accuracy, self-hosted)
  │           + pdfplumber fallback (borderless via "text" strategy)
  │           + Vision LLM (complex/scanned tables)
  │
  ├─ .docx → python-docx (paragraphs, tables, headers)
  │           + docling for complex Word docs
  │
  ├─ .csv  → pandas (structured, typed columns)
  │           + LLM for schema inference on messy CSVs
  │
  ├─ .xlsx → openpyxl (preserves sheets, merged cells, formulas)
  │           + pandas for flat data extraction
  │
  ├─ .png/.jpg → Docling OCR or Vision LLM
  │               + pytesseract fallback
  │
  └─ .html → BeautifulSoup → structured text
```

**Key library choices (2025-2026 benchmarks):**

| Format | Primary Parser | Why |
|--------|---------------|-----|
| PDF | **Docling** (IBM) | 97.9% table accuracy, free, self-hosted, handles merged cells |
| PDF fallback | **LlamaParse** | 6s consistent speed, good for high-volume |
| Word | **python-docx** | Native, fast, preserves structure |
| CSV/Excel | **pandas + openpyxl** | Industry standard, typed |
| Images | **Docling OCR** | Better than raw pytesseract |
| Scanned | **Vision LLM** (GPT-4o/Claude) | Best for degraded scans |

**Output**: Every parser produces the same intermediate format:

```python
@dataclass
class ParsedDocument:
    """Universal intermediate format — all parsers produce this."""
    source_path: str
    source_format: str           # "pdf", "docx", "csv", "xlsx", "image"
    title: str
    total_pages: int             # 1 for CSV/single-sheet

    sections: list[Section]      # Hierarchical text sections
    tables: list[Table]          # Structured tables (headers + rows)
    figures: list[Figure]        # Images/charts with captions
    metadata: dict               # Format-specific metadata

    # Provenance
    parse_method: str            # "docling", "python-docx", "pandas", etc.
    parse_timestamp: str
    confidence: float            # Overall parse quality score
```

### 1b. Universal Chunking

Same contextual chunking strategy, adapted per format:

```python
class ChunkingStrategy:
    """Format-aware chunking rules."""

    PDF:    section-boundary aware, tables never split, 1024 tokens + context prefix
    DOCX:   paragraph-boundary aware, tables never split, same token budget
    CSV:    each row group = one chunk (or full table if small)
    XLSX:   each sheet = separate document, then chunk per section
    IMAGE:  entire OCR output = one chunk (usually small)
```

### 1c. Universal Retrieval

Same hybrid engine for all formats:

```
BM25 + Vector (text-embedding-3-small) + RRF + Reranker (bge-reranker-v2-m3)
```

The retrieval engine is format-agnostic — it operates on chunks.
The only format-specific logic is in the metadata filters (source_format, table_id, etc.).

### 1d. Universal MCP Tools

Tools that work across ALL domains:

```python
# ── Always available ──────────────────────────────────
search(query: str) → str
    "Semantic search across all parsed documents."

read_section(doc_id: str, section_id: str) → str
    "Read a specific section from a specific document."

read_table(doc_id: str, table_id: str) → str
    "Read a specific table with headers and rows."

list_documents() → str
    "List all loaded documents with metadata."

compare_sections(doc_a: str, section_a: str, doc_b: str, section_b: str) → str
    "Compare two sections across documents."

# ── Format-specific (auto-enabled) ────────────────────
vision_extract(doc_id: str, pages: list[int]) → str
    "Extract complex content from page images. PDF/image only."

query_spreadsheet(doc_id: str, sheet: str, filter: str) → str
    "Query CSV/Excel data with natural language filter."

# ── Domain-specific (pluggable) ───────────────────────
lookup_terminology(domain: str, term: str) → str
    "Look up domain terminology (CDISC, GAAP, legal terms, etc.)"
```

### 1e. Universal Guardrails

```python
class InputGuardrails:
    sanitize_query()         # Injection detection, length limits
    validate_document()      # File type check, size limits, malware scan
    validate_schema_type()   # Is the requested extraction type valid?

class OutputGuardrails:
    validate_schema()        # Does output match Pydantic schema?
    check_plausibility()     # Domain-aware sanity checks
    detect_pii()             # Flag PII in output
    check_hallucination()    # Source grounding verification
```

### 1f. Universal Evaluation

```python
class ExtractionScorer:
    grounding_rate()         # % of claims traceable to source
    completeness()           # % of fields populated
    numerical_fidelity()     # Numbers match source?
    cross_field_consistency() # Internal consistency
    domain_accuracy()        # Domain-specific checks (pluggable)
```

---

## 2. The 3-Node Graph (Same for All Domains)

```python
"""The graph is IDENTICAL for every domain.
The domain-specific behavior comes from prompts + schemas + tools."""

def build_graph(domain_config: DomainConfig):

    START → Explorer → Extractor → Reviewer → END
                ↑                       |
                └───── (need_more) ─────┘

    # Explorer: retrieve + assemble context
    #   - Uses domain_config.retrieval_queries
    #   - Uses domain_config.goals
    #   - Tools: search, read_section, read_table, [domain tools]

    # Extractor: extract to schema + validate
    #   - Uses domain_config.extraction_prompt
    #   - Uses domain_config.schema_class
    #   - Runs deterministic validation + domain checks

    # Reviewer: semantic review
    #   - Uses domain_config.review_prompt
    #   - Only checks what validator can't (hallucination, plausibility)
    #   - Decides: DONE or NEED_MORE
```

**The graph never changes. Only the config changes.**

---

## 3. Domain Configuration (Pluggable)

Each domain is a folder with 4 files:

```
domains/
├── clinical_protocol/
│   ├── config.yaml          # Goals, retrieval queries, settings
│   ├── prompts.yaml         # Extraction + review prompts
│   ├── schemas.py           # Pydantic extraction schemas
│   └── knowledge/           # Domain knowledge (CDISC, ICH, etc.)
│       ├── cdisc.json
│       └── ich_guidelines.json
│
├── financial_report/
│   ├── config.yaml
│   ├── prompts.yaml
│   ├── schemas.py           # Revenue, expenses, ratios, etc.
│   └── knowledge/
│       └── gaap.json
│
├── legal_contract/
│   ├── config.yaml
│   ├── prompts.yaml
│   ├── schemas.py           # Parties, obligations, terms, etc.
│   └── knowledge/
│       └── legal_terms.json
│
└── general/                  # Default — works for any document
    ├── config.yaml
    ├── prompts.yaml
    └── schemas.py
```

Example `config.yaml` for a financial report domain:

```yaml
name: financial_report
description: "Extract structured data from financial reports (10-K, 10-Q, annual reports)"

extraction_types:
  - revenue_breakdown
  - expense_analysis
  - balance_sheet
  - cash_flow
  - risk_factors
  - executive_summary

goals:
  revenue_breakdown: "Find all revenue segments, year-over-year comparisons, geographic breakdown"
  expense_analysis: "Find operating expenses, R&D, SGA, cost of revenue"
  balance_sheet: "Find total assets, liabilities, equity, current ratio"
  risk_factors: "Find all disclosed risk factors and forward-looking statements"

retrieval_queries:
  revenue_breakdown:
    - "revenue net sales segment breakdown geographic"
    - "year over year growth decline comparison"
  expense_analysis:
    - "operating expenses cost of revenue research development"
    - "selling general administrative compensation"
  balance_sheet:
    - "total assets liabilities stockholders equity"
    - "current assets current liabilities working capital"

# Domain-specific settings
settings:
  require_source_grounding: true
  numerical_precision: 2        # Decimal places for financial numbers
  currency_detection: true
  fiscal_year_detection: true
```

---

## 4. Project Structure

```
document_intelligence/
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── prompts/                          # GLOBAL prompt templates
│   ├── explorer.yaml
│   ├── extractor.yaml
│   └── reviewer.yaml
│
├── domains/                          # PLUGGABLE domain configs
│   ├── clinical_protocol/
│   ├── financial_report/
│   ├── legal_contract/
│   └── general/
│
├── evals/                            # Evaluation framework
│   ├── score.py
│   ├── cases/                        # Test cases per domain
│   │   ├── clinical/
│   │   └── financial/
│   └── README.md
│
├── app/                              # Core application
│   ├── __init__.py
│   ├── config.py                     # Centralized config
│   ├── main.py                       # Entry point: initialize() + run_query()
│   ├── store.py                      # Multi-document store
│   ├── prompts.py                    # YAML prompt loader
│   │
│   ├── ingestion/                    # Multi-format parsing
│   │   ├── router.py                 # File type → parser routing
│   │   ├── parsers/
│   │   │   ├── pdf.py                # Docling + pdfplumber
│   │   │   ├── docx.py              # python-docx
│   │   │   ├── csv_xlsx.py          # pandas + openpyxl
│   │   │   ├── image.py             # OCR (Docling/pytesseract)
│   │   │   └── html.py              # BeautifulSoup
│   │   ├── chunker.py               # Contextual chunking
│   │   └── models.py                # ParsedDocument, Section, Table
│   │
│   ├── retrieval/
│   │   └── engine.py                 # Hybrid BM25 + vector + reranker
│   │
│   ├── graph/                        # LangGraph pipeline
│   │   ├── builder.py                # 3-node graph (same for all domains)
│   │   └── nodes/
│   │       ├── explorer.py
│   │       ├── extractor.py
│   │       └── reviewer.py
│   │
│   ├── tools/                        # MCP-compatible tools
│   │   ├── search.py
│   │   ├── read_section.py
│   │   ├── read_table.py
│   │   ├── compare.py
│   │   ├── vision.py
│   │   ├── spreadsheet.py
│   │   └── terminology.py
│   │
│   ├── guardrails/
│   │   ├── input.py                  # Query sanitization, file validation
│   │   └── output.py                 # Schema compliance, PII, plausibility
│   │
│   └── validation/
│       └── validator.py              # Deterministic grounding checks
│
└── tests/
    ├── test_ingestion.py
    ├── test_retrieval.py
    ├── test_extraction.py
    └── test_guardrails.py
```

---

## 5. Key Entities

```python
# ── Universal document model ─────────────────────────

@dataclass
class ParsedDocument:
    doc_id: str                  # Unique, immutable
    source_path: str
    source_format: str           # pdf, docx, csv, xlsx, image, html
    title: str
    sections: list[Section]
    tables: list[Table]
    metadata: dict
    parse_confidence: float

@dataclass
class Section:
    section_id: str
    title: str
    content: str
    page_range: list[int]
    level: int                   # Heading depth
    parent_id: str | None

@dataclass
class Table:
    table_id: str
    caption: str
    headers: list[str]
    rows: list[list[str]]
    page_range: list[int]
    source_confidence: float

# ── Domain config (loaded from YAML) ─────────────────

@dataclass
class DomainConfig:
    name: str
    extraction_types: list[str]
    goals: dict[str, str]
    retrieval_queries: dict[str, list[str]]
    schema_map: dict[str, type]  # extraction_type → Pydantic class
    knowledge_dir: Path
    settings: dict

# ── LangGraph state (same for all domains) ────────────

class ExtractionState(TypedDict):
    query: str
    query_type: str              # Maps to domain extraction_type
    domain: str                  # "clinical_protocol", "financial_report", etc.
    doc_ids: list[str]           # Which documents to search

    sections_content: Annotated[dict, merge_dicts]
    tables_content: Annotated[dict, merge_dicts]
    assembled_context: str

    extracted_data: dict
    validation: dict
    signals: Annotated[list, operator.add]

    edge_signal: str             # "done" or "need_more"
    cycle_count: int
    steps: Annotated[list, operator.add]
```

---

## 6. Multi-Document Support

Unlike the current protocol_engine (single document), the new system handles
multiple documents simultaneously:

```python
class DocumentStore:
    """Manages multiple parsed documents."""

    def load(self, path: str) -> str:
        """Parse and index a document. Returns doc_id."""
        doc = parse(path)  # Router picks correct parser
        self._docs[doc.doc_id] = doc
        self._index(doc)   # Add to retrieval index
        return doc.doc_id

    def search(self, query: str, doc_ids: list[str] = None) -> list[Chunk]:
        """Search across all or specific documents."""

    def get_section(self, doc_id: str, section_id: str) -> Section:
        """Read a specific section from a specific document."""

    def compare(self, doc_a: str, sec_a: str, doc_b: str, sec_b: str) -> str:
        """Compare sections across documents."""
```

---

## 7. Technology Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Orchestration** | LangGraph 1.x | Durable execution, state persistence, 90M downloads/mo |
| **Tool protocol** | MCP (Model Context Protocol) | Universal standard, OAuth 2.1, adopted by OpenAI+Anthropic |
| **PDF parsing** | Docling (primary) + pdfplumber | 97.9% table accuracy, free, self-hosted |
| **Word parsing** | python-docx | Native, fast |
| **CSV/Excel** | pandas + openpyxl | Industry standard |
| **OCR** | Docling OCR + Vision LLM fallback | Best accuracy for degraded docs |
| **Embeddings** | text-embedding-3-small | Best price/performance |
| **Reranker** | BAAI/bge-reranker-v2-m3 | Modern, accurate |
| **Retrieval** | LlamaIndex (BM25 + vector + RRF) | Proven hybrid approach |
| **Schemas** | Pydantic v2 | Type-safe, LLM-friendly structured output |
| **Prompts** | YAML files (externalized) | Version-controlled, iterable without deploy |
| **Guardrails** | Custom (input + output) | PII, injection, plausibility |
| **Evaluation** | Custom scorer + LLM-as-judge | 5-metric scoring, CI/CD integration |
| **Tracing** | Langfuse or LangSmith | Full execution traces |
| **Deployment** | ECS Fargate or Docker | Long-running pipelines (30-120s) |
| **State** | PostgreSQL (checkpoints) + Redis (cache) | LangGraph native persistence |

---

## 8. What Moves From Current Project

From `protocol_engine/` → universal `document_intelligence/`:

| Current | New Location | Changes |
|---------|-------------|---------|
| `graph/builder.py` | `app/graph/builder.py` | Parameterized by DomainConfig |
| `graph/nodes/explorer.py` | `app/graph/nodes/explorer.py` | Loads goals/queries from domain config |
| `graph/nodes/extractor.py` | `app/graph/nodes/extractor.py` | Schema from domain config |
| `graph/nodes/reviewer.py` | `app/graph/nodes/reviewer.py` | Prompt from domain config |
| `tools/search.py` | `app/tools/search.py` | Multi-document aware |
| `tools/read_section.py` | `app/tools/read_section.py` | Requires doc_id |
| `tools/vision.py` | `app/tools/vision.py` | Unchanged |
| `retrieval/engine.py` | `app/retrieval/engine.py` | Multi-document index |
| `validation/validator.py` | `app/validation/validator.py` | Domain-pluggable checks |
| `guardrails/` | `app/guardrails/` | + file validation |
| `models/schemas.py` | `domains/clinical_protocol/schemas.py` | Domain-specific |
| `knowledge/` | `domains/clinical_protocol/knowledge/` | Domain-specific |
| `prompts/` | `prompts/` (global) + `domains/*/prompts.yaml` | Split global vs domain |
| `ingestion/` | `app/ingestion/parsers/pdf.py` | + new parsers for docx/csv/xlsx |
| `store.py` | `app/store.py` | Multi-document |
| `config.py` | `app/config.py` | + domain loading |
| `main.py` | `app/main.py` | + domain selection |

---

## 9. How to Build This (Recommended Order)

### Phase 1: Universal Foundation (Week 1-2)
- [ ] Project structure + config
- [ ] Universal document models (ParsedDocument, Section, Table)
- [ ] Multi-format ingestion router
- [ ] PDF parser (port from protocol_engine + add Docling)
- [ ] DOCX parser
- [ ] CSV/XLSX parser
- [ ] Contextual chunker (format-aware)
- [ ] Multi-document store

### Phase 2: Retrieval + Graph (Week 2-3)
- [ ] Hybrid retrieval engine (port from protocol_engine)
- [ ] MCP tools (search, read_section, read_table, compare, vision)
- [ ] 3-node LangGraph (port + parameterize by DomainConfig)
- [ ] Prompt loader (port)
- [ ] Guardrails (port + add file validation)

### Phase 3: Domain: Clinical Protocol (Week 3)
- [ ] Port schemas, knowledge, prompts from current project
- [ ] Domain config YAML
- [ ] Domain-specific validation rules
- [ ] Test against existing protocol PDFs

### Phase 4: Domain: Financial Report (Week 3-4)
- [ ] Financial schemas (revenue, expenses, balance sheet, etc.)
- [ ] Financial prompts + goals
- [ ] GAAP/IFRS knowledge base
- [ ] Test against 10-K/10-Q filings

### Phase 5: Domain: General (Week 4)
- [ ] General-purpose extraction schema
- [ ] Works with any document type
- [ ] No domain knowledge required
- [ ] Good default for unknown document types

### Phase 6: Production (Week 4-5)
- [ ] Evaluation framework + test cases per domain
- [ ] CI/CD pipeline
- [ ] Docker deployment
- [ ] Tracing integration (Langfuse)
- [ ] API endpoint (FastAPI)

---

## Sources

- [Docling vs LlamaParse vs Unstructured Benchmark](https://procycons.com/en/blogs/pdf-data-extraction-benchmark/)
- [Databricks Document Intelligence](https://www.databricks.com/blog/pdfs-production-announcing-state-art-document-intelligence-databricks)
- [LlamaIndex Agentic Document Workflows](https://www.llamaindex.ai/blog/introducing-agentic-document-workflows)
- [MCP Specification Nov 2025](https://modelcontextprotocol.io/specification/2025-11-25)
- [LangGraph Production Guide](https://www.alphabold.com/langgraph-agents-in-production/)
- [AI Agent Framework Comparison 2026](https://dev.to/linou518/the-2026-ai-agent-framework-decision-guide-langgraph-vs-crewai-vs-pydantic-ai-b2h)
- [Multi-Agent Document Intelligence](https://sema4.ai/blog/multi-ai-agent-revolution-in-document-intelligence/)
- [Unstructured.io](https://github.com/Unstructured-IO/unstructured)
- [LangGraph Multi-Agent Tutorial](https://dev.to/sidkul2000/production-ready-multi-agent-systems-with-langgraph-a-complete-tutorial-20j1)
