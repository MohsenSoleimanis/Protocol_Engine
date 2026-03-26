# Protocol Intelligence System

**Agentic AI for clinical trial protocol understanding.**

Processes clinical protocol PDFs and extracts structured, grounded information for downstream use in protocol deviation detection, risk-based monitoring, and intelligent query generation.

## Architecture

```
PDF → Parser (PyMuPDF + pdfplumber, deterministic)
       → Structured JSON (sections, tables, cross-refs)
       → Manifest (1 LLM call: classifies sections by domain, cached)
       → Hybrid Index (BM25 + semantic embeddings on 180+ units)

Query → Orchestrator (Python-driven, iterative)
         │
         ├── Step 1: Domain routing (manifest → section titles → search queries)
         │           Falls back to keyword search if manifest unavailable
         │
         ├── Step 2: Hybrid retrieval (BM25 + semantic + LLM reranker)
         │           Scores all sections + tables, merges results
         │
         ├── Step 3: Page expansion (true section boundaries, not parser's)
         │           §3 parser says [54] → true range 54-58 (next section starts 59)
         │           All tables within range automatically included
         │
         ├── Step 4: Content assembly
         │           Table pages → use PARSED table from JSON (merged, clean)
         │           Non-table pages → raw PyMuPDF text (bullets preserved)
         │           SoA tables → GPT-4o vision (checkmarks, many columns)
         │
         ├── Step 5: Extraction (1 LLM call, schema-driven)
         │           ONE generic prompt + Pydantic schema example = instruction
         │           CDISC SDTM appendix auto-injected for deviation rules
         │           OpenAI json_object mode for guaranteed valid JSON
         │
         ├── Step 6: Completeness check (Python, no LLM)
         │           Counts categories found vs expected
         │           If PARTIAL → round 2 with gap-targeted search
         │
         └── Step 7: Validation (4-level deterministic)
                     Level 1: Source grounding (text exists in context)
                     Level 2: Numerical consistency (38°C in claim matches source)
                     Level 3: Section reference integrity (page markers exist)
                     Level 4: Completeness (endpoints need timing, rules need conditions)
```

## Analysis Types (10)

| Type | Schema | Retrieval Strategy |
|------|--------|--------------------|
| Endpoints | EndpointExtraction | manifest:endpoints → §3 |
| Eligibility | EligibilityExtraction | manifest:eligibility → §5 |
| Safety | SafetyExtraction | manifest:safety → §8 |
| Deviation Rules | DeviationRuleSet + CDISC | manifest:eligibility + soa → §5 + §6 |
| Risk Assessment | RiskAssessment | manifest:safety + endpoints + soa |
| Ambiguity | AmbiguityAnalysis | manifest:eligibility + safety + soa |
| Consistency | ConsistencyCheck | manifest:overview + endpoints + statistical |
| Schedule of Activities | SoAExtraction + Vision | manifest:soa + GPT-4o vision on SoA tables |
| Study Design | StudyDesignExtraction | manifest:study_design + overview |
| General Q&A | GeneralExtraction | user's query → hybrid search |

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env    # add OPENAI_API_KEY
python app.py
# Open http://localhost:8000
```

## LLM Calls

| When | Model | Cost |
|------|-------|------|
| Table repair (parsing, rare) | gpt-4o-mini | ~$0.0005 |
| Manifest (once, cached) | gpt-4o-mini | ~$0.002 |
| Embeddings (once, cached in memory) | text-embedding-3-small | ~$0.0001 |
| LLM Reranker (per query) | gpt-4o-mini | ~$0.001 |
| Extraction (per query, 1-2 rounds) | gpt-4o-mini | ~$0.002-0.004 |
| Vision for SoA tables (optional) | gpt-4o | ~$0.03/table |

**Typical query: ~$0.005, 15-30 seconds**

## Key Design Decisions

1. **Schema-driven extraction**: One generic prompt. The Pydantic schema example teaches the LLM what to extract. Adding a new analysis type = define schema + one JSON example. Zero prompt engineering.

2. **Manifest-based domain routing**: The manifest classifies sections by domain (eligibility, safety, etc). Retrieval looks up which domains each query needs. Works for any protocol because the manifest adapts.

3. **True section boundaries**: Parser says §3 starts at page 54. We compute §3 ENDS at page 58 (where §4 starts). All tables in that range are automatically included. No content lost.

4. **Merged table injection**: Multi-page tables use the PARSED structure from JSON (already merged). Non-table pages use raw PDF text. LLM sees one clean table, not 5 pages of scattered text.

5. **Iterative self-correction**: If extraction misses exploratory endpoints (PARTIAL status), Python detects the gap and triggers a second round with targeted search. No human intervention needed.

6. **4-level deterministic validation**: Source grounding + numerical consistency + reference integrity + completeness. Catches hallucinated thresholds (39°C vs 38°C) with zero LLM calls.

7. **CDISC SDTM knowledge for deviation rules**: The deviation schema auto-injects a CDISC reference table. Maps eligibility criteria to DM.AGE, LB.LBTESTCD, MH.MHDECOD — the variables CluePoints' SMART engine needs.

## Debug Log

Every query writes a full trace to `output/debug_log.txt`:
- Retrieval candidates with scores
- Section expansion (parser range → true range)
- Which tables came from JSON vs raw PDF
- **Complete content sent to LLM**
- **Complete LLM response**
- Every validation check at all 4 levels

## Files

```
app.py                          FastAPI server
orchestrator.py                 Python-driven retrieve-extract loop
config.py                       LLM settings
debug_logger.py                 Full query trace logger
knowledge_base/
  hybrid_retriever.py           BM25 + semantic + LLM reranker
  protocol_store.py             Page-indexed protocol store
  manifest_builder.py           LLM section classification (cached)
extraction/
  extractor.py                  Schema-driven extraction (1 generic prompt)
  schemas.py                    Pydantic models for all 10 types
  validator.py                  4-level deterministic validation
  vision_table.py               GPT-4o vision for complex tables
ingestion/
  run.py                        4-phase PDF parser orchestration
  src/                          Parser components
static/
  index.html                    UI with Analysis Types + Chat
```
