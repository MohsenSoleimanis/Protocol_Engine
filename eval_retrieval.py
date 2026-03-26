"""
Retrieval Evaluation — Run this to see exactly what the retriever finds.

Usage:
    python eval_retrieval.py

Tests each query type and shows:
  - What sections the retriever found
  - What the EXPECTED sections are (ground truth)
  - Precision and recall
"""
import json
import sys
import logging

logging.basicConfig(level=logging.WARNING)

# Load structured JSON
json_path = "output/Prot_000_structured.json"
try:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded: {json_path}")
except FileNotFoundError:
    print(f"ERROR: {json_path} not found. Upload a protocol first.")
    sys.exit(1)

from knowledge_base.llamaindex_retriever import build_retriever

retriever = build_retriever(store, data, os.environ.get("OPENAI_API_KEY", ""))

# Try to build embeddings — if fails, BM25 only
try:
    retriever.build_embeddings()
    print("Embeddings: OK")
except Exception as e:
    print(f"Embeddings failed: {e} — using BM25 only")

print()

# Ground truth: what sections SHOULD be found for each query type
# NOTE: This is a single-protocol regression test for Moderna mRNA-1273-P301.
# For a generalizable eval, build ground truth per protocol or use RAGAS.
# Section IDs and table IDs are specific to this protocol's structure.
GROUND_TRUTH = {
    "endpoints": {
        "query": "objectives endpoints primary secondary exploratory",
        "expected_sections": ["3"],  # §3 has all endpoints
        "expected_tables": ["table_p54_1"],  # endpoints table
    },
    "eligibility": {
        "query": "inclusion exclusion criteria eligibility",
        "expected_sections": ["5.1", "5.1.1", "5.1.2"],
        "expected_tables": [],
    },
    "study_design": {
        "query": "study design phase randomization blinding",
        "expected_sections": ["4", "4.1", "4.1.1", "4.2"],
        "expected_tables": [],
    },
    "safety": {
        "query": "safety adverse events monitoring stopping rules AESI",
        "expected_sections": ["8", "8.3", "8.3.4", "8.3.5"],
        "expected_tables": [],
    },
    "soa": {
        "query": "schedule of activities visits procedures",
        "expected_sections": ["11.1"],
        "expected_tables": ["table_p139_1", "table_p142_1", "table_p145_1"],
    },
    "deviation": {
        "query": "inclusion exclusion criteria eligibility",
        "expected_sections": ["5.1", "5.1.1", "5.1.2"],
        "expected_tables": [],
    },
}

print("=" * 70)
print("RETRIEVAL EVALUATION")
print("=" * 70)

total_precision = 0
total_recall = 0
count = 0

for qtype, gt in GROUND_TRUTH.items():
    query = gt["query"]
    expected_secs = set(gt["expected_sections"])
    expected_tbls = set(gt["expected_tables"])
    expected_all = expected_secs | expected_tbls
    
    # Run retrieval
    results = retriever.retrieve(query, top_k=10, use_reranker=True, query_type=qtype)
    
    # What was found
    found_secs = set()
    found_tbls = set()
    for unit, score in results:
        if unit.unit_type == "section":
            found_secs.add(unit.section_id)
        else:
            found_tbls.add(unit.table_id)
    found_all = found_secs | found_tbls
    
    # Metrics
    true_positives = expected_all & found_all
    precision = len(true_positives) / len(found_all) if found_all else 0
    recall = len(true_positives) / len(expected_all) if expected_all else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    total_precision += precision
    total_recall += recall
    count += 1
    
    # Display
    print(f"\n{qtype.upper()}")
    print(f"  Query: '{query[:60]}'")
    print(f"  Expected: {sorted(expected_all)}")
    print(f"  Found:    {sorted(found_all)}")
    print(f"  Correct:  {sorted(true_positives)}")
    
    missing = expected_all - found_all
    if missing:
        print(f"  MISSING:  {sorted(missing)}")
    
    extra = found_all - expected_all
    if extra:
        print(f"  Extra:    {sorted(extra)} (may be useful context)")
    
    print(f"  Precision: {precision:.0%}  Recall: {recall:.0%}  F1: {f1:.0%}")
    
    # Show ranked results
    print(f"  Top results:")
    for unit, score in results[:5]:
        uid = unit.section_id or unit.table_id
        marker = "✓" if uid in expected_all else " "
        print(f"    {marker} [{score:.2f}] {uid:15s} {unit.title[:50]}")

print(f"\n{'=' * 70}")
print(f"AVERAGE: Precision={total_precision/count:.0%}  Recall={total_recall/count:.0%}")
print(f"{'=' * 70}")

# Also test: what does BM25 alone find vs hybrid?
print(f"\n{'=' * 70}")
print("BM25 vs HYBRID COMPARISON")
print(f"{'=' * 70}")

for qtype in ["endpoints", "eligibility", "study_design"]:
    gt = GROUND_TRUTH[qtype]
    query = gt["query"]
    expected = set(gt["expected_sections"]) | set(gt["expected_tables"])
    
    bm25_results = retriever.bm25_search(query, top_k=8)
    hybrid_results = retriever.hybrid_search(query, top_k=8)
    
    bm25_found = set(u.section_id or u.table_id for u, _ in bm25_results)
    hybrid_found = set(u.section_id or u.table_id for u, _ in hybrid_results)
    
    bm25_recall = len(expected & bm25_found) / len(expected) if expected else 0
    hybrid_recall = len(expected & hybrid_found) / len(expected) if expected else 0
    
    print(f"\n  {qtype:15s}  BM25 recall: {bm25_recall:.0%}  Hybrid recall: {hybrid_recall:.0%}")
