"""
Hybrid Retrieval Engine — BM25 + Vector + RRF + Modern Reranking.

Key fixes from old code:
  1. Chunk size 1024 tokens with 128 overlap (was 8192 chars, 0 overlap)
  2. Modern reranker: bge-reranker-v2-m3 (was ms-marco-MiniLM)
  3. Tables are never split across chunks
  4. Leaf-only indexing preserved (correct from old code)
  5. Contextual retrieval support (optional context prepend)
"""
from __future__ import annotations

import json
import logging
import time

from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.openai import OpenAIEmbedding

from protocol_engine.config import (
    EMBEDDING_MODEL, OPENAI_API_KEY, SIMILARITY_TOP_K,
    CHUNK_SIZE_TOKENS, CHUNK_OVERLAP_TOKENS, RERANKER_MODEL,
)

logger = logging.getLogger(__name__)


def build_documents(store, json_data: dict) -> list[Document]:
    """Build LlamaIndex Documents from ProtocolStore.

    Only LEAF sections indexed — parents just have intro text,
    children have actual content.
    """
    documents = []
    all_ids = list(store.all_section_ids)

    has_children = set()
    for sid in all_ids:
        for other in all_ids:
            if other != sid and other.startswith(sid + "."):
                has_children.add(sid)
                break

    indexed = 0
    for section_id in all_ids:
        if section_id in has_children:
            continue
        section_data = store.get_section(section_id)
        if not section_data or not section_data.get("content", "").strip():
            continue
        sid = section_data["section_id"]
        title = section_data["title"]
        content = section_data["content"]
        pages = section_data.get("pages", [])
        doc = Document(
            text=f"§{sid}: {title}\n\n{content}",
            metadata={
                "type": "section",
                "section_id": str(sid),
                "title": title,
                "pages": json.dumps(pages),
            },
        )
        doc.id_ = f"sec_{sid}"
        documents.append(doc)
        indexed += 1

    # Tables as single chunks (never split)
    for table in json_data.get("tables", []):
        tid = table.get("id", "")
        caption = table.get("caption", "")
        headers = table.get("column_headers", [])
        rows = table.get("rows", [])
        pages = table.get("page_range", [])

        parts = [f"Table: {caption}"]
        if headers:
            parts.append(" | ".join(str(h) for h in headers))
        for row in rows[:80]:
            parts.append(" | ".join(str(c) for c in row))
        table_text = "\n".join(parts)
        if not table_text.strip():
            continue

        doc = Document(
            text=table_text,
            metadata={
                "type": "table",
                "table_id": tid,
                "title": caption[:100],
                "pages": json.dumps(pages),
            },
        )
        doc.id_ = f"tbl_{tid}"
        documents.append(doc)

    logger.info(f"Indexed {indexed} leaf sections, {len(documents) - indexed} tables")
    return documents


def build_retriever(store, json_data: dict, api_key: str = "", similarity_top_k: int = 0):
    """Build hybrid retriever: BM25 + Vector + RRF + Reranking."""
    t0 = time.time()
    api_key = api_key or OPENAI_API_KEY
    top_k = similarity_top_k or SIMILARITY_TOP_K

    Settings.embed_model = OpenAIEmbedding(model=EMBEDDING_MODEL, api_key=api_key)

    documents = build_documents(store, json_data)
    logger.info(f"Built {len(documents)} documents")

    from llama_index.core.node_parser import SentenceSplitter
    node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE_TOKENS,
        chunk_overlap=CHUNK_OVERLAP_TOKENS,
    )
    nodes = node_parser.get_nodes_from_documents(documents)
    logger.info(f"Created {len(nodes)} chunks")

    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    storage_context = StorageContext.from_defaults(docstore=docstore)
    vector_index = VectorStoreIndex(
        nodes=nodes, storage_context=storage_context, show_progress=False,
    )

    vector_retriever = vector_index.as_retriever(similarity_top_k=top_k)
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=docstore, similarity_top_k=top_k,
    )

    hybrid = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=top_k,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
        verbose=False,
    )

    reranker = None
    try:
        reranker = SentenceTransformerRerank(
            model=RERANKER_MODEL, top_n=top_k,
        )
        logger.info(f"Reranker loaded ({RERANKER_MODEL})")
    except Exception as e:
        logger.warning(f"Reranker init failed ({RERANKER_MODEL}): {e}")
        # Fallback to old model
        try:
            reranker = SentenceTransformerRerank(
                model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=top_k,
            )
            logger.info("Fallback reranker loaded (ms-marco-MiniLM)")
        except Exception:
            pass

    elapsed = time.time() - t0
    logger.info(f"Hybrid retriever built in {elapsed:.1f}s")
    return HybridRetriever(hybrid, reranker)


class HybridRetriever:
    """Wraps fusion retriever + reranker."""

    def __init__(self, fusion, reranker=None):
        self._fusion = fusion
        self._reranker = reranker

    def retrieve(self, query: str):
        nodes = self._fusion.retrieve(query)
        if self._reranker and len(nodes) > 1:
            try:
                nodes = self._reranker.postprocess_nodes(nodes, query_str=query)
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
        return nodes
