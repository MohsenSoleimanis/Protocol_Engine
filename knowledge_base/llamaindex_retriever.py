"""
LlamaIndex Hybrid Retriever — BM25 + Vector + RRF + Reranking.

Indexes leaf sections (no parent-child duplication) with clean content.
No manifest enrichment — retrieval queries come from schema field
descriptions at query time, not from domain tags at index time.
"""
from __future__ import annotations
import json, logging, time
from llama_index.core import Document, VectorStoreIndex, StorageContext, Settings
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.embeddings.openai import OpenAIEmbedding

logger = logging.getLogger(__name__)


def _build_documents(store, json_data: dict) -> list[Document]:
    """Build LlamaIndex Documents from ProtocolStore.

    Only LEAF sections indexed (parents have just intro paragraphs,
    children have the actual content — no duplication with content_blocks).
    """
    documents = []

    all_ids = list(store.all_section_ids)
    has_children = set()
    for sid in all_ids:
        for other in all_ids:
            if other != sid and other.startswith(sid + "."):
                has_children.add(sid)
                break

    indexed_count = 0
    skipped_parents = 0
    for section_id in all_ids:
        if section_id in has_children:
            skipped_parents += 1
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
            metadata={"type": "section", "section_id": str(sid),
                      "title": title, "pages": json.dumps(pages)},
        )
        doc.id_ = f"sec_{sid}"
        documents.append(doc)
        indexed_count += 1

    for table in json_data.get("tables", []):
        tid = table.get("id", "")
        caption = table.get("caption", "")
        headers = table.get("column_headers", [])
        rows = table.get("rows", [])
        pages = table.get("page_range", [])
        parts = [f"Table: {caption}"]
        if headers: parts.append(" | ".join(str(h) for h in headers))
        for row in rows[:80]: parts.append(" | ".join(str(c) for c in row))
        table_text = "\n".join(parts)
        if not table_text.strip():
            continue
        doc = Document(
            text=table_text,
            metadata={"type": "table", "table_id": tid,
                      "title": caption[:100], "pages": json.dumps(pages)},
        )
        doc.id_ = f"tbl_{tid}"
        documents.append(doc)

    logger.info(f"Indexed {indexed_count} leaf sections (skipped {skipped_parents} parents), "
                f"{len(documents) - indexed_count} tables")
    return documents


def build_retriever(store, json_data: dict, api_key: str, similarity_top_k: int = 8):
    """Build hybrid retriever: BM25 + Vector + RRF + Reranking."""
    t0 = time.time()
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", api_key=api_key)

    documents = _build_documents(store, json_data)
    logger.info(f"Built {len(documents)} documents "
                f"({sum(1 for d in documents if d.metadata.get('type') == 'section')} sections, "
                f"{sum(1 for d in documents if d.metadata.get('type') == 'table')} tables)")

    from llama_index.core.node_parser import SentenceSplitter
    node_parser = SentenceSplitter(chunk_size=8192, chunk_overlap=0)
    nodes = node_parser.get_nodes_from_documents(documents)
    logger.info(f"Created {len(nodes)} nodes (full documents, no fragmentation)")

    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)
    storage_context = StorageContext.from_defaults(docstore=docstore)
    vector_index = VectorStoreIndex(nodes=nodes, storage_context=storage_context, show_progress=False)

    vector_retriever = vector_index.as_retriever(similarity_top_k=similarity_top_k)
    bm25_retriever = BM25Retriever.from_defaults(docstore=docstore, similarity_top_k=similarity_top_k)

    hybrid_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=similarity_top_k,
        num_queries=1, mode="reciprocal_rerank", use_async=False, verbose=False,
    )

    try:
        reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=similarity_top_k)
        logger.info("Reranker loaded (cross-encoder/ms-marco-MiniLM-L-6-v2)")
    except Exception as e:
        logger.warning(f"Reranker init failed: {e}")
        reranker = None

    elapsed = time.time() - t0
    logger.info(f"Hybrid retriever built in {elapsed:.1f}s (BM25 + Vector + RRF"
                f"{' + Reranker' if reranker else ''})")
    return _FusedRetriever(hybrid_retriever, reranker)


class _FusedRetriever:
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
