"""
Protocol Engine — Centralized configuration.

All settings in one place. Environment variables override defaults.
Supports multiple LLM providers via LiteLLM-compatible model strings.
"""
from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"

# ── LLM Configuration ────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Primary extraction model (structured output)
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
# Vision model (table images)
VLM_MODEL = os.getenv("VLM_MODEL", "gpt-4o")
# Cheap model (routing, summarization)
FAST_MODEL = os.getenv("FAST_MODEL", "gpt-4o-mini")
# Embedding model
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# Reranker model
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

# ── Retrieval Configuration ──────────────────────────────────────────────────
SIMILARITY_TOP_K = int(os.getenv("SIMILARITY_TOP_K", "12"))
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", "1024"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "128"))

# ── Context Engineering ──────────────────────────────────────────────────────
# Token budget for context assembly (not char-based)
CONTEXT_BUDGET_TOKENS = int(os.getenv("CONTEXT_BUDGET_TOKENS", "32000"))
# Relevance thresholds for tiered inclusion
HIGH_RELEVANCE_THRESHOLD = float(os.getenv("HIGH_RELEVANCE_THRESHOLD", "0.8"))
MEDIUM_RELEVANCE_THRESHOLD = float(os.getenv("MEDIUM_RELEVANCE_THRESHOLD", "0.5"))

# ── Graph Configuration ──────────────────────────────────────────────────────
MAX_CYCLES = int(os.getenv("MAX_CYCLES", "2"))
MAX_EXPLORER_TURNS = int(os.getenv("MAX_EXPLORER_TURNS", "8"))
MAX_EXTRACTOR_TURNS = int(os.getenv("MAX_EXTRACTOR_TURNS", "6"))
MAX_REVIEWER_TURNS = int(os.getenv("MAX_REVIEWER_TURNS", "8"))
VISION_CALL_LIMIT = int(os.getenv("VISION_CALL_LIMIT", "6"))

# ── Langfuse (optional tracing) ─────────────────────────────────────────────
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")


def get_openai_client():
    """Return an OpenAI client instance."""
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


def get_langfuse_handler():
    """Return Langfuse callback handler if configured, else None."""
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        return None
    try:
        from langfuse.callback import CallbackHandler
        return CallbackHandler(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
    except ImportError:
        return None
