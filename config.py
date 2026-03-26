"""
Protocol Intelligence System — Configuration.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
VLM_MODEL = os.getenv("VLM_MODEL", "gpt-4o")
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "gpt-4o-mini")  # Always cheap: table repair, manifest
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8192"))
DEBUG_LOG_DIR = str(OUTPUT_DIR)

def get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=OPENAI_API_KEY)


# Langfuse tracing (optional - set env vars to enable)
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

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
