#!/usr/bin/env python3
"""
Protocol Intelligence System — FastAPI Server.

Multi-agent architecture:
  Layer 1: Ingestion (parser) — runs once per PDF
  Layer 2: Explorer Agent — navigates protocol, follows cross-refs (agentic)
  Layer 3: Extractor Agent — structures content into schemas (agentic)
  Layer 4: Reviewer Agent — cross-checks extractions, flags signals (agentic)
  Layer 5: Validator — deterministic 4-level accuracy check

Run: python app.py
Open: http://localhost:8000
"""
from __future__ import annotations
import json
import sys
import time
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, str(Path(__file__).parent))

from config import OUTPUT_DIR, LLM_MODEL, OPENAI_API_KEY
from knowledge_base.protocol_store import ProtocolStore


logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# App State
# ═══════════════════════════════════════════════════════════════════════

app = FastAPI(title="Protocol Intelligence System")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class AppState:
    store: Optional[ProtocolStore] = None
    protocol_name: str = ""
    loaded: bool = False
    retriever = None
    pdf_path: str = ""
    json_data: Optional[dict] = None

state = AppState()


# ═══════════════════════════════════════════════════════════════════════
# Load from cached JSON (fast restart)
# ═══════════════════════════════════════════════════════════════════════

def load_protocol(json_path: str, pdf_path: str = None, force_rebuild: bool = False) -> dict:
    """Load a parsed protocol and build retriever."""
    logger.info(f"Loading protocol from {json_path}")

    # Load store
    state.store = ProtocolStore(json_path)
    meta = state.store.metadata
    state.protocol_name = meta.get("filename", Path(json_path).stem)
    logger.info(f"Store: {meta['total_pages']}p, {meta['total_sections']}s, {meta['total_tables']}t")

    # Discover bookmarks from PDF
    if pdf_path and Path(pdf_path).exists():
        state.store.discover_bookmarks(pdf_path)
    else:
        json_stem = Path(json_path).stem.replace("_structured", "")
        matching_pdfs = list(OUTPUT_DIR.glob(f"{json_stem}*.pdf"))
        if matching_pdfs:
            state.store.discover_bookmarks(str(matching_pdfs[0]))
            logger.info(f"Matched PDF: {matching_pdfs[0].name}")
        else:
            for p in sorted(OUTPUT_DIR.glob("*.pdf")):
                state.store.discover_bookmarks(str(p))
                logger.warning(f"Using fallback PDF for bookmarks: {p.name}")
                break

    # Build retriever
    with open(json_path, encoding="utf-8") as f_json:
        state.json_data = json.load(f_json)
    
    from knowledge_base.llamaindex_retriever import build_retriever
    state.retriever = build_retriever(state.store, state.json_data, OPENAI_API_KEY)
    
    # Store PDF path
    if pdf_path:
        state.pdf_path = pdf_path
    else:
        json_stem = Path(json_path).stem.replace("_structured", "")
        for p in OUTPUT_DIR.glob(f"{json_stem}*.pdf"):
            state.pdf_path = str(p)
            break

    state.loaded = True
    logger.info(f"Ready: {meta['total_sections']} sections, {meta['total_tables']} tables")

    return {
        "protocol_name": state.protocol_name,
        "metadata": meta,
    }


def find_existing_json() -> Path | None:
    """Find pre-parsed structured JSON in output/. Returns most recently modified."""
    candidates = list(OUTPUT_DIR.glob("*_structured.json"))
    if OUTPUT_DIR.joinpath("protocol_structured.json").exists():
        candidates.append(OUTPUT_DIR / "protocol_structured.json")
    
    if not candidates:
        return None
    
    # Pick the most recently modified file
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


# ═══════════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str
    query_type: Optional[str] = None


@app.get("/api/status")
def get_status():
    return {
        "loaded": state.loaded,
        "protocol_name": state.protocol_name,
        "metadata": state.store.metadata if state.store else None,
        "model": LLM_MODEL,
    }


@app.post("/api/load")
async def load_cached():
    """Load from pre-parsed structured JSON (fast, ~1s)."""
    json_path = find_existing_json()
    if not json_path:
        raise HTTPException(404, "No structured JSON found in output/. Upload a PDF first.")
    try:
        result = load_protocol(str(json_path))
        return {"status": "success", **result}
    except Exception as e:
        logger.error(f"Load failed: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF → parse → build retriever → ready."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files accepted.")

    # CLEAN OUTPUT DIRECTORY — remove old protocol's files
    # to prevent cross-contamination between protocols
    for old_file in OUTPUT_DIR.iterdir():
        if old_file.suffix in (".json", ".md"):
            old_file.unlink()
            logger.info(f"Removed old: {old_file.name}")
    # Remove old PDFs too (but not the one being uploaded)
    for old_pdf in OUTPUT_DIR.glob("*.pdf"):
        if old_pdf.name != file.filename:
            old_pdf.unlink()
            logger.info(f"Removed old PDF: {old_pdf.name}")

    pdf_path = OUTPUT_DIR / file.filename
    content = await file.read()
    with open(pdf_path, "wb") as f:
        f.write(content)
    logger.info(f"Uploaded: {file.filename} ({len(content)} bytes)")

    steps = []
    try:
        # Run parser
        steps.append({"label": "Parsing PDF (PyMuPDF + pdfplumber)...", "status": "running"})

        sys.path.insert(0, str(Path(__file__).parent / "ingestion"))
        from ingestion import process_protocol

        output_prefix = str(OUTPUT_DIR / pdf_path.stem)
        document_result, json_path, pipeline_steps = process_protocol(
            pdf_path=str(pdf_path),
            output_prefix=output_prefix,
            with_llm=True,
        )
        steps[-1]["status"] = "done"

        for ps in pipeline_steps:
            steps.append({"label": ps.get("label", ""), "status": ps.get("status", "done")})

        # Load protocol — pass the EXACT pdf and json paths
        steps.append({"label": "Building retriever index...", "status": "running"})
        result = load_protocol(json_path, str(pdf_path), force_rebuild=True)
        steps[-1]["status"] = "done"

        return {"status": "success", **result, "steps": steps}

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        steps.append({"label": str(e), "status": "failed"})
        raise HTTPException(500, str(e))


@app.post("/api/query")
def run_query(request: QueryRequest):
    """
    Run the multi-agent pipeline:
      Explorer Agent (finds content) → Extractor Agent (structures it) → Validation
    Each agent uses OpenAI function calling — the LLM decides which tools to use.
    """
    if not state.loaded:
        raise HTTPException(400, "No protocol loaded.")

    t_start = time.time()

    # Create debug log for this query
    from debug_logger import DebugLog
    debug_log = DebugLog(request.query, str(OUTPUT_DIR))

    # Detect query type
    query_type = request.query_type or _detect_query_type(request.query)

    # ── Run LangGraph multi-agent pipeline: Explorer → Extractor → Validator ──
    from agents.graph import run_query as run_agent_query
    
    agent_result = run_agent_query(
        query=request.query,
        query_type=query_type,
        retriever=state.retriever,
        pdf_path=state.pdf_path,
        json_data=state.json_data,
        store=state.store,
        debug_log=debug_log,
    )
    
    # Build steps from agent results
    all_steps = []
    for step in agent_result.get("steps", []):
        all_steps.append({
            "turn": len(all_steps) + 1,
            "type": "tool_result",
            "name": step.get("agent", "?"),
            "content": f"{step.get('turns', 0)} turns, {step.get('tool_calls', 0)} tool calls"
                       + (f" -- {step.get('content_chars', '')} chars" if step.get('content_chars') else "")
                       + (f" -- {step.get('signals', '')} signals" if step.get('signals') else ""),
            "duration_ms": int(step.get("duration_s", 0) * 1000),
        })
    
    data = agent_result.get("data")
    validation = agent_result.get("validation")
    
    if validation and validation.get("total"):
        all_steps.append({
            "turn": len(all_steps) + 1,
            "type": "validation",
            "name": "Validator",
            "content": f"{validation['verified']}/{validation['total']} verified, {validation['flagged']} flagged",
            "duration_ms": 0,
        })

    total_ms = int((time.time() - t_start) * 1000)

    # Save debug log
    # Log final response to debug log
    debug_log._section("FINAL RESPONSE TO UI")
    debug_log.lines.append(f"  query_type: {query_type}")
    debug_log.lines.append(f"  data keys: {list(data.keys()) if data else 'None'}")
    debug_log.lines.append(f"  data (full JSON):")
    debug_log.lines.append(json.dumps(data, indent=2, default=str)[:10000])
    if validation:
        debug_log.lines.append(f"  validation: {validation.get('verified','?')}/{validation.get('total','?')} verified")
    debug_log.lines.append(f"  signals: {agent_result.get('signals', [])}")
    debug_log.lines.append(f"  total_ms: {total_ms}")

    log_path = debug_log.save()
    logger.info(f"Debug log saved to {log_path}")

    return {
        "status": "success" if data else "error",
        "query_type": query_type,
        "data": data,
        "validation": validation,
        "signals": agent_result.get("signals", []),
        "observations": f"{agent_result.get('total_turns', 0)} agent turns, "
                        f"{len(agent_result.get('steps', []))} agents used",
        "steps": all_steps,
        "total_ms": total_ms,
        "retrieval_rounds": agent_result.get("total_turns", 0),
        "error": agent_result.get("error", ""),
    }


def _detect_query_type(query: str) -> str:
    """Detect query type using registry keywords."""
    from shared.registry import REGISTRY
    q = query.lower()
    for qtype, config in REGISTRY.items():
        if config.detection_keywords and any(kw in q for kw in config.detection_keywords):
            return qtype
    return "general"


@app.post("/api/query/stream")
def run_query_stream(request: QueryRequest):
    """SSE streaming endpoint — streams agent status events in real-time."""
    if not state.loaded:
        raise HTTPException(400, "No protocol loaded.")
    
    import threading
    from event_bus import EventBus
    
    bus = EventBus()
    query_type = request.query_type or _detect_query_type(request.query)
    
    def _run_in_thread():
        """Run the agent pipeline in a background thread, pushing events to bus."""
        try:
            t_start = time.time()
            from debug_logger import DebugLog
            debug_log = DebugLog(request.query, str(OUTPUT_DIR))
            
            from agents.graph import run_query as run_agent_query
            agent_result = run_agent_query(
                query=request.query,
                query_type=query_type,
                retriever=state.retriever,
                pdf_path=state.pdf_path,
                json_data=state.json_data,
                store=state.store,
                debug_log=debug_log,
                event_bus=bus,
            )
            
            total_ms = int((time.time() - t_start) * 1000)
            
            # Build the same response as /api/query
            data = agent_result.get("data")
            validation = agent_result.get("validation")
            all_steps = []
            for step in agent_result.get("steps", []):
                all_steps.append({
                    "turn": len(all_steps) + 1,
                    "type": "tool_result",
                    "name": step.get("agent", "?"),
                    "content": f"{step.get('turns', 0)} turns, {step.get('tool_calls', 0)} tool calls",
                    "duration_ms": int(step.get("duration_s", 0) * 1000),
                })
            
            result = {
                "status": "success" if data else "error",
                "query_type": query_type,
                "data": data,
                "validation": validation,
                "signals": agent_result.get("signals", []),
                "observations": f"{agent_result.get('total_turns', 0)} agent turns",
                "steps": all_steps,
                "total_ms": total_ms,
                "retrieval_rounds": agent_result.get("total_turns", 0),
                "error": agent_result.get("error", ""),
            }
            
            # Save debug log
            debug_log._section("FINAL RESPONSE TO UI")
            debug_log.lines.append(f"  query_type: {query_type}")
            debug_log.lines.append(f"  data keys: {list(data.keys()) if data else 'None'}")
            debug_log.lines.append(f"  total_ms: {total_ms}")
            debug_log.save()
            
            bus.emit_done(result)
            
        except Exception as e:
            logger.error(f"Stream query failed: {e}", exc_info=True)
            bus.emit_error(str(e))
    
    # Start agent pipeline in background thread
    thread = threading.Thread(target=_run_in_thread, daemon=True)
    thread.start()
    
    def event_stream():
        for event in bus.stream(timeout=600):
            yield f"data: {json.dumps(event, default=str)}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/sections")
def get_sections():
    """Get list of protocol sections."""
    if not state.store:
        raise HTTPException(400, "No protocol loaded.")
    sections = []
    for sid in state.store.all_section_ids:
        sec = state.store.get_section(sid)
        if sec:
            sections.append({"id": sid, "title": sec["title"],
                            "pages": sec["pages"], "chars": sec["char_count"]})
    return {"sections": sections, "total": len(sections)}


@app.get("/api/section/{section_id}")
def get_section(section_id: str):
    if not state.store:
        raise HTTPException(400, "No protocol loaded.")
    section = state.store.get_section(section_id)
    if not section:
        raise HTTPException(404, f"Section {section_id} not found.")
    return section


@app.get("/api/search")
def search_content(q: str):
    if not state.store:
        raise HTTPException(400, "No protocol loaded.")
    return {"results": state.store.search(q)}


# ═══════════════════════════════════════════════════════════════════════
# Serve UI
# ═══════════════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    ui_path = Path(__file__).parent / "static" / "index.html"
    if ui_path.exists():
        return HTMLResponse(ui_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Protocol Intelligence System</h1><p>UI not found.</p>")


static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ═══════════════════════════════════════════════════════════════════════
# Startup
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    project_root = Path(__file__).parent

    print(f"\n  Protocol Intelligence System")
    print(f"  {'─' * 40}")
    print(f"  Project: {project_root}")
    print(f"  Output:  {OUTPUT_DIR}")

    json_path = find_existing_json()
    if json_path:
        print(f"  Cache:   {json_path.name}")
    print(f"  Model:   {LLM_MODEL}")
    print(f"  UI:  http://localhost:8000")
    print(f"  API: http://localhost:8000/docs")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8000)
