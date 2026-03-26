"""
Debug Logger — Writes detailed query trace to output/debug_log.txt

Every query creates a full trace showing:
  - What query was asked
  - What the agent decided to do (tools, args)
  - What each tool returned (full content, first 2000 chars)
  - What was collected as context
  - What was sent to extraction LLM (full prompt)
  - What extraction LLM returned (full response)
  - What validator checked and results

Usage:
    from debug_logger import DebugLog
    
    log = DebugLog("Extract Endpoints")
    log.agent_tool("get_section", {"section_id": "3"}, result_text)
    log.agent_context(collected_items)
    log.extraction_input(system_prompt, user_prompt)
    log.extraction_output(raw_response)
    log.validation(results)
    log.save()  # writes to output/debug_log.txt
"""
from __future__ import annotations
import time
import json
from pathlib import Path
from datetime import datetime


class DebugLog:
    
    def __init__(self, query: str, output_dir: str = "output"):
        self.query = query
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.lines: list[str] = []
        
        self._header()
    
    def _header(self):
        self.lines.append("=" * 80)
        self.lines.append(f"QUERY DEBUG LOG — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.lines.append(f"Query: {self.query}")
        self.lines.append("=" * 80)
    
    def _section(self, title: str):
        self.lines.append("")
        self.lines.append(f"{'─' * 40}")
        self.lines.append(f"  {title}")
        self.lines.append(f"{'─' * 40}")
    
    # ── Agent events ─────────────────────────────────────────────
    
    def agent_start(self, system_prompt: str):
        self._section("AGENT SYSTEM PROMPT")
        self.lines.append(system_prompt[:3000])
        if len(system_prompt) > 3000:
            self.lines.append(f"... [{len(system_prompt)} total chars]")
    
    def agent_turn(self, turn: int, tokens: int):
        self._section(f"AGENT TURN {turn} (tokens: {tokens})")
    
    def agent_tool_call(self, name: str, args: dict):
        self.lines.append(f"  TOOL CALL: {name}({json.dumps(args)})")
    
    def agent_tool_result(self, name: str, result: str, duration_ms: int):
        self.lines.append(f"  TOOL RESULT: {name} → {len(result)} chars ({duration_ms}ms)")
        # Show first 2000 chars of result
        preview = result[:2000]
        for line in preview.split("\n"):
            self.lines.append(f"    | {line}")
        if len(result) > 2000:
            self.lines.append(f"    | ... [{len(result)} total chars, showing first 2000]")
    
    def agent_collected(self, name: str, chars: int, total_items: int):
        self.lines.append(f"  COLLECTED: {name} → {chars} chars (total items: {total_items})")
    
    def agent_final(self, content: str, turns: int, tokens: int, cost: float):
        self._section(f"AGENT FINAL OUTPUT ({turns} turns, {tokens} tokens, ${cost:.4f})")
        self.lines.append(f"  Agent's summary text ({len(content)} chars):")
        preview = content[:1500]
        for line in preview.split("\n"):
            self.lines.append(f"    | {line}")
    
    def agent_context(self, collected: list[str], total_chars: int):
        self._section(f"COLLECTED CONTEXT → EXTRACTION ({len(collected)} items, {total_chars} chars)")
        for i, item in enumerate(collected):
            self.lines.append(f"  Item {i+1}: {len(item)} chars")
            preview = item[:800]
            for line in preview.split("\n")[:15]:
                self.lines.append(f"    | {line}")
            if len(item) > 800:
                self.lines.append(f"    | ... [{len(item)} total chars]")
            self.lines.append("")
    
    # ── Orchestrator events ─────────────────────────────────────
    
    def orch_content_assembled(self, content: str, tables_used: list, pages_raw: list):
        """Log what was assembled for the LLM: which tables (parsed) vs which pages (raw)."""
        self._section(f"CONTENT ASSEMBLED FOR LLM ({len(content)} chars)")
        
        if tables_used:
            self.lines.append(f"  PARSED TABLES (merged from JSON, not raw PDF):")
            for t in tables_used:
                tid = t.get("id", "?")
                pages = t.get("page_range", [])
                rows = len(t.get("rows", []))
                caption = t.get("caption", "")[:60]
                self.lines.append(f"    {tid}: pages {pages}, {rows} rows — {caption}")
        
        if pages_raw:
            self.lines.append(f"  RAW PDF PAGES (PyMuPDF text): {pages_raw}")
        
        self.lines.append("")
        self.lines.append(f"  ═══ FULL ASSEMBLED CONTENT ({len(content)} chars) ═══")
        for line in content.split("\n"):
            self.lines.append(f"    | {line}")
        self.lines.append(f"  ═══ END ASSEMBLED CONTENT ═══")
    
    # ── Extraction events ────────────────────────────────────────
    
    def extraction_input(self, system_prompt: str, user_prompt: str, query_type: str):
        self._section(f"EXTRACTION INPUT (type: {query_type})")
        
        self.lines.append(f"  SYSTEM PROMPT ({len(system_prompt)} chars):")
        # Show full system prompt — it contains the domain instructions + schema
        for line in system_prompt.split("\n"):
            self.lines.append(f"    | {line}")
        
        self.lines.append("")
        self.lines.append(f"  USER PROMPT ({len(user_prompt)} chars):")
        if "PROTOCOL CONTEXT:" in user_prompt:
            parts = user_prompt.split("PROTOCOL CONTEXT:", 1)
            self.lines.append(f"    Query + observations ({len(parts[0])} chars):")
            for line in parts[0].split("\n"):
                self.lines.append(f"    | {line}")
            
            ctx = parts[1]
            self.lines.append("")
            self.lines.append(f"    ═══ FULL CONTEXT SENT TO LLM ({len(ctx)} chars) ═══")
            # Show EVERYTHING — this is what the user needs to debug
            for line in ctx.split("\n"):
                self.lines.append(f"    | {line}")
            self.lines.append(f"    ═══ END CONTEXT ═══")
        else:
            for line in user_prompt.split("\n"):
                self.lines.append(f"    | {line}")
    
    def extraction_output(self, raw_response: str, elapsed: float, tokens_in: int, tokens_out: int):
        self._section(f"EXTRACTION OUTPUT ({elapsed:.1f}s, {tokens_in} in, {tokens_out} out)")
        self.lines.append(f"  ═══ FULL LLM RESPONSE ({len(raw_response)} chars) ═══")
        for line in raw_response.split("\n"):
            self.lines.append(f"    | {line}")
        self.lines.append(f"  ═══ END LLM RESPONSE ═══")
    
    def extraction_parsed(self, data: dict | None, schema_valid: bool, errors: str = ""):
        self.lines.append(f"  Parsed: {'YES' if data else 'NO'}")
        self.lines.append(f"  Schema valid: {schema_valid}")
        if errors:
            self.lines.append(f"  Errors: {errors}")
        if data:
            if "endpoints" in data:
                self.lines.append(f"  Endpoints found: {len(data['endpoints'])}")
                for ep in data["endpoints"]:
                    gnd = ep.get("grounding", {})
                    self.lines.append(
                        f"    {ep.get('id','?')}: {ep.get('category','')} | "
                        f"p.{gnd.get('page',0)} §{gnd.get('section_id','')} | "
                        f"obj: {str(ep.get('objective',''))[:60]} | "
                        f"ep: {str(ep.get('endpoint',''))[:80]}"
                    )
            elif "inclusion" in data:
                self.lines.append(f"  Inclusion: {len(data.get('inclusion', []))}")
                for c in data.get("inclusion", [])[:5]:
                    self.lines.append(f"    {c.get('id','?')}: {c.get('automation_level','?')} | {str(c.get('text',''))[:80]}")
                self.lines.append(f"  Exclusion: {len(data.get('exclusion', []))}")
                for c in data.get("exclusion", [])[:5]:
                    self.lines.append(f"    {c.get('id','?')}: {c.get('automation_level','?')} | {str(c.get('text',''))[:80]}")
            elif "rules" in data:
                self.lines.append(f"  Deviation Rules: {len(data['rules'])}")
                for r in data["rules"]:
                    gnd = r.get("grounding", {})
                    self.lines.append(
                        f"    {r.get('rule_id','?')}: {r.get('automation_level','?')} | "
                        f"{r.get('sdtm_domain','?')}.{r.get('sdtm_variable','?')} | "
                        f"cond: {str(r.get('condition',''))[:60]} | "
                        f"src: {str(r.get('source_criterion',''))[:50]} | "
                        f"signal: {str(r.get('smart_signal',''))[:60]}"
                    )
                comp = data.get("computable", 0)
                part = data.get("partial", 0)
                manual = data.get("non_computable", 0)
                self.lines.append(f"  Automation: {comp} FULL, {part} PARTIAL, {manual} MANUAL")
            elif "risks" in data:
                self.lines.append(f"  Risks: {len(data['risks'])}")
            elif "answer" in data:
                self.lines.append(f"  Answer: {str(data['answer'])[:200]}")
                if "claims" in data:
                    self.lines.append(f"  Claims: {len(data['claims'])}")
    
    # ── Validation events ────────────────────────────────────────
    
    def validation(self, results: dict):
        self._section("VALIDATION RESULTS (4-level)")
        self.lines.append(
            f"  Summary: {results.get('verified', 0)}/{results.get('total', 0)} verified, "
            f"{results.get('flagged', 0)} flagged, {results.get('failed', 0)} failed"
        )
        self.lines.append("")
        for d in results.get("details", []):
            verdict = d.get("verdict", "?")
            item = d.get("item", "?")
            checks = d.get("checks", {})
            
            # Color-code verdict
            icon = {"VERIFIED": "[OK]", "FLAGGED": "[??]"}.get(verdict, "[XX]")
            self.lines.append(f"  {icon} {verdict:18s} {item}")
            
            for check_name, check_result in checks.items():
                self.lines.append(f"       {check_name:22s}: {check_result}")
            self.lines.append("")
    
    # ── Save ─────────────────────────────────────────────────────
    
    def save(self):
        elapsed = time.time() - self.start_time
        self.lines.append("")
        self.lines.append(f"{'=' * 80}")
        self.lines.append(f"TOTAL TIME: {elapsed:.1f}s")
        self.lines.append(f"{'=' * 80}")
        self.lines.append("")
        self.lines.append("")
        
        log_path = self.output_dir / "debug_log.txt"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        
        return str(log_path)
