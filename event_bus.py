"""
Event Bus — agents push status events, SSE endpoint streams them to UI.

Usage:
    from event_bus import EventBus
    
    bus = EventBus()
    bus.emit("explorer", "searching", "Searching for eligibility criteria...")
    bus.emit("extractor", "extracting", "Extracting structured data...")
    bus.emit_done(result_dict)
    
    # SSE endpoint:
    for event in bus.stream():
        yield f"data: {json.dumps(event)}\n\n"
"""
from __future__ import annotations
import json
import queue
import threading
import time
from typing import Generator


class EventBus:
    """Thread-safe event bus for streaming agent progress."""
    
    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._done = threading.Event()
    
    def emit(self, agent: str, status: str, detail: str = "", **extra):
        """Push a status event."""
        event = {
            "type": "status",
            "agent": agent,
            "status": status,
            "detail": detail,
            "timestamp": time.time(),
            **extra,
        }
        self._queue.put(event)
    
    def emit_tool(self, agent: str, tool_name: str, tool_input: str = ""):
        """Push a tool call event."""
        self._queue.put({
            "type": "tool",
            "agent": agent,
            "tool": tool_name,
            "input": tool_input[:200],
            "timestamp": time.time(),
        })
    
    def emit_done(self, result: dict):
        """Push the final result and signal done."""
        self._queue.put({
            "type": "result",
            "data": result,
            "timestamp": time.time(),
        })
        self._done.set()
    
    def emit_error(self, error: str):
        """Push an error and signal done."""
        self._queue.put({
            "type": "error",
            "error": error,
            "timestamp": time.time(),
        })
        self._done.set()
    
    def stream(self, timeout: float = 600) -> Generator[dict, None, None]:
        """Yield events until done or timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                event = self._queue.get(timeout=0.5)
                yield event
                if event["type"] in ("result", "error"):
                    return
            except queue.Empty:
                # Send keepalive to prevent connection timeout
                yield {"type": "keepalive", "timestamp": time.time()}
                if self._done.is_set():
                    return
