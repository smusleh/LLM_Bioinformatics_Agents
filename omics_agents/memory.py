"""Long-term memory: a JSON file of findings that persists across agent sessions.

Short-term memory = the conversation (messages list) inside one Agent run.
Long-term memory  = this file, which a *new* agent can recall later (paper: "Memory" module,
EHRAgent / RareAgents style experience stores; a minimal form of continual learning).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .config import MEMORY_FILE


def _load() -> list[dict]:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    return []


def save_finding(topic: str, finding: str, evidence: str = "") -> dict:
    """Persist one verified finding."""
    items = _load()
    items.append(
        {
            "id": len(items) + 1,
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "topic": topic,
            "finding": finding,
            "evidence": evidence,
        }
    )
    MEMORY_FILE.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return {"saved": True, "id": len(items), "total_findings": len(items)}


def recall_findings(query: str = "") -> dict:
    """Return stored findings, optionally filtered by a keyword."""
    items = _load()
    if query:
        q = query.lower()
        items = [i for i in items if q in json.dumps(i).lower()]
    return {"n": len(items), "findings": items}


def clear_memory() -> None:
    if MEMORY_FILE.exists():
        MEMORY_FILE.unlink()


SAVE_SCHEMA = {
    "description": "Save a verified finding to long-term memory so future sessions can reuse it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "finding": {"type": "string"},
            "evidence": {"type": "string", "description": "Tool result / KB source backing it"},
        },
        "required": ["topic", "finding", "evidence"],
        "additionalProperties": False,
    },
}

RECALL_SCHEMA = {
    "description": "Recall findings saved in previous sessions (optionally filter by keyword). "
    "Check this before re-running analyses.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": [],
        "additionalProperties": False,
    },
}
