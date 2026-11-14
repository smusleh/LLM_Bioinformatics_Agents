"""A minimal LLM agent: Perception -> Planning -> Action -> (Memory), in ~100 lines.

We write the tool-use loop by hand (instead of the SDK's beta tool runner) so every step of
the agent's cycle is visible in the notebook trace:

    user task ──► LLM plans (thinking summary) ──► tool_use ──► Python tool runs
         ▲                                                          │
         └──────────────── tool_result appended to messages ◄───────┘

Supports two backends (set LLM_PROVIDER in .env):
  - "anthropic"  : Anthropic API (requires ANTHROPIC_API_KEY and credits)
  - "ollama"     : free local model via Ollama (http://localhost:11434)
"""
from __future__ import annotations

import json
import os
import textwrap

import anthropic
import openai

from .config import (EFFORT, MAX_STEPS, MAX_TOKENS, MODEL,
                     LLM_API_KEY, OLLAMA_BASE_URL, OLLAMA_MODEL, PRICES,
                     PROVIDER)
from .tools import ToolError

_anthropic_client: anthropic.Anthropic | None = None
_openai_client: openai.OpenAI | None = None


def client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env / .env
    return _anthropic_client


def _get_ollama_client() -> openai.OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(base_url=OLLAMA_BASE_URL, api_key=LLM_API_KEY)
    return _openai_client


# --- Ollama response normalisation -------------------------------------------

class _Block:
    """Minimal content block mirroring Anthropic SDK blocks."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _NormalizedResponse:
    """Wraps an OpenAI ChatCompletion to look like an Anthropic BetaMessage."""
    def __init__(self, oai_resp):
        msg = oai_resp.choices[0].message
        content: list[_Block] = []
        if msg.content:
            content.append(_Block(type="text", text=msg.content))
        for tc in (msg.tool_calls or []):
            content.append(_Block(
                type="tool_use",
                id=tc.id,
                name=tc.function.name,
                input=json.loads(tc.function.arguments or "{}"),
            ))
        self.content = content
        reason = oai_resp.choices[0].finish_reason
        self.stop_reason = "tool_use" if reason == "tool_calls" else "end_turn"
        self.usage = _Block(
            input_tokens=oai_resp.usage.prompt_tokens,
            output_tokens=oai_resp.usage.completion_tokens,
        )


def _to_openai_messages(system: str, messages: list) -> list:
    """Convert Anthropic-format message history to OpenAI format for Ollama."""
    result = [{"role": "system", "content": system}]
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if isinstance(content, str):
            result.append({"role": role, "content": content})
            continue
        if role == "user":
            tool_results = [b for b in content
                            if isinstance(b, dict) and b.get("type") == "tool_result"]
            if tool_results:
                for tr in tool_results:
                    result.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_use_id"],
                        "content": str(tr["content"]),
                    })
            else:
                text = " ".join(
                    b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
                    for b in content
                )
                result.append({"role": "user", "content": text})
        elif role == "assistant":
            text_parts, tool_calls = [], []
            for b in content:
                btype = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
                if btype == "text":
                    text_parts.append(b.get("text", "") if isinstance(b, dict) else b.text)
                elif btype == "tool_use":
                    bid = b.get("id") if isinstance(b, dict) else b.id
                    bname = b.get("name") if isinstance(b, dict) else b.name
                    binput = b.get("input") if isinstance(b, dict) else b.input
                    tool_calls.append({
                        "id": bid,
                        "type": "function",
                        "function": {"name": bname, "arguments": json.dumps(binput)},
                    })
            oai_msg: dict = {"role": "assistant", "content": " ".join(text_parts) or None}
            if tool_calls:
                oai_msg["tool_calls"] = tool_calls
            result.append(oai_msg)
    return result


def _to_openai_tools(schemas: list) -> list:
    """Convert Anthropic tool schemas to OpenAI function format."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s.get("description", ""),
                "parameters": s["input_schema"],
            },
        }
        for s in schemas
    ]


def _short(text: str, n: int = 400) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + " …"


class Agent:
    """An LLM agent with a role (system prompt), a toolbox and short-term memory."""

    def __init__(self, name, system, toolbox=None, model=None, effort=EFFORT,
                 max_steps=MAX_STEPS, verbose=True, show_thinking=True, provider=PROVIDER):
        if model is None:
            model = OLLAMA_MODEL if provider == "ollama" else MODEL
        self.name, self.system, self.toolbox = name, system, toolbox
        self.model, self.effort, self.max_steps = model, effort, max_steps
        self.verbose, self.show_thinking = verbose, show_thinking
        self.provider = provider
        self.messages: list = []   # short-term memory (this session's conversation)
        self.tool_log: list = []   # every tool call + result, for auditing / critics
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    # ------------------------------------------------------------------ printing
    def _say(self, prefix, text, width=110):
        if self.verbose:
            body = textwrap.fill(text, width, subsequent_indent="    ")
            print(f"[{self.name}] {prefix} {body}")

    # ------------------------------------------------------------------ backends
    def _call(self):
        if self.provider == "ollama":
            return self._call_ollama()
        return self._call_anthropic()

    def _call_anthropic(self):
        kwargs = dict(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=self.system,
            messages=self.messages,
            output_config={"effort": self.effort},
            thinking={"type": "adaptive",
                      "display": "summarized" if self.show_thinking else "omitted"},
            # If a safety classifier declines, the API retries on a suitable fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if self.toolbox:
            kwargs["tools"] = self.toolbox.schemas
        resp = client().beta.messages.create(**kwargs)
        self.usage["input_tokens"] += resp.usage.input_tokens
        self.usage["output_tokens"] += resp.usage.output_tokens
        return resp

    def _call_ollama(self):
        kwargs = dict(
            model=self.model,
            messages=_to_openai_messages(self.system, self.messages),
        )
        if self.toolbox:
            kwargs["tools"] = _to_openai_tools(self.toolbox.schemas)
        resp = _get_ollama_client().chat.completions.create(**kwargs)
        norm = _NormalizedResponse(resp)
        self.usage["input_tokens"] += norm.usage.input_tokens
        self.usage["output_tokens"] += norm.usage.output_tokens
        return norm

    # ------------------------------------------------------------------ the loop
    def run(self, task: str) -> str:
        """Run the agent on a task until it stops calling tools (or hits max_steps)."""
        self.messages.append({"role": "user", "content": task})
        self._say("TASK:", _short(task, 300))
        for _step in range(self.max_steps):
            resp = self._call()
            self.messages.append({"role": "assistant", "content": resp.content})

            for block in resp.content:  # perception of the model's own output
                if block.type == "thinking" and block.thinking and self.show_thinking:
                    self._say("💭 plan:", _short(block.thinking, 500))
                elif block.type == "tool_use":
                    self._say("🔧 action:", f"{block.name}({json.dumps(block.input)})")

            if resp.stop_reason == "refusal":
                return "[The model declined this request.]"
            if resp.stop_reason == "pause_turn":
                continue
            if resp.stop_reason != "tool_use":
                final = "\n".join(b.text for b in resp.content if b.type == "text").strip()
                if resp.stop_reason == "max_tokens":
                    final += "\n\n[truncated: max_tokens reached]"
                self._say("✅ done", f"({self.usage['input_tokens']} in / "
                          f"{self.usage['output_tokens']} out tokens)")
                return final

            # Action: execute every requested tool, return all results in one user turn
            results = []
            for block in resp.content:
                if block.type != "tool_use":
                    continue
                try:
                    out, is_error = self.toolbox.execute(block.name, block.input), False
                except (ToolError, TypeError, KeyError, ValueError) as e:
                    out, is_error = f"Error: {e}", True
                self.tool_log.append({"tool": block.name, "input": block.input,
                                      "result": out, "error": is_error})
                self._say("   ↳ result:" if not is_error else "   ↳ ERROR:", _short(out, 300))
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": out, "is_error": is_error})
            self.messages.append({"role": "user", "content": results})

        return "[Stopped: reached max_steps without a final answer.]"

    def reset(self):
        self.messages, self.tool_log = [], []

    def cost(self) -> float:
        """Approximate USD cost of everything this agent has run so far."""
        pin, pout = PRICES.get(self.model, (0.0, 0.0))
        return (self.usage["input_tokens"] * pin + self.usage["output_tokens"] * pout) / 1e6


def ask_llm(prompt: str, system: str = "You are a helpful bioinformatics assistant.") -> str:
    """A single LLM call with NO tools - the 'bare LLM' baseline."""
    return Agent("LLM", system, toolbox=None, verbose=False).run(prompt)


def human_gate(question: str) -> str:
    """Human-in-the-loop checkpoint. Returns 'y' or the human's redirect instructions.

    Set AUTO_APPROVE=1 to auto-approve (for headless runs, e.g. nbconvert --execute).
    """
    if os.getenv("AUTO_APPROVE") == "1":
        print(f"[HUMAN GATE] {question}\n  -> auto-approved (AUTO_APPROVE=1)")
        return "y"
    answer = input(f"[HUMAN GATE] {question}\n  Type 'y' to approve, or type new instructions: ")
    return answer.strip() or "y"
