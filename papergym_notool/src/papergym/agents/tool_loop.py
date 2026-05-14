"""Shared tool-calling ReAct loop for agentic agents.

Drives an OpenAI-style native tool-calling loop until the model stops
calling tools (natural end) or the step budget is exhausted. Tool dispatch
is provided by the caller as a `dispatch(name, args) -> observation_string`
callable so this module can be shared by Accumulator (PaperEnv.step) and
any future agent.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..llm import LLMClient

MessageHook = Optional[Callable[[dict], None]]
Dispatch = Callable[[str, dict], str]


def _safe_emit(hook: MessageHook, msg: dict) -> None:
    if hook is None:
        return
    try:
        hook(msg)
    except Exception:
        pass  # never let an observability hook kill the loop


def _to_args(arg_str: Optional[str]) -> Any:
    try:
        return json.loads(arg_str or "{}")
    except json.JSONDecodeError:
        return {"_raw_arguments": arg_str}


@dataclass
class LoopResult:
    status: str                            # "natural_end" | "max_steps_exceeded" | "error"
    final_content: Optional[str] = None    # last assistant text on natural_end
    steps: int = 0
    trace: list[dict] = field(default_factory=list)
    reason: str = ""


def run_tool_loop(
    *, llm: LLMClient, messages: list[dict], tools: list[dict],
    dispatch: Dispatch,
    max_steps: int,
    temperature: float = 0.3,
    on_message: MessageHook = None,
) -> LoopResult:
    """Drive the native tool-calling loop until the agent stops calling tools
    (natural end) or `max_steps` is hit.

    `dispatch(name, args) -> str` routes each tool call to the caller's tool
    surface (e.g., PaperEnv.step for the Accumulator).

    `on_message(msg)` fires once per chat-transcript line:
      {"role":"system",   "content":"..."}                       (initial)
      {"role":"user",     "content":"..."}                       (initial)
      {"role":"assistant","thought":"...","tool_calls":[...],"usage":{...}}
      {"role":"tool",     "tool_call_id":"...","name":"...","content":"..."}
    """
    trace: list[dict] = []
    recent_sigs: deque[str] = deque(maxlen=3)

    for m in messages:
        _safe_emit(on_message, {"role": m.get("role", "?"),
                                  "content": m.get("content", "")})

    for step in range(max_steps):
        reply = llm.chat_with_tools(messages, tools=tools,
                                     temperature=temperature,
                                     tool_choice="auto")
        messages.append(reply.raw_message)
        thought = (reply.content or "").strip()

        tool_calls_emit = [{"id": tc.id, "name": tc.name, "args": _to_args(tc.arguments)}
                            for tc in reply.tool_calls]
        _safe_emit(on_message, {"role": "assistant", "thought": thought,
                                  "tool_calls": tool_calls_emit,
                                  "usage": reply.usage})

        if not reply.tool_calls:
            trace.append({"step": step, "kind": "natural_end",
                          "content": thought[:1500]})
            return LoopResult(status="natural_end",
                              final_content=reply.content,
                              steps=step + 1, trace=trace)

        for tc in reply.tool_calls:
            try:
                args = json.loads(tc.arguments or "{}")
            except json.JSONDecodeError as exc:
                observation = f"[arguments invalid JSON] {exc}"
                trace.append({"step": step, "kind": "tool_call",
                              "name": tc.name, "arguments": tc.arguments[:1500],
                              "error": str(exc)})
                _push_tool_response(messages, tc.id, observation)
                _safe_emit(on_message, {"role": "tool", "tool_call_id": tc.id,
                                          "name": tc.name, "content": observation})
                continue

            sig = json.dumps([tc.name, args], sort_keys=True)
            recent_sigs.append(sig)
            if len(recent_sigs) == 3 and all(s == sig for s in recent_sigs):
                trace.append({"step": step, "kind": "tool_call",
                              "thought": thought,
                              "name": tc.name, "args": args, "error": "loop_detected"})
                return LoopResult(
                    status="error", steps=step + 1, trace=trace,
                    reason="detected loop: same tool+args 3 times in a row",
                )

            try:
                observation = dispatch(tc.name, args)
            except Exception as exc:
                observation = f"[tool error] {tc.name}: {exc}"
            trace.append({"step": step, "kind": "tool_call",
                          "thought": thought,
                          "name": tc.name, "args": args,
                          "observation": observation})
            _push_tool_response(messages, tc.id, observation)
            _safe_emit(on_message, {"role": "tool", "tool_call_id": tc.id,
                                      "name": tc.name, "content": observation})

    return LoopResult(status="max_steps_exceeded", steps=max_steps, trace=trace)


def _push_tool_response(messages: list[dict], tool_call_id: str, content: str) -> None:
    messages.append({"role": "tool", "tool_call_id": tool_call_id,
                     "content": content})
