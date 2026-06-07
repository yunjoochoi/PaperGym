"""Single-call inference-time executability classifier (ports the rubric)."""
from __future__ import annotations

import json
from pathlib import Path

from papergym.agents.base import PromptLoader
from papergym.llm import LLMClient

_PROMPTS = PromptLoader(Path(__file__).resolve().parent / "prompts")


def classify_executability(*, proposal: str, llm: LLMClient) -> dict:
    messages = _PROMPTS.render("executability", proposal=proposal)
    raw = llm.chat(messages, temperature=0.0,
                   response_format={"type": "json_object"})
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {"gym_executable": False, "requires_gpu_training": None,
                "requires_human_eval": None, "blockers": ["parse_error"],
                "reasoning": raw[:200]}
    obj.setdefault("gym_executable", False)
    obj.setdefault("blockers", [])
    return obj
