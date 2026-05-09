from __future__ import annotations

import json
import re
from functools import partial
from typing import TYPE_CHECKING, Callable

from ..base import PromptLoader
from ..tool_loop import run_tool_loop
from ...domain import Domain
from ...llm import LLMClient
from ...log import RunLogger
from ...tools import bash, grep, read

if TYPE_CHECKING:
    from ...env import PaperEnv


ACCUMULATOR_TOOL_FNS = (read, grep, bash)
ACCUMULATOR_TOOLS = [fn.schema for fn in ACCUMULATOR_TOOL_FNS]


def _bind_tools(env: "PaperEnv") -> dict[str, Callable[..., str]]:
    # Rebuilt per run because paper_dir is env-dependent; read takes an
    # absolute path and needs no binding.
    return {
        "Read": read,
        "Grep": partial(grep, paper_dir=env.paper_dir),
        "Bash": partial(bash, paper_dir=env.paper_dir),
    }


def _extract_json(text: str) -> str:
    """Pull a JSON object out of a fenced block or fall back to outer braces."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    s, e = text.find("{"), text.rfind("}")
    return text[s:e + 1] if s != -1 and e != -1 and e > s else text


class Accumulator:
    def __init__(self, *, llm: LLMClient, prompts: PromptLoader,
                 max_steps: int = 100):
        self.llm = llm
        self.prompts = prompts
        self.max_steps = max_steps

    def run(self, *, env: PaperEnv) -> dict:
        messages = self.prompts.render(
            "accumulator",
            paper_dir=str(env.paper_dir),
            max_steps=self.max_steps,
        )
        bound = _bind_tools(env)

        def dispatch(name: str, args: dict) -> str:
            fn = bound.get(name)
            if fn is None:
                return f"[unknown tool {name!r}]"
            return fn(**args)

        with RunLogger(env.paper_dir) as logger:
            on_msg = lambda msg: logger.message(stage="accumulator",
                                                  iter=env.arxiv_id, **msg)
            result = run_tool_loop(
                llm=self.llm, messages=messages, tools=ACCUMULATOR_TOOLS,
                dispatch=dispatch,
                max_steps=self.max_steps,
                on_message=on_msg,
            )
        if result.status != "natural_end" or not result.final_content:
            return {"title": "", "domain": None, "seeds": []}
        try:
            parsed = json.loads(_extract_json(result.final_content))
        except (json.JSONDecodeError, ValueError):
            return {"title": "", "domain": None, "seeds": []}
        if not isinstance(parsed, dict):
            return {"title": "", "domain": None, "seeds": []}
        try:
            domain = Domain(parsed.get("domain", ""))
        except ValueError:
            domain = None
        seeds = parsed.get("seeds", [])
        if not isinstance(seeds, list):
            seeds = []
        return {"title":  str(parsed.get("title", "")).strip(),
                "domain": domain,
                "seeds":  seeds}
