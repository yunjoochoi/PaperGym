from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from ..base import PromptLoader
from ...domain import Domain
from ...llm import LLMClient
from ...log import RunLogger

if TYPE_CHECKING:
    from ...env import PaperEnv


def _extract_json(text: str) -> str:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return m.group(1)
    s, e = text.find("{"), text.rfind("}")
    return text[s:e + 1] if s != -1 and e != -1 and e > s else text


class Accumulator:
    """No-tool ablation: full paper.md is injected into the user message and
    the model emits seeds in a single completion."""

    def __init__(self, *, llm: LLMClient, prompts: PromptLoader,
                 temperature: float = 0.3):
        self.llm = llm
        self.prompts = prompts
        self.temperature = temperature

    def run(self, *, env: PaperEnv) -> dict:
        paper_md = (env.paper_dir / "paper.md").read_text(encoding="utf-8")
        messages = self.prompts.render(
            "accumulator",
            paper_dir=str(env.paper_dir),
            paper_content=paper_md,
        )

        with RunLogger(env.paper_dir) as logger:
            for m in messages:
                logger.message(stage="accumulator", iter=env.arxiv_id,
                               role=m["role"], content=m["content"])
            try:
                raw = self.llm.chat(messages=messages,
                                    temperature=self.temperature,
                                    response_format={"type": "json_object"})
            except Exception:
                raw = self.llm.chat(messages=messages,
                                    temperature=self.temperature)
            logger.message(stage="accumulator", iter=env.arxiv_id,
                           role="assistant", content=raw)

        if not raw:
            return {"title": "", "domain": None, "seeds": []}
        try:
            parsed = json.loads(_extract_json(raw))
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
