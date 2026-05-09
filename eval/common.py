"""Shared score parsing for all stage judges."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from typing import Optional

from papergym.agents.base import PromptLoader
from papergym.llm import LLMClient


# Standard public pricing per 1M tokens (prompt, completion).
GPT5_PRICING = (1.25, 10.0)
SONNET46_PRICING = (3.0, 15.0)


@dataclass
class CostSnapshot:
    """LLMClient cumulative token snapshot for delta-based cost tracking."""
    prompt_tokens: int
    completion_tokens: int

    @classmethod
    def of(cls, llm: LLMClient) -> "CostSnapshot":
        return cls(llm.total_prompt_tokens, llm.total_completion_tokens)


def _delta_pair(before: CostSnapshot,
                 after: CostSnapshot) -> tuple[int, int]:
    return (after.prompt_tokens - before.prompt_tokens,
             after.completion_tokens - before.completion_tokens)


def _price(prompt_tok: int, completion_tok: int,
            pricing: tuple) -> float:
    return (prompt_tok * pricing[0]
             + completion_tok * pricing[1]) / 1_000_000


def cost_summary(*, judge_before: CostSnapshot,
                  judge_after: CostSnapshot,
                  wall_clock_s: float,
                  gen_before: Optional[CostSnapshot] = None,
                  gen_after: Optional[CostSnapshot] = None,
                  gen_pricing: tuple = GPT5_PRICING,
                  judge_pricing: tuple = SONNET46_PRICING) -> dict:
    """Build a cost/tokens dict for embedding in a summary.json.

    Pass gen_before / gen_after for scripts that drive generation;
    judge-only scripts (pairwise, grounding) leave them None and the
    gen fields collapse to zero.
    """
    if (gen_before is None) != (gen_after is None):
        raise ValueError(
            "gen_before and gen_after must both be provided or both None")
    gen_dt = (_delta_pair(gen_before, gen_after)
              if gen_before is not None else (0, 0))
    judge_dt = _delta_pair(judge_before, judge_after)
    gen_cost = _price(*gen_dt, gen_pricing)
    judge_cost = _price(*judge_dt, judge_pricing)
    return {
        "gen_tokens": {"prompt": gen_dt[0], "completion": gen_dt[1]},
        "judge_tokens": {"prompt": judge_dt[0], "completion": judge_dt[1]},
        "gen_cost_usd": round(gen_cost, 4),
        "judge_cost_usd": round(judge_cost, 4),
        "total_cost_usd": round(gen_cost + judge_cost, 4),
        "wall_clock_s": round(wall_clock_s, 2),
    }


# Line-anchored to avoid matching anchor citations inside reasoning prose.
# Restricted to 1-5 so out-of-range outputs fail parsing rather than coerce.
_SCORE_RE = re.compile(
    r"^\s*\*{0,2}\s*Score\s*\*{0,2}\s*[:：]\s*\*{0,2}\s*([1-5])",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class AxisScore:
    # score=0 is the sentinel for parse failure; aggregates treat it as missing.
    score: int
    reasoning: str


def parse_axis(raw: str) -> AxisScore:
    if not raw:
        _log_parse_failure("empty")
        return AxisScore(score=0, reasoning="")
    matches = _SCORE_RE.findall(raw)
    if not matches:
        _log_parse_failure(raw)
        return AxisScore(score=0, reasoning=raw[:200])
    score = int(matches[-1])
    last_score_pos = max((m.start() for m in _SCORE_RE.finditer(raw)),
                          default=len(raw))
    reasoning = raw[:last_score_pos].strip()
    if reasoning.lower().startswith("reasoning:"):
        reasoning = reasoning[len("reasoning:"):].strip()
    return AxisScore(score=score, reasoning=reasoning)


def _log_parse_failure(raw: str, reason: str = "no Score line") -> None:
    print(f"[parse_axis] {reason} | raw={raw!r}", file=sys.stderr)


def judge_axis(*, llm: LLMClient, prompts: PromptLoader,
                prompt_name: str, **fields) -> AxisScore:
    messages = prompts.render(prompt_name, **fields)
    raw = llm.chat(messages, temperature=0.0)
    return parse_axis(raw)
