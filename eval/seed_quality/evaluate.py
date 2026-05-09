"""Score one seed on grounding and specificity.

Each axis is judged in a separate completion. Grounding sees the full
paper text so paraphrase isn't mistaken for hallucination.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

from papergym.agents.base import PromptLoader
from papergym.llm import LLMClient

from ..common import AxisScore, judge_axis


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class SeedJudgement:
    seed_id: str
    paper_id: str
    domain: str
    grounding: AxisScore
    specificity: AxisScore

    def to_dict(self) -> dict:
        return {
            "seed_id": self.seed_id,
            "paper_id": self.paper_id,
            "domain": self.domain,
            "grounding": asdict(self.grounding),
            "specificity": asdict(self.specificity),
        }


def judge_seed(*, judge_llm: LLMClient, seed: dict,
                paper_excerpt: str) -> SeedJudgement:
    prompts = PromptLoader(_PROMPTS_DIR)
    grounding = judge_axis(
        llm=judge_llm, prompts=prompts, prompt_name="grounding",
        problem=seed["problem"], method=seed["method"],
        paper_excerpt=paper_excerpt,
    )
    specificity = judge_axis(
        llm=judge_llm, prompts=prompts, prompt_name="specificity",
        problem=seed["problem"], method=seed["method"],
    )
    return SeedJudgement(
        seed_id=seed["seed_id"], paper_id=seed["paper_id"],
        domain=seed.get("domain", ""),
        grounding=grounding,
        specificity=specificity,
    )
