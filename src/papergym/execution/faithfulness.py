"""Faithfulness judge: does the executed code implement the proposed method?"""
from __future__ import annotations

from pathlib import Path

from eval.common import AxisScore, judge_axis
from papergym.agents.base import PromptLoader
from papergym.llm import LLMClient

_PROMPTS = PromptLoader(Path(__file__).resolve().parent / "prompts")


def judge_faithfulness(*, proposal: str, code: str,
                       judge_llm: LLMClient) -> AxisScore:
    return judge_axis(llm=judge_llm, prompts=_PROMPTS,
                      prompt_name="faithfulness",
                      proposal=proposal[:8000], code=code[:8000])
