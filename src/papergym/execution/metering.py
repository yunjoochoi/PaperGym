"""Meter + budget-enforce all LLM calls made on behalf of a sandboxed agent."""
from __future__ import annotations

import threading

from eval.common import GPT5_PRICING


class BudgetExceeded(RuntimeError):
    pass


class UsageMeter:
    """Wraps one LLMClient; records cumulative tokens/cost across calls and
    blocks once budget_usd is exhausted. pricing = (prompt, completion) USD/1M."""

    def __init__(self, llm, budget_usd: float = 5.0, pricing=GPT5_PRICING):
        self._llm = llm
        self.budget_usd = budget_usd
        self.pricing = pricing
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._lock = threading.Lock()

    @property
    def cost_usd(self) -> float:
        return (self.prompt_tokens * self.pricing[0]
                + self.completion_tokens * self.pricing[1]) / 1_000_000

    def call(self, messages: list, temperature: float = 0.0) -> str:
        with self._lock:
            if self.cost_usd >= self.budget_usd:
                raise BudgetExceeded(
                    f"budget ${self.budget_usd} exhausted (${self.cost_usd:.4f})")
            before_p = self._llm.total_prompt_tokens
            before_c = self._llm.total_completion_tokens
            out = self._llm.chat(messages, temperature=temperature)
            self.calls += 1
            self.prompt_tokens += self._llm.total_prompt_tokens - before_p
            self.completion_tokens += self._llm.total_completion_tokens - before_c
            if self.cost_usd > self.budget_usd:
                raise BudgetExceeded(
                    f"budget ${self.budget_usd} overrun (${self.cost_usd:.4f})")
            return out

    def usage(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost_usd, 6)}
