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
        self.by_source: dict[str, dict[str, int]] = {}
        self._lock = threading.Lock()

    @property
    def cost_usd(self) -> float:
        return (self.prompt_tokens * self.pricing[0]
                + self.completion_tokens * self.pricing[1]) / 1_000_000

    def _record(self, source: str, prompt_delta: int,
                completion_delta: int) -> None:
        bucket = self.by_source.setdefault(
            source, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        bucket["calls"] += 1
        bucket["prompt_tokens"] += prompt_delta
        bucket["completion_tokens"] += completion_delta

    def call(self, messages: list, temperature: float = 0.0,
             source: str = "runtime") -> str:
        with self._lock:
            if self.cost_usd >= self.budget_usd:
                raise BudgetExceeded(
                    f"budget ${self.budget_usd} exhausted (${self.cost_usd:.4f})")
            before_p = self._llm.total_prompt_tokens
            before_c = self._llm.total_completion_tokens
            out = self._llm.chat(messages, temperature=temperature)
            prompt_delta = self._llm.total_prompt_tokens - before_p
            completion_delta = self._llm.total_completion_tokens - before_c
            self.calls += 1
            self.prompt_tokens += prompt_delta
            self.completion_tokens += completion_delta
            self._record(source, prompt_delta, completion_delta)
            if self.cost_usd > self.budget_usd:
                raise BudgetExceeded(
                    f"budget ${self.budget_usd} overrun (${self.cost_usd:.4f})")
            return out

    def call_with_tools(self, messages: list, tools: list,
                        temperature: float = 0.0,
                        tool_choice: str = "auto",
                        source: str = "planner"):
        with self._lock:
            if self.cost_usd >= self.budget_usd:
                raise BudgetExceeded(
                    f"budget ${self.budget_usd} exhausted (${self.cost_usd:.4f})")
            before_p = self._llm.total_prompt_tokens
            before_c = self._llm.total_completion_tokens
            out = self._llm.chat_with_tools(
                messages, tools=tools, temperature=temperature,
                tool_choice=tool_choice)
            prompt_delta = self._llm.total_prompt_tokens - before_p
            completion_delta = self._llm.total_completion_tokens - before_c
            self.calls += 1
            self.prompt_tokens += prompt_delta
            self.completion_tokens += completion_delta
            self._record(source, prompt_delta, completion_delta)
            if self.cost_usd > self.budget_usd:
                raise BudgetExceeded(
                    f"budget ${self.budget_usd} overrun (${self.cost_usd:.4f})")
            return out

    def usage(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost_usd, 6),
                "by_source": self.by_source}


class BudgetedLLM:
    """LLMClient-compatible wrapper that charges planner/tool-loop calls to
    the same meter used by sandbox runtime calls."""

    def __init__(self, meter: UsageMeter):
        self.meter = meter
        self.model = getattr(meter._llm, "model", None)

    @property
    def total_prompt_tokens(self) -> int:
        return self.meter._llm.total_prompt_tokens

    @property
    def total_completion_tokens(self) -> int:
        return self.meter._llm.total_completion_tokens

    def chat(self, messages: list, temperature: float = 0.0,
             response_format: dict | None = None) -> str:
        if response_format is not None:
            # ExecutionAgent does not use response_format; keep this explicit
            # until UsageMeter.call supports forwarding it.
            raise ValueError("BudgetedLLM.chat does not support response_format")
        return self.meter.call(messages, temperature=temperature, source="planner")

    def chat_with_tools(self, messages: list, tools: list,
                        temperature: float = 0.0,
                        tool_choice: str = "auto"):
        return self.meter.call_with_tools(
            messages, tools=tools, temperature=temperature,
            tool_choice=tool_choice, source="planner")
