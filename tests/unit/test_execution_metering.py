import threading
import time

import pytest
from papergym.llm import ChatReply
from papergym.execution.metering import BudgetedLLM, UsageMeter, BudgetExceeded


class _FakeLLM:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._calls = 0
    def chat(self, messages, temperature=0.0):
        self._calls += 1
        self.total_prompt_tokens += 100
        self.total_completion_tokens += 20
        return f"reply{self._calls}"
    def chat_with_tools(self, messages, tools, temperature=0.0,
                        tool_choice="auto"):
        self._calls += 1
        self.total_prompt_tokens += 50
        self.total_completion_tokens += 10
        return ChatReply(content="done", tool_calls=[],
                         raw_message={"role": "assistant", "content": "done"})


def test_meter_records_tokens_and_cost():
    m = UsageMeter(_FakeLLM(), budget_usd=1.0, pricing=(1.0, 2.0))
    out = m.call([{"role": "user", "content": "hi"}])
    assert out == "reply1"
    assert m.calls == 1
    assert m.prompt_tokens == 100 and m.completion_tokens == 20
    assert round(m.cost_usd, 8) == round((100*1 + 20*2)/1_000_000, 8)


def test_budget_enforced():
    m = UsageMeter(_FakeLLM(), budget_usd=0.0, pricing=(1.0, 2.0))
    with pytest.raises(BudgetExceeded):
        m.call([{"role": "user", "content": "hi"}])


def test_usage_dict():
    m = UsageMeter(_FakeLLM(), budget_usd=1.0, pricing=(1.0, 2.0))
    m.call([{"role": "user", "content": "hi"}])
    u = m.usage()
    assert u["calls"] == 1 and "cost_usd" in u
    assert u["by_source"]["runtime"]["calls"] == 1


def test_budgeted_llm_charges_tool_loop_calls():
    meter = UsageMeter(_FakeLLM(), budget_usd=1.0, pricing=(1.0, 2.0))
    wrapped = BudgetedLLM(meter)
    reply = wrapped.chat_with_tools(
        [{"role": "user", "content": "hi"}], tools=[])
    assert reply.content == "done"
    usage = meter.usage()
    assert usage["calls"] == 1
    assert usage["by_source"]["planner"]["calls"] == 1


def test_budgeted_llm_enforces_zero_budget_on_planner_call():
    wrapped = BudgetedLLM(
        UsageMeter(_FakeLLM(), budget_usd=0.0, pricing=(1.0, 2.0)))
    with pytest.raises(BudgetExceeded):
        wrapped.chat_with_tools([{"role": "user", "content": "hi"}], tools=[])


def test_concurrent_calls_are_accounted_without_loss():
    class _SlowLLM:
        def __init__(self):
            self.total_prompt_tokens = 0
            self.total_completion_tokens = 0
        def chat(self, messages, temperature=0.0):
            p = self.total_prompt_tokens
            time.sleep(0.001)                      # widen the race window
            self.total_prompt_tokens = p + 10
            c = self.total_completion_tokens
            time.sleep(0.001)
            self.total_completion_tokens = c + 5
            return "ok"

    m = UsageMeter(_SlowLLM(), budget_usd=1_000.0, pricing=(1.0, 1.0))
    threads = [threading.Thread(
        target=lambda: m.call([{"role": "user", "content": "hi"}]))
        for _ in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert m.calls == 8
    assert m.prompt_tokens == 80          # 8 * 10, no lost updates
    assert m.completion_tokens == 40      # 8 * 5
