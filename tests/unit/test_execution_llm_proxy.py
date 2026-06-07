import json
from papergym.execution.llm_proxy import handle_payload
from papergym.execution.metering import UsageMeter


class _FakeLLM:
    total_prompt_tokens = 0
    total_completion_tokens = 0
    def chat(self, messages, temperature=0.0):
        type(self).total_prompt_tokens += 10
        type(self).total_completion_tokens += 5
        return "ok"


def test_handle_payload_returns_content():
    m = UsageMeter(_FakeLLM(), budget_usd=1.0, pricing=(1.0, 1.0))
    status, body = handle_payload(m, {"messages": [{"role": "user", "content": "hi"}]})
    assert status == 200 and json.loads(body)["content"] == "ok"


def test_handle_payload_budget_error():
    m = UsageMeter(_FakeLLM(), budget_usd=0.0, pricing=(1.0, 1.0))
    status, body = handle_payload(m, {"messages": [{"role": "user", "content": "hi"}]})
    assert status == 429 and "budget" in json.loads(body)["error"].lower()
