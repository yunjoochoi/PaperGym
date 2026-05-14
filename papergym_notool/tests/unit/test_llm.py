from unittest.mock import MagicMock

import papergym.llm as llm_mod
from papergym.llm import LLMClient


def _fake_chat_response(content: str = "hello world",
                        prompt_tokens: int = 10,
                        completion_tokens: int = 5):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


def test_llm_chat_records_cost(monkeypatch):
    fake_completion = MagicMock(return_value=_fake_chat_response())
    monkeypatch.setattr(llm_mod, "completion", fake_completion)

    client = LLMClient(api_key="sk-test", model="gpt-5")
    text = client.chat([{"role": "user", "content": "hi"}])

    assert text == "hello world"
    assert client.total_prompt_tokens == 10
    assert client.total_completion_tokens == 5
    args, kwargs = fake_completion.call_args
    assert kwargs["model"] == "gpt-5"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]


def test_llm_embed_records_cost(monkeypatch):
    fake_resp = MagicMock()
    fake_resp.data = [{"embedding": [0.1, 0.2, 0.3]}]
    fake_resp.usage.total_tokens = 7
    fake_embedding = MagicMock(return_value=fake_resp)
    monkeypatch.setattr(llm_mod, "embedding", fake_embedding)

    client = LLMClient(
        api_key="sk-test", model="gpt-5",
        embedding_model="text-embedding-3-small",
    )
    emb = client.embed("hello")

    assert emb == [0.1, 0.2, 0.3]
    assert client.total_embedding_tokens == 7
    args, kwargs = fake_embedding.call_args
    assert kwargs["model"] == "text-embedding-3-small"
    assert kwargs["input"] == "hello"
