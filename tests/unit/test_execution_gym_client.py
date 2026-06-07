import io, json
from unittest import mock
from papergym.execution import gym_client


def test_metered_llm_call_posts_and_returns_content(monkeypatch):
    monkeypatch.setenv("GYM_LLM_URL", "http://localhost:9000")
    resp = io.BytesIO(json.dumps({"content": "hello"}).encode())
    resp.status = 200
    with mock.patch.object(gym_client.urllib.request, "urlopen",
                           return_value=resp) as u:
        out = gym_client.metered_llm_call([{"role": "user", "content": "hi"}])
    assert out == "hello"
    assert u.called


def test_metered_llm_call_requires_url(monkeypatch):
    monkeypatch.delenv("GYM_LLM_URL", raising=False)
    try:
        gym_client.metered_llm_call([{"role": "user", "content": "hi"}])
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
