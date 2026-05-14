from pathlib import Path
import pytest
from unittest.mock import MagicMock
from papergym.agents.base import PromptLoader, BaseAgent

FIXTURE = Path(__file__).parent.parent / "fixtures" / "prompts"


def test_prompt_loader_renders_jinja():
    loader = PromptLoader(FIXTURE)
    messages = loader.render("dummy", role="observer", problem="hardcoded budget")
    assert messages[0]["role"] == "system"
    assert "observer" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "hardcoded budget" in messages[1]["content"]


def test_base_agent_call_invokes_llm_and_parses_json():
    llm = MagicMock()
    llm.chat.return_value = '{"problems": [{"text": "p1"}]}'
    loader = PromptLoader(FIXTURE)
    agent = BaseAgent(llm=llm, prompts=loader, prompt_name="dummy")

    result = agent.call(role="observer", problem="x")
    assert result == {"problems": [{"text": "p1"}]}
    call_kwargs = llm.chat.call_args.kwargs
    assert call_kwargs.get("response_format") == {"type": "json_object"}


def test_base_agent_raises_on_invalid_json():
    llm = MagicMock()
    llm.chat.return_value = "not json at all"
    loader = PromptLoader(FIXTURE)
    agent = BaseAgent(llm=llm, prompts=loader, prompt_name="dummy")
    with pytest.raises(ValueError, match="invalid JSON"):
        agent.call(role="x", problem="y")
