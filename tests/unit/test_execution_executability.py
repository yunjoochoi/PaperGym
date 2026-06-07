import json
from unittest import mock
from papergym.execution.executability import classify_executability


def test_classify_parses_json_verdict():
    fake = mock.MagicMock()
    fake.chat.return_value = json.dumps({
        "gym_executable": True, "requires_gpu_training": False,
        "requires_human_eval": False, "blockers": [],
        "reasoning": "pure prompting"})
    out = classify_executability(proposal="prompt the model to ...", llm=fake)
    assert out["gym_executable"] is True
    assert out["blockers"] == []
    assert fake.chat.call_args.kwargs["response_format"] == {"type": "json_object"}


def test_classify_defaults_false_on_bad_json():
    fake = mock.MagicMock()
    fake.chat.return_value = "not json"
    out = classify_executability(proposal="x", llm=fake)
    assert out["gym_executable"] is False
    assert "parse_error" in out["blockers"]
