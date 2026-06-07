from unittest import mock
from papergym.execution.faithfulness import judge_faithfulness


def test_returns_axisscore_from_judge_text():
    judge = mock.MagicMock()
    judge.chat.return_value = "The code matches the proposal.\nScore: 4"
    out = judge_faithfulness(proposal="prompt method", code="method.py ...",
                             judge_llm=judge)
    assert out.score == 4
    assert judge.chat.call_args.kwargs["temperature"] == 0.0


def test_parse_failure_is_zero_sentinel():
    judge = mock.MagicMock()
    judge.chat.return_value = "no score line here"
    out = judge_faithfulness(proposal="p", code="c", judge_llm=judge)
    assert out.score == 0
