from unittest import mock
from papergym.execution.task import GSM8KAccuracyTask, TASKS


def _examples():
    return [{"id": "0", "question": "2+2?", "answer": "4"},
            {"id": "1", "question": "3+5?", "answer": "8"}]


def test_gsm8k_score_is_exact_match_accuracy():
    task = GSM8KAccuracyTask(n_examples=2)
    task._splits = {"test": _examples()}
    acc = task.score([{"id": "0", "pred": "4"}, {"id": "1", "pred": "7"}],
                     split="test")
    assert acc == 0.5


def test_gsm8k_parse_answer_extracts_final_integer():
    task = GSM8KAccuracyTask()
    assert task.parse_pred("The answer is 42.") == "42"
    assert task.parse_pred("#### 8") == "8"
    assert task.parse_pred("no number here") == ""


def test_baseline_calls_llm_per_example_and_scores():
    task = GSM8KAccuracyTask(n_examples=2)
    task._splits = {"test": _examples()}
    llm = mock.MagicMock()
    llm.chat.side_effect = ["4", "8"]                 # both correct
    acc = task.run_baseline(llm, split="test")
    assert acc == 1.0
    assert llm.chat.call_count == 2


def test_registry_contains_gsm8k():
    assert "gsm8k_accuracy" in TASKS
