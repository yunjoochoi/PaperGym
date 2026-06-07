from unittest import mock
from papergym.execution.task import GeneratedArithmeticTask, GSM8KAccuracyTask, TASKS


def _examples():
    return [{"id": "0", "question": "2+2?", "answer": "4"},
            {"id": "1", "question": "3+5?", "answer": "8"}]


def test_gsm8k_score_is_exact_match_accuracy():
    task = GSM8KAccuracyTask(n_examples=2)
    task._splits = {"test": _examples()}
    acc = task.score([{"id": "0", "pred": "4"}, {"id": "1", "pred": "7"}],
                     split="test")
    assert acc == 0.5


def test_score_counts_each_expected_id_once():
    task = GSM8KAccuracyTask(n_examples=2)
    task._splits = {"test": _examples()}
    acc = task.score([
        {"id": "0", "pred": "4"},
        {"id": "0", "pred": "4"},
        {"id": "1", "pred": "8"},
    ], split="test")
    assert acc == 1.0


def test_validate_predictions_flags_malformed_submission():
    task = GSM8KAccuracyTask(n_examples=2)
    task._splits = {"test": _examples()}
    flags = task.validate_predictions([
        {"id": "0", "pred": "4"},
        {"id": "0", "pred": "4"},
        {"id": "2", "pred": "9"},
    ], split="test")
    assert "duplicate_prediction_id:0" in flags
    assert "unknown_prediction_id:2" in flags
    assert any(f.startswith("missing_prediction_ids:1") for f in flags)


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


def test_registry_contains_generated_math():
    assert "generated_math_accuracy" in TASKS


def test_generated_math_materialize_writes_splits(tmp_path):
    counts = GeneratedArithmeticTask.materialize(
        data_root=tmp_path, n_test=2, n_dev=1, seed=123)
    assert counts == {"test": 2, "dev": 1}
    task = GeneratedArithmeticTask(data_root=tmp_path)
    assert len(task.examples("test")) == 2
    assert "answer" not in task.inputs("test")[0]
