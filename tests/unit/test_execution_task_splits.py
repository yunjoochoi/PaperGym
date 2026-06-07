import json
from papergym.execution.task import GSM8KAccuracyTask


def _write_splits(tmp_path):
    root = tmp_path / "gsm8k_accuracy"
    root.mkdir(parents=True)
    (root / "test.jsonl").write_text(
        '{"id":"0","question":"2+2?","answer":"4"}\n'
        '{"id":"1","question":"3+5?","answer":"8"}\n')
    (root / "dev.jsonl").write_text(
        '{"id":"d0","question":"1+1?","answer":"2"}\n')
    return tmp_path


def test_load_split_reads_local_jsonl(tmp_path):
    data_root = _write_splits(tmp_path)
    task = GSM8KAccuracyTask(n_examples=2, data_root=data_root)
    test = task.examples(split="test")
    assert [e["id"] for e in test] == ["0", "1"]
    dev = task.examples(split="dev")
    assert dev[0]["answer"] == "2"


def test_inputs_view_strips_answers(tmp_path):
    data_root = _write_splits(tmp_path)
    task = GSM8KAccuracyTask(n_examples=2, data_root=data_root)
    inputs = task.inputs(split="test")
    assert inputs == [{"id": "0", "question": "2+2?"},
                      {"id": "1", "question": "3+5?"}]
    assert all("answer" not in r for r in inputs)


def test_score_uses_named_split(tmp_path):
    data_root = _write_splits(tmp_path)
    task = GSM8KAccuracyTask(n_examples=2, data_root=data_root)
    acc = task.score([{"id": "0", "pred": "4"}, {"id": "1", "pred": "9"}],
                     split="test")
    assert acc == 0.5
