from papergym.execution.scorer import score_effectiveness
from papergym.execution.task import GSM8KAccuracyTask
from papergym.execution.types import RunArtifact


def _task():
    t = GSM8KAccuracyTask(n_examples=2)
    t._splits = {"test": [{"id": "0", "question": "q", "answer": "4"},
                          {"id": "1", "question": "q", "answer": "8"}]}
    return t


def test_effectiveness_is_method_minus_baseline():
    run = RunArtifact(status="natural_end",
                      predictions=[{"id": "0", "pred": "4"},
                                   {"id": "1", "pred": "8"}])
    method_m, eff, flags = score_effectiveness(_task(), run, baseline_metric=0.5)
    assert method_m == 1.0 and eff == 0.5 and flags == []


def test_failed_run_yields_none_not_zero():
    run = RunArtifact(status="max_steps_exceeded", predictions=[])
    method_m, eff, flags = score_effectiveness(_task(), run, baseline_metric=0.5)
    assert method_m is None and eff is None


def test_leaked_code_marked_suspect():
    t = GSM8KAccuracyTask(n_examples=2)
    t._splits = {"test": [{"id": "0", "question": "q", "answer": "4"}]}
    run = RunArtifact(status="natural_end", code="ds = load_dataset('x')",
                      predictions=[{"id": "0", "pred": "4"}])
    method_m, eff, flags = score_effectiveness(t, run, baseline_metric=0.5)
    assert method_m is None and eff is None and flags


def test_duplicate_submission_is_unscored():
    run = RunArtifact(status="natural_end",
                      predictions=[{"id": "0", "pred": "4"},
                                   {"id": "0", "pred": "4"},
                                   {"id": "1", "pred": "8"}])
    method_m, eff, flags = score_effectiveness(_task(), run, baseline_metric=0.5)
    assert method_m is None and eff is None
    assert "duplicate_prediction_id:0" in flags
