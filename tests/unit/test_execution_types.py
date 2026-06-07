from papergym.execution.types import IdeaSpec, RunArtifact, ExecResult


def test_ideaspec_roundtrips_and_defaults():
    idea = IdeaSpec(idea_id="Math_3_Human", condition="Human", topic="Math",
                    title="Self-improving Memory", proposal_text="...method...")
    assert idea.human_exec_scores == {}            # default empty
    d = idea.to_dict()
    assert d["idea_id"] == "Math_3_Human"
    assert IdeaSpec.from_dict(d) == idea


def test_execresult_marks_missing_effectiveness_when_metric_failed():
    run = RunArtifact(status="natural_end", code="print(1)", stdout="ok",
                      predictions=[{"id": "0", "pred": "5"}], steps=2)
    res = ExecResult(idea_id="X", task_id="gsm8k_accuracy",
                     baseline_metric=0.40, method_metric=None,
                     effectiveness=None, faithfulness_score=0,
                     run=run, cost={})
    assert res.effectiveness is None               # None propagates (not 0.0)
    assert res.to_dict()["method_metric"] is None
