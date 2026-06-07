from unittest import mock
from pathlib import Path
from eval.execution.evaluate import run_one_idea
from papergym.execution.task import GSM8KAccuracyTask
from papergym.execution.types import IdeaSpec, RunArtifact


def _idea():
    return IdeaSpec(idea_id="Math_3_Human", condition="Human", topic="Math",
                    title="t", proposal_text="method")


def test_run_one_idea_assembles_execresult(tmp_path):
    task = GSM8KAccuracyTask(n_examples=2)
    task._splits = {"test": [{"id": "0", "question": "q", "answer": "4"},
                             {"id": "1", "question": "q", "answer": "8"}],
                    "dev": [{"id": "d0", "question": "q", "answer": "2"}]}
    gen = mock.MagicMock(total_prompt_tokens=0, total_completion_tokens=0)
    judge = mock.MagicMock(total_prompt_tokens=0, total_completion_tokens=0)
    judge.chat.return_value = "faithful\nScore: 5"

    dummy_server = mock.MagicMock()
    with mock.patch("eval.execution.evaluate.GSM8KAccuracyTask.run_baseline",
                    return_value=0.5), \
         mock.patch("eval.execution.evaluate.run_proxy",
                    return_value=(dummy_server, "http://x", None)), \
         mock.patch("eval.execution.evaluate.ExecutionAgent") as AgentCls:
        AgentCls.return_value.run.return_value = RunArtifact(
            status="natural_end", code="m",
            predictions=[{"id": "0", "pred": "4"}, {"id": "1", "pred": "8"}])
        res = run_one_idea(idea=_idea(), task=task, gen_llm=gen,
                           judge_llm=judge, work_root=tmp_path / "run")

    assert res.baseline_metric == 0.5
    assert res.method_metric == 1.0
    assert res.effectiveness == 0.5
    assert res.faithfulness_score == 5
    assert res.leakage_flags == []
    assert "sandbox_llm" in res.cost
    assert res.sandbox == "local"
    assert res.trustworthy is False
    dummy_server.shutdown.assert_called_once()
