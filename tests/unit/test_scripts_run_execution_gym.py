import json, sys
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_execution_gym as reg  # noqa: E402

from papergym.execution.types import ExecResult, RunArtifact


def test_main_writes_results_and_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    ideas = tmp_path / "si_ideas"; ideas.mkdir()
    (ideas / "Math_3_Human.json").write_text(json.dumps({
        "idea_id": "Math_3_Human", "condition": "Human", "topic": "Math",
        "title": "t", "proposal_text": "method", "human_exec_scores": {}}))
    out = tmp_path / "eval"

    fake_res = ExecResult(idea_id="Math_3_Human", task_id="gsm8k_accuracy",
                          baseline_metric=0.5, method_metric=0.6,
                          effectiveness=0.1, faithfulness_score=4,
                          run=RunArtifact(status="natural_end"),
                          cost={"total_cost_usd": 0.0})
    with mock.patch.object(reg, "run_one_idea", return_value=fake_res), \
         mock.patch.object(reg, "LLMClient") as LC:
        LC.side_effect = [mock.MagicMock(model="gen"), mock.MagicMock(model="judge")]
        reg.main(["--ideas", str(ideas), "--output-dir", str(out),
                  "--n-examples", "2"])

    run_dir = next(out.glob("exec_*"))
    lines = (run_dir / "results.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["effectiveness"] == 0.1
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["n_ideas"] == 1
