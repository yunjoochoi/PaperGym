import json, sys
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import fetch_si_ideas as fsi  # noqa: E402


def test_join_human_exec_scores_means_per_idea():
    exec_data = {
        "idea_id": ["Math_3_Human", "Math_3_Human", "Safety_2_AI"],
        "overall_score": [3, 5, 1],
        "novelty_score": [4, 4, 2],
    }
    means = fsi._mean_exec_scores(exec_data)
    assert means["Math_3_Human"]["overall"] == 4.0
    assert means["Math_3_Human"]["novelty"] == 4.0
    assert means["Safety_2_AI"]["overall"] == 1.0


def test_writes_one_ideaspec_json_per_idea(tmp_path):
    proposals = {"Math_3_Human": "Self-improving memory method ..."}
    means = {"Math_3_Human": {"overall": 4.0}}
    meta = {"Math_3_Human": {"condition": "Human", "topic": "Math",
                             "title": "Self-improving Memory"}}
    out = tmp_path / "si_ideas"
    fsi._write_ideaspecs(proposals, means, meta, out)
    rec = json.loads((out / "Math_3_Human.json").read_text())
    assert rec["condition"] == "Human"
    assert rec["human_exec_scores"]["overall"] == 4.0
    assert rec["proposal_text"].startswith("Self-improving")
