from pathlib import Path
from unittest.mock import patch
import json, sys


def test_sample_envs_writes_arxiv_ids_jsonl(tmp_path: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import sample_envs as se

    fake_search = lambda **kw: [
        {"paperId": "P1", "title": "T1", "year": 2024, "citationCount": 999,
         "externalIds": {"ArXiv": "2401.0001"}}
    ]
    out_path = tmp_path / "arxiv_ids.jsonl"

    with patch.object(se, "_search_for_domain", side_effect=fake_search):
        se.main(argv=["--out", str(out_path),
                       "--budget-per-domain", "1",
                       "--only-domain", "RL",
                       "--all-venues"])

    rows = [json.loads(l) for l in out_path.read_text().strip().splitlines()]
    assert any(r["arxiv_id"] == "2401.0001"
                and r["source_query_domain"] == "RL" for r in rows)
