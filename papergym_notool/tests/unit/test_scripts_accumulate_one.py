import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from papergym.domain import Domain


def _import(scripts_dir: Path):
    sys.path.insert(0, str(scripts_dir))
    import accumulate_one as ao
    return ao


def test_accumulate_one_writes_seeds_and_copies_events(tmp_path: Path, monkeypatch):
    ao = _import(Path(__file__).resolve().parents[2] / "scripts")
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    work = tmp_path / "work"
    library = tmp_path / "library"
    events = tmp_path / "events"
    work.mkdir(); library.mkdir(); events.mkdir()
    arxiv_id = "2401.0001"
    paper_dir = work / arxiv_id
    paper_dir.mkdir()
    (paper_dir / "events.jsonl").write_text('{"x":1}\n')

    fake_run_result = {
        "title": "An Awesome RL Paper",
        "domain": Domain.RL,
        "seeds": [{"problem": "How to explore?", "method": "Use bonus."}],
    }
    fake_emb = [0.1] * 1536

    with patch.object(ao.PaperEnv, "reset", return_value=None), \
         patch.object(ao.Accumulator, "run", return_value=fake_run_result), \
         patch.object(ao.LLMClient, "__init__", return_value=None), \
         patch.object(ao.LLMClient, "embed", return_value=fake_emb):
        ao.main(argv=[
            "--arxiv-id", arxiv_id,
            "--work-root", str(work),
            "--library-root", str(library),
            "--events-dir", str(events),
        ])

    assert (events / f"{arxiv_id}.jsonl").exists()

    rows = [json.loads(l) for l in
             (library / "accumulator_log.jsonl").read_text().strip().splitlines()]
    assert any(r["status"] == "ok" and r["domain"] == "RL"
                and r["title"] == "An Awesome RL Paper" and r["n_seeds"] == 1
                for r in rows)

    from papergym.library import LibraryStore
    store = LibraryStore(library / "shard_0")
    q = np.array(fake_emb, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    seeds = store.retrieve(Domain.RL, q, k=3)
    assert len(seeds) == 1
    assert seeds[0].paper_id == "2401.0001"


def test_accumulate_one_writes_to_assigned_shard(tmp_path: Path, monkeypatch):
    """Different --shard-id values must land in different shard subdirs."""
    ao = _import(Path(__file__).resolve().parents[2] / "scripts")
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    work = tmp_path / "work"
    library = tmp_path / "library"
    events = tmp_path / "events"
    work.mkdir(); library.mkdir(); events.mkdir()

    fake_emb = [0.1] * 1536

    def _fake_run_result(arxiv_id):
        return {"title": f"T-{arxiv_id}", "domain": Domain.RL,
                "seeds": [{"problem": "p", "method": "m"}]}

    with patch.object(ao.PaperEnv, "reset", return_value=None), \
         patch.object(ao.LLMClient, "__init__", return_value=None), \
         patch.object(ao.LLMClient, "embed", return_value=fake_emb):
        # paper A → shard 0
        with patch.object(ao.Accumulator, "run",
                            return_value=_fake_run_result("A")):
            ao.main(argv=[
                "--arxiv-id", "2401.A",
                "--work-root", str(work),
                "--library-root", str(library),
                "--events-dir", str(events),
                "--shard-id", "0",
            ])
        # paper B → shard 1
        with patch.object(ao.Accumulator, "run",
                            return_value=_fake_run_result("B")):
            ao.main(argv=[
                "--arxiv-id", "2401.B",
                "--work-root", str(work),
                "--library-root", str(library),
                "--events-dir", str(events),
                "--shard-id", "1",
            ])

    from papergym.library import LibraryStore
    s0 = LibraryStore(library / "shard_0")
    s1 = LibraryStore(library / "shard_1")
    q = np.array(fake_emb, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-9)
    seeds_0 = s0.retrieve(Domain.RL, q, k=5)
    seeds_1 = s1.retrieve(Domain.RL, q, k=5)
    assert [s.paper_id for s in seeds_0] == ["2401.A"]
    assert [s.paper_id for s in seeds_1] == ["2401.B"]

    rows = [json.loads(l) for l in
             (library / "accumulator_log.jsonl").read_text().strip().splitlines()]
    by_aid = {r["arxiv_id"]: r for r in rows if r.get("status") == "ok"}
    assert by_aid["2401.A"]["shard_id"] == 0
    assert by_aid["2401.B"]["shard_id"] == 1


def test_events_copied_even_when_accumulator_raises(tmp_path: Path, monkeypatch):
    ao = _import(Path(__file__).resolve().parents[2] / "scripts")
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    work = tmp_path / "work"
    library = tmp_path / "library"
    events = tmp_path / "events"
    work.mkdir(); library.mkdir(); events.mkdir()
    arxiv_id = "2401.0009"
    paper_dir = work / arxiv_id
    paper_dir.mkdir()
    (paper_dir / "events.jsonl").write_text('{"role":"system"}\n')

    with patch.object(ao.PaperEnv, "reset", return_value=None), \
         patch.object(ao.Accumulator, "run", side_effect=RuntimeError("boom")), \
         patch.object(ao.LLMClient, "__init__", return_value=None):
        with pytest.raises(SystemExit) as excinfo:
            ao.main(argv=[
                "--arxiv-id", arxiv_id,
                "--work-root", str(work),
                "--library-root", str(library),
                "--events-dir", str(events),
            ])
    assert excinfo.value.code == 1

    assert (events / f"{arxiv_id}.jsonl").exists()
    rows = [json.loads(l) for l in
             (library / "accumulator_log.jsonl").read_text().strip().splitlines()]
    assert any(r["status"] == "error" and r["stage"] == "accumulator"
                and r["arxiv_id"] == arxiv_id for r in rows)


def test_pdf_fetch_failure_is_logged(tmp_path: Path, monkeypatch):
    ao = _import(Path(__file__).resolve().parents[2] / "scripts")
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    work = tmp_path / "work"
    library = tmp_path / "library"
    events = tmp_path / "events"
    work.mkdir(); library.mkdir(); events.mkdir()
    arxiv_id = "2401.0010"

    with patch.object(ao.PaperEnv, "reset",
                       side_effect=RuntimeError("paywall")):
        with pytest.raises(SystemExit) as excinfo:
            ao.main(argv=[
                "--arxiv-id", arxiv_id,
                "--work-root", str(work),
                "--library-root", str(library),
                "--events-dir", str(events),
            ])
    assert excinfo.value.code == 1

    rows = [json.loads(l) for l in
             (library / "accumulator_log.jsonl").read_text().strip().splitlines()]
    assert any(r["status"] == "error" and r["stage"] == "fetch"
                and "paywall" in r["error"] for r in rows)


def test_accumulate_one_skips_invalid_domain(tmp_path: Path, monkeypatch):
    ao = _import(Path(__file__).resolve().parents[2] / "scripts")
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    work = tmp_path / "work"
    library = tmp_path / "library"
    events = tmp_path / "events"
    work.mkdir(); library.mkdir(); events.mkdir()
    arxiv_id = "2401.0002"
    paper_dir = work / arxiv_id
    paper_dir.mkdir()

    with patch.object(ao.PaperEnv, "reset", return_value=None), \
         patch.object(ao.Accumulator, "run",
                       return_value={"title": "T", "domain": None, "seeds": []}), \
         patch.object(ao.LLMClient, "__init__", return_value=None):
        ao.main(argv=[
            "--arxiv-id", arxiv_id,
            "--work-root", str(work),
            "--library-root", str(library),
            "--events-dir", str(events),
        ])

    rows = [json.loads(l) for l in
             (library / "accumulator_log.jsonl").read_text().strip().splitlines()]
    assert any(r["status"] == "skipped" and r["arxiv_id"] == arxiv_id for r in rows)
