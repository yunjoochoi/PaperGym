import json
import sys
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import patch, MagicMock


def _import(scripts_dir: Path):
    sys.path.insert(0, str(scripts_dir))
    import run_accumulator as ra
    return ra


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def test_spawns_one_container_per_arxiv_id(tmp_path: Path, monkeypatch):
    ra = _import(SCRIPTS)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    arxiv_ids_path = tmp_path / "arxiv_ids.jsonl"
    arxiv_ids_path.write_text(
        json.dumps({"arxiv_id": "2401.0001"}) + "\n"
        + json.dumps({"arxiv_id": "2401.0002"}) + "\n"
    )

    fake_run = MagicMock(return_value=MagicMock(returncode=0))
    with patch.object(ra, "subprocess", MagicMock(run=fake_run)), \
         patch.object(ra, "ThreadPoolExecutor", _InlineExecutor):
        ra.main(argv=[
            "--arxiv-ids", str(arxiv_ids_path),
            "--library-root", str(tmp_path / "library"),
            "--events-dir", str(tmp_path / "events"),
            "--max-workers", "2",
            "--spawn-delay-s", "0",
        ])

    assert fake_run.call_count == 2
    cmds = [call.args[0] for call in fake_run.call_args_list]
    aids = [c[c.index("--arxiv-id") + 1] for c in cmds]
    assert set(aids) == {"2401.0001", "2401.0002"}
    # each spawn must include --shard-id
    for c in cmds:
        assert "--shard-id" in c


def test_max_papers_limits_iterations(tmp_path: Path, monkeypatch):
    ra = _import(SCRIPTS)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    arxiv_ids_path = tmp_path / "arxiv_ids.jsonl"
    arxiv_ids_path.write_text(
        "\n".join(json.dumps({"arxiv_id": f"2401.000{i}"}) for i in range(5))
    )

    fake_run = MagicMock(return_value=MagicMock(returncode=0))
    with patch.object(ra, "subprocess", MagicMock(run=fake_run)), \
         patch.object(ra, "ThreadPoolExecutor", _InlineExecutor):
        ra.main(argv=[
            "--arxiv-ids", str(arxiv_ids_path),
            "--library-root", str(tmp_path / "library"),
            "--events-dir", str(tmp_path / "events"),
            "--max-papers", "2",
            "--spawn-delay-s", "0",
        ])

    assert fake_run.call_count == 2


def test_resume_skips_already_done(tmp_path: Path, monkeypatch):
    ra = _import(SCRIPTS)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    arxiv_ids_path = tmp_path / "arxiv_ids.jsonl"
    arxiv_ids_path.write_text(
        "\n".join(json.dumps({"arxiv_id": f"2401.000{i}"}) for i in range(4))
    )
    library = tmp_path / "library"
    library.mkdir()
    log = library / "accumulator_log.jsonl"
    log.write_text(
        json.dumps({"arxiv_id": "2401.0000", "status": "ok"}) + "\n"
        + json.dumps({"arxiv_id": "2401.0001", "status": "skipped"}) + "\n"
        + json.dumps({"arxiv_id": "2401.0002", "status": "error"}) + "\n"
    )

    fake_run = MagicMock(return_value=MagicMock(returncode=0))
    with patch.object(ra, "subprocess", MagicMock(run=fake_run)), \
         patch.object(ra, "ThreadPoolExecutor", _InlineExecutor):
        ra.main(argv=[
            "--arxiv-ids", str(arxiv_ids_path),
            "--library-root", str(library),
            "--events-dir", str(tmp_path / "events"),
            "--max-workers", "2",
            "--spawn-delay-s", "0",
        ])

    cmds = [c.args[0] for c in fake_run.call_args_list]
    aids = [c[c.index("--arxiv-id") + 1] for c in cmds]
    # 0000 (ok) + 0001 (skipped) → skipped. 0002 (error) → retry. 0003 → new.
    assert set(aids) == {"2401.0002", "2401.0003"}


def test_resume_takes_latest_status_per_arxiv_id(tmp_path: Path, monkeypatch):
    """Earlier 'error' followed by later 'ok' must be considered done."""
    ra = _import(SCRIPTS)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    arxiv_ids_path = tmp_path / "arxiv_ids.jsonl"
    arxiv_ids_path.write_text(json.dumps({"arxiv_id": "2401.X"}) + "\n")
    library = tmp_path / "library"
    library.mkdir()
    log = library / "accumulator_log.jsonl"
    log.write_text(
        json.dumps({"arxiv_id": "2401.X", "status": "error"}) + "\n"
        + json.dumps({"arxiv_id": "2401.X", "status": "ok"}) + "\n"
    )

    fake_run = MagicMock(return_value=MagicMock(returncode=0))
    with patch.object(ra, "subprocess", MagicMock(run=fake_run)), \
         patch.object(ra, "ThreadPoolExecutor", _InlineExecutor):
        ra.main(argv=[
            "--arxiv-ids", str(arxiv_ids_path),
            "--library-root", str(library),
            "--events-dir", str(tmp_path / "events"),
            "--max-workers", "2",
            "--spawn-delay-s", "0",
        ])
    assert fake_run.call_count == 0


def test_shard_assignment_is_stable(tmp_path: Path, monkeypatch):
    """Same arxiv_id maps to the same shard regardless of run order."""
    ra = _import(SCRIPTS)
    a = ra._shard_for("2401.0001", 4)
    b = ra._shard_for("2401.0001", 4)
    assert a == b
    assert 0 <= a < 4
    # Different ids spread across shards.
    shards = {ra._shard_for(f"2401.000{i}", 4) for i in range(20)}
    assert len(shards) > 1


def test_shards_distributed_across_pending_papers(tmp_path: Path, monkeypatch):
    """Spawned containers spread across all configured shards."""
    ra = _import(SCRIPTS)
    monkeypatch.setenv("OPENAI_API_KEY", "test")

    arxiv_ids_path = tmp_path / "arxiv_ids.jsonl"
    arxiv_ids_path.write_text(
        "\n".join(json.dumps({"arxiv_id": f"2401.{i:04d}"}) for i in range(20))
    )

    fake_run = MagicMock(return_value=MagicMock(returncode=0))
    with patch.object(ra, "subprocess", MagicMock(run=fake_run)), \
         patch.object(ra, "ThreadPoolExecutor", _InlineExecutor):
        ra.main(argv=[
            "--arxiv-ids", str(arxiv_ids_path),
            "--library-root", str(tmp_path / "library"),
            "--events-dir", str(tmp_path / "events"),
            "--max-workers", "4",
            "--spawn-delay-s", "0",
        ])

    cmds = [c.args[0] for c in fake_run.call_args_list]
    shards = {int(c[c.index("--shard-id") + 1]) for c in cmds}
    # With 20 papers and a stable hash, all 4 shards should be hit.
    assert shards == {0, 1, 2, 3}


# ---- helpers ----

class _InlineExecutor:
    """A drop-in for ThreadPoolExecutor that runs submit() in-process.

    We don't actually need parallelism for unit tests — only the API shape.
    """
    def __init__(self, max_workers=None): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def submit(self, fn, *args, **kwargs):
        f = Future()
        try:
            f.set_result(fn(*args, **kwargs))
        except Exception as exc:
            f.set_exception(exc)
        return f
