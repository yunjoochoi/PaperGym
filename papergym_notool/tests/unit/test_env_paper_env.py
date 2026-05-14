from pathlib import Path
from unittest.mock import MagicMock

import papergym.env.base as base_mod
from papergym.env import PaperEnv


def _make_env(tmp_path: Path, arxiv_id: str = "2401.0001") -> PaperEnv:
    return PaperEnv(arxiv_id=arxiv_id, work_root=tmp_path)


def test_init_is_cheap_no_io(tmp_path: Path, monkeypatch):
    fetch = MagicMock()
    monkeypatch.setattr(base_mod, "fetch_paper_to_disk", fetch)
    env = _make_env(tmp_path)
    assert env.arxiv_id == "2401.0001"
    assert env.paper_dir == tmp_path / "2401.0001"
    fetch.assert_not_called()
    # paper_dir must NOT exist yet — construction is pure path math.
    assert not env.paper_dir.exists()


def test_reset_calls_fetch_and_creates_paper_dir(tmp_path: Path, monkeypatch):
    fetch = MagicMock()
    monkeypatch.setattr(base_mod, "fetch_paper_to_disk", fetch)
    env = _make_env(tmp_path)
    env.reset()
    fetch.assert_called_once_with(arxiv_id="2401.0001", root=tmp_path)
    assert env.paper_dir.exists()


def test_reset_is_idempotent_when_paper_md_exists(tmp_path: Path, monkeypatch):
    """`fetch_paper_to_disk` itself short-circuits when paper.md exists; reset
    should be safely callable multiple times without surprises."""
    calls = []
    def fake_fetch(*, arxiv_id, root):
        calls.append((arxiv_id, root))
        (Path(root) / arxiv_id).mkdir(parents=True, exist_ok=True)
        (Path(root) / arxiv_id / "paper.md").write_text("# x")
    monkeypatch.setattr(base_mod, "fetch_paper_to_disk", fake_fetch)

    env = _make_env(tmp_path)
    env.reset()
    env.reset()
    assert len(calls) == 2  # we don't dedupe at the env level
    assert (env.paper_dir / "paper.md").read_text() == "# x"


def test_close_is_safe_to_call(tmp_path: Path):
    env = _make_env(tmp_path)
    env.close()  # no-op for now, but callable
