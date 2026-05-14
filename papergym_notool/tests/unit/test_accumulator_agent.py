import json
from pathlib import Path
from unittest.mock import MagicMock

import papergym.env.base as env_base
from papergym.agents.accumulator import Accumulator
from papergym.agents.base import PromptLoader
from papergym.domain import Domain
from papergym.env import PaperEnv


def _make_env(tmp_path: Path, arxiv_id: str, paper_body: str,
              monkeypatch) -> PaperEnv:
    """Build a PaperEnv whose reset() writes paper.md without hitting arxiv."""
    monkeypatch.setattr(env_base, "fetch_paper_to_disk",
                        lambda **_: None)
    env = PaperEnv(arxiv_id=arxiv_id, work_root=tmp_path)
    env.paper_dir.mkdir(parents=True, exist_ok=True)
    (env.paper_dir / "paper.md").write_text(paper_body, encoding="utf-8")
    return env


PROMPTS = (Path(__file__).resolve().parents[2]
           / "src" / "papergym" / "agents" / "accumulator")


def test_accumulator_emits_title_domain_and_seeds(tmp_path: Path, monkeypatch):
    env = _make_env(tmp_path, "2401.0001", "# Some paper\n\nbody", monkeypatch)
    final_json = json.dumps({
        "title": "Some Paper: A Cool Method",
        "domain": "MULTIMODAL",
        "seeds": [
            {"problem": "P1", "method": "M1"},
            {"problem": "P2", "method": "M2"},
        ],
    })
    llm = MagicMock()
    llm.chat.return_value = f"```json\n{final_json}\n```"
    acc = Accumulator(llm=llm, prompts=PromptLoader(PROMPTS))
    out = acc.run(env=env)
    assert out["title"] == "Some Paper: A Cool Method"
    assert out["domain"] == Domain.MULTIMODAL
    assert len(out["seeds"]) == 2
    assert out["seeds"][0]["problem"] == "P1"
    assert "# Some paper" in llm.chat.call_args.kwargs["messages"][1]["content"]


def test_accumulator_returns_empty_on_empty_completion(tmp_path: Path, monkeypatch):
    env = _make_env(tmp_path, "2401.0002", "# x", monkeypatch)
    llm = MagicMock()
    llm.chat.return_value = ""
    acc = Accumulator(llm=llm, prompts=PromptLoader(PROMPTS))
    out = acc.run(env=env)
    assert out == {"title": "", "domain": None, "seeds": []}


def test_accumulator_returns_none_domain_on_invalid_emit(tmp_path: Path, monkeypatch):
    env = _make_env(tmp_path, "2401.0003", "# x", monkeypatch)
    final_json = json.dumps({"title": "T", "domain": "NOT_A_DOMAIN", "seeds": []})
    llm = MagicMock()
    llm.chat.return_value = f"```json\n{final_json}\n```"
    acc = Accumulator(llm=llm, prompts=PromptLoader(PROMPTS))
    out = acc.run(env=env)
    assert out["domain"] is None


def test_accumulator_returns_empty_on_unparseable_final(tmp_path: Path, monkeypatch):
    env = _make_env(tmp_path, "2401.0004", "# x", monkeypatch)
    llm = MagicMock()
    llm.chat.return_value = "I could not extract anything useful."
    acc = Accumulator(llm=llm, prompts=PromptLoader(PROMPTS))
    out = acc.run(env=env)
    assert out == {"title": "", "domain": None, "seeds": []}
