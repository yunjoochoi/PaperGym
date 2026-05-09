import json
from pathlib import Path
from unittest.mock import MagicMock

from papergym.agents import PromptLoader, Synthesizer
from papergym.domain import Domain
from papergym.library import Seed


def test_synthesize_returns_method_and_provenance():
    llm = MagicMock()
    llm.chat.return_value = json.dumps({
        "method": "...", "rationale": "...",
        "inspired_by": [{"seed_id": "s1", "domain": "RL",
                          "borrowed_aspect": "credit assignment"}],
    })
    prompts = PromptLoader(Path(__file__).parent.parent.parent / "src" /
                            "papergym" / "agents" / "synthesizer")
    seeds = [Seed("s1", "p", "m", Domain.RL, "Some RL paper", "id1")]
    out = Synthesizer(llm, prompts).run("query", seeds=seeds)
    assert out["method"] == "..."
    assert out["rationale"] == "..."
    assert out["inspired_by"][0]["seed_id"] == "s1"
