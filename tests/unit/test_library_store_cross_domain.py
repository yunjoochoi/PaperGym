from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from papergym.domain import Domain
from papergym.library import LibraryStore, Seed


def test_retrieve_cross_domain_skips_null_paraphrases(tmp_path: Path):
    store = LibraryStore(tmp_path, embedding_dim=4)
    e = np.array([1, 0, 0, 0], dtype=np.float32)
    store.add(Seed("a", "p1", "m1", Domain.LLM_NLP, "t", "id1"), e)
    store.add(Seed("b", "p2", "m2", Domain.RL,      "t", "id2"), e)

    paraphrases = {
        "LLM_NLP": "x", "MULTIMODAL": None, "CV": None, "RL": "y",
        "IR_REC": None, "SPEECH": None, "ROBOTICS": None,
    }
    fake_llm = MagicMock()
    fake_llm.embed.return_value = [1, 0, 0, 0]
    out = store.retrieve_cross_domain(
        paraphrases, raw_query="r", natural_domain=None,
        llm=fake_llm, k=2,
    )
    # With null paraphrases skipped, only LLM_NLP and RL slots issue
    # searches; each global search finds both seeds, so seed-home set
    # covers {LLM_NLP, RL} and origin set is {"LLM_NLP", "RL"}.
    assert {seed.domain for seed, _, _ in out} == {Domain.LLM_NLP, Domain.RL}
    assert {origin for _, origin, _ in out} == {"LLM_NLP", "RL"}
    # Each tuple's lens text matches the slot's paraphrase.
    for _seed, origin, lens in out:
        assert lens == paraphrases[origin]


def test_retrieve_cross_domain_natural_domain_uses_raw_query(tmp_path: Path):
    store = LibraryStore(tmp_path, embedding_dim=4)
    e = np.array([1, 0, 0, 0], dtype=np.float32)
    store.add(Seed("a", "p1", "m1", Domain.LLM_NLP, "t", "id1"), e)

    paraphrases = {d.value: f"para-{d.value}" for d in Domain}
    seen_texts: list[str] = []
    fake_llm = MagicMock()
    fake_llm.embed.side_effect = lambda text: (seen_texts.append(text)
                                                  or [1, 0, 0, 0])
    store.retrieve_cross_domain(
        paraphrases, raw_query="RAW", natural_domain="LLM_NLP",
        llm=fake_llm, k=1,
    )
    # natural_domain slot must embed the raw query, not the paraphrase.
    assert "RAW" in seen_texts
    assert "para-LLM_NLP" not in seen_texts
