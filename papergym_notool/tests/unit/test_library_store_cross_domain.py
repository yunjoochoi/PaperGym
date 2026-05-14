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
    out = store.retrieve_cross_domain(paraphrases, llm=fake_llm, k=2)
    assert {seed.domain for seed, _ in out} == {Domain.LLM_NLP, Domain.RL}
