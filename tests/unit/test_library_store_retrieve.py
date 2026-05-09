from pathlib import Path
import numpy as np
from papergym.domain import Domain
from papergym.library.seed import Seed
from papergym.library.store import LibraryStore


def _unit(*entries):
    v = np.array(entries, dtype=np.float32)
    v[3:] = 0  # zero-pad
    return v / (np.linalg.norm(v) + 1e-9)


def test_retrieve_returns_topk_in_same_domain(tmp_path: Path):
    store = LibraryStore(tmp_path, embedding_dim=8)
    seeds = [
        Seed("a", "pa", "ma", Domain.LLM_NLP, "ta", "id1"),
        Seed("b", "pb", "mb", Domain.LLM_NLP, "tb", "id2"),
        Seed("c", "pc", "mc", Domain.RL,      "tc", "id3"),
    ]
    embs = [
        np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    ]
    for s, e in zip(seeds, embs):
        store.add(s, e)

    query = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    hits = store.retrieve(Domain.LLM_NLP, query, k=2)
    assert [h.seed_id for h in hits] == ["a", "b"]

    # Cross-domain query into RL must NOT bleed into LLM_NLP.
    hits_rl = store.retrieve(Domain.RL, query, k=2)
    assert [h.seed_id for h in hits_rl] == ["c"]


def test_retrieve_empty_domain_returns_empty(tmp_path: Path):
    store = LibraryStore(tmp_path, embedding_dim=8)
    query = np.zeros(8, dtype=np.float32); query[0] = 1
    assert store.retrieve(Domain.SPEECH, query, k=3) == []


def test_retrieve_dedupes_by_paper_id_within_domain(tmp_path: Path):
    """When a single paper contributed multiple seeds in the same domain,
    retrieve() returns at most one seed per paper_id (the highest-scoring)."""
    store = LibraryStore(tmp_path, embedding_dim=8)
    # Same paper, two seeds, both close to query
    s1 = Seed("a", "p1", "m1", Domain.LLM_NLP, "tt", "paper-X")
    s2 = Seed("b", "p2", "m2", Domain.LLM_NLP, "tt", "paper-X")
    s3 = Seed("c", "p3", "m3", Domain.LLM_NLP, "tt", "paper-Y")
    e_close = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    e_close2 = np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    e_far = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
    e_close2 = e_close2 / np.linalg.norm(e_close2)
    e_far = e_far / np.linalg.norm(e_far)
    store.add(s1, e_close)
    store.add(s2, e_close2)
    store.add(s3, e_far)

    query = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    hits = store.retrieve(Domain.LLM_NLP, query, k=2)
    paper_ids = [h.paper_id for h in hits]
    assert paper_ids == ["paper-X", "paper-Y"]   # NOT [paper-X, paper-X]
