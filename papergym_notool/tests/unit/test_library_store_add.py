from pathlib import Path
import numpy as np
from papergym.domain import Domain
from papergym.library.seed import Seed
from papergym.library.store import LibraryStore


def _seed(domain=Domain.LLM_NLP, sid="s1"):
    return Seed(seed_id=sid, problem="p", method="m", domain=domain,
                paper_title="t", paper_id="2401.0001")


def test_add_writes_jsonl_and_indexes_embedding(tmp_path: Path):
    store = LibraryStore(tmp_path)
    emb = np.ones(1536, dtype=np.float32) / np.sqrt(1536)
    store.add(_seed(), emb)

    seeds_path = tmp_path / "seeds.jsonl"
    assert seeds_path.exists()
    line = seeds_path.read_text().strip()
    assert "s1" in line and '"LLM_NLP"' in line


def test_add_multiple_domains_keeps_them_separate(tmp_path: Path):
    store = LibraryStore(tmp_path)
    e = np.ones(1536, dtype=np.float32) / np.sqrt(1536)
    store.add(_seed(domain=Domain.LLM_NLP, sid="a"), e)
    store.add(_seed(domain=Domain.RL, sid="b"), e)

    assert (tmp_path / "faiss" / "LLM_NLP.index").exists()
    assert (tmp_path / "faiss" / "RL.index").exists()


def test_stale_store_add_does_not_overwrite_newer_writer(tmp_path: Path):
    first = LibraryStore(tmp_path)
    second = LibraryStore(tmp_path)
    e = np.ones(1536, dtype=np.float32) / np.sqrt(1536)

    first.add(_seed(sid="a"), e)
    second.add(_seed(sid="b"), e)

    reopened = LibraryStore(tmp_path)
    seeds = reopened._seeds_by_domain[Domain.LLM_NLP]
    assert [s.seed_id for s in seeds] == ["a", "b"]
    assert reopened._faiss[Domain.LLM_NLP].ntotal == 2


def test_load_self_heals_orphan_jsonl_row(tmp_path: Path):
    """Partial-write crash where jsonl ran past FAISS — orphan row is dropped."""
    store = LibraryStore(tmp_path)
    emb = np.ones(1536, dtype=np.float32) / np.sqrt(1536)
    store.add(_seed(sid="a"), emb)
    with (tmp_path / "seeds.jsonl").open("a") as f:
        f.write('{"seed_id":"orphan","problem":"p","method":"m",'
                '"domain":"LLM_NLP","paper_title":"t","paper_id":"id"}\n')
    reopened = LibraryStore(tmp_path)
    seeds = reopened._seeds_by_domain[Domain.LLM_NLP]
    assert [s.seed_id for s in seeds] == ["a"]
    # jsonl was rewritten without the orphan
    rows = (tmp_path / "seeds.jsonl").read_text().strip().splitlines()
    assert len(rows) == 1


def test_load_self_heals_faiss_ahead_of_jsonl(tmp_path: Path):
    """FAISS wrote, jsonl append crashed — trim the trailing FAISS vector."""
    import faiss
    store = LibraryStore(tmp_path)
    emb = np.ones(1536, dtype=np.float32) / np.sqrt(1536)
    store.add(_seed(sid="a"), emb)
    # Manually add an extra vector to FAISS without a matching jsonl row.
    idx = faiss.read_index(str(tmp_path / "faiss" / "LLM_NLP.index"))
    idx.add(emb.reshape(1, -1))
    faiss.write_index(idx, str(tmp_path / "faiss" / "LLM_NLP.index"))
    reopened = LibraryStore(tmp_path)
    assert reopened._faiss[Domain.LLM_NLP].ntotal == 1
