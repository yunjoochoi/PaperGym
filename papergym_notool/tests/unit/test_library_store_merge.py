from pathlib import Path

import numpy as np
import pytest

from papergym.domain import Domain
from papergym.library import LibraryStore, Seed, new_seed_id


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def _add(store: LibraryStore, *, problem: str, method: str, domain: Domain,
         paper_id: str, vec: np.ndarray) -> Seed:
    seed = Seed(seed_id=new_seed_id(), problem=problem, method=method,
                domain=domain, paper_title=f"title-{paper_id}",
                paper_id=paper_id)
    store.add(seed, _norm(vec.astype(np.float32)))
    return seed


def _basis(dim: int, i: int) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def test_open_merged_falls_back_when_no_shards(tmp_path: Path):
    s = LibraryStore(tmp_path / "lib")
    _add(s, problem="p", method="m", domain=Domain.RL, paper_id="A",
          vec=_basis(s._dim, 0))
    merged = LibraryStore.open_merged(tmp_path / "lib")
    # Behaves like a regular store; not read-only since no shards were found.
    assert not getattr(merged, "_read_only", False)
    seeds = merged.retrieve(Domain.RL, _norm(_basis(s._dim, 0)), k=3)
    assert [s.paper_id for s in seeds] == ["A"]


def test_open_merged_combines_two_shards(tmp_path: Path):
    root = tmp_path / "lib"
    s0 = LibraryStore(root / "shard_0")
    s1 = LibraryStore(root / "shard_1")
    _add(s0, problem="p0", method="m0", domain=Domain.RL, paper_id="A",
          vec=_basis(s0._dim, 0))
    _add(s1, problem="p1", method="m1", domain=Domain.RL, paper_id="B",
          vec=_basis(s1._dim, 1))
    _add(s1, problem="p2", method="m2", domain=Domain.CV,  paper_id="C",
          vec=_basis(s1._dim, 2))

    merged = LibraryStore.open_merged(root)
    assert getattr(merged, "_read_only", False) is True

    rl = merged.retrieve(Domain.RL, _norm(_basis(merged._dim, 0)), k=5)
    assert {s.paper_id for s in rl} == {"A", "B"}
    cv = merged.retrieve(Domain.CV, _norm(_basis(merged._dim, 2)), k=5)
    assert [s.paper_id for s in cv] == ["C"]
    speech = merged.retrieve(Domain.SPEECH,
                              _norm(_basis(merged._dim, 0)), k=5)
    assert speech == []


def test_open_merged_faiss_count_matches_seed_count(tmp_path: Path):
    root = tmp_path / "lib"
    s0 = LibraryStore(root / "shard_0")
    s1 = LibraryStore(root / "shard_1")
    for i in range(3):
        _add(s0, problem=f"p{i}", method="m", domain=Domain.RL,
              paper_id=f"A{i}", vec=_basis(s0._dim, i))
    for i in range(2):
        _add(s1, problem=f"q{i}", method="m", domain=Domain.RL,
              paper_id=f"B{i}", vec=_basis(s1._dim, i + 3))
    merged = LibraryStore.open_merged(root)
    assert merged._faiss[Domain.RL].ntotal == 5
    assert len(merged._seeds_by_domain[Domain.RL]) == 5


def test_open_merged_is_read_only(tmp_path: Path):
    root = tmp_path / "lib"
    s0 = LibraryStore(root / "shard_0")
    _add(s0, problem="p", method="m", domain=Domain.RL, paper_id="A",
          vec=_basis(s0._dim, 0))
    merged = LibraryStore.open_merged(root)
    extra = Seed(seed_id=new_seed_id(), problem="p", method="m",
                  domain=Domain.RL, paper_title="t", paper_id="X")
    with pytest.raises(RuntimeError, match="read-only"):
        merged.add(extra, _norm(_basis(merged._dim, 0)))
