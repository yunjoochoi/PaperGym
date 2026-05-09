# PaperGym v3 — Cross-domain Synthesis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate PaperGym from v2 (reproduce-grounded benchmark agent in Docker) to v3 (cross-domain idea synthesis from a 7-domain seed library, no Docker, no benchmarks) for the ICML AI4Research workshop.

**Architecture:** Clean migration via one cleanup PR (preserve v2 on `legacy` branch), then build new components in dependency order: Domain → LibraryStore → Bootstrap → Accumulator → Retrieval → Synthesizer → Driver scripts. Heavy use of mocks for LLM / API calls; real I/O exercised via a small smoke run.

**Tech Stack:** Python 3.11, pytest, FAISS (`IndexFlatIP`), OpenAI (chat + `text-embedding-3-small`), Semantic Scholar API, `arxiv` lib, `docling` (PDF→md), Jinja2 prompts.

**Design doc:** `docs/plans/2026-04-26-papergym-cross-domain-synthesis-design.md`. Read it first.

**Current state to be aware of:**
- 79 unit tests pass on main today.
- Working tree currently has uncommitted changes (deletes of old plan docs, edited `pyproject.toml`). Clean those up before starting Phase 0 (covered in Task 0.0).
- Branches: `main` (active), `feature/singularity` (untouched).

---

## Phase 0 — Branch prep

Goal: clean working tree, snapshot v2 on `legacy`, start migration on `main`.

### Task 0.0 — Tidy working tree

**Files:** working tree only.

**Steps:**
1. Inspect uncommitted state: `git status`. Confirm only deletions of stale plan docs and the `pyproject.toml` edit (pydantic removal).
2. Stage and commit those housekeeping changes:
   ```bash
   git add docs/plans pyproject.toml
   git commit -m "chore: clean stale plan docs and unused pydantic dep"
   ```
3. Confirm `git status` is clean.

### Task 0.1 — Create `legacy` branch

**Steps:**
1. From the current `main` HEAD, create the legacy branch:
   ```bash
   git branch legacy
   git push -u origin legacy
   ```
2. Verify with `git branch -a`. Both `legacy` and `origin/legacy` should appear.

This branch is the v2 snapshot. Never push to it again unless cherry-picking specific bug fixes.

### Task 0.2 — Add the design doc commit

**Files:** `docs/plans/2026-04-26-papergym-cross-domain-synthesis-design.md`, this plan doc.

**Steps:**
1. Stage both:
   ```bash
   git add docs/plans/2026-04-26-papergym-cross-domain-synthesis-design.md \
           docs/plans/2026-04-26-papergym-cross-domain-synthesis-impl.md
   git commit -m "docs: v3 cross-domain synthesis design + impl plan"
   ```

---

## Phase 1 — Cleanup (delete v2)

Goal: remove all v2 components in **one atomic commit**. Tests will be broken/missing after this — that is intentional and expected. Phases 2+ rebuild.

### Task 1.0 — Delete v2 source files

**Files (DELETE):**
```
src/papergym/pipeline.py
src/papergym/run.py
src/papergym/agents/observer.py
src/papergym/agents/implementer.py
src/papergym/agents/methodologist.py
src/papergym/agents/writer.py
src/papergym/agents/reviewer.py
src/papergym/agents/analyst.py
src/papergym/envs/__init__.py
src/papergym/envs/deployment.py
src/papergym/envs/reproduce.py
src/papergym/tools/code_tools.py
src/papergym/tools/scoring.py
src/papergym/tools/retrieval.py
src/papergym/library/bootstrap.py
src/papergym/library/store.py
src/papergym/observation.py
src/papergym/types.py
```

**Files (DELETE):** all v2 prompts:
```
src/papergym/prompts/observer.yaml
src/papergym/prompts/implementer.yaml
src/papergym/prompts/methodologist.yaml
src/papergym/prompts/methodologist_refine.yaml
src/papergym/prompts/writer.yaml
src/papergym/prompts/reviewer.yaml
src/papergym/prompts/analyst.yaml
src/papergym/prompts/paraphrase.yaml
```
(Prompts will be re-added in their new locations under the appropriate component dirs in later phases.)

**Files (DELETE):** all v2 unit + integration tests:
```
tests/unit/test_pipeline.py
tests/unit/test_outer_state.py
tests/unit/test_reproduce.py
tests/unit/test_deployment.py
tests/unit/test_agent_observer.py
tests/unit/test_agent_implementer.py
tests/unit/test_agent_methodologist.py
tests/unit/test_agent_writer.py
tests/unit/test_agent_reviewer.py
tests/unit/test_agent_analyst.py
tests/unit/test_code_tools.py
tests/unit/test_scoring.py
tests/unit/test_library_store.py
tests/unit/test_library_paraphrase.py
tests/unit/test_bootstrap_builder.py
tests/integration/test_docker_smoke.py
```

**Files (DELETE):** v2 scripts + docker:
```
scripts/build_base_image.sh
scripts/run_pipeline.py
docker/Dockerfile.base
```
Then `rmdir docker scripts` if both end up empty (they will repopulate).

**Steps:**
1. `git rm` each path above (use shell glob where possible):
   ```bash
   git rm src/papergym/pipeline.py src/papergym/run.py
   git rm src/papergym/agents/{observer,implementer,methodologist,writer,reviewer,analyst}.py
   git rm -r src/papergym/envs
   git rm src/papergym/tools/{code_tools,scoring,retrieval}.py
   git rm src/papergym/library/{bootstrap,store}.py
   git rm src/papergym/observation.py src/papergym/types.py
   git rm src/papergym/prompts/*.yaml
   git rm tests/unit/{test_pipeline,test_outer_state,test_reproduce,test_deployment}.py
   git rm tests/unit/test_agent_{observer,implementer,methodologist,writer,reviewer,analyst}.py
   git rm tests/unit/{test_code_tools,test_scoring,test_library_store,test_library_paraphrase,test_bootstrap_builder}.py
   git rm tests/integration/test_docker_smoke.py
   git rm -r docker scripts
   ```
2. Verify nothing else broke: `git status -uno`.
3. Confirm `git ls-files src/papergym/agents/` lists only `__init__.py`, `base.py`, `tool_loop.py`, `tool_schemas.py` (the survivors). Same for `src/papergym/library/` (only `__init__.py` survives).

### Task 1.1 — Strip v2-specific tool schemas

**File:** `src/papergym/agents/tool_schemas.py`

**Steps:**
1. Open the file. It currently exports `OBSERVER_TOOLS`, `IMPLEMENTER_TOOLS`, etc.
2. Delete every export. Replace the whole file with:
   ```python
   # Tool schemas are defined per-component in v3 (accumulator/, etc.).
   # This module is intentionally empty for now.
   ```
3. Save.

### Task 1.2 — Strip v2-specific helpers from `paper.py`

**File:** `src/papergym/paper.py`

**Steps:**
1. Read the file. Anything tied to `BaselineRecord`, `parse_final_score`, or `PaperContext`'s benchmark-related fields is v2-only.
2. Reduce `paper.py` to a minimal `PaperContext` dataclass with just `{title, authors, arxiv_id}` — what v3 needs (Bootstrap fills these). Drop anything else.
3. If `paper.py` is no longer worth its own module after slimming, leave it for now — Phase 2 may absorb it.

### Task 1.3 — Strip v2-only bits from `library/__init__.py`

**File:** `src/papergym/library/__init__.py`

**Steps:**
1. Replace contents with `# library package — concrete classes added in Phase 2`.

### Task 1.4 — Audit + adjust `pyproject.toml`

**File:** `pyproject.toml`

**Steps:**
1. Confirm `docker>=7.0` is gone (it was removed earlier).
2. Confirm `faiss-cpu>=1.8`, `arxiv`, `docling`, `requests`, `openai`, `python-dotenv`, `tenacity`, `numpy`, `pyyaml`, `jinja2` are all present.
3. Add `gitpython` is **not** required (v3 uses `subprocess` via Bash tool).
4. No code changes needed if all is well — just verify.

### Task 1.5 — Run tests, expect a partial collapse, commit cleanup

**Steps:**
1. Run: `.venv/bin/python -m pytest tests/unit -q 2>&1 | tail -20`
   - Expected: most tests are gone; surviving tests (`test_llm.py`, `test_paper.py`, `test_fetch_paper.py`, `test_agents_base.py`) should still pass. If one fails because it imported something we deleted, comment-out or delete it (record which).
2. Stage everything:
   ```bash
   git add -A
   git commit -m "refactor: drop v2 reproduce-grounded components for v3 pivot"
   ```

**Stopping point.** After Task 1.5, the codebase has only the survivors:
- `src/papergym/{__init__.py, config.py, llm.py, log.py, paper.py}`
- `src/papergym/agents/{__init__.py, base.py, tool_loop.py, tool_schemas.py}` (tool_schemas now empty)
- `src/papergym/library/__init__.py` (empty)
- `src/papergym/tools/{__init__.py, fetch_paper.py, web_search.py}`
- `src/papergym/prompts/__init__.py`
- `tests/unit/{test_llm.py, test_paper.py, test_fetch_paper.py, test_agents_base.py}`

---

## Phase 2 — Foundations: Domain enum + Seed + LibraryStore

Goal: data model + persistence layer.

### Task 2.0 — Add `Domain` enum

**File (Create):** `src/papergym/domain.py`

**Step 1: Write failing test**

**File (Create):** `tests/unit/test_domain.py`

```python
from papergym.domain import Domain, S2_FIELDS, S2_OVERRIDES


def test_seven_domains_no_generative():
    names = {d.value for d in Domain}
    assert names == {"LLM_NLP", "MULTIMODAL", "CV", "RL",
                     "IR_REC", "SPEECH", "ROBOTICS"}


def test_s2_fields_cover_all_domains():
    for d in Domain:
        assert d in S2_FIELDS
        assert isinstance(S2_FIELDS[d], list) and S2_FIELDS[d]


def test_s2_overrides_route_vlm_to_multimodal():
    assert S2_OVERRIDES["Vision and Language"] == Domain.MULTIMODAL
    assert S2_OVERRIDES["Visual Question Answering"] == Domain.MULTIMODAL
    assert S2_OVERRIDES["Image Captioning"] == Domain.MULTIMODAL
```

**Step 2: Run test to verify it fails**
```bash
.venv/bin/python -m pytest tests/unit/test_domain.py -v
```
Expected: ImportError (module not found).

**Step 3: Implement**

```python
# src/papergym/domain.py
from enum import Enum


class Domain(Enum):
    LLM_NLP    = "LLM_NLP"
    MULTIMODAL = "MULTIMODAL"
    CV         = "CV"
    RL         = "RL"
    IR_REC     = "IR_REC"
    SPEECH     = "SPEECH"
    ROBOTICS   = "ROBOTICS"


# Semantic Scholar `s2FieldsOfStudy` labels per domain.
# Source: https://api.semanticscholar.org/graph/v1 documentation.
S2_FIELDS: dict[Domain, list[str]] = {
    Domain.LLM_NLP:    ["Computational Linguistics", "Natural Language Processing"],
    Domain.MULTIMODAL: ["Multimodal Learning", "Vision and Language"],
    Domain.CV:         ["Computer Vision", "Image Recognition", "Pattern Recognition"],
    Domain.RL:         ["Reinforcement Learning"],
    Domain.IR_REC:     ["Information Retrieval", "Recommender Systems"],
    Domain.SPEECH:     ["Speech Processing", "Audio Processing"],
    Domain.ROBOTICS:   ["Robotics"],
}

# When S2 tags a paper with one of these labels, force the assignment
# regardless of any other tag. Catches the common "VLM mistagged as CV" failure.
S2_OVERRIDES: dict[str, Domain] = {
    "Vision and Language":         Domain.MULTIMODAL,
    "Visual Question Answering":   Domain.MULTIMODAL,
    "Image Captioning":            Domain.MULTIMODAL,
}
```

**Step 4: Run test, expect PASS**
```bash
.venv/bin/python -m pytest tests/unit/test_domain.py -v
```

**Step 5: Commit**
```bash
git add src/papergym/domain.py tests/unit/test_domain.py
git commit -m "feat: add Domain enum + S2 field mapping with override table"
```

### Task 2.1 — Add `Seed` dataclass

**File (Create):** `src/papergym/library/seed.py`

**Step 1: Failing test**

**File (Create):** `tests/unit/test_library_seed.py`
```python
from papergym.domain import Domain
from papergym.library.seed import Seed, new_seed_id


def test_seed_round_trip_to_dict():
    s = Seed(seed_id="abc123",
             problem="long-context attention is quadratic",
             method="sparse attention with sliding window",
             domain=Domain.LLM_NLP,
             paper_title="Longformer",
             paper_id="2004.05150")
    d = s.to_dict()
    assert d["domain"] == "LLM_NLP"
    s2 = Seed.from_dict(d)
    assert s2 == s


def test_new_seed_id_is_12_hex_chars():
    sid = new_seed_id()
    assert len(sid) == 12
    int(sid, 16)  # parses as hex
```

**Step 2: Run, expect ImportError.**

**Step 3: Implement**
```python
# src/papergym/library/seed.py
import uuid
from dataclasses import dataclass, asdict
from ..domain import Domain


def new_seed_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Seed:
    seed_id:     str
    problem:     str
    method:      str
    domain:      Domain
    paper_title: str
    paper_id:    str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["domain"] = self.domain.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Seed":
        d = dict(d)
        d["domain"] = Domain(d["domain"])
        return cls(**d)
```

**Step 4: Run, PASS.**

**Step 5: Commit**
```bash
git add src/papergym/library/seed.py tests/unit/test_library_seed.py
git commit -m "feat: add Seed dataclass (no embedding field; FAISS owns vectors)"
```

### Task 2.2 — `LibraryStore.add` and persistence layout

**File (Create):** `src/papergym/library/store.py`

**Step 1: Failing test**

**File (Create):** `tests/unit/test_library_store_add.py`
```python
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
```

**Step 2: Run, expect ImportError.**

**Step 3: Implement (initial — `add` only; `retrieve` follows in 2.3)**

```python
# src/papergym/library/store.py
import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from ..domain import Domain
from .seed import Seed

EMBEDDING_DIM = 1536  # text-embedding-3-small


class LibraryStore:
    """Domain-partitioned seed library.

    Layout under `root_dir`:
        seeds.jsonl                    # Seed metadata (no embeddings)
        faiss/<DOMAIN>.index           # IndexFlatIP per domain (cosine, brute-force)

    Insertion order in seeds.jsonl matches FAISS index position per domain.
    """

    def __init__(self, root_dir: Path, embedding_dim: int = EMBEDDING_DIM):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "faiss").mkdir(exist_ok=True)
        self._dim = embedding_dim

        self._seeds_path = self.root / "seeds.jsonl"
        self._faiss: dict[Domain, faiss.Index] = {}
        self._seeds_by_domain: dict[Domain, list[Seed]] = {d: [] for d in Domain}
        self._load()

    def _index_path(self, domain: Domain) -> Path:
        return self.root / "faiss" / f"{domain.value}.index"

    def _index(self, domain: Domain) -> faiss.Index:
        if domain in self._faiss:
            return self._faiss[domain]
        path = self._index_path(domain)
        if path.exists():
            idx = faiss.read_index(str(path))
        else:
            idx = faiss.IndexFlatIP(self._dim)
        self._faiss[domain] = idx
        return idx

    def _load(self) -> None:
        if not self._seeds_path.exists():
            return
        for line in self._seeds_path.read_text().splitlines():
            if not line.strip():
                continue
            s = Seed.from_dict(json.loads(line))
            self._seeds_by_domain[s.domain].append(s)

    def add(self, seed: Seed, embedding: np.ndarray) -> None:
        if embedding.shape != (self._dim,):
            raise ValueError(f"expected ({self._dim},), got {embedding.shape}")
        emb = embedding.astype(np.float32, copy=False).reshape(1, -1)
        idx = self._index(seed.domain)
        idx.add(emb)
        self._seeds_by_domain[seed.domain].append(seed)
        with self._seeds_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(seed.to_dict(), ensure_ascii=False) + "\n")
        faiss.write_index(idx, str(self._index_path(seed.domain)))
```

**Step 4: Run, expect PASS.**

**Step 5: Commit**
```bash
git add src/papergym/library/store.py tests/unit/test_library_store_add.py
git commit -m "feat(library): LibraryStore.add — jsonl + per-domain FAISS index"
```

### Task 2.3 — `LibraryStore.retrieve`

**Step 1: Failing test (extend `tests/unit/test_library_store_add.py` or new file)**

**File (Create):** `tests/unit/test_library_store_retrieve.py`
```python
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
```

**Step 2: Run, expect failure (method missing).**

**Step 3: Implement**

Add to `src/papergym/library/store.py`:
```python
    def retrieve(self, domain: Domain, query_emb: np.ndarray, k: int = 3) -> list[Seed]:
        seeds = self._seeds_by_domain[domain]
        if not seeds:
            return []
        idx = self._index(domain)
        if idx.ntotal == 0:
            return []
        q = query_emb.astype(np.float32, copy=False).reshape(1, -1)
        k = min(k, idx.ntotal)
        _scores, positions = idx.search(q, k)
        return [seeds[p] for p in positions[0] if p >= 0]
```

**Step 4: Run, PASS.**

**Step 5: Commit**
```bash
git add src/papergym/library/store.py tests/unit/test_library_store_retrieve.py
git commit -m "feat(library): LibraryStore.retrieve — per-domain top-k cosine"
```

### Task 2.4 — Within-domain dedupe by `paper_id` (per design §4.4 addendum)

**Step 1: Failing test**

**File (extend):** `tests/unit/test_library_store_retrieve.py`
```python
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
```

**Step 2: Run, FAIL (current retrieve returns both seeds from paper-X).**

**Step 3: Implement** — modify `retrieve()`:
```python
    def retrieve(self, domain: Domain, query_emb: np.ndarray, k: int = 3) -> list[Seed]:
        seeds = self._seeds_by_domain[domain]
        if not seeds:
            return []
        idx = self._index(domain)
        if idx.ntotal == 0:
            return []
        q = query_emb.astype(np.float32, copy=False).reshape(1, -1)
        # Over-fetch to allow paper_id dedupe to still return k results.
        over_k = min(k * 5, idx.ntotal)
        _scores, positions = idx.search(q, over_k)
        out: list[Seed] = []
        seen_papers: set[str] = set()
        for p in positions[0]:
            if p < 0:
                continue
            s = seeds[p]
            if s.paper_id in seen_papers:
                continue
            seen_papers.add(s.paper_id)
            out.append(s)
            if len(out) == k:
                break
        return out
```

**Step 4: Run, PASS.**

**Step 5: Commit**
```bash
git add src/papergym/library/store.py tests/unit/test_library_store_retrieve.py
git commit -m "feat(library): dedupe retrieved seeds by paper_id within domain"
```

### Task 2.5 — Update `library/__init__.py` exports

**File:** `src/papergym/library/__init__.py`
```python
from .seed import Seed, new_seed_id
from .store import LibraryStore, EMBEDDING_DIM

__all__ = ["Seed", "new_seed_id", "LibraryStore", "EMBEDDING_DIM"]
```

**Steps:**
1. Save.
2. `.venv/bin/python -c "from papergym.library import Seed, LibraryStore"` — must succeed.
3. Commit:
```bash
git add src/papergym/library/__init__.py
git commit -m "chore: export Seed + LibraryStore from library package"
```

---

## Phase 3 — Bootstrap (Semantic Scholar sampler + paper fetcher)

Goal: `bootstrap/` package + `scripts/bootstrap_papers.py` driver.

### Task 3.0 — S2 client wrapper

**File (Create):** `src/papergym/bootstrap/s2_client.py`

**Step 1: Failing test (mocking `requests`)**

**File (Create):** `tests/unit/test_bootstrap_s2_client.py`
```python
from unittest.mock import MagicMock, patch
from papergym.bootstrap.s2_client import S2Client


def _resp(payload, status=200):
    r = MagicMock(); r.status_code = status; r.json.return_value = payload
    r.raise_for_status = MagicMock()
    return r


def test_search_papers_paginates_until_budget(monkeypatch):
    pages = [
        _resp({"data": [{"paperId": f"p{i}", "title": f"t{i}",
                          "year": 2024, "citationCount": 100 - i,
                          "externalIds": {"ArXiv": f"2401.000{i}"},
                          "s2FieldsOfStudy": [{"category": "Computer Science"}]}
                         for i in range(50)],
                "next": 50}),
        _resp({"data": [{"paperId": f"q{i}", "title": f"x{i}",
                          "year": 2023, "citationCount": 50 - i,
                          "externalIds": {"ArXiv": f"2301.000{i}"},
                          "s2FieldsOfStudy": [{"category": "Computer Science"}]}
                         for i in range(30)]}),
    ]
    iter_pages = iter(pages)
    fake_get = lambda *a, **kw: next(iter_pages)
    monkeypatch.setattr("papergym.bootstrap.s2_client.requests.get", fake_get)

    client = S2Client(api_key=None)
    rows = client.search_papers(query="machine learning", year_range=(2017, 2025), max_results=80)
    assert len(rows) == 80
    assert rows[0]["paperId"] == "p0"
```

**Step 2: Run, ImportError.**

**Step 3: Implement** (only what the tests force):
```python
# src/papergym/bootstrap/s2_client.py
"""Thin wrapper around Semantic Scholar's `/graph/v1/paper/search/bulk`."""
from __future__ import annotations

from typing import Iterator, Optional
import requests
import time

S2_BASE = "https://api.semanticscholar.org/graph/v1"
SEARCH_FIELDS = ("paperId,title,year,citationCount,externalIds,"
                 "s2FieldsOfStudy,authors")


class S2Client:
    def __init__(self, api_key: Optional[str] = None,
                 sleep_s: float = 1.0):
        self.api_key = api_key
        self.sleep_s = sleep_s

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key} if self.api_key else {}

    def search_papers(self, *, query: str, year_range: tuple[int, int],
                      max_results: int) -> list[dict]:
        params = {
            "query": query,
            "fields": SEARCH_FIELDS,
            "year": f"{year_range[0]}-{year_range[1]}",
            "limit": 100,
        }
        out: list[dict] = []
        token: Optional[str] = None
        while len(out) < max_results:
            if token is not None:
                params["token"] = token
            r = requests.get(f"{S2_BASE}/paper/search/bulk", params=params,
                             headers=self._headers(), timeout=30)
            r.raise_for_status()
            data = r.json()
            out.extend(data.get("data", []))
            token = data.get("next") if isinstance(data.get("next"), str) else None
            if data.get("next") is None:
                break
            if not token:
                break
            time.sleep(self.sleep_s)
        return out[:max_results]
```

(Note: in the real S2 bulk-search API the pagination cursor is `token`; the mock test uses an int `next` field for a simpler stub. Adjust the test fixture if the real shape differs at integration time. Keep this scope narrow.)

**Step 4: Run, PASS.**

**Step 5: Commit**
```bash
git add src/papergym/bootstrap/s2_client.py tests/unit/test_bootstrap_s2_client.py
mkdir -p src/papergym/bootstrap
touch src/papergym/bootstrap/__init__.py
git add src/papergym/bootstrap/__init__.py
git commit -m "feat(bootstrap): S2Client.search_papers (bulk + pagination)"
```

### Task 3.1 — Domain-tag resolver (S2 fields → Domain)

**File (Create):** `src/papergym/bootstrap/domain_tag.py`

**Step 1: Failing test**

**File (Create):** `tests/unit/test_bootstrap_domain_tag.py`
```python
from papergym.domain import Domain
from papergym.bootstrap.domain_tag import resolve_domain


def test_override_wins_over_other_fields():
    fields = ["Computer Vision", "Vision and Language"]
    assert resolve_domain(fields, requested=Domain.CV) == Domain.MULTIMODAL


def test_no_match_returns_none():
    assert resolve_domain(["Mathematics"], requested=Domain.LLM_NLP) is None


def test_match_within_requested_domain():
    fields = ["Reinforcement Learning"]
    assert resolve_domain(fields, requested=Domain.RL) == Domain.RL


def test_match_against_wrong_requested_returns_none():
    fields = ["Reinforcement Learning"]
    assert resolve_domain(fields, requested=Domain.SPEECH) is None
```

**Step 2: Run, ImportError.**

**Step 3: Implement**
```python
# src/papergym/bootstrap/domain_tag.py
from typing import Optional
from ..domain import Domain, S2_FIELDS, S2_OVERRIDES


def resolve_domain(s2_fields: list[str], requested: Domain) -> Optional[Domain]:
    """Decide whether an S2 paper belongs to the requested domain.

    Returns the resolved Domain (which may differ from `requested` if an
    override fires, but in that case still must equal `requested` for the
    paper to be accepted), or None if the paper doesn't fit.
    """
    # 1. Override wins. If the override's target == requested, accept.
    for label in s2_fields:
        if label in S2_OVERRIDES:
            return S2_OVERRIDES[label] if S2_OVERRIDES[label] == requested else None
    # 2. Otherwise, look for any of `requested`'s S2 labels.
    if any(f in S2_FIELDS[requested] for f in s2_fields):
        return requested
    return None
```

**Step 4: Run, PASS.**

**Step 5: Commit**
```bash
git add src/papergym/bootstrap/domain_tag.py tests/unit/test_bootstrap_domain_tag.py
git commit -m "feat(bootstrap): resolve_domain — S2 fields → Domain with override"
```

### Task 3.2 — Year × citation grid sampler

**File (Create):** `src/papergym/bootstrap/sampler.py`

**Step 1: Failing test**

**File (Create):** `tests/unit/test_bootstrap_sampler.py`
```python
import random
from papergym.bootstrap.sampler import bucket_grid, sample_paper_grid


def _row(paper_id, year, citations):
    return {"paperId": paper_id, "year": year, "citationCount": citations}


def test_bucket_grid_assigns_year_cohort_and_citation_tier():
    rows = (
        [_row(f"a{i}", 2018, 1000 - i) for i in range(20)] +
        [_row(f"b{i}", 2024, 500 - i) for i in range(20)]
    )
    grid = bucket_grid(rows)
    # 2 year cohorts populated, each with 3 citation tiers => 6 cells
    assert ("2017-19", "top_5") in grid
    assert ("2017-19", "top_5_25") in grid
    assert ("2017-19", "top_25_50") in grid
    assert ("2023-25", "top_5") in grid
    # All entries from 2018 stayed in 2017-19 cohort
    assert all(r["year"] == 2018 for r in grid[("2017-19", "top_5")])


def test_sample_paper_grid_distributes_per_cell(monkeypatch):
    rng = random.Random(0)
    rows = [_row(f"p{i}", 2018, 1000 - i) for i in range(30)]
    sampled = sample_paper_grid(rows, budget=9, rng=rng)
    assert len(sampled) <= 9
```

**Step 2: Run, ImportError.**

**Step 3: Implement**
```python
# src/papergym/bootstrap/sampler.py
from __future__ import annotations
import random
from typing import Iterable

YEAR_BUCKETS = [
    ("2017-19", range(2017, 2020)),
    ("2020-22", range(2020, 2023)),
    ("2023-25", range(2023, 2026)),
]
CITATION_TIERS = [("top_5", 0.0, 0.05),
                  ("top_5_25", 0.05, 0.25),
                  ("top_25_50", 0.25, 0.50)]


def _year_bucket(year: int) -> str | None:
    for name, rng in YEAR_BUCKETS:
        if year in rng:
            return name
    return None


def bucket_grid(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Partition S2 rows into a (year_cohort, citation_tier) grid.

    Citation tiers are computed *within* each year cohort (year-relative
    percentiles) so 2024 papers don't get crushed by 2017 papers' larger
    absolute citation counts.
    """
    by_year: dict[str, list[dict]] = {y: [] for y, _ in YEAR_BUCKETS}
    for r in rows:
        y = r.get("year")
        if y is None:
            continue
        b = _year_bucket(y)
        if b is None:
            continue
        by_year[b].append(r)

    grid: dict[tuple[str, str], list[dict]] = {}
    for cohort, cohort_rows in by_year.items():
        if not cohort_rows:
            continue
        cohort_rows.sort(key=lambda r: -r.get("citationCount", 0))
        n = len(cohort_rows)
        for tier_name, lo, hi in CITATION_TIERS:
            start = int(n * lo)
            end = int(n * hi)
            grid[(cohort, tier_name)] = cohort_rows[start:end]
    return grid


def sample_paper_grid(rows: list[dict], budget: int,
                      rng: random.Random) -> list[dict]:
    """Sample ~budget/9 from each populated cell. Underfilled cells take what
    they have; total can be less than budget if cells underfill."""
    grid = bucket_grid(rows)
    n_cells = max(len(grid), 1)
    per_cell = max(budget // n_cells, 1)
    out: list[dict] = []
    for cell, cell_rows in grid.items():
        take = min(per_cell, len(cell_rows))
        out.extend(rng.sample(cell_rows, take))
    return out
```

**Step 4: Run, PASS.**

**Step 5: Commit**
```bash
git add src/papergym/bootstrap/sampler.py tests/unit/test_bootstrap_sampler.py
git commit -m "feat(bootstrap): year-cohort × citation-tier grid sampler"
```

### Task 3.3 — Paper fetcher (PDF → markdown)

**File (Create):** `src/papergym/bootstrap/fetch.py`

**Step 1: Failing test**

**File (Create):** `tests/unit/test_bootstrap_fetch.py`
```python
from pathlib import Path
from unittest.mock import patch, MagicMock
from papergym.bootstrap.fetch import fetch_paper_to_disk
from papergym.domain import Domain


def test_writes_paper_md_under_domain_arxivid_dir(tmp_path: Path):
    arxiv_id = "2401.12345"
    with patch("papergym.bootstrap.fetch._download_pdf",
                return_value=tmp_path / "tmp.pdf") as dl, \
         patch("papergym.bootstrap.fetch._pdf_to_markdown",
                return_value="# Title\n\nbody") as conv:
        out = fetch_paper_to_disk(arxiv_id=arxiv_id,
                                    domain=Domain.LLM_NLP,
                                    root=tmp_path / "papers")
    assert out == tmp_path / "papers" / "LLM_NLP" / arxiv_id / "paper.md"
    assert out.read_text().startswith("# Title")
    dl.assert_called_once()
    conv.assert_called_once()
```

**Step 2: Run, ImportError.**

**Step 3: Implement**

```python
# src/papergym/bootstrap/fetch.py
from __future__ import annotations
from pathlib import Path
import shutil
import tempfile

import arxiv

from ..domain import Domain
from ..tools.fetch_paper import _pdf_to_markdown  # reuse existing converter


def _download_pdf(arxiv_id: str, dest_dir: Path) -> Path:
    """Download via the arxiv lib. Returns the local pdf path."""
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(search.results())
    dest_dir.mkdir(parents=True, exist_ok=True)
    return Path(paper.download_pdf(dirpath=str(dest_dir)))


def fetch_paper_to_disk(*, arxiv_id: str, domain: Domain, root: Path) -> Path:
    """Fetch one paper. Returns path to written paper.md.
    Layout: <root>/<DOMAIN>/<arxiv_id>/paper.md"""
    out_dir = Path(root) / domain.value / arxiv_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_md = out_dir / "paper.md"
    if paper_md.exists():
        return paper_md

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = _download_pdf(arxiv_id, Path(tmp))
        md = _pdf_to_markdown(pdf_path)
    paper_md.write_text(md, encoding="utf-8")
    return paper_md
```

**Step 4: Run, PASS.**

**Step 5: Commit**
```bash
git add src/papergym/bootstrap/fetch.py tests/unit/test_bootstrap_fetch.py
git commit -m "feat(bootstrap): fetch_paper_to_disk — arxiv PDF → markdown"
```

### Task 3.4 — `scripts/bootstrap_papers.py` driver

**File (Create):** `scripts/bootstrap_papers.py`

**Step 1: Failing test (mocked)**

**File (Create):** `tests/unit/test_scripts_bootstrap.py`
```python
from pathlib import Path
from unittest.mock import patch, MagicMock
import json, sys


def test_bootstrap_main_writes_log_and_papers(tmp_path: Path, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import bootstrap_papers as bp

    fake_search = lambda **kw: [
        {"paperId": "P1", "title": "T1", "year": 2024, "citationCount": 999,
         "externalIds": {"ArXiv": "2401.0001"},
         "s2FieldsOfStudy": [{"category": "Reinforcement Learning"}]}
    ]
    fake_fetch = lambda **kw: tmp_path / "papers" / kw["domain"].value / kw["arxiv_id"] / "paper.md"

    with patch.object(bp, "_search_for_domain", side_effect=fake_search), \
         patch("papergym.bootstrap.fetch.fetch_paper_to_disk", side_effect=fake_fetch) as ff, \
         patch("papergym.bootstrap.fetch._download_pdf"), \
         patch("papergym.bootstrap.fetch._pdf_to_markdown", return_value="# T1"):
        bp.main(argv=["--budget-per-domain", "1", "--root", str(tmp_path),
                       "--only-domain", "RL"])

    log = (tmp_path / "bootstrap_log.jsonl").read_text().strip().splitlines()
    rows = [json.loads(l) for l in log]
    assert any(r["status"] == "ok" and r["domain"] == "RL" for r in rows)
```

**Step 2: Run, ImportError.**

**Step 3: Implement (minimal)**

```python
# scripts/bootstrap_papers.py
"""Bootstrap step: sample ~840 papers across 7 domains, fetch PDFs to disk.

Usage:
    .venv/bin/python scripts/bootstrap_papers.py --root data/papers
"""
import argparse
import json
import random
from pathlib import Path

from papergym.bootstrap.s2_client import S2Client
from papergym.bootstrap.sampler import sample_paper_grid
from papergym.bootstrap.domain_tag import resolve_domain
from papergym.bootstrap.fetch import fetch_paper_to_disk
from papergym.domain import Domain, S2_FIELDS

DEFAULT_BUDGET = {
    Domain.LLM_NLP: 180,
    Domain.MULTIMODAL: 150,
    Domain.CV: 130,
    Domain.RL: 100,
    Domain.IR_REC: 100,
    Domain.SPEECH: 100,
    Domain.ROBOTICS: 80,
}


def _search_for_domain(client: S2Client, domain: Domain,
                        max_results: int) -> list[dict]:
    """Run one or more S2 queries combining the domain's S2 field labels."""
    rows = []
    for label in S2_FIELDS[domain]:
        rows.extend(client.search_papers(query=label,
                                          year_range=(2017, 2025),
                                          max_results=max_results))
    # dedupe by paperId
    seen = set()
    out = []
    for r in rows:
        pid = r.get("paperId")
        if pid and pid not in seen:
            out.append(r); seen.add(pid)
    return out


def _arxiv_id(row: dict) -> str | None:
    eid = row.get("externalIds") or {}
    return eid.get("ArXiv")


def _fields(row: dict) -> list[str]:
    return [f["category"] for f in row.get("s2FieldsOfStudy") or [] if "category" in f]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--budget-per-domain", type=int, default=None,
                    help="override per-domain budget (otherwise use defaults)")
    p.add_argument("--only-domain", default=None,
                    help="restrict to one Domain.value (debugging)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--s2-api-key", default=None)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    client = S2Client(api_key=args.s2_api_key)
    log_path = args.root.parent / "bootstrap_log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    domains = [Domain(args.only_domain)] if args.only_domain else list(Domain)
    for domain in domains:
        budget = args.budget_per_domain or DEFAULT_BUDGET[domain]
        rows = _search_for_domain(client, domain, max_results=budget * 8)
        # Filter to rows that resolve to this domain (override-aware).
        rows = [r for r in rows if resolve_domain(_fields(r), requested=domain) is not None]
        # Drop rows missing arxiv ids (no fetchable PDF).
        rows = [r for r in rows if _arxiv_id(r)]

        for chosen in sample_paper_grid(rows, budget=budget, rng=rng):
            arxiv_id = _arxiv_id(chosen)
            try:
                fetch_paper_to_disk(arxiv_id=arxiv_id, domain=domain, root=args.root)
                status = "ok"
                err = None
            except Exception as exc:
                status = "skipped"
                err = str(exc)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "arxiv_id": arxiv_id, "domain": domain.value,
                    "year": chosen.get("year"),
                    "citations": chosen.get("citationCount"),
                    "title": chosen.get("title"),
                    "status": status, "error": err,
                }, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
```

**Step 4: Run, PASS.**

**Step 5: Commit**
```bash
git add scripts/bootstrap_papers.py tests/unit/test_scripts_bootstrap.py
git commit -m "feat(scripts): bootstrap_papers.py driver"
```

---

## Phase 4 — Accumulator (per-paper agentic seed extraction)

Goal: agentic ReAct over a single paper directory, emit 1–3 seeds.

### Task 4.0 — Accumulator prompt YAML

**File (Create):** `src/papergym/accumulator/prompts/accumulator.yaml`

```yaml
system: |
  You are the Accumulator. Given one paper (markdown + optionally a cloned
  GitHub repo), extract 1–3 distinct main contributions as seeds. Each seed
  must capture (a) the underlying problem the paper addresses, and (b) the
  method that solves it, including what is special / novel about the
  approach. Avoid splitting one idea into multiple sub-seeds; prefer fewer,
  higher-quality seeds.

  ## Workspace contract
  Paper directory:    {{ paper_dir }}
  paper.md            Pre-converted markdown of the PDF.
  repo/               Will exist if you `git clone` it. Otherwise unavailable.
  Title is the H1 on the first line of paper.md.
  Domain (assigned upstream, do NOT re-classify): {{ domain.value }}

  ## Tools
  - Read       : view paper.md and any file under the paper directory
  - Grep       : search inside paper.md or the cloned repo
  - Bash       : `git clone <url> {{ paper_dir }}/repo` (only if you've found a
                 *clearly official* GitHub URL in the paper);
                 also general filesystem exploration: tree / ls / find / wc
  - done       : terminate with `{"seeds": [{"problem": "...", "method": "..."}, ...]}`
                 (1–3 entries; or empty list if extraction failed)

  ## Default flow
  1. Read paper.md end-to-end.
  2. Look for an official GitHub URL (abstract, footnote 1, "Code:" line, etc.).
     If you find one and it is unambiguously the paper's repo, clone it.
     If you are unsure, do NOT clone — proceed paper-only.
  3. (If repo cloned) Use Bash / Grep to locate where the main contribution
     is implemented. Read enough to confirm what's special.
  4. Extract 1–3 seeds. Each seed:
       - problem: 1–2 sentences. The challenge the paper addresses.
       - method: 3–5 sentences. The solution AND what is special about it.
  5. Call `done` with the seeds.

  Budget: {{ max_steps }} tool calls. Plan first, then act.

user: |
  ## Paper
  Title (extract from paper.md H1):
  Domain: {{ domain.value }}
  Paper directory: {{ paper_dir }}

  Begin by reading paper.md.
```

**File (Create):** empty `__init__.py` files for the new packages:
```bash
mkdir -p src/papergym/accumulator/prompts
touch src/papergym/accumulator/__init__.py src/papergym/accumulator/prompts/__init__.py
```

(No test for the YAML itself; covered indirectly by Task 4.2 + 4.3.)

**Commit:**
```bash
git add src/papergym/accumulator
git commit -m "feat(accumulator): add prompt yaml"
```

### Task 4.1 — Accumulator tool dispatch (Read/Grep/Bash on local FS)

**File (Create):** `src/papergym/accumulator/tools.py`

The existing `tool_loop.py` dispatches via a v2 `Deployment` abstraction. v3 runs on local FS — no Deployment. Provide a thin `LocalFS` adapter that satisfies the same shape so we can reuse `tool_loop.py` without rewrite.

**Step 1: Failing test**

**File (Create):** `tests/unit/test_accumulator_tools.py`
```python
from pathlib import Path
from papergym.accumulator.tools import LocalFS


def test_read_returns_file_with_line_numbers(tmp_path: Path):
    f = tmp_path / "x.txt"; f.write_text("alpha\nbeta\n")
    fs = LocalFS()
    out = fs.read(str(f))
    assert "alpha" in out and "beta" in out


def test_bash_returns_stdout(tmp_path: Path):
    fs = LocalFS()
    out = fs.bash("echo hello-papergym", cwd=str(tmp_path))
    assert out.exit_code == 0 and "hello-papergym" in out.stdout
```

**Step 2: Run, ImportError.**

**Step 3: Implement**
```python
# src/papergym/accumulator/tools.py
from __future__ import annotations
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str = ""


class LocalFS:
    """Tool dispatch surface for the Accumulator. Mirrors the shape of v2's
    `Deployment` so `agents/tool_loop.py` can drive it unchanged."""

    def read(self, path: str, offset: int = 0, limit: int = 4000) -> str:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        chunk = lines[offset:offset + limit]
        return "\n".join(f"{i+offset+1:6d}\t{ln}" for i, ln in enumerate(chunk))

    def write(self, path: str, content: str) -> str:
        # Accumulator should not write source files; only allow within paper_dir.
        Path(path).write_text(content, encoding="utf-8")
        return f"wrote {len(content)} chars to {path}"

    def file_exists(self, path: str) -> bool:
        return Path(path).exists()

    def bash(self, command: str, *, cwd: str | None = None,
             timeout: int = 120) -> ExecResult:
        try:
            r = subprocess.run(["bash", "-c", command],
                                cwd=cwd, capture_output=True, text=True,
                                timeout=timeout)
            return ExecResult(exit_code=r.returncode,
                               stdout=r.stdout, stderr=r.stderr)
        except subprocess.TimeoutExpired:
            return ExecResult(exit_code=124, stdout="", stderr="timeout")
```

**Step 4: Run, PASS.**

**Step 5: Commit**
```bash
git add src/papergym/accumulator/tools.py tests/unit/test_accumulator_tools.py
git commit -m "feat(accumulator): LocalFS tool dispatch surface"
```

### Task 4.2 — Tool schemas for the Accumulator

**File (Create):** `src/papergym/accumulator/tool_schemas.py`

Provide OpenAI-compatible function definitions for `Read`, `Grep`, `Bash`, `done`. Refer to v2's `tool_schemas.py` (now empty) for shape; copy from the design's tool list.

**Step 1: Failing test**

**File (Create):** `tests/unit/test_accumulator_tool_schemas.py`
```python
from papergym.accumulator.tool_schemas import ACCUMULATOR_TOOLS


def test_has_required_tools():
    names = [t["function"]["name"] for t in ACCUMULATOR_TOOLS]
    assert set(names) == {"Read", "Grep", "Bash", "done"}


def test_done_schema_requires_seeds_array():
    done = next(t for t in ACCUMULATOR_TOOLS if t["function"]["name"] == "done")
    seeds_schema = done["function"]["parameters"]["properties"]["seeds"]
    assert seeds_schema["type"] == "array"
    assert seeds_schema["items"]["required"] == ["problem", "method"]
```

**Step 2: Run, ImportError.**

**Step 3: Implement**
```python
# src/papergym/accumulator/tool_schemas.py
ACCUMULATOR_TOOLS = [
    {"type": "function", "function": {
        "name": "Read",
        "description": "Read a file (paper.md, cloned repo files). Returns line-numbered content.",
        "parameters": {"type": "object", "properties": {
            "file_path": {"type": "string"},
            "offset":    {"type": "integer", "default": 0},
            "limit":     {"type": "integer", "default": 4000},
        }, "required": ["file_path"]},
    }},
    {"type": "function", "function": {
        "name": "Grep",
        "description": "Search a regex pattern in files under a path.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"},
            "path":    {"type": "string"},
        }, "required": ["pattern", "path"]},
    }},
    {"type": "function", "function": {
        "name": "Bash",
        "description": "Run a bash command. Use for `git clone <url>`, `tree`, `ls`, `find`, `wc`. Do NOT run experiments.",
        "parameters": {"type": "object", "properties": {
            "command":     {"type": "string"},
            "description": {"type": "string"},
            "timeout":     {"type": "integer", "default": 120},
        }, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "done",
        "description": "Terminate with the extracted seeds.",
        "parameters": {"type": "object", "properties": {
            "seeds": {"type": "array", "items": {
                "type": "object",
                "properties": {
                    "problem": {"type": "string"},
                    "method":  {"type": "string"},
                },
                "required": ["problem", "method"],
            }},
        }, "required": ["seeds"]},
    }},
]
```

**Step 4: Run, PASS.**

**Step 5: Commit**
```bash
git add src/papergym/accumulator/tool_schemas.py tests/unit/test_accumulator_tool_schemas.py
git commit -m "feat(accumulator): OpenAI tool schemas (Read/Grep/Bash/done)"
```

### Task 4.3 — Accumulator agent + Grep dispatch hook into `tool_loop`

The current `agents/tool_loop.py` dispatches Read/Write/Bash by name and assumes a `Deployment` (container). For v3 we want it to also dispatch Grep, and to operate on `LocalFS`. Easiest path: add a `dispatch` callable arg to `run_tool_loop` so callers (Accumulator, future synthesizer) provide their own routing. Backward compatibility unimportant — v2 callers are gone.

**Step 1: Failing test (mocked LLM, real LocalFS)**

**File (Create):** `tests/unit/test_accumulator_agent.py`
```python
import json
from pathlib import Path
from unittest.mock import MagicMock
from papergym.accumulator.agent import Accumulator
from papergym.accumulator.tools import LocalFS
from papergym.agents.base import PromptLoader
from papergym.domain import Domain
from papergym.llm import ChatReply, ToolCall


def _reply(*calls):
    msg = MagicMock(); msg.content = None
    tcs = [ToolCall(id=f"c{i}", name=n, arguments=json.dumps(a))
           for i, (n, a) in enumerate(calls)]
    msg.tool_calls = tcs
    return ChatReply(content=None, tool_calls=tcs, raw_message=msg)


def _seq(*replies):
    it = iter(replies); return lambda *a, **kw: next(it)


def test_accumulator_emits_seeds_via_done(tmp_path: Path):
    paper_dir = tmp_path / "LLM_NLP" / "2401.0001"
    paper_dir.mkdir(parents=True)
    (paper_dir / "paper.md").write_text("# Some paper\n\nbody")
    llm = MagicMock()
    llm.chat_with_tools.side_effect = _seq(
        _reply(("Read", {"file_path": str(paper_dir / "paper.md")})),
        _reply(("done", {"seeds": [
            {"problem": "P1", "method": "M1"},
            {"problem": "P2", "method": "M2"},
        ]})),
    )
    PROMPTS = Path(__file__).parent.parent.parent / "src" / "papergym" / "accumulator" / "prompts"
    acc = Accumulator(llm=llm, prompts=PromptLoader(PROMPTS),
                       fs=LocalFS(), max_steps=10)
    out = acc.run(paper_dir=paper_dir, domain=Domain.LLM_NLP)
    assert len(out) == 2
    assert out[0]["problem"] == "P1"
```

**Step 2: Run, ImportError.**

**Step 3: Modify `tool_loop.py`** — accept a `dispatch` callable.

**File (Modify):** `src/papergym/agents/tool_loop.py`. Replace the body of the `_dispatch` function with a thin shim that allows the *caller* to inject its own router. Concretely: change `run_tool_loop`'s signature to add `dispatch: Callable[[str, dict], str]` (required), and have it call `dispatch(name, args)` instead of the v2 `_dispatch(...)`. Delete the v2 `_dispatch`, `_container_to_host`, `_host_to_container`, `Deployment`-typed parameter. Keep loop detection, `done` termination, no-tool-call retry. Leave the message-emission hook (`on_message`) intact — `Phase 5` benefits from it too.

After the rewrite the surface is:
```python
def run_tool_loop(
    *, llm: LLMClient, messages: list[dict], tools: list[dict],
    dispatch: Callable[[str, dict], str],
    max_steps: int,
    temperature: float = 0.3,
    done_tool_name: str = "done",
    on_message: MessageHook = None,
) -> LoopResult:
    ...
```

**Step 4: Implement Accumulator**

**File (Create):** `src/papergym/accumulator/agent.py`
```python
from __future__ import annotations
import json
import re
from pathlib import Path

from ..agents.base import PromptLoader
from ..agents.tool_loop import run_tool_loop
from ..domain import Domain
from ..llm import LLMClient
from .tool_schemas import ACCUMULATOR_TOOLS
from .tools import LocalFS


class Accumulator:
    def __init__(self, *, llm: LLMClient, prompts: PromptLoader,
                 fs: LocalFS, max_steps: int = 100):
        self.llm = llm
        self.prompts = prompts
        self.fs = fs
        self.max_steps = max_steps

    def _dispatch(self, name: str, args: dict, *, paper_dir: Path) -> str:
        if name == "Read":
            return self.fs.read(args["file_path"],
                                 offset=int(args.get("offset", 0)),
                                 limit=int(args.get("limit", 4000)))
        if name == "Grep":
            # implement via bash grep -rn for simplicity
            cmd = f"grep -rn -- {json_quote(args['pattern'])} {json_quote(args['path'])} | head -200"
            r = self.fs.bash(cmd, cwd=str(paper_dir))
            return r.stdout if r.exit_code == 0 else (r.stdout or r.stderr or "(no matches)")
        if name == "Bash":
            r = self.fs.bash(args["command"], cwd=str(paper_dir),
                              timeout=int(args.get("timeout", 120)))
            return f"exit={r.exit_code}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
        return f"[unknown tool {name!r}]"

    def run(self, *, paper_dir: Path, domain: Domain) -> list[dict]:
        messages = self.prompts.render("accumulator",
                                        paper_dir=str(paper_dir),
                                        domain=domain,
                                        max_steps=self.max_steps)
        result = run_tool_loop(
            llm=self.llm, messages=messages, tools=ACCUMULATOR_TOOLS,
            dispatch=lambda name, args: self._dispatch(name, args, paper_dir=paper_dir),
            max_steps=self.max_steps,
        )
        if result.status != "done":
            return []
        return (result.done_args or {}).get("seeds", [])


def json_quote(s: str) -> str:
    """Shell-quote a string for use inside `bash -c`."""
    return json.dumps(s)
```

**Step 5: Run test, PASS.**

Will probably also touch `agents/base.PromptLoader` — confirm it accepts a relative `prompts_dir` and resolves `accumulator.yaml` correctly. If `PromptLoader` is hard-coded to look in `src/papergym/prompts/`, it likely needs a small generalization (constructor takes the directory).

**Step 6: Commit**
```bash
git add src/papergym/agents/tool_loop.py src/papergym/accumulator/agent.py tests/unit/test_accumulator_agent.py
git commit -m "feat(accumulator): Accumulator agent + tool_loop generalized via dispatch"
```

### Task 4.4 — `scripts/run_accumulator.py` driver

**File (Create):** `scripts/run_accumulator.py`

**Step 1: Failing test (mocked Accumulator + library)**

**File (Create):** `tests/unit/test_scripts_accumulator.py`

(Mock `Accumulator.run` to return canned seeds; mock `LLMClient.embed`. Verify the driver writes seeds + embeddings into a fresh `LibraryStore`.)

**Step 2: Run, ImportError.**

**Step 3: Implement**

```python
# scripts/run_accumulator.py
"""Iterate over data/papers/<DOMAIN>/<arxiv_id>/ and accumulate seeds.

Usage:
    .venv/bin/python scripts/run_accumulator.py \
        --papers-root data/papers --library-root data/library \
        [--max-papers 5] [--only-domain RL]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from papergym.accumulator.agent import Accumulator
from papergym.accumulator.tools import LocalFS
from papergym.agents.base import PromptLoader
from papergym.domain import Domain
from papergym.library import LibraryStore, Seed, new_seed_id
from papergym.llm import LLMClient


def _title_from_paper_md(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
        if line:
            return line[:120]
    return path.parent.name


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--papers-root", required=True, type=Path)
    p.add_argument("--library-root", required=True, type=Path)
    p.add_argument("--max-papers", type=int, default=None)
    p.add_argument("--only-domain", default=None)
    p.add_argument("--max-steps", type=int, default=100)
    args = p.parse_args(argv)

    llm = LLMClient(api_key=os.environ["OPENAI_API_KEY"])
    prompts = PromptLoader(Path(__file__).resolve().parents[1] /
                            "src" / "papergym" / "accumulator" / "prompts")
    fs = LocalFS()
    accumulator = Accumulator(llm=llm, prompts=prompts, fs=fs,
                                max_steps=args.max_steps)
    library = LibraryStore(args.library_root)

    log_path = args.library_root / "accumulator_log.jsonl"

    domains = [Domain(args.only_domain)] if args.only_domain else list(Domain)
    n_done = 0
    for domain in domains:
        domain_dir = args.papers_root / domain.value
        if not domain_dir.exists():
            continue
        for paper_dir in sorted(domain_dir.iterdir()):
            if args.max_papers and n_done >= args.max_papers:
                break
            arxiv_id = paper_dir.name
            paper_md = paper_dir / "paper.md"
            if not paper_md.exists():
                continue
            title = _title_from_paper_md(paper_md)
            try:
                raw = accumulator.run(paper_dir=paper_dir, domain=domain)
            except Exception as exc:
                with log_path.open("a") as f:
                    f.write(json.dumps({"arxiv_id": arxiv_id,
                                          "domain": domain.value,
                                          "status": "error",
                                          "error": str(exc)}) + "\n")
                continue

            n_added = 0
            for entry in raw[:3]:
                problem = (entry.get("problem") or "").strip()
                method = (entry.get("method") or "").strip()
                if not problem or not method:
                    continue
                seed = Seed(seed_id=new_seed_id(), problem=problem,
                             method=method, domain=domain,
                             paper_title=title, paper_id=arxiv_id)
                emb = np.array(llm.embed(problem), dtype=np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-9)
                library.add(seed, emb)
                n_added += 1
            with log_path.open("a") as f:
                f.write(json.dumps({"arxiv_id": arxiv_id,
                                      "domain": domain.value,
                                      "status": "ok",
                                      "n_seeds": n_added}) + "\n")
            n_done += 1


if __name__ == "__main__":
    main()
```

**Step 4: Run test, PASS.**

**Step 5: Commit**
```bash
git add scripts/run_accumulator.py tests/unit/test_scripts_accumulator.py
git commit -m "feat(scripts): run_accumulator.py driver"
```

---

## Phase 5 — Retrieval (paraphrase + per-domain query)

### Task 5.0 — Paraphrase prompt yaml

**File (Create):** `src/papergym/retrieval/prompts/paraphrase.yaml`
```yaml
system: |
  You paraphrase a research query across 7 ML domains, *not by substituting
  keywords*, but by first extracting the underlying problem essence and then
  re-instantiating that essence in each domain's natural problem space and
  vocabulary.

  Domains:
    - LLM_NLP    : language models, NLP
    - MULTIMODAL : vision-language and other cross-modal
    - CV         : visual perception (classification, detection, segmentation)
    - RL         : reinforcement learning, policies, environments
    - IR_REC     : information retrieval, recommender systems
    - SPEECH     : speech / audio processing
    - ROBOTICS   : embodied agents, manipulation, planning

  Push hard — even cross-domain mappings that *feel* forced often surface
  useful seeds when essence-aligned. Only emit `null` for a domain when the
  problem essence genuinely has no analog there.

  Output JSON:
    {
      "essence":  "<one sentence stating the abstract challenge>",
      "paraphrases": {
        "LLM_NLP":    "<query rephrased for LLM_NLP>",
        "MULTIMODAL": "...",
        "CV":         "...",
        "RL":         "...",
        "IR_REC":     "...",
        "SPEECH":     "...",
        "ROBOTICS":   null
      }
    }

user: |
  Query: {{ query }}
```

```bash
mkdir -p src/papergym/retrieval/prompts
touch src/papergym/retrieval/__init__.py src/papergym/retrieval/prompts/__init__.py
```

### Task 5.1 — `paraphrase_query(query) -> dict`

**Step 1: Failing test (mock LLM)**

**File (Create):** `tests/unit/test_retrieval_paraphrase.py`
```python
import json
from unittest.mock import MagicMock
from pathlib import Path
from papergym.agents.base import PromptLoader
from papergym.retrieval.paraphrase import paraphrase_query


def test_paraphrase_returns_per_domain_dict():
    llm = MagicMock()
    llm.chat.return_value = json.dumps({
        "essence": "process long sequences within bounded compute",
        "paraphrases": {
            "LLM_NLP":    "efficient long-context inference",
            "MULTIMODAL": "efficient cross-modal long fusion",
            "CV":         "long video / high-res image processing",
            "RL":         "credit assignment over long horizons",
            "IR_REC":     "efficient retrieval over very large corpora",
            "SPEECH":     "long-form / streaming audio inference",
            "ROBOTICS":   None,
        }
    })
    prompts = PromptLoader(Path(__file__).parent.parent.parent / "src" /
                            "papergym" / "retrieval" / "prompts")
    result = paraphrase_query("long-context efficient inference",
                                llm=llm, prompts=prompts)
    assert result["essence"].startswith("process long")
    assert result["paraphrases"]["ROBOTICS"] is None
    assert result["paraphrases"]["LLM_NLP"]
```

**Step 2: Run, ImportError.**

**Step 3: Implement**
```python
# src/papergym/retrieval/paraphrase.py
import json
from ..agents.base import PromptLoader
from ..llm import LLMClient


def paraphrase_query(query: str, *, llm: LLMClient, prompts: PromptLoader) -> dict:
    messages = prompts.render("paraphrase", query=query)
    raw = llm.chat(messages, temperature=0.4,
                    response_format={"type": "json_object"})
    return json.loads(raw)
```

**Step 4: Run, PASS.**

**Step 5: Commit**
```bash
git add src/papergym/retrieval tests/unit/test_retrieval_paraphrase.py
git commit -m "feat(retrieval): paraphrase_query — essence-driven per-domain"
```

### Task 5.2 — `retrieve_cross_domain(...)`

**Step 1: Failing test**

**File (Create):** `tests/unit/test_retrieval_cross_domain.py`
```python
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

from papergym.domain import Domain
from papergym.library import LibraryStore, Seed
from papergym.retrieval.retrieve import retrieve_cross_domain


def test_skips_domains_with_null_paraphrase(tmp_path: Path):
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
    out = retrieve_cross_domain(paraphrases, store=store, llm=fake_llm, k=2)
    assert {seed.domain for seed, _ in out} == {Domain.LLM_NLP, Domain.RL}
```

**Step 2: Run, ImportError.**

**Step 3: Implement**
```python
# src/papergym/retrieval/retrieve.py
from typing import Optional
import numpy as np
from ..domain import Domain
from ..library import LibraryStore, Seed
from ..llm import LLMClient


def retrieve_cross_domain(paraphrases: dict[str, Optional[str]],
                           *, store: LibraryStore, llm: LLMClient, k: int = 3
                          ) -> list[tuple[Seed, str]]:
    out: list[tuple[Seed, str]] = []
    for domain in Domain:
        text = paraphrases.get(domain.value)
        if not text:
            continue
        emb = np.array(llm.embed(text), dtype=np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        for seed in store.retrieve(domain, emb, k=k):
            out.append((seed, domain.value))
    return out
```

**Step 4: Run, PASS.**

**Step 5: Commit**
```bash
git add src/papergym/retrieval/retrieve.py tests/unit/test_retrieval_cross_domain.py
git commit -m "feat(retrieval): retrieve_cross_domain — paraphrase-per-domain top-k"
```

---

## Phase 6 — Synthesizer

### Task 6.0 — Synthesizer prompt yaml

**File (Create):** `src/papergym/synthesizer/prompts/synthesize.yaml`
```yaml
system: |
  You synthesize a method that addresses the user's query by drawing on
  retrieved seeds from up to 7 ML domains. Each seed is a published
  contribution with a (problem, method, domain, paper_title) tuple.

  Your output MUST be JSON:
    {
      "method":    "<3–5 paragraphs describing the proposed method end-to-end>",
      "rationale": "<why this addresses the user query, in 1–3 paragraphs>",
      "inspired_by": [
        { "seed_id": "<from one of the provided seeds>",
          "domain":  "<that seed's domain>",
          "borrowed_aspect": "<short noun phrase: what was taken from this seed>" },
        ...
      ]
    }

  Provenance is mandatory: every concrete mechanism / trick in `method` should
  trace back to at least one seed in `inspired_by`. Cross-domain inspirations
  (i.e., where domain differs from the query's natural domain) are the
  point — feature them when they help.

user: |
  ## Query
  {{ query }}

  ## Retrieved seeds
  {% for s in seeds %}
  ### Seed {{ s.seed_id }} [{{ s.domain.value }}] — "{{ s.paper_title }}"
  - Problem: {{ s.problem }}
  - Method:  {{ s.method }}
  {% endfor %}
```

```bash
mkdir -p src/papergym/synthesizer/prompts
touch src/papergym/synthesizer/__init__.py src/papergym/synthesizer/prompts/__init__.py
```

### Task 6.1 — `synthesize(query, seeds) -> dict`

**Step 1: Failing test (mock LLM)**

**File (Create):** `tests/unit/test_synthesizer.py`
```python
import json
from pathlib import Path
from unittest.mock import MagicMock
from papergym.agents.base import PromptLoader
from papergym.domain import Domain
from papergym.library import Seed
from papergym.synthesizer.synthesize import synthesize


def test_synthesize_returns_method_and_provenance():
    llm = MagicMock()
    llm.chat.return_value = json.dumps({
        "method": "...", "rationale": "...",
        "inspired_by": [{"seed_id": "s1", "domain": "RL",
                          "borrowed_aspect": "credit assignment"}],
    })
    prompts = PromptLoader(Path(__file__).parent.parent.parent / "src" /
                            "papergym" / "synthesizer" / "prompts")
    seeds = [Seed("s1", "p", "m", Domain.RL, "Some RL paper", "id1")]
    out = synthesize("query", seeds=seeds, llm=llm, prompts=prompts)
    assert out["method"] == "..."
    assert out["inspired_by"][0]["seed_id"] == "s1"
```

**Step 2: Run, ImportError.**

**Step 3: Implement**
```python
# src/papergym/synthesizer/synthesize.py
import json
from typing import Iterable
from ..agents.base import PromptLoader
from ..library import Seed
from ..llm import LLMClient


def synthesize(query: str, *, seeds: Iterable[Seed], llm: LLMClient,
               prompts: PromptLoader) -> dict:
    messages = prompts.render("synthesize", query=query, seeds=list(seeds))
    raw = llm.chat(messages, temperature=0.5,
                    response_format={"type": "json_object"})
    return json.loads(raw)
```

**Step 4: Run, PASS.**

**Step 5: Commit**
```bash
git add src/papergym/synthesizer tests/unit/test_synthesizer.py
git commit -m "feat(synthesizer): synthesize — single LLM call with provenance"
```

---

## Phase 7 — `scripts/run_synthesis.py` driver

### Task 7.0 — End-to-end synthesis driver

**File (Create):** `scripts/run_synthesis.py`

**Step 1: Failing test**

**File (Create):** `tests/unit/test_scripts_synthesis.py`
```python
from pathlib import Path
from unittest.mock import patch, MagicMock
import json, sys


def test_main_writes_method_json_to_stdout(tmp_path: Path, capsys, monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import run_synthesis as rs

    with patch.object(rs, "paraphrase_query", return_value={
            "essence": "e", "paraphrases": {d: "x" for d in [
                "LLM_NLP","MULTIMODAL","CV","RL","IR_REC","SPEECH","ROBOTICS"]}}), \
         patch.object(rs, "retrieve_cross_domain", return_value=[]), \
         patch.object(rs, "synthesize", return_value={"method": "M", "rationale": "R", "inspired_by": []}), \
         patch("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        rs.main(argv=["--query", "long-context inference",
                       "--library-root", str(tmp_path)])

    out = capsys.readouterr().out
    assert '"method": "M"' in out
```

**Step 2: Run, ImportError.**

**Step 3: Implement**
```python
# scripts/run_synthesis.py
"""End-to-end synthesis: query → paraphrase → retrieve → synthesize.

Usage:
    .venv/bin/python scripts/run_synthesis.py \
        --query "long-context efficient inference" \
        --library-root data/library
"""
import argparse
import json
import os
from pathlib import Path

from papergym.agents.base import PromptLoader
from papergym.library import LibraryStore
from papergym.llm import LLMClient
from papergym.retrieval.paraphrase import paraphrase_query
from papergym.retrieval.retrieve import retrieve_cross_domain
from papergym.synthesizer.synthesize import synthesize


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--library-root", required=True, type=Path)
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args(argv)

    llm = LLMClient(api_key=os.environ["OPENAI_API_KEY"])
    src_root = Path(__file__).resolve().parents[1] / "src" / "papergym"
    paraphrase_prompts = PromptLoader(src_root / "retrieval" / "prompts")
    synth_prompts = PromptLoader(src_root / "synthesizer" / "prompts")
    library = LibraryStore(args.library_root)

    para = paraphrase_query(args.query, llm=llm, prompts=paraphrase_prompts)
    pairs = retrieve_cross_domain(para["paraphrases"], store=library,
                                    llm=llm, k=args.k)
    seeds = [s for s, _domain_label in pairs]
    result = synthesize(args.query, seeds=seeds, llm=llm,
                         prompts=synth_prompts)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

**Step 4: Run, PASS.**

**Step 5: Commit**
```bash
git add scripts/run_synthesis.py tests/unit/test_scripts_synthesis.py
git commit -m "feat(scripts): run_synthesis.py end-to-end driver"
```

---

## Phase 8 — README + smoke run

### Task 8.0 — Update README

**File (Modify):** `README.md`

Replace the v2 content with v3-aligned content describing:
- Setup
- Three driver scripts (bootstrap_papers.py, run_accumulator.py, run_synthesis.py)
- Layout (`data/papers/<DOMAIN>/<arxiv_id>/...`, `data/library/{seeds.jsonl, faiss/}`)
- Pointer to design + impl docs

(Detailed wording at execution time. Aim for ~80 lines.)

**Commit:**
```bash
git add README.md
git commit -m "docs: rewrite README for v3 cross-domain synthesis"
```

### Task 8.1 — Mini smoke run (3 papers per domain, 1 query)

**Steps (manual):**
1. With OPENAI_API_KEY set:
   ```bash
   .venv/bin/python scripts/bootstrap_papers.py \
       --root data/papers --budget-per-domain 3 --only-domain LLM_NLP
   ```
2. Inspect `data/papers/LLM_NLP/` — expect 3 paper.md files.
3. Run accumulator on those 3:
   ```bash
   .venv/bin/python scripts/run_accumulator.py \
       --papers-root data/papers --library-root data/library \
       --max-papers 3 --only-domain LLM_NLP
   ```
4. Inspect `data/library/seeds.jsonl` — expect ~3-9 lines.
5. Run synthesis with a stub query:
   ```bash
   .venv/bin/python scripts/run_synthesis.py \
       --query "long-context efficient inference" --library-root data/library
   ```
6. Confirm stdout JSON has `method`, `rationale`, `inspired_by`.

If any step fails, file an issue / iterate. Don't add new tests for the smoke output — its purpose is end-to-end smoke, not regression coverage.

### Task 8.2 — Final test run

```bash
.venv/bin/python -m pytest tests/unit -q
```
Expected: every test from Phases 2–7 passes. Roughly 25–30 new tests.

**Commit any cleanup** (test fixture tweaks etc.):
```bash
git add -A
git commit -m "test: stabilize unit suite for v3"
```

---

## Out of scope for this plan

- Section 4 (workshop experiment design) — owned by user.
- Concurrency / parallel Accumulator runs.
- LRU eviction on the lazy-cloned `repo/` dirs.
- Cross-paper deduplication beyond the within-domain `paper_id` dedupe in retrieval.
- Telemetry / cost dashboard. `bootstrap_log.jsonl` and `accumulator_log.jsonl` are enough for now; richer analytics can be layered later.

---

## Notes on execution

- Some tasks reuse the existing `agents/base.PromptLoader`. Confirm at execution time it can take an arbitrary directory path; if not, generalize with a one-line constructor change in Phase 4.
- `tests/integration/` will be empty after Phase 1. That's fine. If a real Docker / Bash smoke is desired later, add a single test that runs `bash scripts/run_accumulator.py --max-papers 1` against a known small paper.
- v2's events.jsonl format (per-message lines) is preserved by reusing `agents/base` + `agents/tool_loop`. Good for free observability of Accumulator runs.
