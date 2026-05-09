import fcntl
import json
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import faiss
import numpy as np

from ..domain import Domain
from .seed import Seed

if TYPE_CHECKING:
    from ..llm import LLMClient

EMBEDDING_DIM = 1536  # text-embedding-3-small


class LibraryStore:
    """Domain-partitioned seed library; jsonl metadata + per-domain FAISS IndexFlatIP."""

    def __init__(self, root_dir: Path, embedding_dim: int = EMBEDDING_DIM):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "faiss").mkdir(exist_ok=True)
        self._dim = embedding_dim

        self._seeds_path = self.root / "seeds.jsonl"
        self._lock_path = self.root / ".lock"
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
        jsonl_seeds: dict[Domain, list[Seed]] = {d: [] for d in Domain}
        if self._seeds_path.exists():
            for line in self._seeds_path.read_text().splitlines():
                if not line.strip():
                    continue
                s = Seed.from_dict(json.loads(line))
                jsonl_seeds[s.domain].append(s)

        # Reconcile a partial-write crash in add() by truncating whichever
        # side (FAISS / jsonl) ran past the other to the shorter length.
        per_domain_truncate: dict[Domain, int] = {}
        for d in Domain:
            idx_path = self._index_path(d)
            if not idx_path.exists():
                self._seeds_by_domain[d] = jsonl_seeds[d]
                continue
            idx = faiss.read_index(str(idx_path))
            n_seeds = len(jsonl_seeds[d])

            if idx.ntotal > n_seeds:
                vecs = idx.reconstruct_n(0, n_seeds) if n_seeds > 0 else None
                new_idx = faiss.IndexFlatIP(self._dim)
                if vecs is not None and n_seeds > 0:
                    new_idx.add(vecs)
                faiss.write_index(new_idx, str(idx_path))
                idx = new_idx
            elif n_seeds > idx.ntotal:
                # Defer the jsonl rewrite until all domains are reconciled
                # so one pass covers every domain at once.
                jsonl_seeds[d] = jsonl_seeds[d][:idx.ntotal]
                per_domain_truncate[d] = idx.ntotal

            self._seeds_by_domain[d] = jsonl_seeds[d]
            self._faiss[d] = idx

        for d in Domain:
            if d not in self._faiss:
                self._seeds_by_domain[d] = jsonl_seeds[d]

        if per_domain_truncate:
            self._rewrite_seeds_jsonl()

    def _rewrite_seeds_jsonl(self) -> None:
        tmp = self._seeds_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for d in Domain:
                for s in self._seeds_by_domain[d]:
                    f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
        os.replace(tmp, self._seeds_path)

    def _reload_from_disk(self) -> None:
        self._faiss = {}
        self._seeds_by_domain = {d: [] for d in Domain}
        self._load()

    def add(self, seed: Seed, embedding: np.ndarray) -> None:
        if getattr(self, "_read_only", False):
            raise RuntimeError(
                "LibraryStore opened via open_merged is read-only; "
                "add() would skip the underlying shards.")
        if embedding.shape != (self._dim,):
            raise ValueError(f"expected ({self._dim},), got {embedding.shape}")
        emb = embedding.astype(np.float32, copy=False).reshape(1, -1)

        # Advisory lock against concurrent add() in the same shard.
        # Reload under the lock so a long-running worker cannot overwrite
        # another worker's more recent FAISS index with stale state.
        with self._lock_path.open("a+") as lock_fp:
            fcntl.flock(lock_fp, fcntl.LOCK_EX)
            try:
                self._reload_from_disk()
                idx = self._index(seed.domain)
                idx.add(emb)
                idx_path = self._index_path(seed.domain)
                tmp_path = idx_path.with_suffix(".index.tmp")
                faiss.write_index(idx, str(tmp_path))
                os.replace(tmp_path, idx_path)
                self._seeds_by_domain[seed.domain].append(seed)
                with self._seeds_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(seed.to_dict(), ensure_ascii=False) + "\n")
            finally:
                fcntl.flock(lock_fp, fcntl.LOCK_UN)

    @classmethod
    def open_merged(cls, root: Path,
                    embedding_dim: int = EMBEDDING_DIM) -> "LibraryStore":
        """Open a (possibly sharded) library as a single read-only view.

        If <root>/shard_*/ subdirs exist, load all of them into one in-memory
        store. Otherwise behave like the regular constructor.
        """
        root = Path(root)
        shard_dirs = sorted(p for p in root.glob("shard_*") if p.is_dir())
        if not shard_dirs:
            return cls(root, embedding_dim=embedding_dim)
        merged = cls.__new__(cls)
        merged.root = root
        merged._dim = embedding_dim
        merged._seeds_path = root / "seeds.jsonl"   # unused on merged view
        merged._lock_path = root / ".lock"
        merged._seeds_by_domain = {d: [] for d in Domain}
        merged._faiss = {}
        for shard in shard_dirs:
            sub = cls(shard, embedding_dim=embedding_dim)
            for d in Domain:
                sub_seeds = sub._seeds_by_domain[d]
                sub_idx = sub._faiss.get(d)
                # Skip the domain on this shard when seeds and FAISS are
                # out of sync; mismatched lengths would break the merged
                # store's position-based seed lookup.
                if sub_idx is None and sub_seeds:
                    continue
                if sub_idx is not None and sub_idx.ntotal != len(sub_seeds):
                    continue
                merged._seeds_by_domain[d].extend(sub_seeds)
                if sub_idx is not None and sub_idx.ntotal > 0:
                    if d not in merged._faiss:
                        merged._faiss[d] = faiss.IndexFlatIP(embedding_dim)
                    vecs = sub_idx.reconstruct_n(0, sub_idx.ntotal)
                    merged._faiss[d].add(vecs)
        merged._read_only = True
        return merged

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

    def retrieve_cross_domain(
        self, paraphrases: dict[str, Optional[str]],
        *, raw_query: str, natural_domain: Optional[str],
        llm: "LLMClient", k: int = 3,
    ) -> list[tuple[Seed, str, str]]:
        """For each domain, embed either the paraphrase for that domain
        or, when the domain matches natural_domain, the raw query
        instead. Each embedding then retrieves the GLOBAL top-k across
        the entire library (no per-domain partitioning), so the
        retrieved seed may live in any domain. Each tuple is
        (seed, origin_domain, lens_text) where origin_domain is the
        slot whose text was embedded and lens_text is the actual text
        embedded; downstream consumers pass lens_text to a synthesizer
        or a lens-aware judge so the paraphrase frame is not lost
        between retrieval and use."""
        all_seeds: list[Seed] = []
        all_vecs: list[np.ndarray] = []
        for d in Domain:
            seeds = self._seeds_by_domain.get(d, [])
            idx = self._faiss.get(d)
            if not seeds or idx is None or idx.ntotal == 0:
                continue
            all_seeds.extend(seeds)
            all_vecs.append(idx.reconstruct_n(0, idx.ntotal))
        if not all_seeds:
            return []
        merged = np.vstack(all_vecs)

        out: list[tuple[Seed, str, str]] = []
        for d in Domain:
            text = (raw_query if natural_domain and d.value == natural_domain
                    else paraphrases.get(d.value))
            if not text:
                continue
            emb = np.array(llm.embed(text), dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            scores = merged @ emb
            over_k = min(k * 5, len(all_seeds))
            top = np.argsort(-scores)[:over_k]
            picked: list[Seed] = []
            seen_papers: set[str] = set()
            for i in top:
                s = all_seeds[int(i)]
                if s.paper_id in seen_papers:
                    continue
                seen_papers.add(s.paper_id)
                picked.append(s)
                if len(picked) == k:
                    break
            for s in picked:
                out.append((s, d.value, text))
        return out
