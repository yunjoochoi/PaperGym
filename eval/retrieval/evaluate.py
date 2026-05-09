"""Stage 2: evaluate retrieval quality with paraphrase ON vs OFF.

For each query we run two retrievals over the same total seed budget:
  - paraphrase_on : Paraphraser produces 7 domain-specific reframings;
                    for each domain we embed either the paraphrase or,
                    when the domain matches the query's natural domain,
                    the raw query itself, then take the global top-k
                    across the entire library (no per-domain partition).
  - paraphrase_off: the raw query is embedded once and used to fetch
                    the global top-k_total from the entire library.

Each retrieved seed receives TWO relevance scores:
  - naive       : judge sees only the raw query and the seed.
  - lens_aware  : judge additionally sees the lens text (paraphrase or
                  raw query) that retrieved the seed, so the
                  paraphrase's mediator role is not lost between
                  retrieval and evaluation.

For OFF seeds the lens is the raw query, so naive and lens_aware are
expected to coincide; ON seeds drawn through a paraphrase lens are
where the two judges can diverge, and that divergence is itself the
quantity of interest.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from papergym.agents.base import PromptLoader
from papergym.agents.paraphraser import Paraphraser
from papergym.domain import Domain
from papergym.library import LibraryStore, Seed
from papergym.llm import LLMClient

from ..common import AxisScore, judge_axis


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class SeedScore:
    seed_id: str
    paper_id: str
    domain: str
    lens: str
    relevance: AxisScore
    relevance_lens: AxisScore


@dataclass
class RetrievalRun:
    seeds: list[SeedScore]


@dataclass
class QueryEvaluation:
    query_id: str
    query_text: str
    natural_domain: Optional[str]
    paraphrase_essence: Optional[str]
    paraphrases: dict[str, Optional[str]]
    on: RetrievalRun
    off: RetrievalRun

    def to_dict(self) -> dict:
        return asdict(self)


def _retrieve_off(library: LibraryStore, llm: LLMClient,
                   query_text: str,
                   k_total: int) -> list[tuple[Seed, str, str]]:
    """Embed the raw query once and fetch the global top-k_total across
    the entire library by cosine similarity. No domain partitioning.
    Returns (seed, origin_domain, lens_text); lens_text is always the
    raw query for OFF, so a lens-aware judge yields the same score as
    the naive judge for OFF seeds (a useful consistency check)."""
    emb = np.array(llm.embed(query_text), dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)

    all_seeds: list[Seed] = []
    all_vecs: list[np.ndarray] = []
    for d in Domain:
        seeds = library._seeds_by_domain.get(d, [])
        idx = library._faiss.get(d)
        if not seeds or idx is None or idx.ntotal == 0:
            continue
        all_seeds.extend(seeds)
        all_vecs.append(idx.reconstruct_n(0, idx.ntotal))
    if not all_seeds:
        return []

    merged = np.vstack(all_vecs)
    scores = merged @ emb
    over_k = min(k_total * 5, len(all_seeds))
    top = np.argsort(-scores)[:over_k]

    out: list[tuple[Seed, str, str]] = []
    seen_papers: set[str] = set()
    for i in top:
        s = all_seeds[int(i)]
        if s.paper_id in seen_papers:
            continue
        seen_papers.add(s.paper_id)
        dom = s.domain.value if hasattr(s.domain, "value") else s.domain
        out.append((s, dom, query_text))
        if len(out) == k_total:
            break
    return out


def _judge_relevance(judge_llm: LLMClient,
                      query_text: str, seed: Seed) -> AxisScore:
    # Domain label is intentionally NOT passed so the rating reflects
    # mechanism transfer alone, not the judge's prior about same- vs
    # cross-domain transfer difficulty.
    prompts = PromptLoader(_PROMPTS_DIR)
    return judge_axis(
        llm=judge_llm, prompts=prompts, prompt_name="relevance",
        query=query_text,
        problem=seed.problem,
        method=seed.method,
    )


def _judge_relevance_lens_aware(
    judge_llm: LLMClient, query_text: str, seed: Seed, lens: str,
) -> AxisScore:
    prompts = PromptLoader(_PROMPTS_DIR)
    return judge_axis(
        llm=judge_llm, prompts=prompts, prompt_name="relevance_lens_aware",
        query=query_text, lens=lens,
        problem=seed.problem,
        method=seed.method,
    )


def evaluate_query(*, query: dict,
                    library: LibraryStore,
                    llm: LLMClient,
                    judge_llm: LLMClient,
                    k: int = 3) -> QueryEvaluation:
    paraphraser = Paraphraser(
        llm=llm, prompts=PromptLoader("papergym.agents.paraphraser"),
    )
    para_result = paraphraser.run(query["text"])
    paraphrases = para_result.get("paraphrases", {}) or {}

    seeds_on = library.retrieve_cross_domain(
        paraphrases,
        raw_query=query["text"],
        natural_domain=query.get("natural_domain"),
        llm=llm, k=k,
    )
    seeds_off = _retrieve_off(library, llm, query["text"],
                                k_total=k * len(list(Domain)))

    def _score_seeds(items: list[tuple[Seed, str, str]]) -> list[SeedScore]:
        scored = []
        for seed, dom, lens in items:
            r_naive = _judge_relevance(judge_llm, query["text"], seed)
            # When the lens equals the raw query (all OFF seeds, plus the
            # natural_domain slot of ON), the lens-aware judge would see
            # identical input as the naive one — reuse the score to
            # avoid a redundant call.
            if lens == query["text"]:
                r_lens = r_naive
            else:
                r_lens = _judge_relevance_lens_aware(
                    judge_llm, query["text"], seed, lens)
            scored.append(SeedScore(
                seed_id=seed.seed_id, paper_id=seed.paper_id,
                domain=dom, lens=lens,
                relevance=r_naive, relevance_lens=r_lens,
            ))
        return scored

    return QueryEvaluation(
        query_id=query["id"],
        query_text=query["text"],
        natural_domain=query.get("natural_domain"),
        paraphrase_essence=para_result.get("essence"),
        paraphrases=paraphrases,
        on=RetrievalRun(seeds=_score_seeds(seeds_on)),
        off=RetrievalRun(seeds=_score_seeds(seeds_off)),
    )
