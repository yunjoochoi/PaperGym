"""Ideate (A/B/C conditions) and judge each output.

All conditions return the same IdeationOutput so judges treat them
uniformly. B and C share the total seed budget so inputs are comparable
across conditions; A receives no seeds.
"""
from __future__ import annotations

import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from papergym.agents.base import BaseAgent, PromptLoader
from papergym.agents.paraphraser import Paraphraser
from papergym.agents.synthesizer import Synthesizer
from papergym.domain import Domain
from papergym.library import LibraryStore, Seed
from papergym.llm import LLMClient

from ..common import AxisScore, judge_axis, parse_axis


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass
class IdeationOutput:
    condition: str
    method: str
    rationale: str
    inspired_by: list[dict] = field(default_factory=list)
    retrieved_seed_ids: list[str] = field(default_factory=list)
    # Kept so the novelty judge can compare the proposal against its
    # own retrieved prior-work signal, not just S2 search results.
    retrieved_seeds: list[Seed] = field(default_factory=list)
    paraphrase_essence: Optional[str] = None
    paraphrases: dict = field(default_factory=dict)


@dataclass
class NoveltyResult:
    # score=0 is the parse-failure sentinel.
    score: int
    reasoning: str


def run_condition_a(*, query: str, llm: LLMClient) -> IdeationOutput:
    prompts = PromptLoader(_PROMPTS_DIR)
    agent = BaseAgent(llm=llm, prompts=prompts,
                       prompt_name="ideation_direct", temperature=0.5)
    out = agent.call(query=query)
    return IdeationOutput(
        condition="A",
        method=out.get("method", ""),
        rationale=out.get("rationale", ""),
        inspired_by=out.get("inspired_by") or [],
    )


def run_condition_b(*, query: str, natural_domain: Optional[str],
                     library: LibraryStore, llm: LLMClient,
                     total_seeds: int = 21) -> Optional[IdeationOutput]:
    """Same-domain retrieval; returns None when natural_domain is missing."""
    if not natural_domain:
        return None
    domain = Domain(natural_domain)
    emb = np.array(llm.embed(query), dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-9)
    seeds = library.retrieve(domain, emb, k=total_seeds)
    lenses = [query] * len(seeds)
    synth = Synthesizer(
        llm=llm,
        prompts=PromptLoader("papergym.agents.synthesizer"),
    )
    out = synth.run(query, seeds=seeds, lenses=lenses)
    return IdeationOutput(
        condition="B",
        method=out.get("method", ""),
        rationale=out.get("rationale", ""),
        inspired_by=out.get("inspired_by") or [],
        retrieved_seed_ids=[s.seed_id for s in seeds],
        retrieved_seeds=list(seeds),
    )


def run_condition_c(*, query: str, library: LibraryStore,
                     llm: LLMClient,
                     natural_domain: Optional[str] = None,
                     k_per_domain: int = 3) -> IdeationOutput:
    paraphraser = Paraphraser(
        llm=llm,
        prompts=PromptLoader("papergym.agents.paraphraser"),
    )
    para = paraphraser.run(query)
    paraphrases = para.get("paraphrases", {}) or {}
    triples = library.retrieve_cross_domain(
        paraphrases,
        raw_query=query, natural_domain=natural_domain,
        llm=llm, k=k_per_domain,
    )
    seeds = [s for s, _, _ in triples]
    lenses = [lens for _, _, lens in triples]
    synth = Synthesizer(
        llm=llm,
        prompts=PromptLoader("papergym.agents.synthesizer"),
    )
    out = synth.run(query, seeds=seeds, lenses=lenses)
    return IdeationOutput(
        condition="C",
        method=out.get("method", ""),
        rationale=out.get("rationale", ""),
        inspired_by=out.get("inspired_by") or [],
        retrieved_seed_ids=[s.seed_id for s in seeds],
        retrieved_seeds=list(seeds),
        paraphrase_essence=para.get("essence"),
        paraphrases=paraphrases,
    )


def run_condition_d(*, query: str, library: LibraryStore,
                     llm: LLMClient, total_seeds: int = 21,
                     seed: int = 0) -> IdeationOutput:
    """Negative control: synthesizer over seeds sampled uniformly at
    random from the entire library, with no embedding-based retrieval.
    Isolates the synthesizer's contribution from the retrieval signal:
    if condition C does not beat D, the embedding retrieval is not the
    source of any quality gain."""
    pool: list[Seed] = []
    for d_seeds in library._seeds_by_domain.values():
        pool.extend(d_seeds)
    rng = random.Random(seed)
    if len(pool) <= total_seeds:
        seeds = list(pool)
    else:
        seeds = rng.sample(pool, total_seeds)
    synth = Synthesizer(
        llm=llm,
        prompts=PromptLoader("papergym.agents.synthesizer"),
    )
    # Inform the synthesizer that seeds were sampled uniformly (no
    # targeted retrieval) so the C-vs-D contrast does not reduce to
    # a difference in retrieval-source disclosure.
    lenses = ["(uniform random sample of the library)"] * len(seeds)
    out = synth.run(query, seeds=seeds, lenses=lenses)
    return IdeationOutput(
        condition="D",
        method=out.get("method", ""),
        rationale=out.get("rationale", ""),
        inspired_by=out.get("inspired_by") or [],
        retrieved_seed_ids=[s.seed_id for s in seeds],
        retrieved_seeds=list(seeds),
    )


_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = "title,authors,venue,year,abstract,citationCount"
_QUERY_RE = re.compile(r"^\s*Query:\s*(.+?)\s*$", re.MULTILINE)


def _s2_search(query: str, *, limit: int = 10,
                api_key: Optional[str] = None) -> list[dict]:
    """Returns [] on error so the judge can still finalize."""
    headers = {"x-api-key": api_key} if api_key else {}
    params = {"query": query, "fields": _S2_FIELDS, "limit": limit}
    for attempt in range(3):
        try:
            r = requests.get(f"{_S2_BASE}/paper/search", params=params,
                              headers=headers, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt * 5)
                continue
            r.raise_for_status()
            return (r.json().get("data") or [])[:limit]
        except requests.RequestException:
            return []
    return []


def _format_papers(papers: list[dict]) -> str:
    if not papers:
        return "No papers found."
    lines = []
    for i, p in enumerate(papers, 1):
        abstract = p.get("abstract") or "<no abstract>"
        lines.append(
            f"{i}. {p.get('title','?')} "
            f"({p.get('venue','?')} {p.get('year','?')}, "
            f"cites: {p.get('citationCount', 0)})\n"
            f"   {abstract}"
        )
    return "\n\n".join(lines)


def _parse_novelty_final(text: str) -> Optional[NoveltyResult]:
    """Treats mixed-format output as a search round, not a finalize.
    Without this, the model can end the loop on round 1 by citing an
    anchor while also issuing a search query."""
    if _QUERY_RE.search(text):
        return None
    score_axis = parse_axis(text)
    if score_axis.score == 0:
        return None
    return NoveltyResult(
        score=score_axis.score,
        reasoning=score_axis.reasoning,
    )


def _format_retrieved_seeds(seeds: list[Seed]) -> str:
    if not seeds:
        return "(none — Condition A has no seeds)"
    lines = []
    for i, s in enumerate(seeds, 1):
        lines.append(
            f"{i}. [{s.domain.value if hasattr(s.domain,'value') else s.domain}] "
            f"{s.paper_title} ({s.paper_id})\n"
            f"   problem: {s.problem}\n"
            f"   method:  {s.method}"
        )
    return "\n\n".join(lines)


def judge_novelty(*, judge_llm: LLMClient, query: str,
                   output: IdeationOutput,
                   max_rounds: int = 10,
                   s2_api_key: Optional[str] = None) -> NoveltyResult:
    """ReAct novelty judge: each round either searches S2 or finalizes.
    The judge also sees the ideation's own retrieved seeds so it can
    detect a proposal that merely rehashes its inputs even when no
    published paper covers the exact recombination."""
    s2_api_key = s2_api_key or os.environ.get("S2_API_KEY")
    prompts = PromptLoader(_PROMPTS_DIR)
    seeds_str = _format_retrieved_seeds(output.retrieved_seeds)

    # Static context lives in the system message; per-round user
    # messages carry only the round counter and last search results,
    # keeping token cost linear over the loop instead of quadratic.
    rendered_static = prompts.render(
        "novelty",
        current_round=0, num_rounds=max_rounds,
        query=query, method=output.method, rationale=output.rationale,
        inspired_seeds=seeds_str,
        last_query_results="(unused in static render)",
    )
    sys_msg = next(m for m in rendered_static if m["role"] == "system")

    history: list[dict] = []
    last_results = "(empty on first round — issue your first search)"

    for round_i in range(max_rounds):
        user_msg = {
            "role": "user",
            "content": (
                f"Round {round_i + 1}/{max_rounds}.\n\n"
                f"Last search results:\n{last_results}\n\n"
                "Either issue a Search (Thought + Query) or Finalize "
                "(Reasoning + Score)."
            ),
        }
        messages = [sys_msg, *history, user_msg]
        text = judge_llm.chat(messages=messages, temperature=0.0)
        history.extend([user_msg, {"role": "assistant", "content": text}])

        # A Query line wins over a Score line in the same response so
        # the model can't shortcut to finalize on round 1 by citing an
        # anchor while also issuing a search.
        m = _QUERY_RE.search(text)
        if m:
            s2_q = m.group(1).strip().strip('"').strip("'")
            last_results = _format_papers(
                _s2_search(s2_q, limit=10, api_key=s2_api_key))
            continue

        final = _parse_novelty_final(text)
        if final is not None:
            return final
        break

    final_user = {
        "role": "user",
        "content": ("Maximum rounds reached. DO NOT issue any more "
                     "queries. Output FINAL now in this exact format:\n"
                     "Reasoning: <2-4 sentences>\n"
                     "Score: <integer 1-5>"),
    }
    messages = [sys_msg, *history, final_user]
    text = judge_llm.chat(messages=messages, temperature=0.0)
    final = _parse_novelty_final(text)
    if final is not None:
        return final
    print(f"[judge_novelty] forced-finalize unparseable | raw={text!r}",
          file=sys.stderr)
    return NoveltyResult(score=0,
                          reasoning=f"unparsable forced-finalize: {text}")


def judge_validity(*, judge_llm: LLMClient, query: str,
                    output: IdeationOutput) -> AxisScore:
    """1-5 Likert: does the method's mechanism engage the query's
    failure mode in a traceable way?"""
    prompts = PromptLoader(_PROMPTS_DIR)
    return judge_axis(
        llm=judge_llm, prompts=prompts, prompt_name="validity",
        query=query, method=output.method, rationale=output.rationale,
    )


def judge_coherence(*, judge_llm: LLMClient, query: str,
                     output: IdeationOutput) -> AxisScore:
    """1-5 rating: does the method hold together as a technically
    valid, implementable ML approach (single-pass mirror of the
    pairwise coherence rubric)?"""
    prompts = PromptLoader(_PROMPTS_DIR)
    return judge_axis(
        llm=judge_llm, prompts=prompts, prompt_name="coherence",
        query=query, method=output.method, rationale=output.rationale,
    )


@dataclass
class PairwiseResult:
    # winner ∈ {"A", "B", "tie"} where A and B refer to the order seen
    # by the judge AFTER position randomization. The caller maps these
    # back to original conditions through condition_for_a /
    # condition_for_b.
    winner: str
    condition_for_a: str
    condition_for_b: str
    reasoning: str


_WINNER_RE = re.compile(
    r"^\s*Winner\s*[:：]\s*(A|B|tie)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def judge_pairwise(*, judge_llm: LLMClient, query: str,
                    method_x: str, method_y: str,
                    condition_x: str, condition_y: str,
                    axis: str, rng: random.Random) -> PairwiseResult:
    """Compare two methods on axis ∈ {"novelty", "validity",
    "coherence"} with randomized A/B positions to guard against
    position bias."""
    if axis not in ("novelty", "validity", "coherence"):
        raise ValueError(f"unknown axis {axis!r}")
    swap = rng.random() < 0.5
    method_a = method_y if swap else method_x
    method_b = method_x if swap else method_y
    cond_a = condition_y if swap else condition_x
    cond_b = condition_x if swap else condition_y

    prompts = PromptLoader(_PROMPTS_DIR)
    messages = prompts.render(
        f"pairwise_{axis}",
        query=query, method_a=method_a, method_b=method_b,
    )
    raw = judge_llm.chat(messages, temperature=0.0)
    m = _WINNER_RE.search(raw)
    winner = m.group(1).strip().lower() if m else None
    if winner == "a":
        winner_label = "A"
    elif winner == "b":
        winner_label = "B"
    elif winner == "tie":
        winner_label = "tie"
    else:
        print(f"[judge_pairwise] no Winner line | raw={raw!r}",
              file=sys.stderr)
        winner_label = "tie"
    reasoning = raw.split("Winner", 1)[0].strip()
    if reasoning.lower().startswith("reasoning:"):
        reasoning = reasoning[len("reasoning:"):].strip()
    return PairwiseResult(
        winner=winner_label,
        condition_for_a=cond_a, condition_for_b=cond_b,
        reasoning=reasoning,
    )


def judge_method_specificity(*, judge_llm: LLMClient, query: str,
                               output: IdeationOutput) -> AxisScore:
    """1-5 Likert specificity score for an ideation method (analogous
    to Stage 1 specificity but with cutoffs adjusted for longer
    ideation method texts)."""
    prompts = PromptLoader(_PROMPTS_DIR)
    return judge_axis(
        llm=judge_llm, prompts=prompts, prompt_name="method_specificity",
        query=query, method=output.method, rationale=output.rationale,
    )


def judge_inspired_by_grounding(*, judge_llm: LLMClient, method: str,
                                  seed_id: str, seed_domain: str,
                                  borrowed_aspect: str) -> AxisScore:
    """1-3 score (1=no, 2=partial, 3=full) on whether the synthesized
    method actually incorporates the cited seed's borrowed_aspect.
    Uses a held-out judge model (different family from the synthesizer)
    to mitigate self-bias.
    """
    prompts = PromptLoader(_PROMPTS_DIR)
    return judge_axis(
        llm=judge_llm, prompts=prompts, prompt_name="inspired_by_grounding",
        method=method, seed_id=seed_id, seed_domain=seed_domain,
        borrowed_aspect=borrowed_aspect,
    )
