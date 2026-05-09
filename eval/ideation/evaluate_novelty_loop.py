"""Novelty-guided iterative synthesis (appendix demonstration).

Wraps run_condition_c with a loop that uses the held-out novelty
judge's structured feedback (score, reasoning, surfaced prior works)
to guide re-synthesis until a novelty threshold is met or the round
budget is exhausted. The main pipeline (evaluate.py) is untouched;
this module is the optional extension demonstrated on a single
walkthrough query in the appendix.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from papergym.agents.base import PromptLoader
from papergym.library import LibraryStore, Seed
from papergym.llm import LLMClient

from ..common import parse_axis
from .evaluate import (
    IdeationOutput,
    NoveltyResult,
    _PROMPTS_DIR,
    _format_papers,
    _format_retrieved_seeds,
    _QUERY_RE,
    _s2_search,
    run_condition_c,
    run_condition_d,
)


_PRIOR_LINE_RE = re.compile(r"^\s*-\s+(.+?)\s*$", re.MULTILINE)
_PRIOR_BLOCK_RE = re.compile(
    r"^\s*Prior\s*works\s*:\s*(.*?)(?=^\s*Score\s*:|\Z)",
    re.MULTILINE | re.DOTALL | re.IGNORECASE,
)


@dataclass
class NoveltyResultWithPriors:
    """Same as NoveltyResult plus the structured prior-works listing the
    judge surfaces during finalize. Used to drive feedback into the next
    synthesis round."""

    score: int
    reasoning: str
    prior_works: list[str] = field(default_factory=list)


def _parse_prior_works(text: str) -> list[str]:
    block_match = _PRIOR_BLOCK_RE.search(text)
    if not block_match:
        return []
    block = block_match.group(1).strip()
    if not block or block.lower().startswith("(none)"):
        return []
    return [m.group(1).strip() for m in _PRIOR_LINE_RE.finditer(block)]


def _parse_novelty_with_priors_final(
    text: str,
) -> Optional[NoveltyResultWithPriors]:
    """Like evaluate._parse_novelty_final but also extracts the Prior
    works listing. Returns None on a search round so the caller continues
    the ReAct loop."""
    if _QUERY_RE.search(text):
        return None
    score_axis = parse_axis(text)
    if score_axis.score == 0:
        return None
    return NoveltyResultWithPriors(
        score=score_axis.score,
        reasoning=score_axis.reasoning,
        prior_works=_parse_prior_works(text),
    )


def judge_novelty_with_priors(
    *,
    judge_llm: LLMClient,
    query: str,
    output: IdeationOutput,
    max_rounds: int = 10,
    s2_api_key: Optional[str] = None,
) -> NoveltyResultWithPriors:
    """Mirrors evaluate.judge_novelty but uses the novelty_with_priors
    prompt so the finalize output includes a structured Prior works list
    the loop can feed back to the synthesizer."""
    s2_api_key = s2_api_key or os.environ.get("S2_API_KEY")
    prompts = PromptLoader(_PROMPTS_DIR)
    seeds_str = _format_retrieved_seeds(output.retrieved_seeds)

    rendered_static = prompts.render(
        "novelty_with_priors",
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
                "(Reasoning + Prior works + Score)."
            ),
        }
        messages = [sys_msg, *history, user_msg]
        text = judge_llm.chat(messages=messages, temperature=0.0)
        history.extend([user_msg, {"role": "assistant", "content": text}])

        m = _QUERY_RE.search(text)
        if m:
            s2_q = m.group(1).strip().strip('"').strip("'")
            last_results = _format_papers(
                _s2_search(s2_q, limit=10, api_key=s2_api_key))
            continue

        final = _parse_novelty_with_priors_final(text)
        if final is not None:
            return final
        break

    # Round budget exhausted — request a final score.
    final_user = {
        "role": "user",
        "content": (
            "Round budget exhausted. Provide your final Reasoning, "
            "Prior works, and Score now (no more searches)."
        ),
    }
    text = judge_llm.chat(
        messages=[sys_msg, *history, final_user], temperature=0.0)
    final = _parse_novelty_with_priors_final(text)
    if final is None:
        # Defensive default — score 0 sentinel handled by caller
        return NoveltyResultWithPriors(
            score=0, reasoning=text[:1000], prior_works=[])
    return final


@dataclass
class NoveltyLoopRound:
    round_num: int
    method: str
    rationale: str
    inspired_by: list[dict]
    novelty_score: int
    novelty_reasoning: str
    prior_works: list[str]
    feedback_used: Optional[str]


@dataclass
class NoveltyLoopOutput:
    final_method: str
    final_rationale: str
    final_inspired_by: list[dict]
    final_score: int
    rounds: list[NoveltyLoopRound]
    converged: bool
    threshold: int
    retrieved_seeds: list[Seed]
    paraphrases: dict
    paraphrase_essence: Optional[str]
    # token + cost accounting (per-llm)
    gen_tokens: dict
    judge_tokens: dict
    gen_cost_usd: float
    judge_cost_usd: float
    total_cost_usd: float


def _build_feedback(
    score: int, reasoning: str, prior_works: list[str],
) -> str:
    """Build the feedback message fed back to the synthesizer next round."""
    works_str = (
        "\n".join(f"  - {p}" for p in prior_works)
        if prior_works else "  (judge did not surface a specific match)"
    )
    return (
        f"Your previous synthesis received a novelty score of "
        f"{score}/5 from a held-out novelty judge that ran a "
        f"Semantic Scholar search.\n\n"
        f"The judge surfaced the following prior work that "
        f"substantively covers your proposed mechanisms:\n"
        f"{works_str}\n\n"
        f"Judge's reasoning: {reasoning}\n\n"
        f"Please synthesize a substantially more novel method for "
        f"the same research problem. You may use the same retrieved "
        f"seeds in non-obvious ways, or introduce new mechanism "
        f"families absent from the surfaced literature. The goal is "
        f"a method whose mechanism combination is not already "
        f"established."
    )


def _resynthesize_with_feedback(
    *,
    query: str,
    llm: LLMClient,
    seeds: list[Seed],
    lenses: list[str],
    feedback: str,
    temperature: float = 0.5,
) -> dict:
    """Renders the synthesizer prompt and appends the feedback as an
    extra user message before chatting. Returns the raw JSON dict
    (method/rationale/inspired_by)."""
    prompts = PromptLoader("papergym.agents.synthesizer")
    items = [{"seed": s, "lens": l} for s, l in zip(seeds, lenses)]
    messages = prompts.render("synthesize", query=query, items=items)
    messages.append({"role": "user", "content": feedback})
    raw = llm.chat(
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return json.loads(raw)


# Standard pricing (per 1M tokens). Override if generator/judge uses
# different model.
_GPT5_PRICING = {"prompt": 1.25, "completion": 10.0}
_SONNET46_PRICING = {"prompt": 3.0, "completion": 15.0}


def _estimate_cost(prompt_tok: int, completion_tok: int,
                    pricing: dict) -> float:
    return (prompt_tok * pricing["prompt"]
             + completion_tok * pricing["completion"]) / 1_000_000


def run_condition_c_with_novelty_loop(
    *,
    query: str,
    library: LibraryStore,
    gen_llm: LLMClient,
    judge_llm: LLMClient,
    natural_domain: Optional[str] = None,
    k_per_domain: int = 3,
    max_rounds: int = 10,
    novelty_threshold: int = 4,
    s2_api_key: Optional[str] = None,
    judge_max_rounds: int = 10,
    gen_pricing: dict = _GPT5_PRICING,
    judge_pricing: dict = _SONNET46_PRICING,
) -> NoveltyLoopOutput:
    """Run condition C, then iteratively re-synthesize until the novelty
    judge issues a score >= threshold or max_rounds is reached.

    Each round runs exactly one synthesis call (gen_llm) and one novelty
    judge call (judge_llm). The judge's structured prior_works listing is
    fed back as the feedback for the next round's synthesis. Round 1 has
    no feedback (standard run_condition_c)."""
    # Initial run reuses run_condition_c so retrieval + paraphrase logic
    # is shared with the main pipeline.
    initial = run_condition_c(
        query=query, library=library, llm=gen_llm,
        natural_domain=natural_domain, k_per_domain=k_per_domain,
    )

    seeds = list(initial.retrieved_seeds)
    # Empty lens reused across rounds: re-running cross-domain
    # retrieval just to reconstruct lens text is not worth it; the
    # lens is only a hint to the synthesizer.
    lenses = [""] * len(seeds)

    rounds: list[NoveltyLoopRound] = []
    current_method = initial.method
    current_rationale = initial.rationale
    current_inspired = initial.inspired_by
    feedback: Optional[str] = None
    converged = False

    for round_i in range(1, max_rounds + 1):
        if round_i > 1:
            assert feedback is not None
            synth_out = _resynthesize_with_feedback(
                query=query, llm=gen_llm,
                seeds=seeds, lenses=lenses,
                feedback=feedback,
            )
            current_method = synth_out.get("method", "")
            current_rationale = synth_out.get("rationale", "")
            current_inspired = synth_out.get("inspired_by") or []

        # Wrap current state in IdeationOutput so judge sees the same
        # context shape as the main pipeline.
        round_output = IdeationOutput(
            condition="C",
            method=current_method,
            rationale=current_rationale,
            inspired_by=current_inspired,
            retrieved_seed_ids=[s.seed_id for s in seeds],
            retrieved_seeds=seeds,
            paraphrase_essence=initial.paraphrase_essence,
            paraphrases=initial.paraphrases,
        )
        novelty = judge_novelty_with_priors(
            judge_llm=judge_llm, query=query, output=round_output,
            max_rounds=judge_max_rounds, s2_api_key=s2_api_key,
        )

        rounds.append(NoveltyLoopRound(
            round_num=round_i,
            method=current_method,
            rationale=current_rationale,
            inspired_by=current_inspired,
            novelty_score=novelty.score,
            novelty_reasoning=novelty.reasoning,
            prior_works=novelty.prior_works,
            feedback_used=feedback,
        ))

        if novelty.score >= novelty_threshold:
            converged = True
            break

        feedback = _build_feedback(
            novelty.score, novelty.reasoning, novelty.prior_works)

    gen_cost = _estimate_cost(
        gen_llm.total_prompt_tokens, gen_llm.total_completion_tokens,
        gen_pricing)
    judge_cost = _estimate_cost(
        judge_llm.total_prompt_tokens, judge_llm.total_completion_tokens,
        judge_pricing)

    return NoveltyLoopOutput(
        final_method=current_method,
        final_rationale=current_rationale,
        final_inspired_by=current_inspired,
        final_score=rounds[-1].novelty_score,
        rounds=rounds,
        converged=converged,
        threshold=novelty_threshold,
        retrieved_seeds=seeds,
        paraphrases=initial.paraphrases,
        paraphrase_essence=initial.paraphrase_essence,
        gen_tokens={"prompt": gen_llm.total_prompt_tokens,
                     "completion": gen_llm.total_completion_tokens},
        judge_tokens={"prompt": judge_llm.total_prompt_tokens,
                       "completion": judge_llm.total_completion_tokens},
        gen_cost_usd=round(gen_cost, 4),
        judge_cost_usd=round(judge_cost, 4),
        total_cost_usd=round(gen_cost + judge_cost, 4),
    )


def run_condition_d_with_novelty_loop(
    *,
    query: str,
    library: LibraryStore,
    gen_llm: LLMClient,
    judge_llm: LLMClient,
    total_seeds: int = 21,
    d_seed: int = 0,
    max_rounds: int = 10,
    novelty_threshold: int = 4,
    s2_api_key: Optional[str] = None,
    judge_max_rounds: int = 10,
    gen_pricing: dict = _GPT5_PRICING,
    judge_pricing: dict = _SONNET46_PRICING,
) -> NoveltyLoopOutput:
    """Mirror of run_condition_c_with_novelty_loop but starts from
    condition D (random-seed control). Random seeds are fixed across
    rounds; the lens "(uniform random sample of the library)" is
    preserved through re-synthesis so the synthesizer continues to
    know these seeds were not targeted."""
    initial = run_condition_d(
        query=query, library=library, llm=gen_llm,
        total_seeds=total_seeds, seed=d_seed,
    )

    seeds = list(initial.retrieved_seeds)
    # Preserve the random-sample disclaimer through every round so the
    # synthesizer does not mistakenly treat random seeds as targeted.
    lenses = ["(uniform random sample of the library)"] * len(seeds)

    rounds: list[NoveltyLoopRound] = []
    current_method = initial.method
    current_rationale = initial.rationale
    current_inspired = initial.inspired_by
    feedback: Optional[str] = None
    converged = False

    for round_i in range(1, max_rounds + 1):
        if round_i > 1:
            assert feedback is not None
            synth_out = _resynthesize_with_feedback(
                query=query, llm=gen_llm,
                seeds=seeds, lenses=lenses,
                feedback=feedback,
            )
            current_method = synth_out.get("method", "")
            current_rationale = synth_out.get("rationale", "")
            current_inspired = synth_out.get("inspired_by") or []

        round_output = IdeationOutput(
            condition="D",
            method=current_method,
            rationale=current_rationale,
            inspired_by=current_inspired,
            retrieved_seed_ids=[s.seed_id for s in seeds],
            retrieved_seeds=seeds,
            paraphrase_essence=None,
            paraphrases={},
        )
        novelty = judge_novelty_with_priors(
            judge_llm=judge_llm, query=query, output=round_output,
            max_rounds=judge_max_rounds, s2_api_key=s2_api_key,
        )

        rounds.append(NoveltyLoopRound(
            round_num=round_i,
            method=current_method,
            rationale=current_rationale,
            inspired_by=current_inspired,
            novelty_score=novelty.score,
            novelty_reasoning=novelty.reasoning,
            prior_works=novelty.prior_works,
            feedback_used=feedback,
        ))

        if novelty.score >= novelty_threshold:
            converged = True
            break

        feedback = _build_feedback(
            novelty.score, novelty.reasoning, novelty.prior_works)

    gen_cost = _estimate_cost(
        gen_llm.total_prompt_tokens, gen_llm.total_completion_tokens,
        gen_pricing)
    judge_cost = _estimate_cost(
        judge_llm.total_prompt_tokens, judge_llm.total_completion_tokens,
        judge_pricing)

    return NoveltyLoopOutput(
        final_method=current_method,
        final_rationale=current_rationale,
        final_inspired_by=current_inspired,
        final_score=rounds[-1].novelty_score,
        rounds=rounds,
        converged=converged,
        threshold=novelty_threshold,
        retrieved_seeds=seeds,
        paraphrases={},
        paraphrase_essence=None,
        gen_tokens={"prompt": gen_llm.total_prompt_tokens,
                     "completion": gen_llm.total_completion_tokens},
        judge_tokens={"prompt": judge_llm.total_prompt_tokens,
                       "completion": judge_llm.total_completion_tokens},
        gen_cost_usd=round(gen_cost, 4),
        judge_cost_usd=round(judge_cost, 4),
        total_cost_usd=round(gen_cost + judge_cost, 4),
    )
