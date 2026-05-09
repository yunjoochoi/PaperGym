"""Novelty-loop ablation benchmark across all 30 problems.

For each query: runs the novelty-guided loop on conditions C
(cross-domain retrieval) and D (random-seed control), then judges
the two loop final methods pairwise on novelty, validity, and
coherence using the same rubrics as the main benchmark
(Section 3.4). Aggregates win rates per axis + cost/rounds.

Saves per-query JSON incrementally so mid-run crashes do not lose
progress; final aggregate.json is written when all queries
complete.

uv run python scripts/loop_benchmark.py
"""
from __future__ import annotations

import json
import random
import statistics
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.ideation.evaluate import judge_pairwise  # noqa: E402
from eval.ideation.evaluate_novelty_loop import (  # noqa: E402
    run_condition_c_with_novelty_loop,
    run_condition_d_with_novelty_loop,
)

from papergym.library import LibraryStore  # noqa: E402
from papergym.llm import LLMClient  # noqa: E402

load_dotenv(override=True)

# Config — tunable.
NOVELTY_THRESHOLD = 4
MAX_ROUNDS = 10
JUDGE_MAX_ROUNDS = 10
PAIRWISE_RNG_SEED = 42


# Per-query event-trace logging via monkey-patched LLMClient.chat.
# When _event_fp is set to an open jsonl file, each chat call appends
# a record (timestamp, role, model, messages, response, elapsed_s,
# usage). Set to None to disable logging temporarily.
_event_fp = None
_orig_chat = LLMClient.chat


def _logged_chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
    t0 = time.time()
    response = _orig_chat(self, messages, **kwargs)
    t1 = time.time()
    if _event_fp is not None:
        try:
            _event_fp.write(json.dumps({
                "ts": t0,
                "elapsed_s": round(t1 - t0, 2),
                "model": self.model,
                "messages": messages,
                "response": response,
                "kwargs": kwargs,
            }, ensure_ascii=False, default=str) + "\n")
            _event_fp.flush()
        except Exception:
            # Logging must never break the benchmark.
            pass
    return response


LLMClient.chat = _logged_chat  # type: ignore[assignment]

queries = yaml.safe_load(Path("data/queries.yaml").read_text())["queries"]
library = LibraryStore.open_merged(Path("data/library"))

run_ts = int(time.time())
out_dir = (_REPO_ROOT / "data" / "eval"
            / f"loop_benchmark_{time.strftime('%Y%m%dT%H%M%S')}")
out_dir.mkdir(parents=True, exist_ok=True)

# Single shared LLM clients so total token / cost numbers accumulate
# across all queries.
gen_llm = LLMClient()
judge_llm = LLMClient(model="openai/bedrock.anthropic.claude-sonnet-4-6")
pair_rng = random.Random(PAIRWISE_RNG_SEED)

print(f"Loop benchmark — n_queries={len(queries)}, threshold={NOVELTY_THRESHOLD}, "
       f"max_rounds={MAX_ROUNDS}")
print(f"  generator: {gen_llm.model}")
print(f"  judge:     {judge_llm.model}")
print(f"  output:    {out_dir}")
t_start = time.time()


# Standard pricing per 1M tokens (must match evaluate_novelty_loop._GPT5_PRICING /
# _SONNET46_PRICING). Re-declared here so we can compute per-call deltas against
# shared LLMClient counters that accumulate across all queries.
_GEN_PRICING = {"prompt": 1.25, "completion": 10.0}
_JUDGE_PRICING = {"prompt": 3.0, "completion": 15.0}


def _snap() -> tuple[int, int, int, int]:
    """Snapshot of cumulative LLM token counters at this moment."""
    return (gen_llm.total_prompt_tokens, gen_llm.total_completion_tokens,
            judge_llm.total_prompt_tokens, judge_llm.total_completion_tokens)


def _delta_cost(s0: tuple[int, int, int, int],
                s1: tuple[int, int, int, int]) -> dict:
    """Per-call delta tokens + estimated cost between two snapshots."""
    dgp, dgc, djp, djc = (s1[i] - s0[i] for i in range(4))
    gen_cost = (dgp * _GEN_PRICING["prompt"]
                 + dgc * _GEN_PRICING["completion"]) / 1_000_000
    judge_cost = (djp * _JUDGE_PRICING["prompt"]
                   + djc * _JUDGE_PRICING["completion"]) / 1_000_000
    return {
        "gen_prompt_tokens": dgp,
        "gen_completion_tokens": dgc,
        "judge_prompt_tokens": djp,
        "judge_completion_tokens": djc,
        "gen_cost_usd": round(gen_cost, 4),
        "judge_cost_usd": round(judge_cost, 4),
        "total_cost_usd": round(gen_cost + judge_cost, 4),
    }


per_query: list[dict] = []
for i, q in enumerate(queries, 1):
    qid = q["id"]
    print(f"\n=== [{i}/{len(queries)}] {qid} (domain={q['natural_domain']}) ===")
    rec: dict = {
        "query_id": qid,
        "natural_domain": q.get("natural_domain"),
        "query_text": q["text"],
    }

    # Open per-query event-trace log; closed in finally below so
    # crashes mid-query still flush whatever events were collected.
    event_path = out_dir / f"{qid}.events.loop.benchmark.jsonl"
    _event_fp = open(event_path, "w", encoding="utf-8")  # noqa: F841

    try:
        s_start = _snap()
        t0 = time.time()
        c_out = run_condition_c_with_novelty_loop(
            query=q["text"], library=library,
            gen_llm=gen_llm, judge_llm=judge_llm,
            natural_domain=q.get("natural_domain"),
            k_per_domain=3,
            max_rounds=MAX_ROUNDS,
            novelty_threshold=NOVELTY_THRESHOLD,
            judge_max_rounds=JUDGE_MAX_ROUNDS,
        )
        c_dt = time.time() - t0
        s_after_c = _snap()
        c_cost = _delta_cost(s_start, s_after_c)
        print(f"  C-loop: rounds={len(c_out.rounds)} "
               f"converged={c_out.converged} score={c_out.final_score}/5 "
               f"cost=${c_cost['total_cost_usd']:.3f} t={c_dt:.0f}s")

        t0 = time.time()
        d_out = run_condition_d_with_novelty_loop(
            query=q["text"], library=library,
            gen_llm=gen_llm, judge_llm=judge_llm,
            total_seeds=21, d_seed=0,
            max_rounds=MAX_ROUNDS,
            novelty_threshold=NOVELTY_THRESHOLD,
            judge_max_rounds=JUDGE_MAX_ROUNDS,
        )
        d_dt = time.time() - t0
        s_after_d = _snap()
        d_cost = _delta_cost(s_after_c, s_after_d)
        print(f"  D-loop: rounds={len(d_out.rounds)} "
               f"converged={d_out.converged} score={d_out.final_score}/5 "
               f"cost=${d_cost['total_cost_usd']:.3f} t={d_dt:.0f}s")

        pair: dict = {}
        for axis in ("novelty", "validity", "coherence"):
            res = judge_pairwise(
                judge_llm=judge_llm, query=q["text"],
                method_x=c_out.final_method, method_y=d_out.final_method,
                condition_x="C", condition_y="D",
                axis=axis, rng=pair_rng,
            )
            if res.winner == "tie":
                verdict = "tie"
            elif res.winner == "A":
                verdict = res.condition_for_a
            else:
                verdict = res.condition_for_b
            pair[axis] = {"verdict": verdict, **asdict(res)}
            print(f"    pairwise {axis}: {verdict}")
        s_after_pair = _snap()
        pair_cost = _delta_cost(s_after_d, s_after_pair)

        rec.update({
            "c_loop": {
                "rounds": len(c_out.rounds),
                "converged": c_out.converged,
                "final_score": c_out.final_score,
                "round_scores": [r.novelty_score for r in c_out.rounds],
                "round_records": [asdict(r) for r in c_out.rounds],
                "final_method": c_out.final_method,
                "final_rationale": c_out.final_rationale,
                "final_inspired_by": c_out.final_inspired_by,
                "retrieved_seeds": [s.to_dict()
                                     for s in c_out.retrieved_seeds],
                "paraphrase_essence": c_out.paraphrase_essence,
                "paraphrases": c_out.paraphrases,
                "wall_clock_s": round(c_dt, 2),
                "cost": c_cost,
            },
            "d_loop": {
                "rounds": len(d_out.rounds),
                "converged": d_out.converged,
                "final_score": d_out.final_score,
                "round_scores": [r.novelty_score for r in d_out.rounds],
                "round_records": [asdict(r) for r in d_out.rounds],
                "final_method": d_out.final_method,
                "final_rationale": d_out.final_rationale,
                "final_inspired_by": d_out.final_inspired_by,
                "retrieved_seeds": [s.to_dict()
                                     for s in d_out.retrieved_seeds],
                "wall_clock_s": round(d_dt, 2),
                "cost": d_cost,
            },
            "pairwise": pair,
            "pairwise_cost": pair_cost,
        })

    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
        print(f"  ERROR on {qid}: {e}")
    finally:
        # Close per-query event log and clear the global pointer so
        # any later (out-of-loop) chat calls do not write into a
        # closed file.
        try:
            _event_fp.close()
        except Exception:
            pass
        _event_fp = None

    per_query.append(rec)
    (out_dir / f"{qid}.json").write_text(
        json.dumps(rec, ensure_ascii=False, indent=2))


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _pair_stats(axis: str) -> dict:
    counts = {"C": 0, "D": 0, "tie": 0}
    for r in per_query:
        if "pairwise" not in r:
            continue
        v = r["pairwise"][axis]["verdict"]
        counts[v] = counts.get(v, 0) + 1
    n = sum(counts.values())
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "C_wins": counts["C"], "D_wins": counts["D"], "ties": counts["tie"],
        "C_win_rate": round(counts["C"] / n, 3),
        "D_win_rate": round(counts["D"] / n, 3),
        "tie_rate": round(counts["tie"] / n, 3),
    }


valid = [r for r in per_query if "error" not in r]
n = len(valid)


def _round_score_dist(cond: str) -> list[dict]:
    """Per-round mean score across queries that reached each round position.
    Returns a list aligned to round_num=1..max."""
    if not valid:
        return []
    max_r = max(r[f"{cond}_loop"]["rounds"] for r in valid)
    dist = []
    for idx in range(max_r):
        scores = [
            r[f"{cond}_loop"]["round_scores"][idx]
            for r in valid
            if idx < len(r[f"{cond}_loop"]["round_scores"])
        ]
        dist.append({
            "round": idx + 1,
            "n_reached": len(scores),
            "mean_score": round(_avg(scores), 2) if scores else None,
            "median_score": (statistics.median(scores)
                              if scores else None),
        })
    return dist


def _converge_round_stats(cond: str) -> dict:
    """Histogram + summary of rounds-to-converge for queries that converged."""
    converged_at = [
        r[f"{cond}_loop"]["rounds"]
        for r in valid
        if r[f"{cond}_loop"]["converged"]
    ]
    hist: dict = {}
    for v in converged_at:
        hist[v] = hist.get(v, 0) + 1
    return {
        "n_converged": len(converged_at),
        "histogram": dict(sorted(hist.items())),
        "mean": round(_avg(converged_at), 2) if converged_at else None,
        "median": (statistics.median(converged_at)
                    if converged_at else None),
        "stdev": (round(statistics.stdev(converged_at), 2)
                   if len(converged_at) > 1 else 0),
    }


def _domain_coverage_stats(cond: str) -> dict:
    """Distinct domain count in final inspired_by per query."""
    coverages = []
    for r in valid:
        domains = {
            e.get("domain")
            for e in r[f"{cond}_loop"]["final_inspired_by"]
            if e.get("domain")
        }
        coverages.append(len(domains))
    if not coverages:
        return {}
    return {
        "mean": round(_avg(coverages), 2),
        "median": statistics.median(coverages),
        "stdev": (round(statistics.stdev(coverages), 2)
                   if len(coverages) > 1 else 0),
        "min": min(coverages),
        "max": max(coverages),
    }


def _wall_clock_stats(cond: str) -> dict:
    times = [r[f"{cond}_loop"]["wall_clock_s"] for r in valid]
    if not times:
        return {}
    return {
        "mean_s": round(_avg(times), 1),
        "median_s": round(statistics.median(times), 1),
        "stdev_s": (round(statistics.stdev(times), 1)
                     if len(times) > 1 else 0),
        "min_s": round(min(times), 1),
        "max_s": round(max(times), 1),
    }

aggregate = {
    "n_problems": len(per_query),
    "n_successful": n,
    "novelty_threshold": NOVELTY_THRESHOLD,
    "max_rounds": MAX_ROUNDS,
    "judge_max_rounds": JUDGE_MAX_ROUNDS,
    "c_mean_rounds": round(_avg([r["c_loop"]["rounds"] for r in valid]), 2),
    "d_mean_rounds": round(_avg([r["d_loop"]["rounds"] for r in valid]), 2),
    "c_convergence_rate": round(
        sum(r["c_loop"]["converged"] for r in valid) / max(n, 1), 3),
    "d_convergence_rate": round(
        sum(r["d_loop"]["converged"] for r in valid) / max(n, 1), 3),
    "c_mean_final_score": round(
        _avg([r["c_loop"]["final_score"] for r in valid]), 2),
    "d_mean_final_score": round(
        _avg([r["d_loop"]["final_score"] for r in valid]), 2),
    "c_mean_cost_usd": round(
        _avg([r["c_loop"]["cost"]["total_cost_usd"] for r in valid]), 4),
    "d_mean_cost_usd": round(
        _avg([r["d_loop"]["cost"]["total_cost_usd"] for r in valid]), 4),
    "c_total_cost_usd": round(
        sum(r["c_loop"]["cost"]["total_cost_usd"] for r in valid), 2),
    "d_total_cost_usd": round(
        sum(r["d_loop"]["cost"]["total_cost_usd"] for r in valid), 2),
    "pairwise_total_cost_usd": round(
        sum(r["pairwise_cost"]["total_cost_usd"]
            for r in valid if "pairwise_cost" in r), 2),
    "c_round_score_dist": _round_score_dist("c"),
    "d_round_score_dist": _round_score_dist("d"),
    "c_convergence_round": _converge_round_stats("c"),
    "d_convergence_round": _converge_round_stats("d"),
    "c_domain_coverage": _domain_coverage_stats("c"),
    "d_domain_coverage": _domain_coverage_stats("d"),
    "c_wall_clock": _wall_clock_stats("c"),
    "d_wall_clock": _wall_clock_stats("d"),
    "pairwise_novelty": _pair_stats("novelty"),
    "pairwise_validity": _pair_stats("validity"),
    "pairwise_coherence": _pair_stats("coherence"),
    "total_wall_clock_s": round(time.time() - t_start, 2),
    "judge_total_tokens": {
        "prompt": judge_llm.total_prompt_tokens,
        "completion": judge_llm.total_completion_tokens,
    },
    "gen_total_tokens": {
        "prompt": gen_llm.total_prompt_tokens,
        "completion": gen_llm.total_completion_tokens,
    },
}

(out_dir / "aggregate.json").write_text(
    json.dumps(aggregate, ensure_ascii=False, indent=2))


print(f"\n=== AGGREGATE (n={n}/{len(per_query)}) ===")
print(f"  C: mean_rounds={aggregate['c_mean_rounds']} "
       f"converged={aggregate['c_convergence_rate']*100:.0f}% "
       f"score={aggregate['c_mean_final_score']} "
       f"cost=${aggregate['c_total_cost_usd']}")
print(f"  D: mean_rounds={aggregate['d_mean_rounds']} "
       f"converged={aggregate['d_convergence_rate']*100:.0f}% "
       f"score={aggregate['d_mean_final_score']} "
       f"cost=${aggregate['d_total_cost_usd']}")
for axis in ("novelty", "validity", "coherence"):
    s = aggregate[f"pairwise_{axis}"]
    if s.get("n", 0) == 0:
        continue
    print(f"  pairwise {axis}: "
           f"C {s['C_win_rate']*100:.0f}% / "
           f"D {s['D_win_rate']*100:.0f}% / "
           f"tie {s['tie_rate']*100:.0f}%")
print(f"\n  total wall-clock: {aggregate['total_wall_clock_s']:.0f}s "
       f"({aggregate['total_wall_clock_s']/3600:.1f} h)")
print(f"  output dir: {out_dir}")
