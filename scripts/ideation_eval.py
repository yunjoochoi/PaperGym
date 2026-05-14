"""Stage 3 runner: ideate + judge for each query × selected condition(s).

Conditions are independent; pass --conditions in any subset to run only
what you need (re-runs after a partial failure stay cheap). D is a
retrieval-removed negative control: same synthesizer as B/C, but with
random library seeds, so C vs D isolates the embedding-retrieval signal.

Usage:
    uv run python scripts/ideation_eval.py \\
        --queries data/queries.yaml \\
        --library data/library \\
        --conditions A,B,C,D \\
        --output-dir data/eval \\
        --judge-model openai/bedrock.anthropic.claude-sonnet-4-6 \\
        --k-per-domain 3
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.common import CostSnapshot, cost_summary  # noqa: E402
from eval.ideation import (  # noqa: E402
    run_condition_a, run_condition_b, run_condition_c, run_condition_d,
    judge_novelty, judge_validity, judge_coherence,
)

from papergym.library import LibraryStore  # noqa: E402
from papergym.llm import LLMClient  # noqa: E402

load_dotenv(override=True)


def _run_one(cond: str, *, query: dict,
              library: LibraryStore, llm: LLMClient, k_per_domain: int):
    text = query["text"]
    if cond == "A":
        return run_condition_a(query=text, llm=llm)
    if cond == "B":
        return run_condition_b(
            query=text, natural_domain=query.get("natural_domain"),
            library=library, llm=llm, total_seeds=k_per_domain * 7,
        )
    if cond == "C":
        return run_condition_c(
            query=text, library=library, llm=llm,
            natural_domain=query.get("natural_domain"),
            k_per_domain=k_per_domain,
        )
    if cond == "D":
        # Fixed seed=0: same 21-seed random pool across every query,
        # matching the loop ablation convention (evaluate_novelty_loop)
        # and the paper's reproducibility statement.
        return run_condition_d(
            query=text, library=library, llm=llm,
            total_seeds=k_per_domain * 7,
            seed=0,
        )
    raise ValueError(f"unknown condition {cond!r}")


def _judge(cond: str, query_text: str, output, judge_llm: LLMClient) -> dict:
    return {
        "condition": cond,
        "method": output.method,
        "rationale": output.rationale,
        "inspired_by": output.inspired_by,
        "retrieved_seed_ids": output.retrieved_seed_ids,
        "paraphrase_essence": output.paraphrase_essence,
        "novelty": asdict(judge_novelty(judge_llm=judge_llm,
                                          query=query_text, output=output)),
        "validity": asdict(judge_validity(judge_llm=judge_llm,
                                            query=query_text,
                                            output=output)),
        "coherence": asdict(judge_coherence(judge_llm=judge_llm,
                                              query=query_text,
                                              output=output)),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True, type=Path)
    p.add_argument("--library", required=True, type=Path)
    p.add_argument("--conditions", default="A,B,C,D",
                    help="Comma-separated subset of A,B,C,D. "
                          "D = synthesizer over random library seeds "
                          "(retrieval-removed negative control).")
    p.add_argument("--output-dir", type=Path, default=Path("data/eval"))
    p.add_argument("--k-per-domain", type=int, default=3)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    if "OPENAI_API_KEY" not in os.environ:
        print("error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    conds = [c.strip().upper() for c in args.conditions.split(",")
              if c.strip()]
    for c in conds:
        if c not in ("A", "B", "C", "D"):
            print(f"error: unknown condition {c!r}", file=sys.stderr)
            sys.exit(1)

    queries = yaml.safe_load(args.queries.read_text())["queries"]
    if args.limit:
        queries = queries[:args.limit]
    print(f"queries: {len(queries)}, conditions: {conds}", file=sys.stderr)

    library = LibraryStore.open_merged(args.library)
    llm = LLMClient()
    judge = LLMClient(model=args.judge_model) if args.judge_model else llm
    if judge.model == llm.model:
        # Self-bias guard: an LLM judging its own family is a known
        # source of inflated scores. Force a different judge model.
        print(f"error: judge and ideation share the same model "
              f"({llm.model!r}); pass --judge-model with a different "
              f"family (e.g. anthropic.claude-sonnet-4-6 vs gpt-5).",
              file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / f"ideation_{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir.mkdir(parents=True)
    out_path = run_dir / "evaluations.jsonl"
    print(f"writing to {out_path}", file=sys.stderr)

    gen_t0 = CostSnapshot.of(llm)
    judge_t0 = CostSnapshot.of(judge)
    wall_t0 = time.time()

    n_done = 0
    with out_path.open("w", encoding="utf-8") as fp:
        for q in queries:
            rec: dict = {
                "query_id": q["id"],
                "query_text": q["text"],
                "natural_domain": q.get("natural_domain"),
            }
            q_gen_t0 = CostSnapshot.of(llm)
            q_judge_t0 = CostSnapshot.of(judge)
            q_wall_t0 = time.time()
            for cond in conds:
                try:
                    out = _run_one(cond, query=q, library=library, llm=llm,
                                    k_per_domain=args.k_per_domain)
                except Exception as e:
                    print(f"  error {q['id']} {cond}: {e}", file=sys.stderr)
                    rec[cond] = None
                    continue
                if out is None:
                    rec[cond] = None
                    continue
                rec[cond] = _judge(cond, q["text"], out, judge)
            rec["cost"] = cost_summary(
                judge_before=q_judge_t0, judge_after=CostSnapshot.of(judge),
                gen_before=q_gen_t0, gen_after=CostSnapshot.of(llm),
                wall_clock_s=time.time() - q_wall_t0,
            )
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fp.flush()
            n_done += 1
            print(f"  {n_done}/{len(queries)} {q['id']} done "
                   f"(${rec['cost']['total_cost_usd']:.3f}, "
                   f"{rec['cost']['wall_clock_s']:.1f}s)",
                   file=sys.stderr)

    total_cost = cost_summary(
        judge_before=judge_t0, judge_after=CostSnapshot.of(judge),
        gen_before=gen_t0, gen_after=CostSnapshot.of(llm),
        wall_clock_s=time.time() - wall_t0,
    )
    summary = _summarise(out_path, conds)
    summary["cost_total"] = total_cost
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n=== summary ===", file=sys.stderr)
    print(json.dumps(summary, indent=2, ensure_ascii=False), file=sys.stderr)


def _summarise(eval_path: Path, conds: list[str]) -> dict:
    by_cond: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list))

    for line in eval_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for cond in conds:
            r = rec.get(cond)
            if not r:
                continue
            n = r["novelty"]
            if n.get("score"):
                by_cond[cond]["novelty"].append(n["score"])
            v = r.get("validity") or {}
            if v.get("score"):
                by_cond[cond]["validity"].append(v["score"])
            c = r.get("coherence") or {}
            if c.get("score"):
                by_cond[cond]["coherence"].append(c["score"])

    def _stat(xs):
        if not xs:
            return {"n": 0, "mean": None, "median": None, "stdev": None,
                    "p25": None, "p75": None, "hist": {}}
        qs = statistics.quantiles(xs, n=4) if len(xs) > 1 else [xs[0]] * 3
        # Histogram per bin catches inflation hidden by aggregate stats.
        hist: dict[str, int] = {}
        for x in xs:
            k = str(int(x)) if float(x).is_integer() else f"{x:.2f}"
            hist[k] = hist.get(k, 0) + 1
        return {
            "n": len(xs),
            "mean": round(statistics.mean(xs), 3),
            "median": round(statistics.median(xs), 3),
            "stdev": (round(statistics.stdev(xs), 3)
                      if len(xs) > 1 else None),
            "p25": round(qs[0], 3),
            "p75": round(qs[2], 3),
            "hist": dict(sorted(hist.items())),
        }

    return {
        cond: {key: _stat(vals) for key, vals in metrics.items()}
        for cond, metrics in by_cond.items()
    }


if __name__ == "__main__":
    main()
