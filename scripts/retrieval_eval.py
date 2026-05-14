"""Stage 2 runner: paraphrase ON vs OFF retrieval comparison over a query set.

Usage:
    uv run python scripts/retrieval_eval.py \\
        --queries data/queries.yaml \\
        --library data/library \\
        --output-dir data/eval \\
        --judge-model openai/bedrock.anthropic.claude-sonnet-4-6 \\
        --k 3
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml
from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.common import CostSnapshot, cost_summary  # noqa: E402
from eval.retrieval import evaluate_query  # noqa: E402

from papergym.library import LibraryStore  # noqa: E402
from papergym.llm import LLMClient  # noqa: E402

load_dotenv(override=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--queries", required=True, type=Path,
                    help="YAML file with `queries: [...]` list.")
    p.add_argument("--library", required=True, type=Path,
                    help="Path to a seed library root (with shard_*).")
    p.add_argument("--output-dir", type=Path, default=Path("data/eval"))
    p.add_argument("--k", type=int, default=3,
                    help="Top-k seeds per domain.")
    p.add_argument("--judge-model", default=None,
                    help="LITELLM model for the relevance judge. "
                         "Defaults to $LITELLM_MODEL.")
    p.add_argument("--limit", type=int, default=None,
                    help="Process at most N queries (smoke test).")
    args = p.parse_args(argv)

    if "OPENAI_API_KEY" not in os.environ:
        print("error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    queries = yaml.safe_load(args.queries.read_text())["queries"]
    if args.limit:
        queries = queries[:args.limit]
    print(f"queries: {len(queries)}", file=sys.stderr)

    library = LibraryStore.open_merged(args.library)
    llm = LLMClient()
    judge = LLMClient(model=args.judge_model) if args.judge_model else llm
    if judge.model == llm.model:
        # Self-bias guard: paraphraser and judge sharing a model family
        # inflates relevance scores via shared lexical priors.
        print(f"error: judge and paraphraser share the same model "
              f"({llm.model!r}); pass --judge-model with a different "
              f"family.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / f"retrieval_{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir.mkdir(parents=True)
    out_path = run_dir / "evaluations.jsonl"
    print(f"writing to {out_path}", file=sys.stderr)

    gen_t0 = CostSnapshot.of(llm)
    judge_t0 = CostSnapshot.of(judge)
    wall_t0 = time.time()

    n_done = 0
    with out_path.open("w", encoding="utf-8") as fp:
        for q in queries:
            try:
                result = evaluate_query(
                    query=q, library=library,
                    llm=llm, judge_llm=judge, k=args.k,
                )
            except Exception as e:
                print(f"  error {q['id']}: {e}", file=sys.stderr)
                continue
            fp.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            fp.flush()
            n_done += 1
            print(f"  {n_done}/{len(queries)} {q['id']} done",
                  file=sys.stderr)

    summary = _summarise(out_path)
    summary["cost_total"] = cost_summary(
        judge_before=judge_t0, judge_after=CostSnapshot.of(judge),
        gen_before=gen_t0, gen_after=CostSnapshot.of(llm),
        wall_clock_s=time.time() - wall_t0,
    )
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n=== summary ===", file=sys.stderr)
    print(json.dumps(summary, indent=2, ensure_ascii=False), file=sys.stderr)


def _summarise(eval_path: Path) -> dict:
    by_mode: dict[str, dict[str, list[float]]] = {
        "on": defaultdict(list),
        "off": defaultdict(list),
    }

    for line in eval_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for mode in ("on", "off"):
            run = rec[mode]
            naive = [s["relevance"]["score"] for s in run["seeds"]
                      if s["relevance"]["score"] > 0]
            if naive:
                by_mode[mode]["relevance_mean"].append(
                    sum(naive) / len(naive))
            lens = [s["relevance_lens"]["score"] for s in run["seeds"]
                     if s["relevance_lens"]["score"] > 0]
            if lens:
                by_mode[mode]["relevance_lens_mean"].append(
                    sum(lens) / len(lens))
            # Seed home-domain coverage: how many of the 7 ML domains
            # the retrieved seeds actually came from. Cross-domain
            # retrieval should surface seeds from multiple domains;
            # naive single-query retrieval often concentrates in 1-2.
            home_domains = {s.get("domain") for s in run["seeds"]
                             if s.get("domain")}
            if home_domains:
                by_mode[mode]["seed_home_coverage"].append(len(home_domains))

    def _stat(xs: list[float]) -> dict:
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
        mode: {key: _stat(vals) for key, vals in metrics.items()}
        for mode, metrics in by_mode.items()
    }


if __name__ == "__main__":
    main()
