"""Stage 3 single-pass coherence judge — backfills per-condition
coherence onto an existing ideation_<ts>/evaluations.jsonl produced
when the runner only scored novelty and validity.

Mirrors the per-condition novelty/validity already in tab:per-cond
so coherence reporting is symmetric across the three axes.

Usage:
    uv run python scripts/coherence_per_condition_eval.py \\
        --ideation-jsonl data/eval/ideation_<ts>/evaluations.jsonl \\
        --conditions A,B,C,D \\
        --judge-model openai/bedrock.anthropic.claude-sonnet-4-6
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

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.common import CostSnapshot, cost_summary  # noqa: E402
from eval.ideation import IdeationOutput, judge_coherence  # noqa: E402
from papergym.llm import LLMClient  # noqa: E402

load_dotenv(override=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ideation-jsonl", required=True, type=Path)
    p.add_argument("--conditions", default="A,B,C,D")
    p.add_argument("--judge-model", default=None,
                    help="Defaults to JUDGE_MODEL from .env.")
    p.add_argument("--limit", type=int, default=None,
                    help="Process at most N records (smoke test).")
    p.add_argument("--output-dir", type=Path, default=None)
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

    judge_model = args.judge_model or os.environ.get("JUDGE_MODEL")
    if not judge_model:
        print("error: pass --judge-model or set JUDGE_MODEL in .env",
              file=sys.stderr)
        sys.exit(1)
    judge = LLMClient(model=judge_model)
    seed_gen_model = os.environ.get("LITELLM_MODEL")
    if seed_gen_model and judge.model == seed_gen_model:
        print(f"error: judge ({judge.model!r}) matches LITELLM_MODEL; "
              f"pass --judge-model with a different family.",
              file=sys.stderr)
        sys.exit(1)

    in_path = args.ideation_jsonl
    recs = [json.loads(l) for l in in_path.read_text().splitlines()
            if l.strip()]
    if args.limit:
        recs = recs[:args.limit]
    print(f"loaded {len(recs)} ideation records, conditions={conds}",
          file=sys.stderr)

    out_dir = (args.output_dir if args.output_dir
                else in_path.parent.parent /
                    f"coherence_per_cond_{time.strftime('%Y%m%dT%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "judgements.jsonl"
    print(f"writing to {out_path}", file=sys.stderr)

    judge_t0 = CostSnapshot.of(judge)
    wall_t0 = time.time()

    n_done = 0
    with out_path.open("w") as fp:
        for rec in recs:
            qid = rec["query_id"]
            qtext = rec["query_text"]
            for cond in conds:
                cond_rec = rec.get(cond)
                if not cond_rec or not cond_rec.get("method"):
                    continue
                # Build a minimal IdeationOutput just for the judge —
                # only method/rationale are read.
                fake_out = IdeationOutput(
                    condition=cond,
                    method=cond_rec["method"],
                    rationale=cond_rec.get("rationale", ""),
                )
                try:
                    score = judge_coherence(judge_llm=judge,
                                              query=qtext, output=fake_out)
                except Exception as e:
                    print(f"  err {qid} {cond}: {e}", file=sys.stderr)
                    continue
                fp.write(json.dumps({
                    "query_id":  qid,
                    "condition": cond,
                    **asdict(score),
                }, ensure_ascii=False) + "\n")
                fp.flush()
                n_done += 1

    print(f"\n{n_done} judgments complete", file=sys.stderr)
    summary = _summarise(out_path, conds)
    summary["cost_total"] = cost_summary(
        judge_before=judge_t0, judge_after=CostSnapshot.of(judge),
        wall_clock_s=time.time() - wall_t0,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== summary ===", file=sys.stderr)
    print(json.dumps(summary, indent=2, ensure_ascii=False),
          file=sys.stderr)


def _summarise(jsonl_path: Path, conds: list[str]) -> dict:
    by_cond: dict[str, list[int]] = defaultdict(list)
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("score"):
                by_cond[r["condition"]].append(r["score"])

    out: dict = {}
    for cond in conds:
        xs = by_cond.get(cond, [])
        if not xs:
            out[cond] = {"n": 0}
            continue
        hist: dict[str, int] = {}
        for x in xs:
            k = str(int(x))
            hist[k] = hist.get(k, 0) + 1
        out[cond] = {
            "n":      len(xs),
            "mean":   round(statistics.mean(xs), 3),
            "median": round(statistics.median(xs), 3),
            "stdev":  (round(statistics.stdev(xs), 3)
                        if len(xs) > 1 else None),
            "hist":   dict(sorted(hist.items())),
        }
    return out


if __name__ == "__main__":
    main()
