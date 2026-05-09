"""Stage 3 pairwise coherence judge for C vs D.

Tests whether random-seed methods (D) are technically less
coherent than cross-domain methods (C). Uses a held-out judge
(different model family from the synthesizer).

Usage:
    uv run python scripts/coherence_pairwise_eval.py \\
        --ideation-jsonl data/eval/ideation_<ts>/evaluations.jsonl \\
        --judge-model openai/bedrock.anthropic.claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.common import CostSnapshot, cost_summary  # noqa: E402
from eval.ideation import judge_pairwise  # noqa: E402

from papergym.llm import LLMClient  # noqa: E402

load_dotenv(override=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ideation-jsonl", required=True, type=Path)
    p.add_argument("--pair", default="C-D",
                    help="Single pair X-Y to compare (default: C-D).")
    p.add_argument("--judge-model", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                    help="Process at most N records (smoke test).")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)

    if "OPENAI_API_KEY" not in os.environ:
        print("error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    cond_x, cond_y = args.pair.split("-")
    judge = LLMClient(model=args.judge_model)
    seed_gen_model = os.environ.get("LITELLM_MODEL")
    if seed_gen_model and judge.model == seed_gen_model:
        print(f"error: judge ({judge.model!r}) matches LITELLM_MODEL; "
              f"pass --judge-model with a different family.",
              file=sys.stderr)
        sys.exit(1)

    rng = random.Random(args.seed)
    in_path = args.ideation_jsonl
    recs = [json.loads(l) for l in in_path.read_text().splitlines()
            if l.strip()]
    if args.limit:
        recs = recs[:args.limit]
    print(f"loaded {len(recs)} records, pair={cond_x}-{cond_y}",
          file=sys.stderr)

    out_dir = (args.output_dir if args.output_dir
                else in_path.parent.parent /
                    f"coherence_{time.strftime('%Y%m%dT%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "judgements.jsonl"
    print(f"writing to {out_path}", file=sys.stderr)

    judge_t0 = CostSnapshot.of(judge)
    wall_t0 = time.time()

    n_done = 0
    with out_path.open("w") as fp:
        for rec in recs:
            qid = rec["query_id"]
            if not (rec.get(cond_x, {}).get("method")
                    and rec.get(cond_y, {}).get("method")):
                continue
            try:
                result = judge_pairwise(
                    judge_llm=judge, query=rec["query_text"],
                    method_x=rec[cond_x]["method"],
                    method_y=rec[cond_y]["method"],
                    condition_x=cond_x, condition_y=cond_y,
                    axis="coherence", rng=rng,
                )
            except Exception as e:
                print(f"  err {qid}: {e}", file=sys.stderr)
                continue
            fp.write(json.dumps({
                "query_id": qid, "x": cond_x, "y": cond_y,
                **asdict(result),
            }, ensure_ascii=False) + "\n")
            fp.flush()
            n_done += 1

    print(f"\n{n_done} judgments complete", file=sys.stderr)
    summary = _summarise(out_path, cond_x, cond_y)
    summary["cost_total"] = cost_summary(
        judge_before=judge_t0, judge_after=CostSnapshot.of(judge),
        wall_clock_s=time.time() - wall_t0,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== summary ===", file=sys.stderr)
    print(json.dumps(summary, indent=2, ensure_ascii=False),
          file=sys.stderr)


def _summarise(jsonl_path: Path, cond_x: str, cond_y: str) -> dict:
    counts = Counter()
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            w = r["winner"]
            if w == "tie":
                counts["tie"] += 1
                continue
            resolved = (r["condition_for_a"] if w == "A"
                        else r["condition_for_b"])
            if resolved == cond_x:
                counts["x_win"] += 1
            elif resolved == cond_y:
                counts["y_win"] += 1
    n = sum(counts.values())
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        f"{cond_x}_win": counts["x_win"],
        f"{cond_y}_win": counts["y_win"],
        "tie": counts["tie"],
        f"{cond_x}_win_rate": round(counts["x_win"] / n, 3),
        f"{cond_y}_win_rate": round(counts["y_win"] / n, 3),
        "tie_rate": round(counts["tie"] / n, 3),
    }


if __name__ == "__main__":
    main()
