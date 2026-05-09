"""Stage 3 inspired_by grounding judge.

For each cited seed in conditions C and D, score whether the
synthesized method actually incorporates the seed's claimed
borrowed_aspect. Uses a judge model from a different family
than the synthesizer (self-bias guard).

Usage:
    uv run python scripts/inspired_by_grounding_eval.py \\
        --ideation-jsonl data/eval/ideation_<ts>/evaluations.jsonl \\
        --conditions C,D \\
        --judge-model openai/bedrock.anthropic.claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.common import CostSnapshot, cost_summary  # noqa: E402
from eval.ideation import judge_inspired_by_grounding  # noqa: E402

from papergym.llm import LLMClient  # noqa: E402

load_dotenv(override=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ideation-jsonl", required=True, type=Path)
    p.add_argument("--conditions", default="C,D",
                    help="Comma-separated condition labels to evaluate.")
    p.add_argument("--judge-model", required=True,
                    help="LITELLM model for the grounding judge. "
                         "Must differ from the synthesizer family.")
    p.add_argument("--max-cites-per-method", type=int, default=None,
                    help="Cap citations judged per method (None = all).")
    p.add_argument("--limit", type=int, default=None,
                    help="Evaluate at most N records (smoke test).")
    p.add_argument("--output-dir", type=Path, default=None)
    args = p.parse_args(argv)

    if "OPENAI_API_KEY" not in os.environ:
        print("error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    judge = LLMClient(model=args.judge_model)
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
    print(f"loaded {len(recs)} records, conditions={conds}", file=sys.stderr)

    out_dir = (args.output_dir if args.output_dir
                else in_path.parent.parent /
                    f"grounding_{time.strftime('%Y%m%dT%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "judgements.jsonl"
    print(f"writing to {out_path}", file=sys.stderr)

    judge_t0 = CostSnapshot.of(judge)
    wall_t0 = time.time()

    n_done = 0
    with out_path.open("w") as fp:
        for rec in recs:
            qid = rec["query_id"]
            for cond in conds:
                cd = rec.get(cond) or {}
                method = cd.get("method", "")
                if not method:
                    continue
                ib = cd.get("inspired_by") or []
                # de-duplicate by seed_id (B-condition outlier had
                # 152 entries collapsing to 10 unique IDs)
                seen = set()
                unique_ib = []
                for e in ib:
                    sid = e.get("seed_id")
                    if not sid or sid in seen:
                        continue
                    seen.add(sid)
                    unique_ib.append(e)
                if args.max_cites_per_method is not None:
                    unique_ib = unique_ib[:args.max_cites_per_method]

                for entry in unique_ib:
                    try:
                        score = judge_inspired_by_grounding(
                            judge_llm=judge,
                            method=method,
                            seed_id=entry.get("seed_id", ""),
                            seed_domain=entry.get("domain", ""),
                            borrowed_aspect=entry.get("borrowed_aspect", ""),
                        )
                    except Exception as e:
                        print(f"  err {qid} {cond} {entry.get('seed_id')}: {e}",
                              file=sys.stderr)
                        continue
                    fp.write(json.dumps({
                        "query_id": qid,
                        "condition": cond,
                        "seed_id": entry.get("seed_id"),
                        "domain": entry.get("domain"),
                        "borrowed_aspect": entry.get("borrowed_aspect"),
                        **asdict(score),
                    }, ensure_ascii=False) + "\n")
                    fp.flush()

            n_done += 1
            if n_done % 5 == 0:
                print(f"  {n_done}/{len(recs)} records done",
                      file=sys.stderr)

    summary = _summarise(out_path)
    summary["cost_total"] = cost_summary(
        judge_before=judge_t0, judge_after=CostSnapshot.of(judge),
        wall_clock_s=time.time() - wall_t0,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== summary ===", file=sys.stderr)
    print(json.dumps(summary, indent=2, ensure_ascii=False), file=sys.stderr)


def _summarise(jsonl_path: Path) -> dict:
    by_cond: dict[str, list[int]] = defaultdict(list)
    if jsonl_path.exists():
        for line in jsonl_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            sc = r.get("score") or 0
            if sc:  # 0 = parse failure
                by_cond[r["condition"]].append(sc)

    out = {}
    for cond, scores in by_cond.items():
        if not scores:
            continue
        out[cond] = {
            "n": len(scores),
            "mean": round(statistics.mean(scores), 3),
            "median": statistics.median(scores),
            "stdev": (round(statistics.stdev(scores), 3)
                       if len(scores) > 1 else None),
            "hist": dict(sorted(Counter(scores).items())),
            "fraction_full": round(
                sum(1 for s in scores if s == 3) / len(scores), 3),
            "fraction_no": round(
                sum(1 for s in scores if s == 1) / len(scores), 3),
        }
    return out


if __name__ == "__main__":
    main()
