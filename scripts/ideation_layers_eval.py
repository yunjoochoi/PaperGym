"""Layer 1 (pairwise) + Layer 2 (method specificity) evaluation over an
existing Stage 3 ideation jsonl. Runs without re-generating methods.

Layer 1 — pairwise win-rate across condition pairs (default C vs A,
C vs D, C vs B). Method positions are randomized per call to guard
against judge position bias.

Layer 2 — method specificity Likert (1-5) reusing the Stage 1 entity-
density rubric, adapted for the longer ideation method texts.

Usage:
    uv run python scripts/ideation_layer12_eval.py \\
        --ideation-jsonl data/eval/ideation_<ts>/evaluations.jsonl \\
        --pairs C-A,C-D,C-B \\
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
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.common import CostSnapshot, cost_summary  # noqa: E402
from eval.ideation import (  # noqa: E402
    judge_pairwise, judge_method_specificity, IdeationOutput,
)

from papergym.llm import LLMClient  # noqa: E402

load_dotenv(override=True)


def _output_from_record(rec: dict, cond: str) -> IdeationOutput:
    """Reconstruct an IdeationOutput from a saved jsonl record."""
    c = rec[cond]
    return IdeationOutput(
        condition=cond,
        method=c.get("method", ""),
        rationale=c.get("rationale", ""),
        inspired_by=c.get("inspired_by") or [],
        retrieved_seed_ids=c.get("retrieved_seed_ids") or [],
        retrieved_seeds=[],   # not needed for these judges
        paraphrase_essence=c.get("paraphrase_essence"),
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ideation-jsonl", required=True, type=Path)
    p.add_argument("--pairs", default="C-A,C-D,C-B",
                    help="Comma-separated X-Y pairs for pairwise judging.")
    p.add_argument("--axes", default="novelty,validity",
                    help="Comma-separated pairwise axes.")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None,
                    help="Defaults to a sibling dir of --ideation-jsonl.")
    args = p.parse_args(argv)

    if "OPENAI_API_KEY" not in os.environ:
        print("error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    pairs = [tuple(p.split("-")) for p in args.pairs.split(",") if p.strip()]
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]

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
    print(f"loaded {len(recs)} queries", file=sys.stderr)

    out_dir = (args.output_dir if args.output_dir
                else in_path.parent.parent /
                    f"layer12_{time.strftime('%Y%m%dT%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pw_path = out_dir / "pairwise.jsonl"
    spec_path = out_dir / "specificity.jsonl"
    print(f"writing to {out_dir}", file=sys.stderr)

    judge_t0 = CostSnapshot.of(judge)
    wall_t0 = time.time()

    n_done = 0
    with pw_path.open("w") as pw_fp, spec_path.open("w") as spec_fp:
        for rec in recs:
            qid = rec["query_id"]
            qtext = rec["query_text"]
            present = [c for c in ("A", "B", "C", "D")
                        if rec.get(c) and rec[c].get("method")]

            # Layer 1: pairwise comparisons
            for x, y in pairs:
                if x not in present or y not in present:
                    continue
                ox = _output_from_record(rec, x)
                oy = _output_from_record(rec, y)
                for axis in axes:
                    try:
                        result = judge_pairwise(
                            judge_llm=judge, query=qtext,
                            method_x=ox.method, method_y=oy.method,
                            condition_x=x, condition_y=y,
                            axis=axis, rng=rng,
                        )
                    except Exception as e:
                        print(f"  pairwise {qid} {x}-{y} {axis} error: {e}",
                              file=sys.stderr)
                        continue
                    pw_fp.write(json.dumps({
                        "query_id": qid, "axis": axis,
                        "x": x, "y": y,
                        **asdict(result),
                    }, ensure_ascii=False) + "\n")
                    pw_fp.flush()

            # Layer 2: method specificity per condition
            for cond in present:
                out = _output_from_record(rec, cond)
                try:
                    score = judge_method_specificity(
                        judge_llm=judge, query=qtext, output=out)
                except Exception as e:
                    print(f"  specificity {qid} {cond} error: {e}",
                          file=sys.stderr)
                    continue
                spec_fp.write(json.dumps({
                    "query_id": qid, "condition": cond,
                    "specificity": asdict(score),
                }, ensure_ascii=False) + "\n")
                spec_fp.flush()

            n_done += 1
            if n_done % 5 == 0:
                print(f"  {n_done}/{len(recs)} queries done", file=sys.stderr)

    summary = _summarise(pw_path, spec_path)
    summary["cost_total"] = cost_summary(
        judge_before=judge_t0, judge_after=CostSnapshot.of(judge),
        wall_clock_s=time.time() - wall_t0,
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print("\n=== summary ===", file=sys.stderr)
    print(json.dumps(summary, indent=2, ensure_ascii=False), file=sys.stderr)


def _summarise(pw_path: Path, spec_path: Path) -> dict:
    # Layer 1: per-(pair, axis) win rate, mapped back through the random
    # position swap so the reported winner refers to the original X/Y.
    pairwise: dict = defaultdict(lambda: Counter())
    if pw_path.exists():
        for line in pw_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["x"], r["y"], r["axis"])
            w = r["winner"]
            if w == "tie":
                pairwise[key]["tie"] += 1
            else:
                # Map shown-position ("A"/"B") back to original X/Y.
                resolved = (r["condition_for_a"] if w == "A"
                            else r["condition_for_b"])
                if resolved == r["x"]:
                    pairwise[key]["x_win"] += 1
                elif resolved == r["y"]:
                    pairwise[key]["y_win"] += 1

    pairwise_out: dict = {}
    for (x, y, axis), c in pairwise.items():
        n = c["x_win"] + c["y_win"] + c["tie"]
        if n == 0:
            continue
        pairwise_out[f"{x}-vs-{y}__{axis}"] = {
            "n": n,
            f"{x}_win": c["x_win"],
            f"{y}_win": c["y_win"],
            "tie": c["tie"],
            f"{x}_win_rate": round(c["x_win"] / n, 3),
            f"{y}_win_rate": round(c["y_win"] / n, 3),
            "tie_rate": round(c["tie"] / n, 3),
        }

    # Layer 2: per-condition mean specificity.
    by_cond: dict = defaultdict(list)
    if spec_path.exists():
        for line in spec_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            sc = r["specificity"]["score"]
            if sc:
                by_cond[r["condition"]].append(sc)
    spec_out = {}
    for cond, scores in by_cond.items():
        if not scores:
            continue
        spec_out[cond] = {
            "n": len(scores),
            "mean": round(statistics.mean(scores), 3),
            "median": statistics.median(scores),
            "stdev": (round(statistics.stdev(scores), 3)
                       if len(scores) > 1 else None),
            "hist": dict(sorted(Counter(scores).items())),
        }

    return {"pairwise": pairwise_out, "specificity": spec_out}


if __name__ == "__main__":
    main()
