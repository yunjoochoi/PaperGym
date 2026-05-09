"""Sample N papers and score every seed across the configured libraries.

Designed to compare A군 (no-tool) and C군 (full-agent) seeds head-to-head:
the same paper sample is evaluated for whichever libraries are supplied.

Usage:
    uv run python scripts/seed_quality_eval.py \\
        --library A=/home/shaush/__research/PaperGym_notool/data/library \\
        --library C=/home/shaush/__research/PaperGym/data/library \\
        --papers-cache /home/shaush/__research/papers_cache \\
        --output-dir data/eval \\
        --n-papers 50 \\
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
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

# Top-level eval/ package isn't installed via pip; ensure repo root is on path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.common import CostSnapshot, cost_summary  # noqa: E402
from eval.seed_quality import judge_seed  # noqa: E402

from papergym.llm import LLMClient  # noqa: E402

load_dotenv(override=True)


def _parse_library_arg(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--library must be NAME=PATH, got {spec!r}")
    name, raw_path = spec.split("=", 1)
    return name, Path(raw_path).expanduser()


def _read_seeds(library_root: Path) -> list[dict]:
    seeds: list[dict] = []
    for shard in sorted(library_root.glob("shard_*")):
        path = shard / "seeds.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    return seeds


def _paper_text(papers_cache: Path, arxiv_id: str) -> str:
    """Read full paper.md from the cache; empty string if missing."""
    p = papers_cache / arxiv_id / "paper.md"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--library", action="append", required=True,
                    type=_parse_library_arg,
                    help="NAME=/path/to/library — repeatable, e.g. "
                         "--library A=... --library C=...")
    p.add_argument("--papers-cache", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("data/eval"))
    p.add_argument("--n-papers", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge-model", default=None,
                    help="LITELLM model for the judge LLM. Defaults to "
                         "$LITELLM_MODEL.")
    args = p.parse_args(argv)

    if "OPENAI_API_KEY" not in os.environ:
        print("error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    libraries = dict(args.library)
    print(f"libraries: {list(libraries)}", file=sys.stderr)

    # Group seeds by paper across all libraries; restrict to common papers.
    by_paper: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for name, root in libraries.items():
        for s in _read_seeds(root):
            by_paper[s["paper_id"]].setdefault(name, []).append(s)
    common = [pid for pid, groups in by_paper.items()
              if set(groups) == set(libraries)]
    print(f"papers in all libraries: {len(common)} (of {len(by_paper)} total)",
          file=sys.stderr)

    rng = random.Random(args.seed)
    rng.shuffle(common)
    sample = common[:args.n_papers]
    print(f"sampling {len(sample)} papers", file=sys.stderr)

    judge = LLMClient(model=args.judge_model)
    seed_gen_model = os.environ.get("LITELLM_MODEL")
    if seed_gen_model and judge.model == seed_gen_model:
        # Self-bias guard: a judge sharing the seed-generator's model is
        # known to inflate scores; pass a different family.
        print(f"error: judge model ({judge.model!r}) matches "
              f"LITELLM_MODEL used to generate the seeds. Pass "
              f"--judge-model with a different family.", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / time.strftime("%Y%m%dT%H%M%S")
    run_dir.mkdir(parents=True)
    out_path = run_dir / "judgements.jsonl"
    print(f"writing to {out_path}", file=sys.stderr)

    judge_t0 = CostSnapshot.of(judge)
    wall_t0 = time.time()

    n_done = 0
    with out_path.open("w", encoding="utf-8") as out_fp:
        for pid in sample:
            paper_md = _paper_text(args.papers_cache, pid)
            if not paper_md:
                print(f"  skip {pid}: paper.md missing in cache",
                      file=sys.stderr)
                continue
            for name, seeds in by_paper[pid].items():
                for seed in seeds:
                    try:
                        j = judge_seed(judge_llm=judge, seed=seed,
                                        paper_excerpt=paper_md)
                    except Exception as e:
                        print(f"  error {name} {pid} {seed['seed_id']}: {e}",
                              file=sys.stderr)
                        continue
                    rec = {"group": name, **j.to_dict()}
                    out_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_fp.flush()
            n_done += 1
            if n_done % 5 == 0:
                print(f"  {n_done}/{len(sample)} papers done",
                      file=sys.stderr)

    summary = _summarise(out_path)
    summary["cost_total"] = cost_summary(
        judge_before=judge_t0, judge_after=CostSnapshot.of(judge),
        wall_clock_s=time.time() - wall_t0,
    )
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n=== summary ===", file=sys.stderr)
    print(json.dumps(summary, indent=2, ensure_ascii=False), file=sys.stderr)
    print(f"\nwrote {summary_path}", file=sys.stderr)


def _summarise(judgements_path: Path) -> dict:
    # Aggregate at paper level first so a library producing more seeds
    # per paper does not weight that paper more heavily, and so the
    # paired (paper, condition) structure is preserved for downstream
    # paired analyses.
    per_paper: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))

    for line in judgements_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        g, pid = rec["group"], rec["paper_id"]
        for axis in ("grounding", "specificity"):
            sc = rec[axis]["score"]
            if sc:
                per_paper[g][pid][axis].append(sc)

    summary = {}
    for g, paper_scores in per_paper.items():
        # Per-paper mean per axis becomes the unit of analysis.
        axis_papers: dict[str, list[float]] = defaultdict(list)
        for pid, axis_to_scores in paper_scores.items():
            for axis, scs in axis_to_scores.items():
                axis_papers[axis].append(statistics.mean(scs))
        summary[g] = {
            "papers_evaluated": len(paper_scores),
            "unit": "paper-level mean (one observation per paper)",
            "axes": {axis: _stat(means) for axis, means in axis_papers.items()},
        }
    return summary


def _stat(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0, "mean": None, "median": None, "stdev": None,
                "p25": None, "p75": None, "hist": {}}
    qs = statistics.quantiles(scores, n=4) if len(scores) > 1 else [scores[0]] * 3
    # Histogram per integer score reveals inflation that mean/std hide
    # (e.g., a judge that defaults to 5 with rare 1s and a judge that
    # spreads 3-5 evenly produce similar means but very different
    # distributions).
    hist: dict[str, int] = {}
    for s in scores:
        k = str(int(s)) if float(s).is_integer() else f"{s:.2f}"
        hist[k] = hist.get(k, 0) + 1
    return {
        "n": len(scores),
        "mean": round(statistics.mean(scores), 3),
        "median": statistics.median(scores),
        "stdev": (round(statistics.stdev(scores), 3)
                  if len(scores) > 1 else None),
        "p25": round(qs[0], 3),
        "p75": round(qs[2], 3),
        "hist": dict(sorted(hist.items())),
    }


if __name__ == "__main__":
    main()
