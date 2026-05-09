"""Stage 1 negative control: re-judge grounding with shuffled papers.

Takes an existing Stage 1 run directory; for each judged seed, pairs the
seed's text with a randomly chosen DIFFERENT paper.md and re-judges
grounding only. If the judge actually reads the paper, grounding should
collapse from ~5 to 1-2; if scores stay high the judge is rubber-stamping
on prior knowledge and the original eval is unreliable.

Usage:
    uv run python scripts/seed_shuffled.py \\
        --judgements data/eval/20260502T184507/judgements.jsonl \\
        --library A=/home/shaush/__research/PaperGym_notool/data/library \\
        --library C=/home/shaush/__research/PaperGym/data/library \\
        --papers-cache /home/shaush/__research/papers_cache \\
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

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from eval.common import CostSnapshot, cost_summary, judge_axis  # noqa: E402
from papergym.agents.base import PromptLoader  # noqa: E402
from papergym.llm import LLMClient  # noqa: E402

load_dotenv(override=True)


def _parse_library_arg(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"--library must be NAME=PATH, got {spec!r}")
    name, raw_path = spec.split("=", 1)
    return name, Path(raw_path).expanduser()


def _read_seeds(library_root: Path) -> dict[str, dict]:
    by_seed_id: dict[str, dict] = {}
    for shard in sorted(library_root.glob("shard_*")):
        path = shard / "seeds.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            by_seed_id[s["seed_id"]] = s
    return by_seed_id


def _paper_text(papers_cache: Path, arxiv_id: str) -> str:
    p = papers_cache / arxiv_id / "paper.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--judgements", required=True, type=Path,
                    help="Path to a Stage 1 judgements.jsonl whose grounding "
                          "scores will be re-judged against shuffled papers.")
    p.add_argument("--library", action="append", required=True,
                    type=_parse_library_arg)
    p.add_argument("--papers-cache", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("data/eval"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--judge-model", default=None)
    args = p.parse_args(argv)

    if "OPENAI_API_KEY" not in os.environ:
        print("error: OPENAI_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    libraries = dict(args.library)
    seeds_by_group: dict[str, dict[str, dict]] = {
        name: _read_seeds(root) for name, root in libraries.items()
    }

    judgements = [json.loads(l) for l in
                   args.judgements.read_text(encoding="utf-8").splitlines()
                   if l.strip()]
    print(f"loaded {len(judgements)} judgements", file=sys.stderr)

    # All papers per group, used as the shuffle pool.
    papers_per_group: dict[str, list[str]] = defaultdict(list)
    for j in judgements:
        papers_per_group[j["group"]].append(j["paper_id"])
    for g in papers_per_group:
        papers_per_group[g] = sorted(set(papers_per_group[g]))

    rng = random.Random(args.seed)
    judge = LLMClient(model=args.judge_model)
    seed_gen_model = os.environ.get("LITELLM_MODEL")
    if seed_gen_model and judge.model == seed_gen_model:
        print(f"error: judge model ({judge.model!r}) matches "
              f"LITELLM_MODEL; pass --judge-model with a different family.",
              file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / f"shuffled_{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir.mkdir(parents=True)
    out_path = run_dir / "judgements.jsonl"
    print(f"writing to {out_path}", file=sys.stderr)

    grounding_prompts = PromptLoader(_REPO_ROOT / "eval" / "seed_quality" /
                                       "prompts")

    judge_t0 = CostSnapshot.of(judge)
    wall_t0 = time.time()

    n_done = 0
    with out_path.open("w", encoding="utf-8") as fp:
        for j in judgements:
            g, pid, sid = j["group"], j["paper_id"], j["seed_id"]
            seed = seeds_by_group.get(g, {}).get(sid)
            if seed is None:
                continue
            other_papers = [p for p in papers_per_group[g] if p != pid]
            if not other_papers:
                continue
            shuffled_pid = rng.choice(other_papers)
            paper_md = _paper_text(args.papers_cache, shuffled_pid)
            if not paper_md:
                continue
            try:
                score = judge_axis(
                    llm=judge, prompts=grounding_prompts,
                    prompt_name="grounding",
                    problem=seed["problem"], method=seed["method"],
                    paper_excerpt=paper_md,
                )
            except Exception as e:
                print(f"  error {g} {sid}: {e}", file=sys.stderr)
                continue
            fp.write(json.dumps({
                "group": g, "seed_id": sid,
                "true_paper_id": pid, "shuffled_paper_id": shuffled_pid,
                "grounding_true": j["grounding"],
                "grounding_shuffled": {"score": score.score,
                                          "reasoning": score.reasoning},
            }, ensure_ascii=False) + "\n")
            fp.flush()
            n_done += 1
            if n_done % 25 == 0:
                print(f"  {n_done} done", file=sys.stderr)

    summary = _summarise(out_path)
    summary["cost_total"] = cost_summary(
        judge_before=judge_t0, judge_after=CostSnapshot.of(judge),
        wall_clock_s=time.time() - wall_t0,
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n=== summary ===", file=sys.stderr)
    print(json.dumps(summary, indent=2, ensure_ascii=False), file=sys.stderr)


def _summarise(path: Path) -> dict:
    by_group: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        true_s = r["grounding_true"]["score"]
        shuf_s = r["grounding_shuffled"]["score"]
        if true_s and shuf_s:
            by_group[r["group"]].append((true_s, shuf_s))

    summary: dict = {}
    for g, pairs in by_group.items():
        true_scores = [t for t, _ in pairs]
        shuf_scores = [s for _, s in pairs]
        diffs = [t - s for t, s in pairs]
        # Sign sanity: if the judge actually reads the paper, the
        # shuffled-paper score should be lower than the true-paper score
        # for almost every seed.
        n_lower = sum(1 for d in diffs if d > 0)
        summary[g] = {
            "n": len(pairs),
            "true_mean": round(statistics.mean(true_scores), 3),
            "shuffled_mean": round(statistics.mean(shuf_scores), 3),
            "drop_mean": round(statistics.mean(diffs), 3),
            "drop_stdev": (round(statistics.stdev(diffs), 3)
                            if len(diffs) > 1 else None),
            "shuffled_lt_true_count": n_lower,
            "shuffled_lt_true_pct": round(n_lower / len(pairs) * 100, 1),
        }
    return summary


if __name__ == "__main__":
    main()
