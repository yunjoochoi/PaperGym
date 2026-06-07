"""Run the execution gym over Si et al. ideas: baseline+agent exec+score."""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from eval.execution.evaluate import run_one_idea
from papergym.execution.task import TASKS, GSM8KAccuracyTask
from papergym.execution.types import IdeaSpec
from papergym.llm import LLMClient

load_dotenv(override=True)

# topic -> default Task for the MVP (extend as more Tasks land in the registry).
TOPIC_TASK = {"Math": "gsm8k_accuracy"}


def _summarise(results: list) -> dict:
    effs = [r["effectiveness"] for r in results if r["effectiveness"] is not None]
    faiths = [r["faithfulness_score"] for r in results if r["faithfulness_score"]]
    return {
        "n_ideas": len(results),
        "n_scored": len(effs),
        "effectiveness_mean": round(statistics.mean(effs), 4) if effs else None,
        "faithfulness_mean": round(statistics.mean(faiths), 4) if faiths else None,
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ideas", type=Path, default=Path("data/si_ideas"))
    p.add_argument("--output-dir", type=Path, default=Path("data/eval"))
    p.add_argument("--n-examples", type=int, default=50)
    p.add_argument("--use-docker", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")

    gen_llm = LLMClient(model=os.environ.get("LITELLM_MODEL"))
    judge_llm = LLMClient(model=os.environ.get("JUDGE_MODEL"))
    if judge_llm.model == gen_llm.model:
        sys.exit("judge model must differ from generator (self-bias control)")

    run_dir = args.output_dir / f"exec_{time.strftime('%Y%m%dT%H%M%S')}"
    run_dir.mkdir(parents=True)

    paths = sorted(pp for pp in args.ideas.glob("*.json")
                   if pp.name != "executability.jsonl")
    if args.limit:
        paths = paths[:args.limit]

    results = []
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as fp:
        for path in paths:
            idea = IdeaSpec.from_dict(json.loads(path.read_text()))
            task_id = TOPIC_TASK.get(idea.topic)
            if task_id is None:
                continue                                   # no MVP task for topic
            task = TASKS[task_id](n_examples=args.n_examples)
            res = run_one_idea(idea=idea, task=task, gen_llm=gen_llm,
                               judge_llm=judge_llm,
                               work_root=run_dir / idea.idea_id,
                               use_docker=args.use_docker)
            fp.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
            fp.flush()
            results.append(res.to_dict())

    summary = _summarise(results)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
