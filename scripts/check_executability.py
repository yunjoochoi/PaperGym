"""Classify every data/si_ideas/*.json for inference-time executability."""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from papergym.execution.executability import classify_executability
from papergym.execution.types import IdeaSpec
from papergym.llm import LLMClient

load_dotenv(override=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ideas", type=Path, default=Path("data/si_ideas"))
    p.add_argument("--out", type=Path,
                   default=Path("data/si_ideas/executability.jsonl"))
    args = p.parse_args(argv)

    llm = LLMClient()
    n_exec = 0
    with args.out.open("w", encoding="utf-8") as fp:
        for path in sorted(args.ideas.glob("*.json")):
            if path.name == "executability.jsonl":
                continue
            idea = IdeaSpec.from_dict(json.loads(path.read_text()))
            verdict = classify_executability(proposal=idea.proposal_text,
                                             llm=llm)
            rec = {"idea_id": idea.idea_id, "topic": idea.topic,
                   "condition": idea.condition, **verdict}
            fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fp.flush()
            n_exec += int(bool(verdict.get("gym_executable")))
    print(f"executable: {n_exec} ideas -> {args.out}")


if __name__ == "__main__":
    main()
