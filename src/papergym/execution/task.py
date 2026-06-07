"""Standard-harness Task abstraction for the execution gym."""
from __future__ import annotations

import re
from typing import Optional

from papergym.llm import LLMClient

_INT_RE = re.compile(r"(-?\d[\d,]*)")


class Task:
    """A standard task: fixed data + baseline + objective metric. Subclasses
    set task_id/topic and implement load_examples/parse_pred/score and the
    baseline prompt. predictions = [{"id","pred"}]."""
    task_id: str = "base"
    topic: str = ""

    def __init__(self, n_examples: int = 50, seed: int = 0):
        self.n_examples = n_examples
        self.seed = seed
        self._examples: Optional[list] = None

    # ---- data ----
    def load_examples(self) -> list:
        raise NotImplementedError

    def examples(self) -> list:
        if self._examples is None:
            self._examples = self.load_examples()
        return self._examples

    # ---- metric ----
    def parse_pred(self, raw: str) -> str:
        raise NotImplementedError

    def score(self, predictions: list) -> float:
        raise NotImplementedError

    # ---- baseline ----
    def baseline_prompt(self, ex: dict) -> str:
        raise NotImplementedError

    def run_baseline(self, llm: LLMClient) -> float:
        preds = []
        for ex in self.examples():
            raw = llm.chat([{"role": "user", "content": self.baseline_prompt(ex)}],
                           temperature=0.0)
            preds.append({"id": ex["id"], "pred": self.parse_pred(raw)})
        return self.score(preds)

    # ---- agent-facing manifest ----
    def manifest(self) -> dict:
        return {"task_id": self.task_id, "topic": self.topic,
                "n_examples": self.n_examples,
                "predictions_format": '[{"id": str, "pred": str}]',
                "description": self.__doc__ or ""}


class GSM8KAccuracyTask(Task):
    """GSM8K grade-school math word problems; exact-match accuracy on the
    final integer answer. Baseline = direct chain-of-thought prompting."""
    task_id = "gsm8k_accuracy"
    topic = "Math"

    def load_examples(self) -> list:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ds = ds.select(range(min(self.n_examples, len(ds))))
        out = []
        for i, row in enumerate(ds):
            gold = self.parse_pred(row["answer"])
            out.append({"id": str(i), "question": row["question"], "answer": gold})
        return out

    def parse_pred(self, raw: str) -> str:
        if not raw:
            return ""
        nums = _INT_RE.findall(raw)
        return nums[-1].replace(",", "") if nums else ""

    def score(self, predictions: list) -> float:
        gold = {ex["id"]: ex["answer"] for ex in self.examples()}
        if not gold:
            return 0.0
        hits = sum(1 for p in predictions
                   if gold.get(p["id"]) == self.parse_pred(str(p["pred"])))
        return hits / len(gold)

    def baseline_prompt(self, ex: dict) -> str:
        return (f"Solve this problem. Think step by step, then give the final "
                f"answer as a single number on the last line.\n\n{ex['question']}")


TASKS = {GSM8KAccuracyTask.task_id: GSM8KAccuracyTask}
