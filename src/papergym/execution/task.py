"""Standard-harness Task abstraction for the execution gym."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Optional

from papergym.llm import LLMClient

_INT_RE = re.compile(r"(-?\d[\d,]*)")

DEFAULT_DATA_ROOT = Path("data/datasets")


class Task:
    """A standard task: fixed data + baseline + objective metric. Subclasses
    set task_id/topic and implement parse_pred and the baseline prompt.
    predictions = [{"id","pred"}]."""
    task_id: str = "base"
    topic: str = ""

    def __init__(self, n_examples: int = 50, seed: int = 0,
                 data_root: Path = DEFAULT_DATA_ROOT):
        self.n_examples = n_examples
        self.seed = seed
        self.data_root = Path(data_root)
        self._splits: dict = {}

    # ---- data (local jsonl splits) ----
    def split_path(self, split: str) -> Path:
        return self.data_root / self.task_id / f"{split}.jsonl"

    def load_split(self, split: str) -> list:
        path = self.split_path(split)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run scripts/prefetch_datasets.py first")
        return [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]

    def examples(self, split: str = "test") -> list:
        if split not in self._splits:
            self._splits[split] = self.load_split(split)
        return self._splits[split]

    def inputs(self, split: str = "test") -> list:
        """Label-free view handed to the agent (no 'answer')."""
        return [{"id": e["id"], "question": e["question"]}
                for e in self.examples(split)]

    # ---- metric ----
    def parse_pred(self, raw: str) -> str:
        raise NotImplementedError

    def score(self, predictions: list, split: str = "test") -> float:
        gold = {ex["id"]: ex["answer"] for ex in self.examples(split)}
        if not gold:
            return 0.0
        # Score each expected id at most once. Formal submission validation
        # lives in validate_predictions(); this guard keeps direct calls from
        # inflating accuracy with duplicate rows.
        pred_by_id = {}
        for p in predictions or []:
            if not isinstance(p, dict):
                continue
            pid = str(p.get("id", ""))
            if pid not in pred_by_id:
                pred_by_id[pid] = p.get("pred", "")
        hits = sum(
            1 for pid, answer in gold.items()
            if answer == self.parse_pred(str(pred_by_id.get(pid, "")))
        )
        return hits / len(gold)

    def validate_predictions(self, predictions: list,
                             split: str = "test") -> list[str]:
        """Return submission-format flags. Empty means scorable."""
        expected_ids = [str(ex["id"]) for ex in self.examples(split)]
        expected = set(expected_ids)
        flags: list[str] = []
        if not isinstance(predictions, list):
            return ["predictions_not_list"]

        seen: set[str] = set()
        for i, p in enumerate(predictions):
            if not isinstance(p, dict):
                flags.append(f"prediction_{i}_not_object")
                continue
            if "id" not in p:
                flags.append(f"prediction_{i}_missing_id")
                continue
            if "pred" not in p:
                flags.append(f"prediction_{i}_missing_pred")
            pid = str(p.get("id", ""))
            if pid in seen:
                flags.append(f"duplicate_prediction_id:{pid}")
            seen.add(pid)
            if pid not in expected:
                flags.append(f"unknown_prediction_id:{pid}")

        missing = expected - seen
        if missing:
            preview = ",".join(sorted(missing)[:10])
            flags.append(f"missing_prediction_ids:{preview}")
        return flags

    # ---- baseline ----
    def baseline_prompt(self, ex: dict) -> str:
        raise NotImplementedError

    def run_baseline(self, llm: LLMClient, split: str = "test") -> float:
        preds = []
        for ex in self.examples(split):
            raw = llm.chat([{"role": "user", "content": self.baseline_prompt(ex)}],
                           temperature=0.0)
            preds.append({"id": ex["id"], "pred": self.parse_pred(raw)})
        return self.score(preds, split=split)

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

    def parse_pred(self, raw: str) -> str:
        if not raw:
            return ""
        nums = _INT_RE.findall(raw)
        return nums[-1].replace(",", "") if nums else ""

    def baseline_prompt(self, ex: dict) -> str:
        return (f"Solve this problem. Think step by step, then give the final "
                f"answer as a single number on the last line.\n\n{ex['question']}")

    @classmethod
    def materialize(cls, data_root: Path = DEFAULT_DATA_ROOT,
                    n_test: int = 50, n_dev: int = 50) -> dict:
        """Pull GSM8K once from HF and write disjoint local dev/test splits.
        Labels live ONLY in these host files (never baked into the image)."""
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        out = data_root / cls.task_id
        out.mkdir(parents=True, exist_ok=True)
        tmp = cls()
        def _row(rid, row):
            return {"id": str(rid), "question": row["question"],
                    "answer": tmp.parse_pred(row["answer"])}
        test = [_row(i, ds[i]) for i in range(min(n_test, len(ds)))]
        dev = [_row(f"d{i}", ds[i])
               for i in range(n_test, min(n_test + n_dev, len(ds)))]
        for name, rows in (("test", test), ("dev", dev)):
            (out / f"{name}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n")
        return {"test": len(test), "dev": len(dev)}


class GeneratedArithmeticTask(Task):
    """Synthetic GSM-style arithmetic word problems with host-generated
    labels. Used for trustworthy smoke benchmarks because answers are not
    recoverable from a public dataset lookup."""
    task_id = "generated_math_accuracy"
    topic = "Math"

    def parse_pred(self, raw: str) -> str:
        if not raw:
            return ""
        nums = _INT_RE.findall(raw)
        return nums[-1].replace(",", "") if nums else ""

    def baseline_prompt(self, ex: dict) -> str:
        return (f"Solve this problem. Think step by step, then give the final "
                f"answer as a single number on the last line.\n\n{ex['question']}")

    @classmethod
    def materialize(cls, data_root: Path = DEFAULT_DATA_ROOT,
                    n_test: int = 50, n_dev: int = 50,
                    seed: int = 0) -> dict:
        rng = random.Random(seed)
        out = data_root / cls.task_id
        out.mkdir(parents=True, exist_ok=True)

        def _make(rid: str) -> dict:
            apples = rng.randint(3, 40)
            buys = rng.randint(2, 30)
            gives = rng.randint(1, min(apples + buys - 1, 25))
            boxes = rng.randint(2, 12)
            per_box = rng.randint(2, 9)
            answer = apples + buys - gives + boxes * per_box
            q = (
                f"Mina has {apples} stickers. She buys {buys} more, gives "
                f"{gives} to a friend, then opens {boxes} packs with "
                f"{per_box} stickers each. How many stickers does Mina have?"
            )
            return {"id": rid, "question": q, "answer": str(answer)}

        test = [_make(str(i)) for i in range(n_test)]
        dev = [_make(f"d{i}") for i in range(n_dev)]
        for name, rows in (("test", test), ("dev", dev)):
            (out / f"{name}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n")
        return {"test": len(test), "dev": len(dev)}


TASKS = {
    GSM8KAccuracyTask.task_id: GSM8KAccuracyTask,
    GeneratedArithmeticTask.task_id: GeneratedArithmeticTask,
}
