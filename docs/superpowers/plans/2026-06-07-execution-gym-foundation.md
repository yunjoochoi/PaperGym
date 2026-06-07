# Execution Gym Foundation (P1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimum execution gym that takes ONE inference-only research-idea proposal, has an LLM agent implement+run its experiment in an isolated sandbox against a standard task, and emits an objective effectiveness score plus a faithfulness check — plus the Si et al. data acquisition and full-43 executability pre-check that everything downstream depends on.

**Architecture:** Reuse PaperGym's existing primitives (`LLMClient`, `run_tool_loop`, `fn.schema` tool pattern, `judge_axis`/`parse_axis`, script conventions). Add a new `papergym.execution` package: a `Task` abstraction (dataset + baseline + objective metric), an isolated `Sandbox` (Local for dev/tests, Docker for real runs — PaperGym has NO execution sandbox today, so this is net-new), an `ExecutionAgent` that drives `run_tool_loop` with a constrained write/run/finish action space, an objective `Scorer`, and a `faithfulness` judge. Orchestration + a timestamped runner mirror `scripts/ideation_eval.py`.

**Tech Stack:** Python 3.11, litellm (via `LLMClient`), `datasets` (HF), Docker, pytest + `unittest.mock` (no pytest-mock plugin — match existing tests), `gdown` for Si et al. archives.

**MVP concrete task (swappable):** `gsm8k_accuracy` — exact-match accuracy on a GSM8K subset. Chosen because it is objective, cheap, deterministic, and maps to appendix idea H3 (Math, known human-execution overall 3.0) so P2 can anchor it. Swapping in another `Task` (e.g. ACP/GovReport summarization) requires only a new `Task` subclass.

---

## File Structure

```
src/papergym/execution/
  __init__.py            # exports
  types.py               # IdeaSpec, RunArtifact, ExecResult dataclasses
  task.py                # Task base + GSM8KAccuracyTask + TASKS registry
  sandbox.py             # Sandbox protocol, LocalSandbox, DockerSandbox
  executability.py       # classify_executability(proposal) -> dict (ports the ① rubric)
  agent.py               # execution tools (WriteFile/RunPython/ReadFile/Finish) + ExecutionAgent
  scorer.py              # score_effectiveness(task, run) -> ExecResult fields
  faithfulness.py        # judge_faithfulness(proposal, code) -> AxisScore
  prompts/
    faithfulness.yaml    # judge prompt (ends with 'Score: <1-5>')
    executability.yaml   # classifier prompt
eval/execution/
  __init__.py
  evaluate.py            # run_one_idea(...) orchestration -> ExecResult
scripts/
  fetch_si_ideas.py      # gdown 3 idea archives + exec archive, join human scores -> data/si_ideas/
  check_executability.py # run classifier over all 43 -> data/si_ideas/executability.jsonl
  run_execution_gym.py   # top-level runner -> data/eval/exec_<ts>/{results.jsonl,summary.json}
docker/
  Dockerfile.exec        # execution sandbox image
tests/unit/
  test_execution_types.py
  test_execution_task.py
  test_execution_sandbox_local.py
  test_execution_sandbox_docker.py
  test_execution_executability.py
  test_execution_agent.py
  test_execution_scorer.py
  test_execution_faithfulness.py
  test_execution_evaluate.py
  test_scripts_fetch_si_ideas.py
  test_scripts_run_execution_gym.py
```

**Reuse contracts (verified against the codebase):**
- `from papergym.llm import LLMClient, ChatReply, ToolCall` — `chat(messages, temperature=0.0, response_format=None) -> str`, `chat_with_tools(messages, tools, temperature, tool_choice="auto") -> ChatReply`, cumulative `total_prompt_tokens`/`total_completion_tokens`.
- `from papergym.agents.tool_loop import run_tool_loop, LoopResult` — `run_tool_loop(*, llm, messages, tools, dispatch, max_steps, temperature=0.3, on_message=None) -> LoopResult`. Dispatch keys are the **schema** names (capitalized). Tool fns carry a `.schema` attribute.
- `from papergym.agents.base import PromptLoader` — `PromptLoader(Path)` = filesystem dir, `PromptLoader("dotted.pkg")` = package; `.render(name, **fields) -> [system,user] messages` from `<name>.yaml`.
- `from eval.common import judge_axis, parse_axis, AxisScore, CostSnapshot, cost_summary` — `judge_axis(*, llm, prompts, prompt_name, **fields) -> AxisScore`; `AxisScore(score:int, reasoning:str)`; **score=0 is the parse-failure sentinel — treat as missing**; prompt YAML must end with a `Score: <1-5>` line.
- Script conventions (`scripts/accumulate_one.py`, `scripts/ideation_eval.py`): `load_dotenv(override=True)` at import; argparse `main(argv=None)`; `_log(path, **fields)` appends one JSON line; timestamped `run_dir = output_dir / f"exec_{time.strftime('%Y%m%dT%H%M%S')}"`; stream `results.jsonl` with `flush()`; final `summary.json` via `json.dumps(indent=2)`.

---

## Task 1: Core datatypes

**Files:**
- Create: `src/papergym/execution/__init__.py`
- Create: `src/papergym/execution/types.py`
- Test: `tests/unit/test_execution_types.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_execution_types.py
from papergym.execution.types import IdeaSpec, RunArtifact, ExecResult


def test_ideaspec_roundtrips_and_defaults():
    idea = IdeaSpec(idea_id="Math_3_Human", condition="Human", topic="Math",
                    title="Self-improving Memory", proposal_text="...method...")
    assert idea.human_exec_scores == {}            # default empty
    d = idea.to_dict()
    assert d["idea_id"] == "Math_3_Human"
    assert IdeaSpec.from_dict(d) == idea


def test_execresult_marks_missing_effectiveness_when_metric_failed():
    run = RunArtifact(status="natural_end", code="print(1)", stdout="ok",
                      predictions=[{"id": "0", "pred": "5"}], steps=2)
    res = ExecResult(idea_id="X", task_id="gsm8k_accuracy",
                     baseline_metric=0.40, method_metric=None,
                     effectiveness=None, faithfulness_score=0,
                     run=run, cost={})
    assert res.effectiveness is None               # None propagates (not 0.0)
    assert res.to_dict()["method_metric"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_execution_types.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'papergym.execution'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/papergym/execution/__init__.py
from .types import IdeaSpec, RunArtifact, ExecResult

__all__ = ["IdeaSpec", "RunArtifact", "ExecResult"]
```

```python
# src/papergym/execution/types.py
"""Core datatypes shared across the execution gym."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class IdeaSpec:
    """One research-idea proposal to execute, plus any known human-execution
    outcome (for validation in P2). human_exec_scores maps metric name ->
    mean human score, e.g. {"overall": 3.0, "novelty": 4.1}."""
    idea_id: str
    condition: str            # "AI" | "Human" | "AI_Rerank"
    topic: str
    title: str
    proposal_text: str
    human_exec_scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IdeaSpec":
        return cls(
            idea_id=d["idea_id"], condition=d["condition"], topic=d["topic"],
            title=d["title"], proposal_text=d["proposal_text"],
            human_exec_scores=d.get("human_exec_scores", {}) or {},
        )


@dataclass
class RunArtifact:
    """Output of one agent execution run inside the sandbox."""
    status: str                                  # mirrors LoopResult.status
    code: str = ""                               # the method.py the agent wrote
    stdout: str = ""
    predictions: list = field(default_factory=list)   # [{"id","pred"}, ...]
    steps: int = 0
    trace: list = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecResult:
    """Final per-idea result. effectiveness/method_metric are Optional:
    None means the run failed to produce a scorable artifact (NOT 0.0).
    faithfulness_score uses the parse_axis 1-5 scale with 0 = parse failure."""
    idea_id: str
    task_id: str
    baseline_metric: Optional[float]
    method_metric: Optional[float]
    effectiveness: Optional[float]               # method_metric - baseline_metric
    faithfulness_score: int
    run: RunArtifact
    cost: dict

    def to_dict(self) -> dict:
        d = asdict(self)
        return d
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_execution_types.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/papergym/execution/__init__.py src/papergym/execution/types.py tests/unit/test_execution_types.py
git commit -m "feat(execution): core datatypes (IdeaSpec/RunArtifact/ExecResult)"
```

---

## Task 2: Si et al. data fetch + join

**Files:**
- Create: `scripts/fetch_si_ideas.py`
- Test: `tests/unit/test_scripts_fetch_si_ideas.py`

Pulls the 3 idea-proposal archives (Human / AI / AI_Rerank, Google Drive links in `AI-Researcher/README.md` L157–163) and joins each idea's mean human **execution** scores from `AI-Researcher/reviews_execution/data_points_all_execution.json` (column-oriented: keys are field names, values are length-181 lists; join key `idea_id`, e.g. `Safety_2_AI`). Writes one `data/si_ideas/<idea_id>.json` per idea (an `IdeaSpec` dict). Download is isolated behind `_download(file_id, dest)` so tests stub it.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_scripts_fetch_si_ideas.py
import json, sys
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import fetch_si_ideas as fsi  # noqa: E402


def test_join_human_exec_scores_means_per_idea():
    # column-oriented like the real file: two reviews of one idea
    exec_data = {
        "idea_id": ["Math_3_Human", "Math_3_Human", "Safety_2_AI"],
        "overall_score": [3, 5, 1],
        "novelty_score": [4, 4, 2],
    }
    means = fsi._mean_exec_scores(exec_data)
    assert means["Math_3_Human"]["overall"] == 4.0     # (3+5)/2
    assert means["Math_3_Human"]["novelty"] == 4.0
    assert means["Safety_2_AI"]["overall"] == 1.0


def test_writes_one_ideaspec_json_per_idea(tmp_path):
    proposals = {"Math_3_Human": "Self-improving memory method ..."}
    means = {"Math_3_Human": {"overall": 4.0}}
    meta = {"Math_3_Human": {"condition": "Human", "topic": "Math",
                             "title": "Self-improving Memory"}}
    out = tmp_path / "si_ideas"
    fsi._write_ideaspecs(proposals, means, meta, out)
    rec = json.loads((out / "Math_3_Human.json").read_text())
    assert rec["condition"] == "Human"
    assert rec["human_exec_scores"]["overall"] == 4.0
    assert rec["proposal_text"].startswith("Self-improving")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_scripts_fetch_si_ideas.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fetch_si_ideas'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/fetch_si_ideas.py
"""Fetch Si et al. (NoviScl/AI-Researcher) idea proposals + join human
execution scores into data/si_ideas/<idea_id>.json (IdeaSpec dicts).

Idea-proposal archives (Google Drive, from AI-Researcher/README.md):
  Human:     1Z2Nd7WNNks-eCoqUgPzx1_ovYqU8OiPx
  AI:        1AFjSUCTj4wL081R2b17nVuNSs7xvzh8F
  AI_Rerank: 15r3TzFd-6dPXdSMx0shZ4q3ljDOEzp8I
  Execution: 1PpxeTz_-xHHcMXyUwv1Oed1avTaiD5vv
The exact in-archive layout is resolved at runtime by _load_proposals();
adjust _PROPOSAL_GLOB if the archive nests differently.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

DRIVE_IDS = {
    "Human": "1Z2Nd7WNNks-eCoqUgPzx1_ovYqU8OiPx",
    "AI": "1AFjSUCTj4wL081R2b17nVuNSs7xvzh8F",
    "AI_Rerank": "15r3TzFd-6dPXdSMx0shZ4q3ljDOEzp8I",
}
_EXEC_METRICS = ("overall_score", "novelty_score", "excitement_score",
                 "effectiveness_score", "soundness_score")


def _download(file_id: str, dest: Path) -> Path:
    """Download a Google Drive file via gdown. Isolated for test stubbing."""
    import gdown
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(dest), quiet=False)
    return dest


def _mean_exec_scores(exec_data: dict) -> dict:
    """Column-oriented review dump -> {idea_id: {metric: mean}}.
    Skips non-numeric / missing cells. Metric names drop the _score suffix."""
    ids = exec_data["idea_id"]
    by_idea = defaultdict(lambda: defaultdict(list))
    for metric in _EXEC_METRICS:
        col = exec_data.get(metric)
        if not col:
            continue
        for idea_id, val in zip(ids, col):
            try:
                by_idea[str(idea_id).strip()][metric].append(float(val))
            except (TypeError, ValueError):
                continue
    out = {}
    for idea_id, metrics in by_idea.items():
        out[idea_id] = {m.replace("_score", ""): statistics.mean(v)
                        for m, v in metrics.items() if v}
    return out


def _write_ideaspecs(proposals: dict, means: dict, meta: dict,
                     out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for idea_id, text in proposals.items():
        m = meta.get(idea_id, {})
        rec = {
            "idea_id": idea_id,
            "condition": m.get("condition", _condition_of(idea_id)),
            "topic": m.get("topic", _topic_of(idea_id)),
            "title": m.get("title", idea_id),
            "proposal_text": text,
            "human_exec_scores": means.get(idea_id, {}),
        }
        (out_dir / f"{idea_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2))
        n += 1
    return n


def _condition_of(idea_id: str) -> str:
    return "AI" if idea_id.endswith("_AI") else (
        "AI_Rerank" if idea_id.endswith("_AI_Rerank") else "Human")


def _topic_of(idea_id: str) -> str:
    return idea_id.split("_")[0]


def _load_proposals(extract_dir: Path) -> dict:
    """Map idea_id -> proposal text from an extracted archive. Each idea is a
    file whose stem is the idea_id (txt/md/json). JSON files are read for a
    'proposal'/'text'/'idea' field, else dumped whole."""
    proposals = {}
    for p in sorted(extract_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".txt", ".md", ".json"):
            continue
        idea_id = p.stem.strip()
        if p.suffix.lower() == ".json":
            try:
                obj = json.loads(p.read_text())
                text = (obj.get("proposal") or obj.get("text")
                        or obj.get("idea") or json.dumps(obj))
            except json.JSONDecodeError:
                text = p.read_text()
        else:
            text = p.read_text()
        proposals[idea_id] = text
    return proposals


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ai-researcher", type=Path,
                   default=Path("../AI-Researcher"),
                   help="Path to the NoviScl/AI-Researcher checkout.")
    p.add_argument("--out", type=Path, default=Path("data/si_ideas"))
    p.add_argument("--download-dir", type=Path, default=Path("data/si_raw"))
    args = p.parse_args(argv)

    exec_json = (args.ai_researcher / "reviews_execution"
                 / "data_points_all_execution.json")
    means = _mean_exec_scores(json.loads(exec_json.read_text()))

    proposals = {}
    for cond, fid in DRIVE_IDS.items():
        archive = args.download_dir / f"{cond}.zip"
        if not archive.exists():
            _download(fid, archive)
        extract_dir = args.download_dir / cond
        if not extract_dir.exists():
            import shutil
            shutil.unpack_archive(str(archive), str(extract_dir))
        proposals.update(_load_proposals(extract_dir))

    n = _write_ideaspecs(proposals, means, {}, args.out)
    print(f"wrote {n} ideas to {args.out}; "
          f"{len(means)} have human exec scores")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_scripts_fetch_si_ideas.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Add gdown dependency**

Edit `pyproject.toml` `[project].dependencies`, add `"gdown>=5.0"`. Run: `uv sync`

- [ ] **Step 6: Commit**

```bash
git add scripts/fetch_si_ideas.py tests/unit/test_scripts_fetch_si_ideas.py pyproject.toml
git commit -m "feat(execution): fetch Si et al. idea proposals + join human exec scores"
```

---

## Task 3: Executability classifier (extend ① to all 43)

**Files:**
- Create: `src/papergym/execution/executability.py`
- Create: `src/papergym/execution/prompts/executability.yaml`
- Create: `scripts/check_executability.py`
- Test: `tests/unit/test_execution_executability.py`

Ports the ① workflow rubric into a single LLM call: given a proposal, return whether it is inference-only auto-executable (no GPU training, no human study, public data, automatic metrics). The classifier returns a dict; the script runs it over every `data/si_ideas/*.json`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_execution_executability.py
import json
from unittest import mock
from papergym.execution.executability import classify_executability


def test_classify_parses_json_verdict():
    fake = mock.MagicMock()
    fake.chat.return_value = json.dumps({
        "gym_executable": True, "requires_gpu_training": False,
        "requires_human_eval": False, "blockers": [],
        "reasoning": "pure prompting"})
    out = classify_executability(proposal="prompt the model to ...", llm=fake)
    assert out["gym_executable"] is True
    assert out["blockers"] == []
    # called with JSON response_format and temperature 0
    assert fake.chat.call_args.kwargs["response_format"] == {"type": "json_object"}


def test_classify_defaults_false_on_bad_json():
    fake = mock.MagicMock()
    fake.chat.return_value = "not json"
    out = classify_executability(proposal="x", llm=fake)
    assert out["gym_executable"] is False
    assert "parse_error" in out["blockers"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_execution_executability.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/papergym/execution/prompts/executability.yaml
system: |
  You assess whether a research idea can be executed AUTOMATICALLY by an LLM
  agent that only makes inference-time API calls (NO GPU training, NO human
  annotators, public/standard datasets, automatic metrics).
  Reply with a single JSON object with keys:
  gym_executable (bool), requires_gpu_training (bool),
  requires_human_eval (bool), blockers (list of strings), reasoning (string).
  Set gym_executable true ONLY if it needs no model training, no human raters,
  and uses public data with automatic metrics.
user: |
  Idea proposal:
  {{ proposal }}
```

```python
# src/papergym/execution/executability.py
"""Single-call inference-time executability classifier (ports the ① rubric)."""
from __future__ import annotations

import json
from pathlib import Path

from papergym.agents.base import PromptLoader
from papergym.llm import LLMClient

_PROMPTS = PromptLoader(Path(__file__).resolve().parent / "prompts")


def classify_executability(*, proposal: str, llm: LLMClient) -> dict:
    messages = _PROMPTS.render("executability", proposal=proposal)
    raw = llm.chat(messages, temperature=0.0,
                   response_format={"type": "json_object"})
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {"gym_executable": False, "requires_gpu_training": None,
                "requires_human_eval": None, "blockers": ["parse_error"],
                "reasoning": raw[:200]}
    obj.setdefault("gym_executable", False)
    obj.setdefault("blockers", [])
    return obj
```

Note: `classify_executability` takes `llm` as a keyword. The test calls it positionally-by-keyword (`llm=fake`); keep the signature keyword-only friendly: `def classify_executability(*, proposal, llm)`. Update the first test's call to `classify_executability(proposal="...", llm=fake)` (already written that way).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_execution_executability.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write the runner over all ideas**

```python
# scripts/check_executability.py
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
```

- [ ] **Step 6: Commit**

```bash
git add src/papergym/execution/executability.py src/papergym/execution/prompts/executability.yaml scripts/check_executability.py tests/unit/test_execution_executability.py
git commit -m "feat(execution): inference-time executability classifier + 43-idea runner"
```

---

## Task 4: Task abstraction + GSM8K instance

**Files:**
- Create: `src/papergym/execution/task.py`
- Test: `tests/unit/test_execution_task.py`

A `Task` provides: the dataset (`load_examples`), the **baseline** runner (direct prompting), the objective `score(predictions)` metric, and a `manifest()` the agent reads. Predictions are `[{"id","pred"}]`. `GSM8KAccuracyTask` uses exact-match on the parsed integer answer.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_execution_task.py
from unittest import mock
from papergym.execution.task import GSM8KAccuracyTask, TASKS


def _examples():
    return [{"id": "0", "question": "2+2?", "answer": "4"},
            {"id": "1", "question": "3+5?", "answer": "8"}]


def test_gsm8k_score_is_exact_match_accuracy():
    task = GSM8KAccuracyTask(n_examples=2)
    task._examples = _examples()                      # inject fixtures
    acc = task.score([{"id": "0", "pred": "4"}, {"id": "1", "pred": "7"}])
    assert acc == 0.5


def test_gsm8k_parse_answer_extracts_final_integer():
    task = GSM8KAccuracyTask()
    assert task.parse_pred("The answer is 42.") == "42"
    assert task.parse_pred("#### 8") == "8"
    assert task.parse_pred("no number here") == ""


def test_baseline_calls_llm_per_example_and_scores():
    task = GSM8KAccuracyTask(n_examples=2)
    task._examples = _examples()
    llm = mock.MagicMock()
    llm.chat.side_effect = ["4", "8"]                 # both correct
    acc = task.run_baseline(llm)
    assert acc == 1.0
    assert llm.chat.call_count == 2


def test_registry_contains_gsm8k():
    assert "gsm8k_accuracy" in TASKS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_execution_task.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/papergym/execution/task.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_execution_task.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Add datasets dependency**

Edit `pyproject.toml` `[project].dependencies`, add `"datasets>=2.18"`. Run: `uv sync`

- [ ] **Step 6: Commit**

```bash
git add src/papergym/execution/task.py tests/unit/test_execution_task.py pyproject.toml
git commit -m "feat(execution): Task abstraction + GSM8K accuracy task"
```

---

## Task 5: Sandbox protocol + LocalSandbox

**Files:**
- Create: `src/papergym/execution/sandbox.py`
- Test: `tests/unit/test_execution_sandbox_local.py`

`Sandbox` lifecycle mirrors `PaperEnv` (cheap `__init__`, side effects in `reset()`, teardown in `close()`; usable as a context manager). It exposes `write_file(rel, content)`, `read_file(rel)`, `run_python(rel, timeout)` returning `(exit_code, stdout, stderr)`. `LocalSandbox` runs in an isolated tmp workdir via `subprocess` (same transport as the existing `bash` tool) — for dev/tests only; not isolated.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_execution_sandbox_local.py
from papergym.execution.sandbox import LocalSandbox


def test_init_is_cheap_no_io(tmp_path):
    sb = LocalSandbox(work_root=tmp_path / "run1")
    assert not (tmp_path / "run1").exists()           # __init__ does no I/O


def test_write_run_read_roundtrip(tmp_path):
    with LocalSandbox(work_root=tmp_path / "run1") as sb:
        sb.write_file("m.py", "open('out.txt','w').write('hi'); print('done')")
        rc, out, err = sb.run_python("m.py", timeout=30)
        assert rc == 0 and "done" in out
        assert sb.read_file("out.txt") == "hi"


def test_timeout_returns_124(tmp_path):
    with LocalSandbox(work_root=tmp_path / "run2") as sb:
        sb.write_file("loop.py", "while True: pass")
        rc, out, err = sb.run_python("loop.py", timeout=1)
        assert rc == 124
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_execution_sandbox_local.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/papergym/execution/sandbox.py
"""Isolated experiment sandboxes for the execution gym.

PaperGym has NO execution sandbox today (its `bash` tool is a bare host
subprocess). LocalSandbox keeps that transport for dev/tests; DockerSandbox
(Task 6) adds real isolation for untrusted agent-written code.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Protocol


class Sandbox(Protocol):
    def reset(self) -> None: ...
    def close(self) -> None: ...
    def write_file(self, rel: str, content: str) -> None: ...
    def read_file(self, rel: str) -> str: ...
    def run_python(self, rel: str, timeout: int = 600) -> tuple[int, str, str]: ...


class LocalSandbox:
    """Runs in an isolated host tmp workdir. NOT secure isolation — dev/tests
    only. __init__ does no I/O (mirrors PaperEnv); reset() creates the dir."""

    def __init__(self, *, work_root: Path):
        self.work_root = Path(work_root)
        self._ready = False

    def __enter__(self) -> "LocalSandbox":
        self.reset()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def reset(self) -> None:
        self.work_root.mkdir(parents=True, exist_ok=True)
        self._ready = True

    def close(self) -> None:
        self._ready = False  # local dir is left for inspection; runner cleans up

    def _ensure(self) -> None:
        if not self._ready:
            self.reset()

    def write_file(self, rel: str, content: str) -> None:
        self._ensure()
        p = self.work_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def read_file(self, rel: str) -> str:
        return (self.work_root / rel).read_text(encoding="utf-8")

    def run_python(self, rel: str, timeout: int = 600) -> tuple[int, str, str]:
        self._ensure()
        try:
            r = subprocess.run(["python", rel], cwd=str(self.work_root),
                               capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "timeout"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_execution_sandbox_local.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/papergym/execution/sandbox.py tests/unit/test_execution_sandbox_local.py
git commit -m "feat(execution): Sandbox protocol + LocalSandbox (dev transport)"
```

---

## Task 6: DockerSandbox + execution image

**Files:**
- Modify: `src/papergym/execution/sandbox.py`
- Create: `docker/Dockerfile.exec`
- Test: `tests/unit/test_execution_sandbox_docker.py`

Same `Sandbox` interface, but `run_python` executes inside a Docker container with the workdir bind-mounted and provider env forwarded (mirrors `run_accumulator._spawn_one`). Docker calls go through `subprocess`, so tests mock `subprocess.run`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_execution_sandbox_docker.py
from pathlib import Path
from unittest import mock
from papergym.execution import sandbox as sb_mod
from papergym.execution.sandbox import DockerSandbox


def test_run_python_invokes_docker_run_with_mount(tmp_path):
    sb = DockerSandbox(work_root=tmp_path / "run", image="papergym-exec:test")
    sb.reset()
    fake = mock.MagicMock(returncode=0, stdout="ok", stderr="")
    with mock.patch.object(sb_mod.subprocess, "run", return_value=fake) as run:
        rc, out, err = sb.run_python("m.py", timeout=120)
    assert rc == 0 and out == "ok"
    argv = run.call_args.args[0]
    assert argv[0] == "docker" and "run" in argv
    # workdir bind-mounted to /work and image present
    assert any(str(sb.work_root) + ":/work" in a for a in argv)
    assert "papergym-exec:test" in argv


def test_forwards_provider_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LITELLM_MODEL", "gpt-5")
    sb = DockerSandbox(work_root=tmp_path / "run", image="img")
    sb.reset()
    fake = mock.MagicMock(returncode=0, stdout="", stderr="")
    with mock.patch.object(sb_mod.subprocess, "run", return_value=fake) as run:
        sb.run_python("m.py")
    argv = run.call_args.args[0]
    joined = " ".join(argv)
    assert "OPENAI_API_KEY" in joined and "LITELLM_MODEL" in joined
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_execution_sandbox_docker.py -v`
Expected: FAIL with `ImportError: cannot import name 'DockerSandbox'`

- [ ] **Step 3: Add DockerSandbox to sandbox.py**

```python
# append to src/papergym/execution/sandbox.py
import os

_FORWARDED_ENV = ("OPENAI_API_KEY", "OPENAI_API_BASE", "ANTHROPIC_API_KEY",
                  "LITELLM_MODEL", "EMBEDDING_MODEL")


class DockerSandbox:
    """Runs agent-written code inside a Docker container. workdir is bind-
    mounted at /work; provider env is forwarded so in-container code can call
    the gym's LLM. __init__ does no I/O; reset() creates the host workdir."""

    def __init__(self, *, work_root: Path, image: str = "papergym-exec:latest",
                 network: str = "bridge"):
        self.work_root = Path(work_root)
        self.image = image
        self.network = network
        self._ready = False

    def __enter__(self) -> "DockerSandbox":
        self.reset(); return self

    def __exit__(self, *exc) -> None:
        self.close()

    def reset(self) -> None:
        self.work_root.mkdir(parents=True, exist_ok=True)
        self._ready = True

    def close(self) -> None:
        self._ready = False

    def write_file(self, rel: str, content: str) -> None:
        if not self._ready:
            self.reset()
        p = self.work_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def read_file(self, rel: str) -> str:
        return (self.work_root / rel).read_text(encoding="utf-8")

    def _env_flags(self) -> list[str]:
        flags = []
        for k in _FORWARDED_ENV:
            if os.environ.get(k):
                flags += ["-e", f"{k}={os.environ[k]}"]
        return flags

    def run_python(self, rel: str, timeout: int = 600) -> tuple[int, str, str]:
        if not self._ready:
            self.reset()
        argv = (["docker", "run", "--rm", f"--network={self.network}",
                 "-v", f"{self.work_root}:/work", "-w", "/work"]
                + self._env_flags()
                + [self.image, "python", rel])
        try:
            r = subprocess.run(argv, capture_output=True, text=True,
                               timeout=timeout)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "timeout"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_execution_sandbox_docker.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write the execution image**

```dockerfile
# docker/Dockerfile.exec
FROM python:3.11-slim
WORKDIR /papergym
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir datasets scikit-learn numpy
# Pre-stage GSM8K so parallel sandboxes don't race the HF Hub.
RUN python -c "from datasets import load_dataset; load_dataset('openai/gsm8k','main')"
WORKDIR /work
```

- [ ] **Step 6: Build the image and commit**

Run: `docker build -f docker/Dockerfile.exec -t papergym-exec:latest .`
Expected: image built (or skip if Docker unavailable; LocalSandbox covers dev).

```bash
git add src/papergym/execution/sandbox.py docker/Dockerfile.exec tests/unit/test_execution_sandbox_docker.py
git commit -m "feat(execution): DockerSandbox + execution image"
```

---

## Task 7: Execution action space + ExecutionAgent

**Files:**
- Create: `src/papergym/execution/agent.py`
- Test: `tests/unit/test_execution_agent.py`

Defines four tools (`WriteFile`, `RunPython`, `ReadFile`, `Finish`) each carrying a `.schema` (the existing pattern), bound to a sandbox via a dispatch closure, driven by `run_tool_loop`. The agent is told to implement the idea as `method.py` that writes `predictions.json` (`[{"id","pred"}]`). `ExecutionAgent.run(idea, task, sandbox)` returns a `RunArtifact`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_execution_agent.py
import json
from pathlib import Path
from unittest import mock
from papergym.execution.agent import ExecutionAgent, EXECUTION_TOOLS
from papergym.execution.sandbox import LocalSandbox
from papergym.execution.task import GSM8KAccuracyTask
from papergym.execution.types import IdeaSpec
from papergym.llm import ChatReply, ToolCall


def _idea():
    return IdeaSpec(idea_id="Math_3_Human", condition="Human", topic="Math",
                    title="t", proposal_text="Prefix each question with 'Hint:'")


def test_tools_expose_capitalized_schema_names():
    names = {t.schema["function"]["name"] for t in EXECUTION_TOOLS}
    assert names == {"WriteFile", "RunPython", "ReadFile", "Finish"}


def test_agent_writes_runs_and_collects_predictions(tmp_path):
    task = GSM8KAccuracyTask(n_examples=1)
    task._examples = [{"id": "0", "question": "2+2?", "answer": "4"}]
    # method.py that writes a correct prediction without needing an LLM
    method = ("import json;"
              "json.dump([{'id':'0','pred':'4'}], open('predictions.json','w'))")
    calls = [
        ChatReply(content="", tool_calls=[ToolCall(
            id="a", name="WriteFile",
            arguments=json.dumps({"path": "method.py", "content": method}))],
            raw_message={"role": "assistant"}),
        ChatReply(content="", tool_calls=[ToolCall(
            id="b", name="RunPython",
            arguments=json.dumps({"path": "method.py"}))],
            raw_message={"role": "assistant"}),
        ChatReply(content="", tool_calls=[ToolCall(
            id="c", name="Finish", arguments=json.dumps({"summary": "done"}))],
            raw_message={"role": "assistant"}),
        ChatReply(content="ok", tool_calls=[], raw_message={"role": "assistant"}),
    ]
    llm = mock.MagicMock()
    llm.chat_with_tools.side_effect = calls
    with LocalSandbox(work_root=tmp_path / "run") as sb:
        agent = ExecutionAgent(llm=llm, max_steps=10)
        run = agent.run(idea=_idea(), task=task, sandbox=sb)
    assert run.predictions == [{"id": "0", "pred": "4"}]
    assert "method.py" in run.code
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_execution_agent.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/papergym/execution/agent.py
"""Execution agent: drives run_tool_loop with a write/run/read/finish action
space so an LLM can implement and run an idea's experiment in a sandbox."""
from __future__ import annotations

import json
from dataclasses import dataclass

from papergym.agents.tool_loop import run_tool_loop
from papergym.llm import LLMClient

from .sandbox import Sandbox
from .task import Task
from .types import IdeaSpec, RunArtifact


# ---- tool functions: each carries a .schema (capitalized name) ----
def _write_file(path: str, content: str, *, sandbox: Sandbox) -> str:
    sandbox.write_file(path, content)
    return f"wrote {path} ({len(content)} chars)"


def _run_python(path: str, *, sandbox: Sandbox, timeout: int = 600) -> str:
    rc, out, err = sandbox.run_python(path, timeout=timeout)
    return f"exit={rc}\nstdout:\n{out[:4000]}\nstderr:\n{err[:2000]}"


def _read_file(path: str, *, sandbox: Sandbox) -> str:
    try:
        return sandbox.read_file(path)[:4000]
    except FileNotFoundError:
        return f"[no such file] {path}"


def _finish(summary: str = "") -> str:
    return f"FINISHED: {summary}"


_write_file.schema = {"type": "function", "function": {
    "name": "WriteFile", "description": "Write a text file in the sandbox.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"]}}}
_run_python.schema = {"type": "function", "function": {
    "name": "RunPython",
    "description": "Run a python file in the sandbox. RUN EXPERIMENTS here.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "timeout": {"type": "integer", "default": 600}},
        "required": ["path"]}}}
_read_file.schema = {"type": "function", "function": {
    "name": "ReadFile", "description": "Read a file from the sandbox.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}}
_finish.schema = {"type": "function", "function": {
    "name": "Finish", "description": "Call when predictions.json is written.",
    "parameters": {"type": "object", "properties": {
        "summary": {"type": "string"}}}}}

EXECUTION_TOOL_FNS = (_write_file, _run_python, _read_file, _finish)
EXECUTION_TOOLS = list(EXECUTION_TOOL_FNS)


_SYSTEM = """You are a research-execution agent. Implement the METHOD from the
idea proposal as `method.py`, run it, and write predictions to
`predictions.json` in the format {fmt}. You may call the gym's LLM from inside
method.py via `from papergym.llm import LLMClient; LLMClient().chat(msgs)`.
The dataset is available via the task description below. Do NOT change the
proposed method; only implement its experiment. Call Finish when
predictions.json exists. NO model training — inference-time only."""


@dataclass
class _Tools:
    sandbox: Sandbox

    def dispatch(self, name: str, args: dict) -> str:
        if name == "WriteFile":
            return _write_file(args["path"], args["content"], sandbox=self.sandbox)
        if name == "RunPython":
            return _run_python(args["path"], sandbox=self.sandbox,
                               timeout=int(args.get("timeout", 600)))
        if name == "ReadFile":
            return _read_file(args["path"], sandbox=self.sandbox)
        if name == "Finish":
            return _finish(args.get("summary", ""))
        return f"[unknown tool] {name}"


class ExecutionAgent:
    def __init__(self, *, llm: LLMClient, max_steps: int = 40,
                 temperature: float = 0.2):
        self.llm = llm
        self.max_steps = max_steps
        self.temperature = temperature

    def run(self, *, idea: IdeaSpec, task: Task, sandbox: Sandbox) -> RunArtifact:
        tools = _Tools(sandbox)
        system = _SYSTEM.format(fmt=task.manifest()["predictions_format"])
        # stage the dataset for the agent's method.py to read
        sandbox.write_file("examples.json",
                           json.dumps([{"id": e["id"], "question": e["question"]}
                                       for e in task.examples()]))
        user = (f"IDEA PROPOSAL:\n{idea.proposal_text}\n\n"
                f"TASK: {json.dumps(task.manifest())}\n"
                f"Dataset rows (id, question) are in examples.json in your "
                f"sandbox. Write predictions for every row.")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        result = run_tool_loop(
            llm=self.llm, messages=messages,
            tools=[fn.schema for fn in EXECUTION_TOOL_FNS],
            dispatch=tools.dispatch, max_steps=self.max_steps,
            temperature=self.temperature)

        code = ""
        try:
            code = "method.py\n" + sandbox.read_file("method.py")
        except FileNotFoundError:
            pass
        predictions = []
        try:
            predictions = json.loads(sandbox.read_file("predictions.json"))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return RunArtifact(status=result.status, code=code,
                           predictions=predictions, steps=result.steps,
                           trace=result.trace, error=result.reason)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_execution_agent.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/papergym/execution/agent.py tests/unit/test_execution_agent.py
git commit -m "feat(execution): execution action space + ExecutionAgent"
```

---

## Task 8: Objective scorer

**Files:**
- Create: `src/papergym/execution/scorer.py`
- Test: `tests/unit/test_execution_scorer.py`

`score_effectiveness(task, run, baseline_metric)` computes `method_metric = task.score(run.predictions)` and `effectiveness = method_metric - baseline_metric`, returning `None` for both when the run produced no predictions (so a failed run is *missing*, not 0.0).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_execution_scorer.py
from papergym.execution.scorer import score_effectiveness
from papergym.execution.task import GSM8KAccuracyTask
from papergym.execution.types import RunArtifact


def _task():
    t = GSM8KAccuracyTask(n_examples=2)
    t._examples = [{"id": "0", "question": "q", "answer": "4"},
                   {"id": "1", "question": "q", "answer": "8"}]
    return t


def test_effectiveness_is_method_minus_baseline():
    run = RunArtifact(status="natural_end",
                      predictions=[{"id": "0", "pred": "4"},
                                   {"id": "1", "pred": "8"}])
    method_m, eff = score_effectiveness(_task(), run, baseline_metric=0.5)
    assert method_m == 1.0 and eff == 0.5


def test_failed_run_yields_none_not_zero():
    run = RunArtifact(status="max_steps_exceeded", predictions=[])
    method_m, eff = score_effectiveness(_task(), run, baseline_metric=0.5)
    assert method_m is None and eff is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_execution_scorer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```python
# src/papergym/execution/scorer.py
"""Objective effectiveness scoring: method metric vs baseline."""
from __future__ import annotations

from typing import Optional

from .task import Task
from .types import RunArtifact


def score_effectiveness(task: Task, run: RunArtifact,
                        baseline_metric: float
                        ) -> tuple[Optional[float], Optional[float]]:
    """Returns (method_metric, effectiveness). Both None when the run
    produced no predictions — a failed run is missing data, not a 0 score."""
    if not run.predictions:
        return None, None
    method_metric = task.score(run.predictions)
    return method_metric, method_metric - baseline_metric
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_execution_scorer.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/papergym/execution/scorer.py tests/unit/test_execution_scorer.py
git commit -m "feat(execution): objective effectiveness scorer"
```

---

## Task 9: Faithfulness judge

**Files:**
- Create: `src/papergym/execution/faithfulness.py`
- Create: `src/papergym/execution/prompts/faithfulness.yaml`
- Test: `tests/unit/test_execution_faithfulness.py`

Reuses `judge_axis`: a judge LLM rates how faithfully the executed `method.py` implements the proposal's method (1-5; the same role as Si et al.'s `faithfulness_score`). Returns `AxisScore` (0 = parse failure).

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_execution_faithfulness.py
from unittest import mock
from papergym.execution.faithfulness import judge_faithfulness


def test_returns_axisscore_from_judge_text():
    judge = mock.MagicMock()
    judge.chat.return_value = "The code matches the proposal.\nScore: 4"
    out = judge_faithfulness(proposal="prompt method", code="method.py ...",
                             judge_llm=judge)
    assert out.score == 4
    assert judge.chat.call_args.kwargs["temperature"] == 0.0


def test_parse_failure_is_zero_sentinel():
    judge = mock.MagicMock()
    judge.chat.return_value = "no score line here"
    out = judge_faithfulness(proposal="p", code="c", judge_llm=judge)
    assert out.score == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_execution_faithfulness.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write minimal implementation**

```yaml
# src/papergym/execution/prompts/faithfulness.yaml
system: |
  You judge whether executed code faithfully implements a proposed research
  method, WITHOUT adding or removing core methodological components.
  Reason briefly, then end with exactly one line:
  Score: <1-5>
  where 5 = fully faithful, 1 = the executed method differs fundamentally.
user: |
  PROPOSED METHOD:
  {{ proposal }}

  EXECUTED CODE:
  {{ code }}
```

```python
# src/papergym/execution/faithfulness.py
"""Faithfulness judge: does the executed code implement the proposed method?"""
from __future__ import annotations

from pathlib import Path

from eval.common import AxisScore, judge_axis
from papergym.agents.base import PromptLoader
from papergym.llm import LLMClient

_PROMPTS = PromptLoader(Path(__file__).resolve().parent / "prompts")


def judge_faithfulness(*, proposal: str, code: str,
                       judge_llm: LLMClient) -> AxisScore:
    return judge_axis(llm=judge_llm, prompts=_PROMPTS,
                      prompt_name="faithfulness",
                      proposal=proposal[:8000], code=code[:8000])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_execution_faithfulness.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/papergym/execution/faithfulness.py src/papergym/execution/prompts/faithfulness.yaml tests/unit/test_execution_faithfulness.py
git commit -m "feat(execution): faithfulness judge (proposal vs executed code)"
```

---

## Task 10: Orchestration — run_one_idea

**Files:**
- Create: `eval/execution/__init__.py`
- Create: `eval/execution/evaluate.py`
- Test: `tests/unit/test_execution_evaluate.py`

Wires baseline → agent execution → effectiveness → faithfulness → cost into one `ExecResult`. Generator and judge are separate `LLMClient`s; cost via `CostSnapshot`/`cost_summary`.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_execution_evaluate.py
from unittest import mock
from pathlib import Path
from eval.execution.evaluate import run_one_idea
from papergym.execution.task import GSM8KAccuracyTask
from papergym.execution.types import IdeaSpec, RunArtifact


def _idea():
    return IdeaSpec(idea_id="Math_3_Human", condition="Human", topic="Math",
                    title="t", proposal_text="method")


def test_run_one_idea_assembles_execresult(tmp_path):
    task = GSM8KAccuracyTask(n_examples=2)
    task._examples = [{"id": "0", "question": "q", "answer": "4"},
                      {"id": "1", "question": "q", "answer": "8"}]
    gen = mock.MagicMock(total_prompt_tokens=0, total_completion_tokens=0)
    judge = mock.MagicMock(total_prompt_tokens=0, total_completion_tokens=0)
    judge.chat.return_value = "faithful\nScore: 5"

    with mock.patch("eval.execution.evaluate.GSM8KAccuracyTask.run_baseline",
                    return_value=0.5), \
         mock.patch("eval.execution.evaluate.ExecutionAgent") as AgentCls:
        AgentCls.return_value.run.return_value = RunArtifact(
            status="natural_end", code="m",
            predictions=[{"id": "0", "pred": "4"}, {"id": "1", "pred": "8"}])
        res = run_one_idea(idea=_idea(), task=task, gen_llm=gen,
                           judge_llm=judge, work_root=tmp_path / "run")

    assert res.baseline_metric == 0.5
    assert res.method_metric == 1.0
    assert res.effectiveness == 0.5
    assert res.faithfulness_score == 5
    assert "total_cost_usd" in res.cost
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_execution_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.execution'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/execution/__init__.py
from .evaluate import run_one_idea

__all__ = ["run_one_idea"]
```

```python
# eval/execution/evaluate.py
"""Orchestrate one idea: baseline -> agent execution -> score -> faithfulness."""
from __future__ import annotations

import time
from pathlib import Path

from eval.common import CostSnapshot, cost_summary
from papergym.execution.agent import ExecutionAgent
from papergym.execution.faithfulness import judge_faithfulness
from papergym.execution.scorer import score_effectiveness
from papergym.execution.sandbox import LocalSandbox, DockerSandbox
from papergym.execution.task import Task, GSM8KAccuracyTask  # noqa: F401 (patch target)
from papergym.execution.types import ExecResult, IdeaSpec
from papergym.llm import LLMClient


def run_one_idea(*, idea: IdeaSpec, task: Task, gen_llm: LLMClient,
                 judge_llm: LLMClient, work_root: Path,
                 use_docker: bool = False, image: str = "papergym-exec:latest",
                 max_steps: int = 40) -> ExecResult:
    t0 = time.time()
    gen_before = CostSnapshot.of(gen_llm)
    judge_before = CostSnapshot.of(judge_llm)

    baseline_metric = task.run_baseline(gen_llm)

    sb = (DockerSandbox(work_root=work_root, image=image) if use_docker
          else LocalSandbox(work_root=work_root))
    with sb:
        run = ExecutionAgent(llm=gen_llm, max_steps=max_steps).run(
            idea=idea, task=task, sandbox=sb)

    method_metric, effectiveness = score_effectiveness(task, run, baseline_metric)
    faith = judge_faithfulness(proposal=idea.proposal_text, code=run.code,
                               judge_llm=judge_llm)

    cost = cost_summary(
        judge_before=judge_before, judge_after=CostSnapshot.of(judge_llm),
        gen_before=gen_before, gen_after=CostSnapshot.of(gen_llm),
        wall_clock_s=time.time() - t0)

    return ExecResult(idea_id=idea.idea_id, task_id=task.task_id,
                      baseline_metric=baseline_metric,
                      method_metric=method_metric, effectiveness=effectiveness,
                      faithfulness_score=faith.score, run=run, cost=cost)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_execution_evaluate.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add eval/execution/__init__.py eval/execution/evaluate.py tests/unit/test_execution_evaluate.py
git commit -m "feat(execution): run_one_idea orchestration"
```

---

## Task 11: Top-level runner

**Files:**
- Create: `scripts/run_execution_gym.py`
- Test: `tests/unit/test_scripts_run_execution_gym.py`

Argparse `main(argv=None)` mirroring `scripts/ideation_eval.py`: load one or more `IdeaSpec`s from `data/si_ideas/`, map topic→`Task`, run `run_one_idea`, stream `results.jsonl`, write `summary.json`. Enforces the self-bias guard (judge model ≠ generator model) like the existing scripts.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_scripts_run_execution_gym.py
import json, sys
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_execution_gym as reg  # noqa: E402

from papergym.execution.types import ExecResult, RunArtifact


def test_main_writes_results_and_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    ideas = tmp_path / "si_ideas"; ideas.mkdir()
    (ideas / "Math_3_Human.json").write_text(json.dumps({
        "idea_id": "Math_3_Human", "condition": "Human", "topic": "Math",
        "title": "t", "proposal_text": "method", "human_exec_scores": {}}))
    out = tmp_path / "eval"

    fake_res = ExecResult(idea_id="Math_3_Human", task_id="gsm8k_accuracy",
                          baseline_metric=0.5, method_metric=0.6,
                          effectiveness=0.1, faithfulness_score=4,
                          run=RunArtifact(status="natural_end"),
                          cost={"total_cost_usd": 0.0})
    with mock.patch.object(reg, "run_one_idea", return_value=fake_res), \
         mock.patch.object(reg, "LLMClient") as LC:
        LC.side_effect = [mock.MagicMock(model="gen"), mock.MagicMock(model="judge")]
        reg.main(["--ideas", str(ideas), "--output-dir", str(out),
                  "--n-examples", "2"])

    run_dir = next(out.glob("exec_*"))
    lines = (run_dir / "results.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["effectiveness"] == 0.1
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["n_ideas"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_scripts_run_execution_gym.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_execution_gym'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/run_execution_gym.py
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

# topic -> default Task for the MVP (extend as more Tasks land in Task registry).
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_scripts_run_execution_gym.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full execution-gym test suite**

Run: `uv run pytest tests/unit/test_execution_*.py tests/unit/test_scripts_fetch_si_ideas.py tests/unit/test_scripts_run_execution_gym.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/run_execution_gym.py tests/unit/test_scripts_run_execution_gym.py
git commit -m "feat(execution): top-level execution-gym runner"
```

---

## Task 12: Smoke run (manual, gated on real keys + Docker)

**Files:** none (verification only)

- [ ] **Step 1: Fetch data + executability over 43**

Run:
```bash
uv run python scripts/fetch_si_ideas.py --ai-researcher ../AI-Researcher
uv run python scripts/check_executability.py
uv run python scripts/prefetch_datasets.py        # pull HF (GSM8K) download forward — fail-fast, warms cache
```
Expected: `data/si_ideas/*.json` written; `executability.jsonl` reports the executable count (the full-43 extension of ①); `prefetched gsm8k_accuracy: N examples` confirms the HF cache is warm before any LLM spend.

- [ ] **Step 2: Execute one Math idea end-to-end**

Run:
```bash
uv run python scripts/run_execution_gym.py --limit 1 --n-examples 20
```
Expected: a `data/eval/exec_<ts>/` with `results.jsonl` (one record: `baseline_metric`, `method_metric`, `effectiveness`, `faithfulness_score`) and `summary.json`. This is the P1 success signal: **one inference-only idea executed + scored automatically.**

- [ ] **Step 3: Record outcome**

Append a one-line note to `docs/REPRODUCE.md` mapping "P1 smoke: execute one idea" → `scripts/run_execution_gym.py` → `summary.json`.

```bash
git add docs/REPRODUCE.md
git commit -m "docs: record P1 execution-gym smoke run"
```

---

## Self-Review

**Spec coverage (against `2026-06-07-execution-gym-design.md`):**
- §5 Execution Engine (action space, model-pool, standard harness) → Tasks 4–7. ✓
- §5 faithfulness control → Task 9 + faithfulness in Task 10. ✓
- §5.1 executability pre-check (extend ① to 43) → Task 3 + Task 12 Step 1. ✓
- §6 dual reward — *effectiveness only* in P1 (objective metric, Task 8). VRN novelty guard is **P3** (out of P1 scope). Noted gap-by-design.
- §7 E1 data dependency (Si et al. proposals + human exec scores) → Task 2. The E1 *analysis* (reproduce the gap) is **P2**.
- §11 reproducibility (timestamped run dirs, jsonl+summary, cost) → Tasks 10–12. ✓

**Deferred to later plans (intentional, not gaps):** VRN scorer (P3), E1 gap-reproduction analysis (P2), test-time feedback loop (P4), RL (P5), additional `Task`s beyond GSM8K (P2+ as coverage demands).

**Placeholder scan:** No TBD/TODO; every code step has complete code. ✓

**Type consistency:** `IdeaSpec`/`RunArtifact`/`ExecResult` (Task 1) used identically in Tasks 7/8/10/11. `Sandbox` interface (Task 5) implemented by `LocalSandbox`/`DockerSandbox` (Tasks 5/6) and consumed in Tasks 7/10. Tool dispatch keys are capitalized schema names (`WriteFile`/`RunPython`/`ReadFile`/`Finish`) consistently in Task 7. `score_effectiveness` returns `(method_metric, effectiveness)` and is called that way in Task 10. `judge_faithfulness(*, proposal, code, judge_llm)` signature matches its call in Task 10. ✓

**Known risk carried from spec §9:** `LocalSandbox` runs agent code on the host (not isolated) — acceptable for dev/tests; real runs must use `--use-docker` (Task 6). The execution-faithfulness confounder is measured (Task 9), not eliminated, in P1.
