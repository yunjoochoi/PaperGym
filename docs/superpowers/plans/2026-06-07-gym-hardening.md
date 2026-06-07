# Gym Hardening (P1.5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Commit steps are intentionally OMITTED — the human controls commits.

**Goal:** Make the execution gym's effectiveness number *trustworthy* and the sandbox *safe* by closing three validity/security holes found in P1: (#1) test-label leakage, (#2) unmetered in-sandbox LLM cost, (#3) sandbox path/host escape — plus materialize datasets into a repo-local dev/test split.

**Architecture:** One hardened-sandbox design fixes all three: (a) a filesystem path-jail confines all file I/O to the workspace; (b) provider API keys are NEVER placed in the sandbox — the only LLM path is a host-side **metering proxy** (`UsageMeter` + stdlib `http.server`) that records tokens/cost and enforces a budget; (c) datasets are materialized once into `data/datasets/<task>/{dev,test}.jsonl`, the agent receives only **test inputs** (no labels) plus a labeled **dev** split to develop on, and the host-side evaluator holds the test labels; (d) a static leakage guard scans the agent's code for dataset-reload / direct-provider-call patterns as defense-in-depth.

**Tech Stack:** Python 3.11, stdlib `http.server`/`urllib` (no new deps), `datasets` (materialize only), pytest + `unittest.mock`.

**Scope note (honest):** Unit tests cover all structural guarantees (no labels staged, no keys forwarded, path traversal rejected, usage metered, budget enforced, guard detects patterns, proxy/client logic). True network-egress isolation for `DockerSandbox` (blocking huggingface.co while allowing the proxy) is an OPS step verified manually in Task 6 — the code makes it *possible and default-restricted*, but the firewall/network policy itself is not unit-testable here.

---

## File Structure

```
src/papergym/execution/
  sandbox.py        # MODIFY: path-jail, drop provider keys, forward GYM_LLM_URL
  task.py           # MODIFY: DATA_ROOT, materialize(), load_split(), examples(split)
  agent.py          # MODIFY: stage test-inputs + labeled dev; prompt uses metered_llm_call
  scorer.py         # MODIFY: score against held-out test split; surface leakage flags
  integrity.py      # NEW: scan_for_leakage(code) -> list[str]
  metering.py       # NEW: UsageMeter (tokens/cost/budget) + BudgetExceeded
  llm_proxy.py      # NEW: handle_payload() + run_proxy() (stdlib http.server)
  gym_client.py     # NEW: metered_llm_call() used INSIDE the sandbox
eval/execution/
  evaluate.py       # MODIFY: start proxy, no keys in sandbox, fold metered cost, test split
scripts/
  prefetch_datasets.py  # MODIFY: materialize data/datasets/<task>/{dev,test}.jsonl
  run_execution_gym.py  # MODIFY: pass budget; (no behavior change otherwise)
docker/
  Dockerfile.exec   # MODIFY: do NOT bake labeled dataset; keep libs only
tests/unit/
  test_execution_sandbox_jail.py        # NEW
  test_execution_task_splits.py         # NEW
  test_execution_integrity.py           # NEW
  test_execution_metering.py            # NEW
  test_execution_llm_proxy.py           # NEW
  test_execution_gym_client.py          # NEW
  (existing sandbox/task/agent/evaluate tests updated as needed)
```

**Reuse contracts (verified):** `LLMClient.chat(messages, temperature=0.0)->str`, cumulative `total_prompt_tokens`/`total_completion_tokens`; `eval.common.CostSnapshot`/`cost_summary`/`GPT5_PRICING`. Tool loop and agent unchanged except the system prompt + staged files.

---

## Task 1: Path-jailed sandbox + no provider keys (#3)

**Files:**
- Modify: `src/papergym/execution/sandbox.py`
- Test: `tests/unit/test_execution_sandbox_jail.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_execution_sandbox_jail.py
import pytest
from papergym.execution.sandbox import LocalSandbox, DockerSandbox, SandboxPathError


def test_local_rejects_parent_traversal(tmp_path):
    with LocalSandbox(work_root=tmp_path / "run") as sb:
        with pytest.raises(SandboxPathError):
            sb.write_file("../escape.txt", "x")
        with pytest.raises(SandboxPathError):
            sb.read_file("../../etc/passwd")
        with pytest.raises(SandboxPathError):
            sb.run_python("../m.py")


def test_local_rejects_absolute_path(tmp_path):
    with LocalSandbox(work_root=tmp_path / "run") as sb:
        with pytest.raises(SandboxPathError):
            sb.write_file("/tmp/x.txt", "x")


def test_local_allows_in_workspace(tmp_path):
    with LocalSandbox(work_root=tmp_path / "run") as sb:
        sb.write_file("sub/m.py", "print('ok')")
        rc, out, _ = sb.run_python("sub/m.py")
        assert rc == 0 and "ok" in out


def test_docker_forwards_proxy_url_not_provider_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("GYM_LLM_URL", "http://host.docker.internal:9000")
    sb = DockerSandbox(work_root=tmp_path / "run", image="img")
    flags = sb._env_flags()
    joined = " ".join(flags)
    assert "GYM_LLM_URL" in joined
    assert "OPENAI_API_KEY" not in joined and "sk-secret" not in joined
```

- [ ] **Step 2: Run, expect FAIL** — `ImportError: cannot import name 'SandboxPathError'`.
Run: `uv run pytest tests/unit/test_execution_sandbox_jail.py -v`

- [ ] **Step 3: Implement** — edit `src/papergym/execution/sandbox.py`.

Add near the top (after imports):
```python
class SandboxPathError(ValueError):
    """Raised when a sandbox file path would escape the workspace."""


def _resolve_in(work_root: Path, rel: str) -> Path:
    """Resolve rel against work_root and confirm it stays inside it.
    Rejects absolute paths and parent-traversal."""
    if rel != rel.strip() or rel.startswith(("/", "\\")) or ":" in rel[:3]:
        raise SandboxPathError(f"absolute or suspicious path: {rel!r}")
    root = Path(work_root).resolve()
    target = (root / rel).resolve()
    if root != target and root not in target.parents:
        raise SandboxPathError(f"path escapes workspace: {rel!r}")
    return target
```

In `LocalSandbox`, replace the bodies of `write_file`, `read_file`, `run_python` to route through `_resolve_in`:
```python
    def write_file(self, rel: str, content: str) -> None:
        self._ensure()
        p = _resolve_in(self.work_root, rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def read_file(self, rel: str) -> str:
        return _resolve_in(self.work_root, rel).read_text(encoding="utf-8")

    def run_python(self, rel: str, timeout: int = 600) -> tuple[int, str, str]:
        self._ensure()
        _resolve_in(self.work_root, rel)          # validate; cwd is work_root
        try:
            r = subprocess.run(["python", rel], cwd=str(self.work_root),
                               capture_output=True, text=True, timeout=timeout)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "timeout"
```

In `DockerSandbox`: apply the SAME `_resolve_in` guard in `write_file`/`read_file` (host-side I/O), and change `_FORWARDED_ENV` + `_env_flags` to forward ONLY the gym proxy vars, NEVER provider keys:
```python
# replace the module-level _FORWARDED_ENV
_FORWARDED_ENV = ("GYM_LLM_URL", "GYM_JOB_TOKEN")   # NEVER provider keys
```
(`_env_flags` body stays the same — it loops over `_FORWARDED_ENV`.) Also guard `DockerSandbox.write_file`/`read_file` with `_resolve_in` exactly like LocalSandbox.

- [ ] **Step 4: Run, expect PASS (4 passed).** Then run the existing sandbox tests to catch regressions:
Run: `uv run pytest tests/unit/test_execution_sandbox_local.py tests/unit/test_execution_sandbox_docker.py tests/unit/test_execution_sandbox_jail.py -v`
NOTE: the existing `test_execution_sandbox_docker.py::test_forwards_provider_env` asserts OPENAI_API_KEY IS forwarded — that behavior is now intentionally REMOVED. Update that test to assert provider keys are NOT forwarded and `GYM_LLM_URL` IS (mirror `test_docker_forwards_proxy_url_not_provider_keys`). Re-run; expect all green.

---

## Task 2: Local dataset materialization + dev/test split (#1 structural, HF-local)

**Files:**
- Modify: `src/papergym/execution/task.py`
- Modify: `scripts/prefetch_datasets.py`
- Test: `tests/unit/test_execution_task_splits.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_execution_task_splits.py
import json
from papergym.execution.task import GSM8KAccuracyTask


def _write_splits(tmp_path):
    root = tmp_path / "gsm8k_accuracy"
    root.mkdir(parents=True)
    (root / "test.jsonl").write_text(
        '{"id":"0","question":"2+2?","answer":"4"}\n'
        '{"id":"1","question":"3+5?","answer":"8"}\n')
    (root / "dev.jsonl").write_text(
        '{"id":"d0","question":"1+1?","answer":"2"}\n')
    return tmp_path


def test_load_split_reads_local_jsonl(tmp_path):
    data_root = _write_splits(tmp_path)
    task = GSM8KAccuracyTask(n_examples=2, data_root=data_root)
    test = task.examples(split="test")
    assert [e["id"] for e in test] == ["0", "1"]
    dev = task.examples(split="dev")
    assert dev[0]["answer"] == "2"


def test_inputs_view_strips_answers(tmp_path):
    data_root = _write_splits(tmp_path)
    task = GSM8KAccuracyTask(n_examples=2, data_root=data_root)
    inputs = task.inputs(split="test")
    assert inputs == [{"id": "0", "question": "2+2?"},
                      {"id": "1", "question": "3+5?"}]
    assert all("answer" not in r for r in inputs)


def test_score_uses_named_split(tmp_path):
    data_root = _write_splits(tmp_path)
    task = GSM8KAccuracyTask(n_examples=2, data_root=data_root)
    acc = task.score([{"id": "0", "pred": "4"}, {"id": "1", "pred": "9"}],
                     split="test")
    assert acc == 0.5
```

- [ ] **Step 2: Run, expect FAIL** (TypeError: unexpected `data_root` / `split`).

- [ ] **Step 3: Implement** — edit `src/papergym/execution/task.py`.

Add at module top:
```python
import json
from pathlib import Path

DEFAULT_DATA_ROOT = Path("data/datasets")
```
Change `Task.__init__` and add split-aware data access (replace the data section of `Task`):
```python
    def __init__(self, n_examples: int = 50, seed: int = 0,
                 data_root: Path = DEFAULT_DATA_ROOT):
        self.n_examples = n_examples
        self.seed = seed
        self.data_root = Path(data_root)
        self._splits: dict = {}                      # split -> list

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
```
Update `score`, `run_baseline` to take `split="test"`:
```python
    def score(self, predictions: list, split: str = "test") -> float:
        gold = {ex["id"]: ex["answer"] for ex in self.examples(split)}
        if not gold:
            return 0.0
        hits = sum(1 for p in predictions
                   if gold.get(p["id"]) == self.parse_pred(str(p["pred"])))
        return hits / len(gold)

    def run_baseline(self, llm: LLMClient, split: str = "test") -> float:
        preds = []
        for ex in self.examples(split):
            raw = llm.chat([{"role": "user", "content": self.baseline_prompt(ex)}],
                           temperature=0.0)
            preds.append({"id": ex["id"], "pred": self.parse_pred(raw)})
        return self.score(preds, split=split)
```
Remove `GSM8KAccuracyTask.load_examples` (HF-at-runtime) and ADD a `materialize` classmethod used only by the prefetch script:
```python
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
        def _row(i, row):
            return {"id": str(i), "question": row["question"],
                    "answer": tmp.parse_pred(row["answer"])}
        test = [_row(i, ds[i]) for i in range(min(n_test, len(ds)))]
        dev = [_row(f"d{i}", ds[i]) for i in range(n_test, min(n_test + n_dev, len(ds)))]
        for name, rows in (("test", test), ("dev", dev)):
            (out / f"{name}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n")
        return {"test": len(test), "dev": len(dev)}
```
Keep `parse_pred`, `baseline_prompt`, `manifest`, `TASKS` as-is. (Note: existing `tests/unit/test_execution_task.py` injects `task._examples` and calls `score(...)`/`run_baseline(...)` without a split — update those tests to set `task._splits = {"test": _examples()}` and pass `split="test"`, OR keep back-compat by having `examples()` fall back to `self._splits.get("test")`. Simplest: update the existing test to the new split API.)

- [ ] **Step 4: Rewrite `scripts/prefetch_datasets.py`** to materialize local splits:
```python
"""Materialize each registered task's dataset into repo-local dev/test splits
(data/datasets/<task>/{dev,test}.jsonl). Pulls HF ONCE here so the run loop
never touches the network, and so test LABELS live only on the host (never in
the sandbox image)."""
import argparse
from pathlib import Path

from dotenv import load_dotenv

from papergym.execution.task import TASKS, DEFAULT_DATA_ROOT

load_dotenv(override=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--n-test", type=int, default=50)
    p.add_argument("--n-dev", type=int, default=50)
    p.add_argument("--tasks", nargs="*", default=None)
    args = p.parse_args(argv)
    for tid in (args.tasks or list(TASKS)):
        counts = TASKS[tid].materialize(data_root=args.data_root,
                                        n_test=args.n_test, n_dev=args.n_dev)
        print(f"materialized {tid}: {counts}")


if __name__ == "__main__":
    main()
```
Update `tests/unit/test_scripts_prefetch_datasets.py`: patch `TASKS["gsm8k_accuracy"].materialize` (a classmethod) to return `{"test":1,"dev":1}` and assert the printed line; e.g.
```python
monkeypatch.setattr(pf.TASKS["gsm8k_accuracy"], "materialize",
                    classmethod(lambda cls, **k: {"test": 1, "dev": 1}))
pf.main(["--tasks", "gsm8k_accuracy"])
assert "materialized gsm8k_accuracy" in capsys.readouterr().out
```

- [ ] **Step 5: Run** `uv run pytest tests/unit/test_execution_task_splits.py tests/unit/test_execution_task.py tests/unit/test_scripts_prefetch_datasets.py -v` — all green.

---

## Task 3: Leakage-safe data flow + static guard (#1 data flow)

**Files:**
- Create: `src/papergym/execution/integrity.py`
- Modify: `src/papergym/execution/agent.py` (stage inputs-only test + labeled dev; prompt)
- Modify: `src/papergym/execution/scorer.py` (score test split; surface leakage flags)
- Modify: `docker/Dockerfile.exec` (do NOT bake labeled dataset)
- Test: `tests/unit/test_execution_integrity.py` (+ update `test_execution_agent.py`, `test_execution_scorer.py`)

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_execution_integrity.py
from papergym.execution.integrity import scan_for_leakage


def test_flags_dataset_reload_and_direct_provider():
    code = ("from datasets import load_dataset\n"
            "import openai\n"
            "ds = load_dataset('openai/gsm8k','main',split='test')\n")
    flags = scan_for_leakage(code)
    assert any("load_dataset" in f for f in flags)
    assert any("openai" in f for f in flags)


def test_clean_code_has_no_flags():
    code = ("from papergym.execution.gym_client import metered_llm_call\n"
            "metered_llm_call([{'role':'user','content':'hi'}])\n")
    assert scan_for_leakage(code) == []
```

- [ ] **Step 2: Run, expect FAIL** (ModuleNotFoundError).

- [ ] **Step 3: Implement** `src/papergym/execution/integrity.py`:
```python
"""Static leakage / cheating guard for agent-written experiment code."""
from __future__ import annotations

import re

# patterns that indicate label leakage or bypassing the metered LLM path
_FORBIDDEN = {
    "load_dataset": re.compile(r"\bload_dataset\b"),
    "datasets-import": re.compile(r"\bimport\s+datasets\b|from\s+datasets\b"),
    "huggingface": re.compile(r"huggingface|hf_hub|datasets\.load"),
    "openai-direct": re.compile(r"\bimport\s+openai\b|from\s+openai\b"),
    "anthropic-direct": re.compile(r"\bimport\s+anthropic\b|from\s+anthropic\b"),
    "litellm-direct": re.compile(r"\bimport\s+litellm\b|from\s+litellm\b"),
    "papergym-llm-direct": re.compile(r"papergym\.llm"),
    "raw-network": re.compile(r"\brequests\.(get|post)\b|urllib\.request"),
}


def scan_for_leakage(code: str) -> list[str]:
    """Return a list of human-readable flags; empty == clean. The only blessed
    LLM path is papergym.execution.gym_client.metered_llm_call (allowed)."""
    flags = []
    for name, rx in _FORBIDDEN.items():
        if rx.search(code or ""):
            flags.append(f"forbidden pattern: {name}")
    return flags
```

- [ ] **Step 4: Update `agent.py`** — stage label-free test inputs + labeled dev, and change the system prompt. In `ExecutionAgent.run`, replace the staging + prompt:
```python
        sandbox.write_file("test_inputs.json",
                           json.dumps(task.inputs(split="test")))
        sandbox.write_file("dev.json",
                           json.dumps(task.examples(split="dev")))
        user = (f"IDEA PROPOSAL:\n{idea.proposal_text}\n\n"
                f"TASK: {json.dumps(task.manifest())}\n"
                f"`dev.json` (id, question, ANSWER) is for developing/tuning your "
                f"method. `test_inputs.json` (id, question — NO answers) is what "
                f"you must predict. Write predictions.json = [{{'id','pred'}}] for "
                f"every test_inputs row.")
```
And change `_SYSTEM` to forbid label access and mandate the metered client:
```python
_SYSTEM = """You are a research-execution agent. Implement the METHOD from the
idea proposal as `method.py`, run it, and write `predictions.json` in the
format {fmt}. Call the LLM ONLY via:
  from papergym.execution.gym_client import metered_llm_call
  text = metered_llm_call([{{"role":"user","content":"..."}}])
Do NOT import datasets/openai/anthropic/litellm/papergym.llm, do NOT download
any dataset, do NOT read test answers — they do not exist in your sandbox.
Develop on dev.json (has answers); predict test_inputs.json (no answers).
Call Finish when predictions.json exists. NO model training."""
```
(`task.manifest()` line that staged `examples.json` is removed/replaced by the two writes above.)

- [ ] **Step 5: Update `scorer.py`** to score the test split and surface leakage flags:
```python
"""Objective effectiveness scoring: method metric vs baseline (held-out test)."""
from __future__ import annotations

from typing import Optional

from .integrity import scan_for_leakage
from .task import Task
from .types import RunArtifact


def score_effectiveness(task: Task, run: RunArtifact, baseline_metric: float,
                        split: str = "test"
                        ) -> tuple[Optional[float], Optional[float], list]:
    """Returns (method_metric, effectiveness, leakage_flags). If the agent's
    code trips the leakage guard, the run is marked suspect: metrics are
    returned as None (a leaked 100% is NOT a real score)."""
    flags = scan_for_leakage(run.code)
    if flags or not run.predictions:
        return None, None, flags
    method_metric = task.score(run.predictions, split=split)
    return method_metric, method_metric - baseline_metric, flags
```
Update `tests/unit/test_execution_scorer.py`: the task fixture now uses `task._splits = {"test": [...]}` (not `_examples`), `score_effectiveness` returns a 3-tuple, and add a test that leaked code (`run.code` containing `load_dataset`) returns `(None, None, [non-empty])`.

- [ ] **Step 6: Edit `docker/Dockerfile.exec`** — REMOVE the labeled-dataset bake (the `RUN python -c "... load_dataset('openai/gsm8k','main')"` line). The sandbox must not contain test labels. Keep the lib install. Add a comment: `# datasets/labels are NOT baked in — labels live host-side only (data/datasets/<task>/test.jsonl)`.

- [ ] **Step 7: Run** `uv run pytest tests/unit/test_execution_integrity.py tests/unit/test_execution_agent.py tests/unit/test_execution_scorer.py -v` — all green (update agent/scorer tests as noted).

---

## Task 4: UsageMeter + budget (#2 core)

**Files:**
- Create: `src/papergym/execution/metering.py`
- Test: `tests/unit/test_execution_metering.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_execution_metering.py
import pytest
from papergym.execution.metering import UsageMeter, BudgetExceeded


class _FakeLLM:
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self._calls = 0
    def chat(self, messages, temperature=0.0):
        self._calls += 1
        self.total_prompt_tokens += 100
        self.total_completion_tokens += 20
        return f"reply{self._calls}"


def test_meter_records_tokens_and_cost():
    m = UsageMeter(_FakeLLM(), budget_usd=1.0, pricing=(1.0, 2.0))
    out = m.call([{"role": "user", "content": "hi"}])
    assert out == "reply1"
    assert m.calls == 1
    assert m.prompt_tokens == 100 and m.completion_tokens == 20
    # cost = (100*1 + 20*2)/1e6
    assert round(m.cost_usd, 8) == round((100*1 + 20*2)/1_000_000, 8)


def test_budget_enforced():
    m = UsageMeter(_FakeLLM(), budget_usd=0.0, pricing=(1.0, 2.0))
    with pytest.raises(BudgetExceeded):
        m.call([{"role": "user", "content": "hi"}])


def test_usage_dict():
    m = UsageMeter(_FakeLLM(), budget_usd=1.0, pricing=(1.0, 2.0))
    m.call([{"role": "user", "content": "hi"}])
    u = m.usage()
    assert u["calls"] == 1 and "cost_usd" in u
```

- [ ] **Step 2: Run, expect FAIL** (ModuleNotFoundError).

- [ ] **Step 3: Implement** `src/papergym/execution/metering.py`:
```python
"""Meter + budget-enforce all LLM calls made on behalf of a sandboxed agent."""
from __future__ import annotations

from eval.common import GPT5_PRICING


class BudgetExceeded(RuntimeError):
    pass


class UsageMeter:
    """Wraps one LLMClient; records cumulative tokens/cost across calls and
    blocks once budget_usd is exhausted. pricing = (prompt, completion) USD/1M."""

    def __init__(self, llm, budget_usd: float = 5.0, pricing=GPT5_PRICING):
        self._llm = llm
        self.budget_usd = budget_usd
        self.pricing = pricing
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def cost_usd(self) -> float:
        return (self.prompt_tokens * self.pricing[0]
                + self.completion_tokens * self.pricing[1]) / 1_000_000

    def call(self, messages: list, temperature: float = 0.0) -> str:
        if self.cost_usd >= self.budget_usd:
            raise BudgetExceeded(
                f"budget ${self.budget_usd} exhausted (${self.cost_usd:.4f})")
        before_p = self._llm.total_prompt_tokens
        before_c = self._llm.total_completion_tokens
        out = self._llm.chat(messages, temperature=temperature)
        self.calls += 1
        self.prompt_tokens += self._llm.total_prompt_tokens - before_p
        self.completion_tokens += self._llm.total_completion_tokens - before_c
        if self.cost_usd > self.budget_usd:
            raise BudgetExceeded(
                f"budget ${self.budget_usd} overrun (${self.cost_usd:.4f})")
        return out

    def usage(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "cost_usd": round(self.cost_usd, 6)}
```

- [ ] **Step 4: Run, expect PASS (3 passed).**

---

## Task 5: Metering proxy + in-sandbox client + runner wiring (#2 transport)

**Files:**
- Create: `src/papergym/execution/llm_proxy.py`
- Create: `src/papergym/execution/gym_client.py`
- Modify: `eval/execution/evaluate.py`
- Test: `tests/unit/test_execution_llm_proxy.py`, `tests/unit/test_execution_gym_client.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_execution_llm_proxy.py
import json
from papergym.execution.llm_proxy import handle_payload
from papergym.execution.metering import UsageMeter, BudgetExceeded


class _FakeLLM:
    total_prompt_tokens = 0
    total_completion_tokens = 0
    def chat(self, messages, temperature=0.0):
        type(self).total_prompt_tokens += 10
        type(self).total_completion_tokens += 5
        return "ok"


def test_handle_payload_returns_content():
    m = UsageMeter(_FakeLLM(), budget_usd=1.0, pricing=(1.0, 1.0))
    status, body = handle_payload(m, {"messages": [{"role": "user", "content": "hi"}]})
    assert status == 200 and json.loads(body)["content"] == "ok"


def test_handle_payload_budget_error():
    m = UsageMeter(_FakeLLM(), budget_usd=0.0, pricing=(1.0, 1.0))
    status, body = handle_payload(m, {"messages": [{"role": "user", "content": "hi"}]})
    assert status == 429 and "budget" in json.loads(body)["error"].lower()
```

```python
# tests/unit/test_execution_gym_client.py
import io, json
from unittest import mock
from papergym.execution import gym_client


def test_metered_llm_call_posts_and_returns_content(monkeypatch):
    monkeypatch.setenv("GYM_LLM_URL", "http://localhost:9000")
    resp = io.BytesIO(json.dumps({"content": "hello"}).encode())
    resp.status = 200
    with mock.patch.object(gym_client.urllib.request, "urlopen",
                           return_value=resp) as u:
        out = gym_client.metered_llm_call([{"role": "user", "content": "hi"}])
    assert out == "hello"
    assert u.called


def test_metered_llm_call_requires_url(monkeypatch):
    monkeypatch.delenv("GYM_LLM_URL", raising=False)
    try:
        gym_client.metered_llm_call([{"role": "user", "content": "hi"}])
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass
```

- [ ] **Step 2: Run, expect FAIL** (ModuleNotFoundError).

- [ ] **Step 3: Implement** `src/papergym/execution/llm_proxy.py`:
```python
"""Host-side metering proxy: the ONLY LLM path for sandboxed agent code.
Holds the real LLMClient (and provider keys) on the host; the sandbox sees
only an http URL. Records tokens/cost and enforces budget via UsageMeter."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .metering import BudgetExceeded, UsageMeter


def handle_payload(meter: UsageMeter, payload: dict) -> tuple[int, str]:
    """Pure request logic (unit-tested). Returns (status, json_body)."""
    messages = payload.get("messages") or []
    temperature = float(payload.get("temperature", 0.0))
    try:
        content = meter.call(messages, temperature=temperature)
    except BudgetExceeded as exc:
        return 429, json.dumps({"error": str(exc)})
    except Exception as exc:                     # provider/transport error
        return 500, json.dumps({"error": str(exc)})
    return 200, json.dumps({"content": content, "usage": meter.usage()})


def run_proxy(meter: UsageMeter, host: str = "127.0.0.1", port: int = 0):
    """Start the proxy in a background thread. Returns (server, url, thread).
    port=0 lets the OS pick a free port (read it from server.server_address)."""
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):               # silence
            return
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(n) or b"{}")
            status, body = handle_payload(meter, payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body.encode())

    server = ThreadingHTTPServer((host, port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://{host}:{server.server_address[1]}"
    return server, url, thread
```

`src/papergym/execution/gym_client.py`:
```python
"""In-sandbox client: the agent's ONLY way to reach an LLM. Talks to the host
metering proxy at $GYM_LLM_URL. No provider keys exist in the sandbox."""
from __future__ import annotations

import json
import os
import urllib.request


def metered_llm_call(messages: list, temperature: float = 0.0) -> str:
    url = os.environ.get("GYM_LLM_URL")
    if not url:
        raise RuntimeError("GYM_LLM_URL not set — no LLM available in sandbox")
    data = json.dumps({"messages": messages, "temperature": temperature}).encode()
    req = urllib.request.Request(url + "/chat", data=data,
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    body = json.loads(resp.read())
    if getattr(resp, "status", 200) != 200 or "content" not in body:
        raise RuntimeError(f"gym LLM error: {body.get('error', body)}")
    return body["content"]
```
(Note: the gym_client test patches `gym_client.urllib.request.urlopen`; the response stub sets `.status` and a JSON `content`. Keep `import urllib.request` at module top so the patch target resolves. The proxy POST path is `/chat` — match it in `do_POST` by ignoring path or checking `self.path == "/chat"`; for the MVP ignore path.)

- [ ] **Step 4: Wire the runner** — edit `eval/execution/evaluate.py` `run_one_idea`:
  - import `from papergym.execution.llm_proxy import run_proxy` and `from papergym.execution.metering import UsageMeter`.
  - Build `meter = UsageMeter(gen_llm, budget_usd=budget_usd)`; `server, url, _ = run_proxy(meter)`.
  - Set the sandbox env so in-container/subprocess code sees it: `os.environ["GYM_LLM_URL"] = url` for the duration (LocalSandbox inherits host env; DockerSandbox `_env_flags` forwards `GYM_LLM_URL`; for Docker also pass `--add-host=host.docker.internal:host-gateway` and use that host — see Task 6).
  - Run baseline + agent on the **test split** (`task.run_baseline(gen_llm, split="test")`; agent already stages test inputs).
  - `method_metric, effectiveness, leakage_flags = score_effectiveness(task, run, baseline_metric, split="test")`.
  - After the run: `server.shutdown()`; fold metered usage into cost: add `cost["sandbox_llm"] = meter.usage()` and `cost["total_cost_usd"] += meter.usage()["cost_usd"]`.
  - Add `leakage_flags` to the returned `ExecResult` (extend the dataclass in `types.py` with `leakage_flags: list = field(default_factory=list)` and `budget_usd` param threaded through). 
  - Signature gains `budget_usd: float = 5.0`.
  Update `tests/unit/test_execution_evaluate.py`: patch `run_proxy` to return a dummy `(server_with_shutdown, "http://x", None)`, set `task._splits={"test":[...], "dev":[...]}`, patch `ExecutionAgent`, assert `leakage_flags == []` and `sandbox_llm` in cost.

- [ ] **Step 5: Run** `uv run pytest tests/unit/test_execution_llm_proxy.py tests/unit/test_execution_gym_client.py tests/unit/test_execution_evaluate.py -v` — all green.

---

## Task 6: Full suite + manual Docker egress verification

**Files:** none (verification) + `types.py` field already added in Task 5.

- [ ] **Step 1: Full unit suite** — Run: `uv run pytest tests/unit -q`. Expected: ALL PASS (existing P1 tests updated for the new split/scorer/sandbox APIs).

- [ ] **Step 2 (manual, needs Docker): network egress lockdown.** Document + verify that the experiment container can reach ONLY the proxy:
  - Create a dedicated docker network and run the container with it; ensure huggingface.co is unreachable from inside while `GYM_LLM_URL` is reachable. Example check:
    ```bash
    docker run --rm --add-host=host.docker.internal:host-gateway papergym-exec:latest \
      python -c "import urllib.request as u; u.urlopen('https://huggingface.co',timeout=5)" ; echo "exit=$?"
    ```
    Expected once locked down: non-zero exit (blocked). Record the network policy used (firewall rule / `--network` config) in `docs/REPRODUCE.md`.
  - Until egress is locked, the static leakage guard (Task 3) + no-labels-in-sandbox (Tasks 2–3) are the active defenses; note this in the run output.

- [ ] **Step 3: Append a note to `docs/REPRODUCE.md`** mapping "gym hardening (leakage/cost/sandbox)" to the new modules and the manual egress check.

---

## Self-Review

**Threat coverage:**
- #1 leakage → Task 2 (labels only in host-side local splits; HF removed from runtime), Task 3 (agent gets test *inputs* only + labeled dev; labels never staged; image no longer bakes labels; static guard), Task 6 (network egress). ✓ structural; network lockdown is manual.
- #2 cost → Task 4 (UsageMeter+budget), Task 5 (proxy is the only LLM path; provider keys removed from sandbox in Task 1; metered usage folded into cost). ✓
- #3 sandbox boundary → Task 1 (path-jail on read/write/run for BOTH sandboxes; provider keys no longer forwarded). ✓

**Placeholder scan:** none — every step has complete code or an exact command.

**Type consistency:** `score_effectiveness` now returns a 3-tuple `(method_metric, effectiveness, leakage_flags)` — Task 5 updates the only caller (`evaluate.py`) and the test. `Task.score`/`run_baseline`/`examples` gain `split=` (default "test") — back-compatible for positional pred arg; callers updated. `ExecResult` gains `leakage_flags`; `run_one_idea` gains `budget_usd`. `metered_llm_call`/`handle_payload`/`UsageMeter.call` signatures are consistent across Tasks 4–5.

**Carried risk:** LocalSandbox still runs agent code as a host subprocess (dev only) — it inherits host env, so for Local the "no provider keys" guarantee depends on the runner not exporting them; real/untrusted runs MUST use DockerSandbox + Task 6 egress lockdown. Documented.

---

## Known residuals (adversarial review, 2026-06-07) → P1.6 follow-up

The review confirmed the **structural** data-flow is fixed (no labels staged/imaged; scored on host-held test labels; path-jail `_resolve_in` is robust incl. symlinks; no cost double-count; provider keys absent from the *Docker* forward list). But these gaps remain — close them in a P1.6 before producing any *real* effectiveness number:

- **R1 (leakage/boundary, Critical): the executed process is unconfined.** `_resolve_in` validates path *arguments* only; `run_python` then runs `method.py` as a host subprocess with no FS confinement. On LocalSandbox it can `open('/etc/passwd')` or read host `data/datasets/<task>/test.jsonl` directly → fake 100%. Fix: scored runs MUST use DockerSandbox; for Local, pass an explicit `env=` minus provider keys and treat as dev-only/never-scored.
- **R2 (cost/keys, Critical for Local): LocalSandbox subprocess inherits host env** → `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` visible in-sandbox (demonstrated), so `method.py` can call providers directly, bypassing the meter. Fix: spawn with sanitized `env=` (drop provider keys, keep `GYM_LLM_URL`).
- **R3 (Docker functional bug, Important): in `--use-docker`, the in-container `metered_llm_call` cannot reach the host proxy.** Proxy binds `127.0.0.1`; `evaluate.py` forwards that literal URL; `docker run` has no `--add-host=host.docker.internal:host-gateway` and no URL rewrite. Result: every agent LLM call fails in Docker mode. Fix: add `--add-host` and rewrite `GYM_LLM_URL` host→`host.docker.internal` for the container.
- **R4 (leakage, Important): static guard is evadable + scans only `method.py`.** Misses `__import__('datasets')`, dynamic string building, HF-by-IP, `open()` of host label files, and any cheating placed in a second file (`import cheat`). Fix: scan ALL written `*.py`; treat guard as defense-in-depth only — the real defense is network egress lockdown (Task 6, still manual/unimplemented).
- **R5 (cost, Important): network egress not actually blocked** — only documented (Task 6). Until a firewall/`--network` policy is in place, the agent can re-download GSM8K (incl. by IP) at runtime. Any "100%" must be treated as suspect.
- **R6 (concurrency, Should): `UsageMeter` is not thread-safe** under `ThreadingHTTPServer` (unguarded read-modify-write + delta from a shared global counter). Fix: add a lock or run the proxy single-threaded.
- **R7 (reporting, Minor): `budget_usd` not threaded through `run_execution_gym.py` (no CLI flag); `--n-examples` is now a dead/misleading knob** (actual scored size = `len(test.jsonl)` from materialize). Fix: add `--budget-usd`, drop/relabel `--n-examples`, reconcile `manifest()`.

**Verdict:** honest-evaluation posture improved and unit-green (104 passed), but the gym is **not yet adversarially safe** and Docker LLM calls are broken — R1–R5 must land before trusting a real number.
