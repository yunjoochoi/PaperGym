# Gym Hardening P1.6 — close adversarial residuals (R1–R7)

> Subagent-driven. Commit steps OMITTED (human controls commits). TDD per task.

**Goal:** Close the code-fixable residuals the adversarial review found in P1.5 so a *real* effectiveness number is trustworthy and Docker mode actually works. R5 (true network-egress block) is partly OPS — code provides the isolated-network construction + defense-in-depth; the firewall verification stays manual.

**Residuals → tasks:**
- R1 process confinement → Task 1 (mark Local as untrustworthy; scored runs flagged) + R2.
- R2 LocalSandbox key inheritance → Task 1 (sanitized subprocess `env=`).
- R3 Docker proxy unreachable → Task 2 (`--add-host` + GYM_LLM_URL host rewrite).
- R4 guard scans only method.py / evadable → Task 3 (scan ALL written files).
- R5 egress → Task 2 (isolated `--network` default + doc) + R4 defense-in-depth; firewall = manual.
- R6 UsageMeter not thread-safe → Task 4 (lock).
- R7 budget/n_examples reporting → Task 5 (CLI `--budget-usd`, reconcile, `trustworthy` in summary).

---

## Task 1: LocalSandbox env-strip + trustworthy flag (R1, R2)

**Files:** Modify `src/papergym/execution/sandbox.py`, `src/papergym/execution/types.py`, `eval/execution/evaluate.py`; Test `tests/unit/test_execution_sandbox_env.py` (new); update `tests/unit/test_execution_evaluate.py`.

- [ ] **Step 1 — failing test** `tests/unit/test_execution_sandbox_env.py`:
```python
from papergym.execution.sandbox import LocalSandbox


def test_local_subprocess_env_drops_provider_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-secret")
    monkeypatch.setenv("GYM_LLM_URL", "http://localhost:1234")
    with LocalSandbox(work_root=tmp_path / "run") as sb:
        sb.write_file("m.py",
                      "import os,json;"
                      "json.dump({k:os.environ.get(k) for k in "
                      "['OPENAI_API_KEY','ANTHROPIC_API_KEY','GYM_LLM_URL']},"
                      "open('out.json','w'))")
        sb.run_python("m.py")
        import json
        seen = json.loads(sb.read_file("out.json"))
    assert seen["OPENAI_API_KEY"] is None
    assert seen["ANTHROPIC_API_KEY"] is None
    assert seen["GYM_LLM_URL"] == "http://localhost:1234"   # proxy still reachable
```

- [ ] **Step 2 — run, expect FAIL** (keys leak through).

- [ ] **Step 3 — implement.** In `sandbox.py` add a module-level constant + helper:
```python
_PROVIDER_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_BASE",
                  "EMBEDDING_MODEL", "LITELLM_MODEL")


def _sandboxed_env() -> dict:
    """Host env minus provider secrets (keep GYM_LLM_URL/PATH/etc.)."""
    import os
    return {k: v for k, v in os.environ.items() if k not in _PROVIDER_KEYS}
```
In `LocalSandbox.run_python`, pass `env=_sandboxed_env()` to `subprocess.run(...)`.

- [ ] **Step 4 — trustworthy flag.** In `types.py` add to `ExecResult` (after `leakage_flags`):
```python
    sandbox: str = "local"
    trustworthy: bool = False
```
In `eval/execution/evaluate.py` `run_one_idea`: compute `kind = "docker" if use_docker else "local"`; pass `sandbox=kind, trustworthy=use_docker` into `ExecResult` (a Local run is dev-only → not a trustworthy score even though the number is computed). Update `tests/unit/test_execution_evaluate.py`: assert `res.sandbox == "local"` and `res.trustworthy is False` for the default (Local) path.

- [ ] **Step 5 — run** `uv run pytest tests/unit/test_execution_sandbox_env.py tests/unit/test_execution_evaluate.py tests/unit -q` — all green.

---

## Task 2: Docker proxy reachability + isolated network (R3, R5-code)

**Files:** Modify `src/papergym/execution/sandbox.py`; update `tests/unit/test_execution_sandbox_docker.py`/`_jail` as needed; Test additions inline.

- [ ] **Step 1 — failing test** (add to `tests/unit/test_execution_sandbox_docker.py`):
```python
def test_docker_adds_host_gateway_and_rewrites_proxy_host(tmp_path, monkeypatch):
    monkeypatch.setenv("GYM_LLM_URL", "http://127.0.0.1:9000")
    sb = DockerSandbox(work_root=tmp_path / "run", image="img")
    sb.reset()
    fake = mock.MagicMock(returncode=0, stdout="", stderr="")
    with mock.patch.object(sb_mod.subprocess, "run", return_value=fake) as run:
        sb.run_python("m.py")
    argv = run.call_args.args[0]
    joined = " ".join(argv)
    assert "--add-host=host.docker.internal:host-gateway" in argv
    # container can't reach 127.0.0.1 of the host -> rewrite to host.docker.internal
    assert "host.docker.internal:9000" in joined
    assert "127.0.0.1:9000" not in joined
```

- [ ] **Step 2 — run, expect FAIL.**

- [ ] **Step 3 — implement.** In `DockerSandbox`:
  - In `_env_flags`, when forwarding `GYM_LLM_URL`, rewrite host `127.0.0.1`/`localhost` → `host.docker.internal`:
```python
    def _env_flags(self) -> list[str]:
        flags = []
        for k in _FORWARDED_ENV:
            v = os.environ.get(k)
            if not v:
                continue
            if k == "GYM_LLM_URL":
                v = v.replace("127.0.0.1", "host.docker.internal") \
                     .replace("localhost", "host.docker.internal")
            flags += ["-e", f"{k}={v}"]
        return flags
```
  - In `run_python`, add `--add-host` to the argv (after `--rm`):
```python
        argv = (["docker", "run", "--rm",
                 "--add-host=host.docker.internal:host-gateway",
                 f"--network={self.network}",
                 "-v", f"{self.work_root}:/work", "-w", "/work"]
                + self._env_flags()
                + [self.image, "python", rel])
```
  - Keep `network` param (default `"bridge"`); document that a locked-down run sets `network` to a dedicated gym network whose only egress is the proxy (firewall = manual, Task 6 / R5).

- [ ] **Step 4 — run** the docker sandbox tests + full suite — green. (Existing `test_run_python_invokes_docker_run_with_mount` should still pass; if it asserted exact argv length, relax it to membership checks.)

---

## Task 3: Scan ALL written files, not just method.py (R4)

**Files:** Modify `src/papergym/execution/agent.py`; update `tests/unit/test_execution_agent.py`; Test new behavior.

- [ ] **Step 1 — failing test** (add to `tests/unit/test_execution_agent.py`): the agent writes a clean `method.py` that `import cheat`, and a `cheat.py` containing `load_dataset`; assert the collected `run.code` includes BOTH files' contents (so the downstream guard can catch cheat.py).
```python
def test_run_code_includes_all_written_py_files(tmp_path):
    import json
    from papergym.execution.agent import ExecutionAgent
    from papergym.execution.sandbox import LocalSandbox
    from papergym.execution.task import GSM8KAccuracyTask
    from papergym.execution.types import IdeaSpec
    from papergym.llm import ChatReply, ToolCall
    from unittest import mock
    task = GSM8KAccuracyTask(n_examples=1)
    task._splits = {"test": [{"id": "0", "question": "2+2?", "answer": "4"}],
                    "dev": [{"id": "d0", "question": "1+1?", "answer": "2"}]}
    calls = [
        ChatReply(content="", tool_calls=[ToolCall(id="a", name="WriteFile",
            arguments=json.dumps({"path": "cheat.py", "content": "load_dataset('x')"}))],
            raw_message={"role": "assistant"}),
        ChatReply(content="", tool_calls=[ToolCall(id="b", name="WriteFile",
            arguments=json.dumps({"path": "method.py",
                "content": "import cheat\nimport json; json.dump([{'id':'0','pred':'4'}], open('predictions.json','w'))"}))],
            raw_message={"role": "assistant"}),
        ChatReply(content="", tool_calls=[ToolCall(id="c", name="RunPython",
            arguments=json.dumps({"path": "method.py"}))], raw_message={"role": "assistant"}),
        ChatReply(content="", tool_calls=[ToolCall(id="d", name="Finish",
            arguments=json.dumps({"summary": "x"}))], raw_message={"role": "assistant"}),
        ChatReply(content="ok", tool_calls=[], raw_message={"role": "assistant"}),
    ]
    llm = mock.MagicMock(); llm.chat_with_tools.side_effect = calls
    with LocalSandbox(work_root=tmp_path / "run") as sb:
        run = ExecutionAgent(llm=llm, max_steps=10).run(
            idea=IdeaSpec(idea_id="i", condition="Human", topic="Math",
                          title="t", proposal_text="p"), task=task, sandbox=sb)
    assert "cheat.py" in run.code and "load_dataset" in run.code
    assert "method.py" in run.code
```

- [ ] **Step 2 — run, expect FAIL** (only method.py collected today).

- [ ] **Step 3 — implement** in `agent.py`: after the tool loop, collect every `*.py` file the agent wrote into `run.code`. Track writes in the dispatch and concatenate, OR list the workspace. Simplest robust: have `_Tools` accumulate written paths, then read them back:
```python
@dataclass
class _Tools:
    sandbox: Sandbox
    written: list = field(default_factory=list)   # add import: from dataclasses import field

    def dispatch(self, name, args):
        if name == "WriteFile":
            if str(args.get("path", "")).endswith(".py"):
                self.written.append(args["path"])
            return _write_file(args["path"], args["content"], sandbox=self.sandbox)
        ...
```
After the loop, build `code` from all written .py files (dedup, keep order), falling back to method.py:
```python
        seen, parts = set(), []
        for path in tools.written + ["method.py"]:
            if path in seen:
                continue
            seen.add(path)
            try:
                parts.append(f"# {path}\n" + sandbox.read_file(path))
            except FileNotFoundError:
                continue
        code = "\n\n".join(parts)
```
Keep predictions.json collection unchanged. (`run.code` now contains all written python — the scorer's `scan_for_leakage` already runs on `run.code`, so cheat.py is now scanned.)

- [ ] **Step 4 — run** `tests/unit/test_execution_agent.py` + full suite — green. (The existing agent test still passes: it writes method.py only.)

---

## Task 4: UsageMeter thread-safety (R6)

**Files:** Modify `src/papergym/execution/metering.py`; Test add.

- [ ] **Step 1 — failing/representative test** (add to `tests/unit/test_execution_metering.py`): drive `UsageMeter.call` from 8 threads over a fake LLM and assert `calls == 8` and token totals are exactly `8 * per_call` (no lost updates). Use a fake LLM whose `chat` does a tiny `time.sleep(0)` between reading/writing its counters to widen races.

- [ ] **Step 2 — run** (may be flaky-fail without a lock).

- [ ] **Step 3 — implement.** Add `import threading`; in `__init__` `self._lock = threading.Lock()`; wrap the budget-check + chat + accounting in `with self._lock:` so the read-modify-write of counters (and the delta off the shared client) is atomic. (Note: serializing LLM calls in the proxy is acceptable — correctness over throughput.)

- [ ] **Step 4 — run** the metering tests + full suite — green.

---

## Task 5: Runner budget flag + reporting reconcile + trustworthy in summary (R7)

**Files:** Modify `scripts/run_execution_gym.py`; update `tests/unit/test_scripts_run_execution_gym.py`.

- [ ] **Step 1 — failing test** (update `tests/unit/test_scripts_run_execution_gym.py`): pass `--budget-usd 2.5` and assert `run_one_idea` was called with `budget_usd=2.5`; assert `summary.json` includes `n_trustworthy` (count of results with `trustworthy=True`). Keep the existing fake `ExecResult` but set `trustworthy=False`/`sandbox="local"` on it.

- [ ] **Step 2 — run, expect FAIL.**

- [ ] **Step 3 — implement** in `run_execution_gym.py`:
  - add `p.add_argument("--budget-usd", type=float, default=5.0)`.
  - pass `budget_usd=args.budget_usd` and `use_docker=args.use_docker` into `run_one_idea`.
  - `--n-examples`: the scored size is fixed by the materialized test split, so relabel its help to "(agent hint only; scored size = len(test.jsonl))" — keep the flag for back-compat but it no longer controls scoring. (Task constructor still accepts it.)
  - extend `_summarise` to add `"n_trustworthy": sum(1 for r in results if r.get("trustworthy"))` and `"untrustworthy_note"` when any result has `trustworthy=False` (e.g. "ran on LocalSandbox — not a scored result; use --use-docker").

- [ ] **Step 4 — run** the runner test + FULL suite `uv run pytest tests/unit -q` — all green.

---

## Task 6 (manual, ops): network egress lockdown (R5) — unchanged from P1.5 Task 6
Verify the experiment container reaches ONLY the proxy (block huggingface.co etc.). Record the network policy in `docs/REPRODUCE.md`.

---

## Self-Review
- R1 → Local marked `trustworthy=False`; scored numbers should come from Docker. R2 → Local subprocess env stripped of provider keys (verified). R3 → `--add-host` + host rewrite (Docker LLM now reachable). R4 → all written `*.py` scanned. R6 → lock. R7 → `--budget-usd`, reconciled reporting, `n_trustworthy`. R5 → isolated-network construction in code; firewall verification manual (Task 6).
- Placeholders: none. Type consistency: `ExecResult` gains `sandbox`/`trustworthy` (defaults keep old constructions valid); `run_one_idea` passes them; runner reads them from `to_dict()`.
- Carried: true FS/network confinement still depends on Docker + the manual egress policy; LocalSandbox remains dev-only by design (now explicitly flagged untrustworthy).
