# PaperGym — `env/` Refactor Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Introduce a `PaperEnv` abstraction that consolidates the
"paper-as-environment" concept — currently scattered across `bootstrap/`,
`scripts/run_accumulator.py`, and `agents/accumulator/LocalFS`. The
abstraction is **container-side** (lives inside the Docker container, no
docker spawn inside the env class) so per-step overhead stays at zero.
Workshop framing: "we adopt the Gym API where it carries information
(reset, step); observation/reward are deferred to future RL extensions."

**Non-goals (deferred to future PRs):**
- Reward signal and `(obs, reward, done)` step tuple
- Any RL-style training loop
- Event-logging extension (latency, retry_count, $ cost) — separate PR

**Tech stack:** Python 3.11, pytest, existing `litellm`-based `LLMClient`,
existing `tool_loop.run_tool_loop` driver. No new dependencies.

**Current state to be aware of:**
- `litellm` migration + `tool_loop` natural-end refactor just landed (uncommitted/recent commit). All tests should be green before starting this plan.
- `bootstrap/` currently holds: `paper_search.py`, `sampler.py`, `fetch.py`. `fetch_paper_to_disk()` is what `accumulate_one.py` calls inside the container.
- `agents/accumulator/__init__.py` defines `LocalFS` (Read/Bash/file_exists/write) and `Accumulator._dispatch` that routes Read/Grep/Bash to `LocalFS`. We will fold this dispatch into `PaperEnv.step`.
- `scripts/run_accumulator.py` (host) spawns one Docker container per arxiv_id. **This stays as-is** — env code does not spawn containers.

---

## Architecture

```
src/papergym/env/
├── __init__.py        # exports PaperEnv, sample_envs, make
├── base.py            # PaperEnv class
└── sampler.py         # moved from bootstrap/sampler.py + paper_search.py
```

`bootstrap/` keeps `fetch.py` only (called by `PaperEnv.reset`); other
files move under `env/sampler.py` for thematic coherence ("environment
sampling").

### `PaperEnv` interface

```python
class PaperEnv:
    arxiv_id: str
    paper_dir: Path

    def __init__(self, arxiv_id: str, work_root: Path): ...
    def reset(self) -> None:
        """Deterministically prepare paper.md from arxiv_id."""
    def step(self, tool_name: str, args: dict) -> str:
        """Dispatch Read/Grep/Bash; return observation string."""
    def close(self) -> None: ...
```

No `Observation` dataclass: the agent receives `paper_dir` via the prompt
template (`{{ paper_dir }}`) and reads `paper.md` via the `Read` tool.

`step()`'s return is the same string contract that `tool_loop.dispatch`
already expects; the call site in `Accumulator.run` becomes:
```python
result = run_tool_loop(..., dispatch=env.step, ...)
```

---

## Phase 0 — Pre-flight

### Task 0.0 — Verify baseline is green

**Steps:**
1. `git status` — confirm clean working tree (post-litellm refactor committed).
2. `pytest tests/unit -q` — all tests pass.
3. Note current test count for later comparison.

### Task 0.1 — Branch + commit this plan

**Steps:**
1. Create branch: `git checkout -b feature/env-refactor`
2. Commit this plan doc:
   ```bash
   git add docs/plans/2026-04-27-papergym-env-refactor.md
   git commit -m "docs: env/ refactor plan"
   ```

---

## Phase 1 — Create `env/` package

Goal: introduce `PaperEnv` without yet wiring it into the accumulator.
After this phase, `env/` exists and has tests, but old code paths are
still live.

### Task 1.0 — Skeleton + test scaffolding

**Files:**
- `src/papergym/env/__init__.py` (new)
- `src/papergym/env/base.py` (new)
- `tests/unit/test_env_paper_env.py` (new)

**Steps:**
1. Create `env/base.py` with `PaperEnv`:
   - `__init__(arxiv_id, work_root)` stores both, sets `self.paper_dir = work_root / arxiv_id`. No I/O.
   - `reset()` calls `fetch_paper_to_disk(arxiv_id, root=work_root)`. Idempotent: if `paper.md` already exists, skip refetch.
   - `step(tool_name, args)` implements Read/Grep/Bash dispatch. Lift the body of `Accumulator._dispatch` here. Use `subprocess.run` directly (don't depend on `LocalFS`).
   - `close()` no-op for now (future hook for cleanup).
2. `__init__.py` re-exports `PaperEnv`.
3. Tests:
   - `test_paper_env_reset_creates_paper_md` (monkeypatch `fetch_paper_to_disk`)
   - `test_paper_env_step_read_returns_line_numbered_content` (use a real tmp file)
   - `test_paper_env_step_bash_runs_command_in_paper_dir`
   - `test_paper_env_step_unknown_tool_returns_error_string`
   - `test_paper_env_reset_is_idempotent_when_paper_md_exists`

**Verify:** new tests pass, all old tests still pass.

### Task 1.1 — Move bootstrap modules to `env/sampler.py`

**Files:**
- `src/papergym/env/sampler.py` (new — combines `bootstrap/sampler.py` + `bootstrap/paper_search.py`)
- `src/papergym/bootstrap/__init__.py` (edited — re-export from `env.sampler` for backward compat with existing imports/tests)
- Existing tests that import from `papergym.bootstrap.sampler` / `papergym.bootstrap.paper_search` keep working via the re-export.

**Steps:**
1. Create `env/sampler.py` with the content of both bootstrap modules, merged. Keep all public function names identical (`sample_papers`, `search_papers`, etc. — match what's actually exported today; verify with `grep`).
2. Update `bootstrap/__init__.py`:
   ```python
   # Back-compat shim. New code should import from papergym.env.sampler.
   from ..env.sampler import *  # noqa
   ```
3. Delete the original `bootstrap/sampler.py` and `bootstrap/paper_search.py` files.
4. **`bootstrap/fetch.py` stays.** It's used at runtime by `PaperEnv.reset` and is not "sampling" logic.
5. Update `scripts/bootstrap_papers.py` to import from `papergym.env.sampler` directly (preferred), but the re-export keeps it from breaking if missed.

**Verify:**
- `pytest tests/unit/test_bootstrap_sampler.py tests/unit/test_bootstrap_paper_search.py -q` — green via re-export.
- `pytest tests/unit/test_scripts_bootstrap.py -q` — green.
- `grep -rn "from papergym.bootstrap" src tests scripts` — note remaining imports for follow-up cleanup (don't have to fix now).

---

## Phase 2 — Wire `PaperEnv` into Accumulator

Goal: replace `LocalFS` + `Accumulator._dispatch` with `env.step`. Single
source of truth for tool dispatch.

### Task 2.0 — Switch Accumulator to use `PaperEnv`

**Files:**
- `src/papergym/agents/accumulator/__init__.py` (edited)
- `tests/unit/test_accumulator_agent.py` (edited)

**Steps:**
1. Change `Accumulator.__init__` signature: `(*, llm, prompts, max_steps=100)` — drop the `fs: LocalFS` parameter.
2. Change `Accumulator.run` signature: `run(*, env: PaperEnv) -> dict`.
   - Replace `paper_dir = ...` with `paper_dir = env.paper_dir`.
   - Replace the `dispatch=lambda n, a: self._dispatch(n, a, paper_dir=...)` call with `dispatch=env.step`.
   - Remove `_dispatch` method entirely.
3. Remove `LocalFS` class from this file. (If anything else imports it, remove those imports — should be only tests.)
4. Update `test_accumulator_agent.py`:
   - Construct `env = PaperEnv(arxiv_id="2401.0001", work_root=tmp_path)`; call `env.reset()` won't actually fetch (we monkeypatch `fetch_paper_to_disk` to no-op or just write a file).
   - Pass `env=env` to `acc.run`.
   - Drop `LocalFS` import.

**Verify:** `pytest tests/unit -q` — all green. Test count should equal baseline + new env tests.

### Task 2.1 — Update `accumulate_one.py` to construct PaperEnv

**Files:**
- `scripts/accumulate_one.py` (edited)

**Steps:**
1. Replace:
   ```python
   from papergym.agents.accumulator import Accumulator, LocalFS
   ...
   accumulator = Accumulator(llm=llm, prompts=prompts, fs=LocalFS(), max_steps=...)
   result = accumulator.run(paper_dir=paper_dir)
   ```
   with:
   ```python
   from papergym.agents.accumulator import Accumulator
   from papergym.env import PaperEnv
   ...
   env = PaperEnv(arxiv_id=args.arxiv_id, work_root=args.work_root)
   env.reset()  # fetch paper.md
   accumulator = Accumulator(llm=llm, prompts=prompts, max_steps=...)
   try:
       result = accumulator.run(env=env)
   finally:
       env.close()
   ```
2. Move the existing `try: fetch_paper_to_disk(...) except` block — it's now redundant because `env.reset()` does the fetch. The fetch error handling needs to be preserved: wrap `env.reset()` in try/except and `_log` on failure as before.
3. Drop the now-unused `from papergym.bootstrap.fetch import fetch_paper_to_disk` import (or keep if anything else needs it — verify with grep).

**Verify:**
- `pytest tests/unit/test_scripts_accumulate_one.py -q` — green. Test will need updates if it asserts on the old `fs=` kwarg or `paper_dir=` kwarg.
- Manually inspect: `accumulate_one.py` reads top-to-bottom cleanly.

### Task 2.2 — Remove `LocalFS` from accumulator tool schema test

**Files:**
- `tests/unit/test_accumulator_tools.py` (edited or deleted)

**Steps:**
1. If this test only verifies `LocalFS.read`/`bash`/etc., either:
   - Migrate the assertions to `PaperEnv.step` (preferred — same coverage, new home), or
   - Delete it if duplicated by `test_env_paper_env.py`.
2. Decide based on actual content; do not blindly delete.

**Verify:** `pytest tests/unit -q` green; coverage for tool dispatch logic preserved (one of the two tests covers it).

---

## Phase 3 — Cleanup + docs

### Task 3.0 — Remove `bootstrap/` re-export shim (optional, deferred)

Leave the re-export in place. Removing it is a breaking change for any
external scripts; revisit in a later cleanup PR.

### Task 3.1 — Update `accumulator.yaml` workspace contract (optional)

**Files:** `src/papergym/agents/accumulator/accumulator.yaml`

The current prompt says "Paper directory: {{ paper_dir }} / paper.md / repo/".
This still works — `env.paper_dir` is what gets passed in. No change needed
unless the language drifts (e.g., we want to mention "env"). Skip this task
unless behavior changes.

### Task 3.2 — Add a brief paragraph to README (or design doc) about `env/`

**Files:** `README.md` (or new `docs/plans/2026-04-27-papergym-env-refactor.md` design notes section).

**Steps:**
1. Add 5-10 lines explaining: "Each paper is treated as an environment
   (`PaperEnv`). `reset()` deterministically prepares `paper.md`;
   `step(tool, args)` dispatches Read/Grep/Bash. Container isolation is
   handled externally by `run_accumulator.py` per paper. No reward signal
   yet — see future work."
2. This is the workshop-facing framing; keep it tight.

### Task 3.3 — Final verification

**Steps:**
1. `pytest tests/unit -q` — all green.
2. `grep -rn "LocalFS" src/ tests/ scripts/` — should return nothing.
3. `grep -rn "from papergym.bootstrap.sampler\|from papergym.bootstrap.paper_search" src/ tests/ scripts/` — should be empty (all migrated to `env.sampler`); the re-export shim catches stragglers but new code paths shouldn't rely on it.
4. Manual smoke (optional, requires real OPENAI_API_KEY + docker image): run `accumulate_one.py` against one known arxiv_id end-to-end. If nontrivial to set up, skip — unit tests cover the contract.

---

## Risks & mitigation

| Risk | Mitigation |
|---|---|
| `bootstrap/` import path breaks something we missed | Re-export shim in `bootstrap/__init__.py` catches it. Verify with `grep` before declaring done. |
| `PaperEnv.step` Bash semantics drift from current `LocalFS.bash` | Lift the function body verbatim, including timeout/cwd handling. Add a test that asserts cwd and exit code propagation. |
| Tests that rely on `Accumulator(fs=LocalFS())` constructor | Update each call site; there are only a couple. Caught at test time. |
| `env.reset()` re-fetch on retry | Idempotency check: `paper.md` exists → skip. Add a test for this. |

## Out-of-scope reminders

- No `Observation` dataclass.
- No `reward` / `done` in `step()`.
- No host-side container-managing variant of PaperEnv.
- No event-logging extension. That's a separate PR after this one lands.
