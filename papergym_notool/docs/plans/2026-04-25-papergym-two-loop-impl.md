# PaperGym Two-Loop Architecture Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add nested two-loop architecture (Methodologist↔Implementer for ideas, ExperimentAnalyst↔Implementer for empirical breadth) under a 3-step outer envelope; reuse one container per paper.

**Architecture:** See `docs/plans/2026-04-25-papergym-two-loop-design.md`. Pipeline gains an outer loop. Inside each outer step, Loop A picks a working method (Δ > baseline × 0.01); Loop B expands evidence around that method (multi-seed, ablation, fidelity check). New `ExperimentAnalyst` agent. One container per paper, worktree reset between Loop A attempts.

**Tech Stack:** Python 3.11, pytest, Jinja2, OpenAI API, Singularity/Docker.

**Commit policy:** User owns commits. Each task ends at "stage for user commit" — do NOT run `git commit`. Only `git add`.

---

## Task 1 — Dataclasses: `ExperimentSpec`, `ExperimentAnalysis`, extended `IterationContext`

**Files:**
- Modify: `src/papergym/types.py`

**Step 1: Add `ExperimentSpec` and `ExperimentAnalysis` dataclasses**

Add to `src/papergym/types.py`:

```python
@dataclass
class ExperimentSpec:
    kind: Literal["seed_sweep", "ablation", "additional_metric",
                  "additional_dataset", "fidelity_fix"]
    description: str
    seeds: Optional[list[int]] = None
    metric_override: Optional[str] = None
    dataset_override: Optional[str] = None


@dataclass
class ExperimentAnalysis:
    satisfied: bool
    fidelity_ok: bool
    fidelity_issues: list[str] = field(default_factory=list)
    breadth_gaps: list[str] = field(default_factory=list)
    next_experiment: Optional[ExperimentSpec] = None
    key_observations: list[str] = field(default_factory=list)
```

**Step 2: Extend `IterationContext` with `experiment_spec` field**

Add to existing `IterationContext`:

```python
    experiment_spec: Optional["ExperimentSpec"] = None
```

(Place after `method` field. Forward-ref string OK since ExperimentSpec is in same file.)

**Step 3: Add `OuterRecord` and `OuterState` dataclasses (replaces ad-hoc Pipeline state)**

```python
@dataclass
class OuterRecord:
    outer: int
    status: Literal["completed", "loop_a_failed"]
    method: Optional[MethodSpec] = None
    chosen_impl: Optional[ImplResult] = None
    evidence: list[ImplResult] = field(default_factory=list)
    analysis: Optional[ExperimentAnalysis] = None
    candidates: list = field(default_factory=list)  # only filled on loop_a_failed


@dataclass
class OuterState:
    evidence: list[ImplResult] = field(default_factory=list)
    deep_observations: list[str] = field(default_factory=list)
    failed_methods: list["MethodSpec"] = field(default_factory=list)
    history: list[OuterRecord] = field(default_factory=list)

    @property
    def best_outer(self) -> Optional[OuterRecord]:
        completed = [r for r in self.history if r.status == "completed"]
        if not completed:
            return None
        return max(completed, key=lambda r: r.chosen_impl.new_score or float("-inf"))
```

**Step 4: No tests** (pure dataclass scaffolding per project convention).

**Step 5: Stage**

```bash
git add src/papergym/types.py
```

---

## Task 2 — Implementer prompt: optional `experiment_spec` block + worktree advisory

**Files:**
- Modify: `src/papergym/prompts/implementer.yaml`

**Step 1a: Add a worktree-state advisory in the `system:` template**

Add near the bottom of the `system:` block, after the default-flow numbered list:

```yaml
  Worktree state: this container is reused across baseline + all your prior
  attempts on this paper. The repo at /workspace/repo may carry modifications
  from earlier attempts. If continuing on top of a polluted state would
  confuse you, you can either:
    - re-clone fresh: Bash("rm -rf /workspace/repo && git clone <github_url> /workspace/repo")
      (cheap — model weights and pip deps are cached separately and survive)
    - partial recovery: Bash("cd /workspace/repo && git checkout -- <file>")
  Use Bash("cd /workspace/repo && git status") to inspect first.
```

**Step 1b: Add `{% if ctx.experiment_spec %}` block in the `user:` template**

Insert after the existing method/baseline section in the `user:` template, before "## Workspace inside container":

```yaml
  {% if ctx.experiment_spec %}
  ## Additional experiment instructions
  Goal: {{ ctx.experiment_spec.description }}
  Spec kind: {{ ctx.experiment_spec.kind }}
  {% if ctx.experiment_spec.seeds %}Seeds to use: {{ ctx.experiment_spec.seeds | join(", ") }}{% endif %}
  {% if ctx.experiment_spec.metric_override %}Alternative metric: {{ ctx.experiment_spec.metric_override }}{% endif %}
  {% if ctx.experiment_spec.dataset_override %}Alternative dataset: {{ ctx.experiment_spec.dataset_override }}{% endif %}

  The repo is freshly cloned and dependencies are already installed in this
  container. First apply the method described above, then perform the
  experiment described here. Print FINAL_SCORE per the spec — when running
  multiple seeds, also print FINAL_SCORE_MEAN and FINAL_SCORE_STD on
  separate lines so the host can parse aggregated results.
  {% endif %}
```

**Step 2a: Update `paper.parse_final_score` to also accept `FINAL_SCORE_MEAN`**

Modify `src/papergym/paper.py`:

```python
FINAL_SCORE_RE = re.compile(r"FINAL_SCORE(?:_MEAN)?:\s*([0-9]*\.?[0-9]+)")
```

(So `FINAL_SCORE: 0.55` and `FINAL_SCORE_MEAN: 0.55` both parse to the same `new_score`.)

**Step 2b: Add a helper `parse_score_stats(output: str)` returning mean and std if present**

```python
FINAL_SCORE_STD_RE = re.compile(r"FINAL_SCORE_STD:\s*([0-9]*\.?[0-9]+)")

def parse_score_stats(output: str) -> dict:
    """Return {'mean': float, 'std': float | None} parsed from output."""
    m = FINAL_SCORE_RE.search(output)
    if not m:
        raise ValueError("no FINAL_SCORE line in output")
    s = FINAL_SCORE_STD_RE.search(output)
    return {"mean": float(m.group(1)), "std": float(s.group(1)) if s else None}
```

**Step 3: Test**

Add to `tests/unit/test_paper.py` (create if missing):

```python
from papergym.paper import parse_final_score, parse_score_stats

def test_parse_final_score_plain():
    assert parse_final_score("FINAL_SCORE: 0.567") == 0.567

def test_parse_final_score_mean_variant():
    assert parse_final_score("FINAL_SCORE_MEAN: 0.55") == 0.55

def test_parse_score_stats_with_std():
    out = "FINAL_SCORE_MEAN: 0.55\nFINAL_SCORE_STD: 0.02\n"
    assert parse_score_stats(out) == {"mean": 0.55, "std": 0.02}

def test_parse_score_stats_no_std():
    assert parse_score_stats("FINAL_SCORE: 0.42") == {"mean": 0.42, "std": None}
```

**Step 4: Run tests**

```bash
.venv/bin/pytest tests/unit/test_paper.py -v
```

Expected: 4 pass.

**Step 5: Stage**

```bash
git add src/papergym/prompts/implementer.yaml src/papergym/paper.py tests/unit/test_paper.py
```

---

## Task 3 — `ExperimentAnalyst` agent

**Files:**
- Create: `src/papergym/agents/analyst.py`
- Create: `src/papergym/prompts/analyst.yaml`
- Create: `tests/unit/test_agent_analyst.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_agent_analyst.py
import json
from pathlib import Path
from unittest.mock import MagicMock

from papergym.agents.analyst import ExperimentAnalyst
from papergym.agents.base import PromptLoader
from papergym.paper import PaperContext
from papergym.types import (ImplResult, MethodSpec, ExperimentAnalysis,
                              ExperimentSpec)

PROMPTS = Path(__file__).parent.parent.parent / "src" / "papergym" / "prompts"


def make_method():
    return MethodSpec(method="confidence-conditioned budget forcing",
                      observable_signal="accuracy curve plateaus later",
                      rationale="fixed budget over-allocates on easy items")


def test_analyst_returns_satisfied_when_breadth_complete():
    llm = MagicMock()
    llm.chat.return_value = json.dumps({
        "satisfied": True,
        "fidelity_ok": True,
        "fidelity_issues": [],
        "breadth_gaps": [],
        "next_experiment": None,
        "key_observations": [
            "Method beats baseline by 5% on AIME24 (mean 0.55 ± 0.02 over 3 seeds)",
            "Ablation shows budget gating is the load-bearing component",
        ],
    })
    analyst = ExperimentAnalyst(llm=llm, prompts=PromptLoader(PROMPTS))
    evidence = [
        ImplResult(status="done", new_score=0.55, steps=12),
        ImplResult(status="done", new_score=0.54, steps=10),
        ImplResult(status="done", new_score=0.56, steps=11),
    ]
    result = analyst.evaluate(method=make_method(), evidence=evidence,
                                paper=PaperContext(title="s1"))
    assert isinstance(result, ExperimentAnalysis)
    assert result.satisfied is True
    assert result.next_experiment is None
    assert "Method beats baseline" in result.key_observations[0]


def test_analyst_proposes_seed_sweep_when_only_one_run():
    llm = MagicMock()
    llm.chat.return_value = json.dumps({
        "satisfied": False,
        "fidelity_ok": True,
        "fidelity_issues": [],
        "breadth_gaps": ["variance: only one seed"],
        "next_experiment": {
            "kind": "seed_sweep",
            "description": "Re-run with seeds 1,2,3 and report FINAL_SCORE_MEAN/STD.",
            "seeds": [1, 2, 3],
        },
        "key_observations": [],
    })
    analyst = ExperimentAnalyst(llm=llm, prompts=PromptLoader(PROMPTS))
    result = analyst.evaluate(
        method=make_method(),
        evidence=[ImplResult(status="done", new_score=0.55, steps=8)],
        paper=PaperContext(title="s1"),
    )
    assert result.satisfied is False
    assert result.next_experiment.kind == "seed_sweep"
    assert result.next_experiment.seeds == [1, 2, 3]


def test_analyst_flags_fidelity_drift():
    llm = MagicMock()
    llm.chat.return_value = json.dumps({
        "satisfied": False,
        "fidelity_ok": False,
        "fidelity_issues": [
            "Method calls for confidence-conditioned budget; code uses fixed budget * 1.5",
        ],
        "breadth_gaps": [],
        "next_experiment": {
            "kind": "fidelity_fix",
            "description": "Reimplement: gate budget on per-token confidence as the method specifies.",
        },
        "key_observations": [],
    })
    analyst = ExperimentAnalyst(llm=llm, prompts=PromptLoader(PROMPTS))
    result = analyst.evaluate(
        method=make_method(),
        evidence=[ImplResult(status="done", new_score=0.51, steps=6,
                              trace=[{"tool": "Bash", "args": {"command": "sed ..."}}])],
        paper=PaperContext(title="s1"),
    )
    assert result.fidelity_ok is False
    assert result.next_experiment.kind == "fidelity_fix"
```

**Step 2: Run test → expect ImportError**

```bash
.venv/bin/pytest tests/unit/test_agent_analyst.py -v
```

**Step 3: Write `prompts/analyst.yaml`**

```yaml
system: |
  You are the Experiment Analyst for PaperGym. After each Implementer run on
  a candidate method, you decide:
    1. Empirical breadth — does the accumulated evidence cover the 4 axes
       (tasks/metrics/variance/ablation) at a level a strict ICML reviewer
       would accept?
    2. Code–idea fidelity — looking at the Implementer trace, does the code
       it wrote actually instantiate the method NL spec, or did it drift
       (e.g., method = "confidence-gated budget", code = "budget *= 1.5")?
    3. Key observations — what does the evidence reveal about the method's
       actual behavior? (Will be fed back to Methodologist next outer.)

  Be calibrated. If evidence is one run with one metric, you are NOT
  satisfied. Conversely, do NOT demand exhaustive coverage on early outers —
  ~3 seeds + 1 ablation + 1 alt metric is a healthy minimum.

  When breadth is incomplete OR fidelity has drifted, propose ONE concrete
  next_experiment for the Implementer (not several).

user: |
  ## Paper
  {{ paper.title }}

  ## Method under evaluation
  {{ method.method }}
  Rationale: {{ method.rationale }}
  Observable signal: {{ method.observable_signal }}

  ## Evidence so far ({{ evidence | length }} runs)
  {% for r in evidence %}
  - run {{ loop.index }}: status={{ r.status }}, new_score={{ r.new_score }}, steps={{ r.steps }}{% if r.reason %}, reason={{ r.reason }}{% endif %}
  {% endfor %}

  ## Implementer trace summary (latest run)
  {{ latest_trace_summary }}

  Decide: satisfied? fidelity_ok? what's the gap? what's the next experiment?

  Return JSON with EXACTLY these keys (no extras):
  {
    "satisfied": bool,
    "fidelity_ok": bool,
    "fidelity_issues": [str],
    "breadth_gaps": [str],
    "next_experiment": null OR {
      "kind": "seed_sweep" | "ablation" | "additional_metric" | "additional_dataset" | "fidelity_fix",
      "description": str,
      "seeds": [int] | null,
      "metric_override": str | null,
      "dataset_override": str | null
    },
    "key_observations": [str]
  }
```

**Step 4: Write `agents/analyst.py`**

```python
from .base import BaseAgent, PromptLoader
from ..llm import LLMClient
from ..paper import PaperContext
from ..types import (ExperimentAnalysis, ExperimentSpec, ImplResult,
                      MethodSpec)


def _summarize_trace(impl: ImplResult, max_steps: int = 8) -> str:
    if not impl.trace:
        return "(no trace)"
    lines = []
    for step in impl.trace[:max_steps]:
        tool = step.get("tool", "?")
        args = step.get("args", {})
        cmd = args.get("command") or args.get("file_path") or args.get("query") or ""
        lines.append(f"- {tool}: {str(cmd)[:120]}")
    if len(impl.trace) > max_steps:
        lines.append(f"... ({len(impl.trace) - max_steps} more steps)")
    return "\n".join(lines)


class ExperimentAnalyst(BaseAgent):
    def __init__(self, llm: LLMClient, prompts: PromptLoader):
        super().__init__(llm=llm, prompts=prompts, prompt_name="analyst",
                         temperature=0.3)

    def evaluate(self, *, method: MethodSpec, evidence: list[ImplResult],
                 paper: PaperContext) -> ExperimentAnalysis:
        latest = evidence[-1] if evidence else None
        result = self.call(
            method=method, evidence=evidence, paper=paper,
            latest_trace_summary=_summarize_trace(latest) if latest else "(no runs)",
        )
        spec = None
        if result.get("next_experiment"):
            ne = result["next_experiment"]
            spec = ExperimentSpec(
                kind=ne["kind"],
                description=ne["description"],
                seeds=ne.get("seeds"),
                metric_override=ne.get("metric_override"),
                dataset_override=ne.get("dataset_override"),
            )
        return ExperimentAnalysis(
            satisfied=result["satisfied"],
            fidelity_ok=result["fidelity_ok"],
            fidelity_issues=result.get("fidelity_issues", []),
            breadth_gaps=result.get("breadth_gaps", []),
            next_experiment=spec,
            key_observations=result.get("key_observations", []),
        )
```

**Step 5: Run tests**

```bash
.venv/bin/pytest tests/unit/test_agent_analyst.py -v
```

Expected: 3 pass.

**Step 6: Stage**

```bash
git add src/papergym/agents/analyst.py src/papergym/prompts/analyst.yaml tests/unit/test_agent_analyst.py
```

---

## Task 4 — Pipeline rewrite: outer envelope + Loop A + Loop B + container reuse

**Files:**
- Modify: `src/papergym/pipeline.py`
- Modify: `tests/unit/test_pipeline.py`

This is the biggest change. Rewrite `Pipeline.run()` to orchestrate the two-loop alternation. Drop `deployment_factory(it)`; accept a single `deployment` instance.

**Step 1: Rewrite Pipeline constructor signature**

Replace `deployment_factory` and `implementer_factory` parameters with a single shared `deployment`:

```python
class Pipeline:
    def __init__(self, paper: PaperContext, baseline_score: float,
                 observer, methodologist, implementer, analyst,
                 scorer, writer, reviewer,
                 store: LibraryStore, retriever: LibraryRetriever,
                 embed_fn: Callable[[str], list[float]],
                 prompts: PromptLoader,
                 experiment_dir: Path,
                 deployment: Deployment,           # NEW: single shared
                 outer_iters: int = 3,
                 loop_a_max: int = 10,
                 loop_b_max: int = 10,
                 loop_a_threshold_ratio: float = 0.01,
                 flags: Optional[AblationFlags] = None,
                 paraphrase_llm=None,
                 baseline_log_path: Optional[Path] = None,
                 baseline_pdf_path: Optional[Path] = None,
                 baseline_repo_path: Optional[Path] = None,
                 logger=None):
        ...
        self.deployment = deployment
        self.analyst = analyst
        self.outer_iters = outer_iters
        self.loop_a_max = loop_a_max
        self.loop_b_max = loop_b_max
        self.loop_a_threshold_ratio = loop_a_threshold_ratio
```

Drop the old `max_iters`, `deployment_factory`, `implementer_factory` fields.

**Step 2: Rewrite `Pipeline.run()` to the two-loop architecture**

```python
def run(self) -> PipelineState:
    state = PipelineState(paper=self.paper, baseline_score=self.baseline_score)
    outer_state = OuterState()
    threshold = self.baseline_score * self.loop_a_threshold_ratio

    # Build initial Observer signals once (paper-level, not iter-level)
    signals = self.observer.collect_signals(
        repo_path=self.baseline_repo_path,
        paper_text_path=self.baseline_pdf_path,
        log_path=self.baseline_log_path,
    )

    for outer in range(self.outer_iters):
        ctx = IterationContext(paper=self.paper, iter=outer)

        # Observer (re-evaluate problems each outer with deep_observations context)
        problems, _, traj = self._run_stage(
            "observer", outer, self.observer,
            self.observer.observe, ctx, signals=signals,
        )
        self._emit("stage_end", stage="observer", iter=outer, output={
            "n_problems": len(problems)}, trajectory=traj)
        if not problems:
            outer_state.history.append(OuterRecord(outer=outer, status="loop_a_failed"))
            continue
        ctx.problem = problems[0]

        # Retrieve seeds (library)
        ctx.retrieved_seeds, bucket_dict = self._retrieve(ctx.problem.problem)
        ctx.retrieved_seed_buckets.strong = bucket_dict["strong"]
        ctx.retrieved_seed_buckets.variant = bucket_dict["variant"]
        ctx.retrieved_seed_buckets.avoid = bucket_dict["avoid"]

        # ── Loop A ────────────────────────────────────────────────
        chosen = None
        candidates = []
        for a in range(self.loop_a_max):
            method = self.methodologist.propose(
                ctx,                                 # methodologist already reads ctx
                # extra context: prev failures + deep observations
                # (need methodologist signature update — Task 4b below)
            )
            ctx.method = method
            impl = self.implementer.run(ctx, workspace="/workspace")
            candidates.append((method, impl))
            if impl.status == "done" and impl.new_score is not None \
                    and (impl.new_score - self.baseline_score) > threshold:
                chosen = (method, impl)
                break

        if chosen is None:
            outer_state.failed_methods.extend(m for m, _ in candidates)
            outer_state.history.append(OuterRecord(
                outer=outer, status="loop_a_failed", candidates=candidates))
            continue

        chosen_method, chosen_impl = chosen
        ctx.method = chosen_method
        ctx.impl = chosen_impl

        # ── Loop B ────────────────────────────────────────────────
        evidence = [chosen_impl]
        analysis = None
        for b in range(self.loop_b_max):
            analysis = self.analyst.evaluate(
                method=chosen_method, evidence=evidence, paper=self.paper)
            if analysis.satisfied:
                break
            ctx.experiment_spec = analysis.next_experiment
            impl = self.implementer.run(ctx, workspace="/workspace")
            evidence.append(impl)
        ctx.experiment_spec = None  # clear for cleanliness

        outer_state.evidence.extend(evidence)
        if analysis and analysis.key_observations:
            outer_state.deep_observations.extend(analysis.key_observations)
        outer_state.history.append(OuterRecord(
            outer=outer, status="completed", method=chosen_method,
            chosen_impl=chosen_impl, evidence=evidence, analysis=analysis))

    # ── Finalize: Writer + web_search + Reviewer (1x) ─────────────
    best = outer_state.best_outer
    if best is None:
        return state

    ctx = IterationContext(paper=self.paper, iter=-1,
                            method=best.method, impl=best.chosen_impl)
    ctx.score = self.scorer.score(ctx, baseline_score=self.baseline_score)
    state.history.append(IterationRecord(iter=-1, ctx=ctx))
    seed = self._make_seed(ctx)
    if not self.flags.disable_library:
        self.store.add(seed)

    ctx.ideation_doc = self.writer.write(ctx, baseline_score=self.baseline_score)
    web_results = web_search.semantic_scholar_search(
        ctx.method.method, k=self.flags.novelty_k)

    if self.flags.disable_reviewer:
        state.accepted_seed = seed
        return state

    ri = ReviewerInput(
        ideation_doc=ctx.ideation_doc,
        delta_bench=ctx.score.delta_bench,
        benchmark=self.paper.benchmark_name or "(unspecified)",
        web_search_results=web_results,
    )
    review = self.reviewer.review(ri)
    if review.verdict == "accept":
        state.accepted_seed = seed
    return state

# No pipeline-forced cleanup. Agent re-clones via Bash if needed.
```

**Step 2b: Methodologist needs to see `failed_methods` and `deep_observations`**

Update `methodologist.propose()` and prompt to render those. Add to `IterationContext` (already done in Task 1 via `OuterState` — but we need pipeline to inject these into ctx before each call):

In Pipeline's Loop A iteration, before `methodologist.propose(ctx)`:

```python
ctx.prior_attempts = [
    PriorAttempt(problem=ctx.problem.problem, method=m.method,
                 delta_bench=(i.new_score or 0) - self.baseline_score,
                 verdict="reject",
                 feedback=f"loop_a iter {idx}: {i.reason or i.status}")
    for idx, (m, i) in enumerate(candidates)
] + [
    PriorAttempt(problem=ctx.problem.problem, method=m.method,
                 delta_bench=0.0, verdict="reject",
                 feedback="failed in earlier outer")
    for m in outer_state.failed_methods
]
# attach deep observations as a new ctx field
ctx.deep_observations = list(outer_state.deep_observations)
```

Add `deep_observations: list[str] = field(default_factory=list)` to `IterationContext` (Task 1 step 2 follow-up).

Update `prompts/methodologist.yaml` to render new sections:

```yaml
{% if ctx.deep_observations %}
## Deep observations from prior outer iterations (do NOT ignore these)
{% for o in ctx.deep_observations %}
- {{ o }}
{% endfor %}
{% endif %}
```

**Step 3: Update tests**

Most existing `test_pipeline.py` tests assume `target=...` (already migrated to `paper=`) and the old single-loop signature. Update `make_pipe()` helper:
- Remove `deployment_factory` / `implementer_factory` mock setup
- Add `deployment` (`MagicMock()`) and `analyst` (`MagicMock()`) parameters
- Default `outer_iters=1, loop_a_max=2, loop_b_max=1` so tests stay fast

Add new tests:

```python
def test_pipeline_outer_envelope_runs_three_times_when_all_succeed(tmp_path):
    """Outer loop runs N times even after early acceptance — Reviewer is final."""
    paper = make_paper()
    deployment = MagicMock()
    analyst = MagicMock()
    analyst.evaluate.return_value = ExperimentAnalysis(
        satisfied=True, fidelity_ok=True, key_observations=["obs"])
    pipe, _ = make_pipe(
        paper, deployment=deployment, analyst=analyst,
        observer_problems=[Problem(problem="p", observation="o", source_signal="code")],
        method=MethodSpec(method="m", observable_signal="s", rationale="r"),
        impl=ImplResult(status="done", new_score=0.6, steps=2),  # > baseline 0.5 by 0.1 > threshold
        score=ScoreResult(delta_bench=0.1),
        review=ReviewVerdict(verdict="accept", scores={}, feedback=""),
        tmp_path=tmp_path, outer_iters=3,
    )
    with patch("papergym.pipeline.web_search.semantic_scholar_search", return_value=[]):
        state = pipe.run()
    # Loop A succeeds first try each outer, Loop B satisfies first call each outer
    # Implementer should be called 3 times (one per outer's Loop A initial attempt)
    assert pipe.implementer.run.call_count == 3
    assert analyst.evaluate.call_count == 3


def test_pipeline_loop_a_rejects_until_threshold(tmp_path):
    """Loop A keeps trying until (Δ > baseline × threshold_ratio)."""
    paper = make_paper()
    impls = iter([
        ImplResult(status="done", new_score=0.50, steps=1),  # Δ=0, fail
        ImplResult(status="done", new_score=0.52, steps=1),  # Δ=0.02 > 0.5*0.01=0.005, pass
    ])
    implementer = MagicMock()
    implementer.run.side_effect = lambda *a, **kw: next(impls)
    analyst = MagicMock()
    analyst.evaluate.return_value = ExperimentAnalysis(
        satisfied=True, fidelity_ok=True, key_observations=[])

    deployment = MagicMock()
    store = LibraryStore(tmp_path / "s.jsonl", tmp_path / "e.pkl")
    pipe = Pipeline(
        paper=paper, baseline_score=0.5,
        observer=_observer_returning([Problem(problem="p", observation="o",
                                                source_signal="code")]),
        methodologist=_methodologist_returning(
            MethodSpec(method="m", observable_signal="s", rationale="r")),
        implementer=implementer,
        analyst=analyst,
        scorer=MagicMock(score=MagicMock(return_value=ScoreResult(delta_bench=0.02))),
        writer=MagicMock(write=MagicMock(return_value="# doc")),
        reviewer=MagicMock(review=MagicMock(return_value=ReviewVerdict(
            verdict="accept", scores={}, feedback=""))),
        store=store,
        retriever=LibraryRetriever(store, score_high=0.05, score_low=0.01, k=5),
        embed_fn=lambda t: [0.0],
        prompts=PromptLoader(PROMPTS),
        experiment_dir=tmp_path, deployment=deployment,
        outer_iters=1, loop_a_max=5, loop_b_max=1,
    )
    with patch("papergym.pipeline.web_search.semantic_scholar_search", return_value=[]):
        pipe.run()
    assert implementer.run.call_count == 2  # 1 reject + 1 accept


def test_pipeline_loop_a_all_fail_skips_loop_b(tmp_path):
    """If all Loop A attempts fail, Loop B is skipped and outer advances."""
    paper = make_paper()
    implementer = MagicMock()
    implementer.run.return_value = ImplResult(status="done", new_score=0.50, steps=1)  # Δ=0
    analyst = MagicMock()
    deployment = MagicMock()
    store = LibraryStore(tmp_path / "s.jsonl", tmp_path / "e.pkl")
    pipe = Pipeline(
        paper=paper, baseline_score=0.5,
        observer=_observer_returning([Problem(problem="p", observation="o",
                                                source_signal="code")]),
        methodologist=_methodologist_returning(
            MethodSpec(method="m", observable_signal="s", rationale="r")),
        implementer=implementer, analyst=analyst,
        scorer=MagicMock(), writer=MagicMock(), reviewer=MagicMock(),
        store=store,
        retriever=LibraryRetriever(store, score_high=0.05, score_low=0.01, k=5),
        embed_fn=lambda t: [0.0], prompts=PromptLoader(PROMPTS),
        experiment_dir=tmp_path, deployment=deployment,
        outer_iters=1, loop_a_max=3, loop_b_max=5,
    )
    with patch("papergym.pipeline.web_search.semantic_scholar_search", return_value=[]):
        pipe.run()
    assert implementer.run.call_count == 3  # all 3 Loop A attempts
    analyst.evaluate.assert_not_called()    # Loop B skipped


def test_pipeline_loop_b_expands_until_satisfied(tmp_path):
    """Loop B runs until analyst.satisfied=True, propagating ExperimentSpec."""
    # ... 3 expansion calls then satisfied; assert experiment_spec was set
```

(Add helpers `_observer_returning`, `_methodologist_returning` for cleaner tests.)

**Step 4: Update existing test `test_pipeline_uses_fresh_deployment_per_iteration`**

Rename to `test_pipeline_reuses_single_deployment_across_iterations` and assert `deployment.start.assert_not_called()` (Pipeline doesn't start it; container is started by the caller).

**Step 5: Run tests**

```bash
.venv/bin/pytest tests/unit/test_pipeline.py -v
```

Iterate until green.

**Step 6: Stage**

```bash
git add src/papergym/pipeline.py src/papergym/types.py \
        src/papergym/prompts/methodologist.yaml \
        tests/unit/test_pipeline.py
```

---

## Task 5 — `run_pipeline.py`: single deployment per paper + analyst wiring

**Files:**
- Modify: `scripts/run_pipeline.py`

**Step 1: Replace per-iter `deployment_factory` lambda with one deployment per paper**

Current `_run_for_paper` already creates `dep` for baseline. Restructure so `dep` lives across baseline + Pipeline:

```python
def _run_for_paper(title: str, authors: str, args, cfg: Config, image: str):
    paper = PaperContext(title=title, authors=authors)
    hint_slug = _paper_slug_hint(title)
    experiment_dir = cfg.experiment_root / f"{hint_slug}_experiment_result"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    paper.repo_slug = hint_slug

    workspace_host = experiment_dir / "workspace"   # single workspace per paper
    dep = make_deployment(
        backend=args.backend, image=image,
        unit_name=hint_slug,
        gpus=_normalize_gpus(args.gpus),
        workspace_host=workspace_host,
        experiment_result_host=experiment_dir,
        hf_cache_host=cfg.hf_cache_path,
    )

    with RunLogger(experiment_dir) as logger:
        logger.event("pipeline_start", paper=title)
        dep.start()
        try:
            baseline_score = _run_baseline_in_dep(paper, args, cfg, logger,
                                                    experiment_dir, dep)
            llm = LLMClient(api_key=cfg.openai_api_key, chat_model=cfg.chat_model,
                             embedding_model=cfg.embedding_model)
            prompts = PromptLoader(...)
            store = LibraryStore(cfg.library_path, cfg.embeddings_cache_path)
            retriever = LibraryRetriever(store, ...)
            implementer = Implementer(llm=llm, prompts=prompts, deployment=dep,
                                       max_steps=args.implementer_max_steps)

            pipe = Pipeline(
                paper=paper, baseline_score=baseline_score,
                observer=Observer(llm=llm, prompts=prompts),
                methodologist=Methodologist(llm=llm, prompts=prompts),
                implementer=implementer,
                analyst=ExperimentAnalyst(llm=llm, prompts=prompts),
                scorer=Scorer(),
                writer=Writer(llm=llm, prompts=prompts),
                reviewer=Reviewer(llm=llm, prompts=prompts),
                store=store, retriever=retriever, embed_fn=llm.embed,
                prompts=prompts, experiment_dir=experiment_dir,
                deployment=dep,
                outer_iters=args.outer_iters,
                loop_a_max=args.loop_a_max,
                loop_b_max=args.loop_b_max,
                flags=AblationFlags(
                    disable_library=args.disable_library,
                    disable_compounding=args.disable_compounding,
                    disable_reviewer=args.disable_reviewer,
                    disable_reproduce=args.disable_reproduce,
                ),
                paraphrase_llm=llm,
                baseline_log_path=experiment_dir / "baseline" / "stdout.log",
                baseline_repo_path=workspace_host,
                logger=logger,
            )
            state = pipe.run()
            logger.event("pipeline_end",
                          accepted=state.accepted_seed is not None)
            if state.accepted_seed:
                doc_dir = experiment_dir / "ideation_docs"
                doc_dir.mkdir(parents=True, exist_ok=True)
                (doc_dir / f"{state.accepted_seed.seed_id}.md").write_text(
                    state.history[-1].ctx.ideation_doc or "")
        finally:
            dep.stop()
```

`_run_baseline` becomes `_run_baseline_in_dep` and takes the externally-managed `dep` instead of creating its own.

**Step 2: New CLI args**

```python
p.add_argument("--outer-iters", type=int, default=3)
p.add_argument("--loop-a-max", type=int, default=10)
p.add_argument("--loop-b-max", type=int, default=10)
```

Replace the old `--max-iters` with `--outer-iters`. Bootstrap mode (next task) overrides these.

**Step 3: Stage**

```bash
git add scripts/run_pipeline.py
```

---

## Task 6 — Bootstrap mode: outer=1, Loop B skipped, Reviewer skipped

**Files:**
- Modify: `scripts/run_pipeline.py`
- Modify: `src/papergym/pipeline.py` (just verify ablation flag interaction)

**Step 1: Add `--bootstrap` shortcut flag**

In `parse_args`:

```python
p.add_argument("--bootstrap", action="store_true",
               help="Bootstrap shortcut: --outer-iters=1 --loop-b-max=0 "
                    "--disable-library --disable-reviewer (overrides those flags).")
```

After parsing:

```python
if args.bootstrap:
    args.outer_iters = 1
    args.loop_b_max = 0
    args.disable_library = True
    args.disable_reviewer = True
```

**Step 2: Update Pipeline to honor `loop_b_max=0`**

Already works if Loop B's `for b in range(0)` simply doesn't execute. Add an early-out so Analyst is never instantiated/called when `loop_b_max == 0`:

```python
if self.loop_b_max == 0 or self.analyst is None:
    # bootstrap path: no expansion, no analyst
    evidence = [chosen_impl]
    analysis = None
else:
    # ... existing Loop B logic
```

**Step 3: Update `bootstrap/seed_surveys.yaml` README block in main README to reference `--bootstrap`**

In `README.md`, replace bootstrap example:

```bash
.venv/bin/python scripts/run_pipeline.py \
    --papers bootstrap/seed_surveys.yaml --bootstrap \
    --implementer-max-steps 500
```

**Step 4: Update slurm wrapper**

`scripts/slurm_papergym.sh`'s usage comment for bootstrap should reference `--bootstrap` flag.

**Step 5: Test bootstrap path stays single-shot**

Add to `tests/unit/test_pipeline.py`:

```python
def test_bootstrap_path_skips_loop_b_and_reviewer(tmp_path):
    paper = make_paper()
    implementer = MagicMock()
    implementer.run.return_value = ImplResult(status="done", new_score=0.6, steps=1)
    analyst = MagicMock()  # should NOT be called
    reviewer = MagicMock()  # should NOT be called

    store = LibraryStore(tmp_path / "s.jsonl", tmp_path / "e.pkl")
    pipe = Pipeline(
        paper=paper, baseline_score=0.5,
        observer=_observer_returning([Problem(problem="p", observation="o",
                                                source_signal="code")]),
        methodologist=_methodologist_returning(
            MethodSpec(method="m", observable_signal="s", rationale="r")),
        implementer=implementer, analyst=analyst,
        scorer=MagicMock(score=MagicMock(return_value=ScoreResult(delta_bench=0.1))),
        writer=MagicMock(write=MagicMock(return_value="# doc")),
        reviewer=reviewer,
        store=store,
        retriever=LibraryRetriever(store, score_high=0.05, score_low=0.01, k=5),
        embed_fn=lambda t: [0.0], prompts=PromptLoader(PROMPTS),
        experiment_dir=tmp_path, deployment=MagicMock(),
        outer_iters=1, loop_a_max=2, loop_b_max=0,
        flags=AblationFlags(disable_library=True, disable_reviewer=True),
    )
    with patch("papergym.pipeline.web_search.semantic_scholar_search", return_value=[]):
        pipe.run()
    analyst.evaluate.assert_not_called()
    reviewer.review.assert_not_called()
    assert store.all() == []   # disable_library — nothing stored
```

**Step 6: Run tests**

```bash
.venv/bin/pytest tests/unit -q
```

Expected: all pass.

**Step 7: Stage**

```bash
git add scripts/run_pipeline.py src/papergym/pipeline.py \
        tests/unit/test_pipeline.py README.md scripts/slurm_papergym.sh
```

---

## Task 7 — README + design doc cross-links

**Files:**
- Modify: `README.md`

**Step 1: Add a "Pipeline architecture" subsection under "End-to-End Workflow"**

After the existing flow ASCII diagram, add:

```markdown
### Pipeline architecture (ideation mode)

Per paper, the agent runs a 3-step **outer envelope**, alternating two
inner loops:

- **Loop A (Methodologist ↔ Implementer, max 10 attempts)** — proposes a
  candidate method, runs it once. Exits as soon as Δ > baseline × 0.01;
  otherwise rejects and tries another method.
- **Loop B (ExperimentAnalyst ↔ Implementer, max 10 attempts)** — fixes the
  chosen method, expands evidence (multi-seed, ablation, alt metrics) and
  checks code-vs-idea fidelity. Exits when the analyst is satisfied.

Loop B's deep observations feed back into Loop A in the next outer step,
letting Methodologist redesign on the basis of empirical findings rather
than just reviewer rejections. Reviewer is called once at the end.

Bootstrap mode (`--bootstrap`) runs a single outer step, skips Loop B and
Reviewer, and just stores the seed.
```

**Step 2: Update flag table**

Add `--outer-iters`, `--loop-a-max`, `--loop-b-max`, `--bootstrap` to the ablation flag listing.

**Step 3: Stage**

```bash
git add README.md
```

---

## Acceptance criteria

- [ ] `pytest tests/unit -q` → all pass
- [ ] `scripts/run_pipeline.py --paper "<title>" --outer-iters 1 --loop-a-max 2 --loop-b-max 1` runs end-to-end on a small test paper without crashing
- [ ] `scripts/run_pipeline.py --papers bootstrap/seed_surveys.yaml --bootstrap` populates `library/seeds.jsonl` with one seed per paper, no Reviewer calls in `events.jsonl`
- [ ] One paper = one container start + stop in `events.jsonl` (verify per-paper container reuse)
- [ ] Loop B's `next_experiment` actually changes Implementer behavior (verify via `events.jsonl` trajectories on a `seed_sweep` spec)
- [ ] `git status` shows everything staged but no commits made (user owns commits)
