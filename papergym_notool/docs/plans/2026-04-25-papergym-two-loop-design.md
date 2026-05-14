# PaperGym Two-Loop Refinement Design

Supplements `2026-04-23-papergym-redesign.md`. Adds explicit experiment-refinement loop so empirical breadth no longer depends on Reviewer rejection feedback.

## Motivation

Current pipeline = `Implementer (single shot) → Writer → Reviewer`. Reviewer reliably rejects with "needs more breadth" (no variance, single metric, no ablation). Methodologist's job is method redesign, not experiment expansion, so the rejection feedback bounces between the wrong two stages and the loop diverges instead of converges.

Reference: AI-Researcher uses an `Experiment Analyser → further_plan → ML Agent` loop to expand experiments after the first run, but caps at 2 iters and skips multi-seed variance / stress / overhead. PaperGym extends this with explicit alternation.

## Architecture

```
state = {
    evidence: list[ImplResult],          # all runs across all outers
    deep_observations: list[str],        # B's findings, fed back to A
    failed_methods: list[MethodSpec],    # A's rejects, fed forward
    history: list[OuterRecord],
}

for outer in range(3):                                 # outer envelope
    # ── Loop A: Methodologist ↔ Implementer (max 10) ─────────────────
    method_candidates = []
    for a in range(10):
        method = methodologist.propose(
            paper, problem, retrieved_seeds,
            deep_observations=state.deep_observations,
            prev_attempts=method_candidates + state.failed_methods,
        )
        impl = implementer.run(method, mode="initial")    # single seed/metric
        method_candidates.append((method, impl))
        if impl.status == "done" and (impl.new_score - baseline) > baseline * 0.01:
            chosen_method, chosen_impl = method, impl
            break
        if a == 4:    # nudge after 5th reject
            inject_signal_to_methodologist("5 rejects so far, propose a bolder change")
    else:
        # all 10 rejected — Loop B skipped this outer
        state.failed_methods.extend(m for m, _ in method_candidates)
        state.history.append(OuterRecord(outer=outer, status="loop_a_failed",
                                           candidates=method_candidates))
        continue

    # ── Loop B: ExperimentAnalyst ↔ Implementer (max 10) ──────────────
    evidence = [chosen_impl]
    analysis = None
    for b in range(10):
        analysis = analyst.evaluate(
            method=chosen_method, evidence=evidence, paper=paper,
        )
        if analysis.satisfied:
            break
        impl = implementer.run(chosen_method, mode="expand",
                                spec=analysis.next_experiment)
        evidence.append(impl)
    state.evidence.extend(evidence)
    state.deep_observations.extend(analysis.key_observations)
    state.history.append(OuterRecord(outer=outer, status="completed",
                                       method=chosen_method, evidence=evidence,
                                       analysis=analysis))

# ── one-shot finalization ─────────────────────────────────────────────
ideation_doc = writer.write(state)
web_results  = web_search(state.best_method)
verdict      = reviewer.review(ideation_doc, state.evidence, web_results)   # 1x
```

## Decisions (locked)

| | Decision |
|---|---|
| Loop A exit threshold | `Δ > baseline_score × 0.01` |
| Loop A all-fail | Skip Loop B, accumulate `failed_methods`, advance outer |
| Loop A nudge | After 5th reject, inject "propose bolder change" signal into next Methodologist call |
| Loop B exit | `analyst.satisfied = True` |
| Outer envelope | 3 |
| Per-outer Implementer cap | None (max 20 = A 10 + B 10) |
| Reviewer | Called once after outer loop ends |
| Bootstrap mode | `outer=1`, Loop B skipped, Reviewer skipped (single-shot to populate library) |
| Container model | One container per paper (reused across baseline + all iters); no pipeline-forced cleanup — agent re-clones or partial-resets via Bash if needed |

## New components

### `ExperimentAnalyst` agent (new)

Three responsibilities:
1. **Empirical breadth gap detection** — 4 axes: tasks/metrics/variance/ablation. Compare current evidence against ICML rubric.
2. **Code-idea fidelity check** — read Implementer's trace (Read/Write/Bash sequence). Does the code actually instantiate the proposed method, or did Implementer drift (e.g., method = "confidence-conditioned budget", code = `budget *= 1.5` constant)?
3. **Key observation extraction** — what does the accumulated evidence reveal about the method's actual behavior? (fed back to Methodologist next outer)

### Dataclasses

```python
@dataclass
class ExperimentSpec:
    """What the next Implementer expansion run should do."""
    kind: Literal["seed_sweep", "ablation", "additional_metric",
                  "additional_dataset", "fidelity_fix"]
    description: str               # NL spec the Implementer follows
    seeds: list[int] | None = None
    metric_override: str | None = None
    dataset_override: str | None = None

@dataclass
class ExperimentAnalysis:
    satisfied: bool
    fidelity_ok: bool
    fidelity_issues: list[str]
    breadth_gaps: list[str]
    next_experiment: ExperimentSpec | None
    key_observations: list[str]
```

### Implementer (no mode split)

Implementer signature unchanged. The branch happens via an optional field on `IterationContext`:

```python
@dataclass
class IterationContext:
    ...
    method: Optional[MethodSpec] = None
    experiment_spec: Optional[ExperimentSpec] = None   # NEW
```

Pipeline:
- Loop A → `ctx.experiment_spec = None`. Implementer does default flow (single seed/metric).
- Loop B → `ctx.experiment_spec = analysis.next_experiment`. Implementer prompt renders an extra block:

```yaml
{% if ctx.experiment_spec %}
## Additional experiment instructions
{{ ctx.experiment_spec.description }}
The repo is freshly cloned in this container; first apply the method above,
then perform the experiment described here. Print FINAL_SCORE per the spec
(may include FINAL_SCORE_MEAN / FINAL_SCORE_STD when running multiple seeds).
{% endif %}
```

Each call still = one fresh container = one `ImplResult`. Container reuse is a separate optimization, not part of this design.

### Pipeline state extensions

Add to `IterationContext` / new `OuterState`:
- `deep_observations: list[str]`  — Loop B → Loop A transfer
- `failed_methods: list[MethodSpec]` — Loop A → next outer transfer
- `evidence: list[ImplResult]` — accumulated empirical record

## Bootstrap path

Single-shot kept identical to current (after redesign) behavior:

```python
if mode == "bootstrap":
    method = methodologist.propose(...)
    impl = implementer.run(method, mode="initial")
    if impl.status == "done":
        store.add(_make_seed(impl, method, problem))
    return
```

No outer loop, no Loop B, no Reviewer. Bootstrap goal = populate `library/seeds.jsonl` with `(problem_embedding, method_NL, Δ-sign)` triples for retrieval; deep validation is wasted at this stage.

## Removed assumptions

- Reviewer rejection no longer drives experiment expansion — it only enforces the final accept/reject of the integrated narrative.
- Methodologist no longer bears responsibility for "more experiments" — it only does idea redesign.

## Container model — per-paper reuse (β)

**One container per paper, reused across baseline + every Loop A/B iter.** Not per-iteration. The cleanliness boundary is between *papers*, not between iterations of the same paper.

- One `SingularityDeployment` (or `DockerDeployment`) instance per paper, started once before Loop A iter 0, stopped after the outer loop finishes.
- baseline does the initial `git clone` + dep install + first benchmark inside that container.
- Each Loop A iter applies its method via Implementer's Read/Write/Bash tools on the same `/workspace/repo` checkout.
- Each Loop B iter operates on the chosen method's modified worktree (or with Analyst's `experiment_spec`).
- **No pipeline-driven cleanup**. The agent decides — if a previous attempt polluted the worktree, the agent calls `Bash("rm -rf /workspace/repo && git clone <github_url> /workspace/repo")` itself (re-clone is cheap; model weights / pip deps / HF cache live elsewhere in the container and survive). Or the agent does partial recovery via `git checkout -- specific_file.py`. Pipeline never forces it.
- Implementer prompt mentions this option so the LLM knows it has the choice.

Pipeline change vs current code:
- Drop `deployment_factory(it)` pattern. `_run_for_paper` creates one `dep`, starts it, passes it to baseline + Pipeline. Pipeline holds `self.deployment` and uses it directly for every Implementer call.
- `Implementer.__init__(deployment=...)` already takes a deployment — reuse one instance across all calls.

## Cost envelope

Per paper, worst case = `3 outers × (10 + 10) = 60 Implementer calls`, but only **1 container start + 1 container stop**. Per-call cost = LLM steps + `git checkout` (cheap) + benchmark run. Model download + dep install paid only once at baseline. Realistic walltime: dominated by per-call inference, not container churn.

Cross-paper isolation preserved: each paper still gets its own container instance with its own host-mounted workspace, so model/deps/cache mixing across papers is impossible.

Primary cost lever: Loop B's `analyst.satisfied` threshold. Set it to favor "directional empirical breadth" over "exhaustive coverage" for first runs.
