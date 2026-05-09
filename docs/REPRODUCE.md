# Reproducing the paper

Each paper number maps to the script behind it and the file it was read from. [`scripts/reproduce_paper.sh`](../scripts/reproduce_paper.sh) runs every step in order.

Run all commands from the repo root with `OPENAI_API_KEY` and Bedrock credentials set ([README setup](../README.md#setup)).

Stage 2/3 read from [`data/library/`](../data/library/) and [`data/queries.yaml`](../data/queries.yaml). The released library ships with the repo; rebuild it via [Bootstrap](../README.md#bootstrap) if needed.

## Conventions

- `<JUDGE>` = `openai/bedrock.anthropic.claude-sonnet-4-6` (held-out judge family, different from the GPT-5 generator)
- `<LIB>` = `data/library` (path to the built seed library)
- `<RUN>` = the timestamped directory the previous step wrote to (e.g. `data/eval/ideation_20260507T090814`)

## Stage 1 — Tool-augmented seed extraction (Section 3.2)

The A condition (no-tool extraction baseline) lives in [PaperGym_notool](https://github.com/yunjoochoi/PaperGym_notool); the judges below run here and score both libraries.

| Paper claim | Script | Output file | Field |
|---|---|---|---|
| Specificity 4.22 → 4.76 (n=30 papers, paper-level mean) | [`scripts/seed_quality_eval.py`](../scripts/seed_quality_eval.py) | `data/eval/20260502T230414/summary.json` | `A.axes.specificity.mean`, `C.axes.specificity.mean` |
| Grounding 4.82 (both conditions) | same | same | `A.axes.grounding.mean`, `C.axes.grounding.mean` |
| Shuffle control: grounding → 1.00, drop ≈ 3.82 (n=85 A, n=82 C) | [`scripts/seed_shuffled.py`](../scripts/seed_shuffled.py) | `data/eval/shuffled_20260503T080925/summary.json` | `A.true_mean`, `A.shuffled_mean`, `A.drop_mean` |

```bash
# A condition = direct extraction (companion repo); C = tool-augmented (this repo)
uv run python scripts/seed_quality_eval.py \
  --library A=path/to/no-tool-library --library C=path/to/tool-library \
  --papers-cache data/papers_cache \
  --judge-model <JUDGE>

# Shuffle control re-uses the previous judgements file
uv run python scripts/seed_shuffled.py \
  --judgements data/eval/20260502T230414/judgements.jsonl \
  --library A=... --library C=... \
  --papers-cache data/papers_cache \
  --judge-model <JUDGE>
```

## Stage 2 — Cross-domain retrieval (Section 3.3)

| Paper claim | Script | Output file | Field |
|---|---|---|---|
| Coverage 7.00 ± 0.00 (paraphrase mode) | [`scripts/retrieval_eval.py`](../scripts/retrieval_eval.py) | `data/eval/retrieval_<ts>/summary.json` | `on.seed_home_coverage.mean` |
| Coverage 5.03 ± 1.22 (single probe) | same | same | `off.seed_home_coverage.mean` |
| 28/30 strict improvement | derived from `evaluations.jsonl` per-query records | same dir, `evaluations.jsonl` | per-query `on.seeds[*].domain` vs `off.seeds[*].domain` |

```bash
uv run python scripts/retrieval_eval.py \
  --queries data/queries.yaml \
  --library <LIB> \
  --judge-model <JUDGE>
```

## Stage 3 — Method synthesis (Section 3.4)

Stage 3 chains off one ideation run — the four downstream judges all read its `evaluations.jsonl`.

| Paper claim | Script | Output file | Field |
|---|---|---|---|
| Per-condition novelty A 3.97 / B 4.07 / C 4.13 / D 4.13 | [`scripts/ideation_eval.py`](../scripts/ideation_eval.py) | `<RUN>/summary.json` | `A.novelty.mean`, `B.novelty.mean`, ... |
| Per-condition validity (4.97–5.00) | same | same | `*.validity.mean` |
| Per-condition coherence A 3.63 / B 3.47 / C 3.33 / D 3.33 | [`scripts/coherence_per_condition_eval.py`](../scripts/coherence_per_condition_eval.py) | `data/eval/coherence_per_cond_<ts>/summary.json` | `A.mean`, `B.mean`, `C.mean`, `D.mean` |
| Pairwise novelty C-vs-A 60/40, C-vs-B 67/30, C-vs-D 47/53 | [`scripts/ideation_layers_eval.py`](../scripts/ideation_layers_eval.py) | `data/eval/layer12_<ts>/summary.json` | `pairwise.C-vs-A__novelty.C_win_rate`, etc. |
| Pairwise validity A 50% / C 20% / tie 30% (C-vs-A) | same | same | `pairwise.C-vs-A__validity.*` |
| Pairwise coherence C 17/30 vs D 13/30 (57% vs 43%) | [`scripts/coherence_pairwise_eval.py`](../scripts/coherence_pairwise_eval.py) | `data/eval/coherence_<ts>/summary.json` | `C_win`, `D_win`, `C_win_rate` |
| Attributed coverage A 0 / B 1.00 / C 3.70 / D 3.97; D ≥ C in 21/30 | derived from ideation `evaluations.jsonl` | same | per-record `inspired_by[*].domain` |
| Incorporation 1-3 averages C 2.79 / D 2.73 (n=187, n=191) | [`scripts/inspired_by_grounding_eval.py`](../scripts/inspired_by_grounding_eval.py) | `data/eval/grounding_<ts>/summary.json` | `C.mean`, `D.mean`, `C.fraction_no` |
| McNemar per-problem 1/30 (C) vs 8/30 (D), p ≈ 0.016 | derived from grounding `judgements.jsonl` | same | per-record `score == 1` aggregated by `query_id` |

```bash
# (a) Synthesize and rate per-condition (A, B, C, D × 30 problems)
uv run python scripts/ideation_eval.py \
  --queries data/queries.yaml \
  --library <LIB> \
  --conditions A,B,C,D \
  --judge-model <JUDGE>
# Capture the new run dir, e.g. data/eval/ideation_20260507T090814
RUN=$(ls -td data/eval/ideation_* | head -1)

# (b) Pairwise novelty + validity for three condition pairs
uv run python scripts/ideation_layers_eval.py \
  --ideation-jsonl $RUN/evaluations.jsonl \
  --pairs C-A,C-D,C-B --axes novelty,validity \
  --judge-model <JUDGE>

# (c) Pairwise coherence (C vs D)
uv run python scripts/coherence_pairwise_eval.py \
  --ideation-jsonl $RUN/evaluations.jsonl \
  --pair C-D --judge-model <JUDGE>

# (d) Per-condition single-pass coherence (A, B, C, D)
uv run python scripts/coherence_per_condition_eval.py \
  --ideation-jsonl $RUN/evaluations.jsonl \
  --conditions A,B,C,D --judge-model <JUDGE>

# (e) Inspired-by incorporation (1-3 scale)
uv run python scripts/inspired_by_grounding_eval.py \
  --ideation-jsonl $RUN/evaluations.jsonl \
  --conditions C,D --judge-model <JUDGE>
```

## Appendix A — Q23 walkthrough (STAMM)

STAMM and GMRP method texts, attributed seeds, and per-axis scores all live in the canonical Stage 3 ideation run's Q23 record. Pairwise verdicts are the Q23 rows of the main `layer12` and `coherence` files.

| Paper claim | Source | Field (filter `query_id == "Q23"`) |
|---|---|---|
| STAMM method, 10 attributed seeds across 5 domains, novelty 4/5 | `data/eval/ideation_20260503T131152/evaluations.jsonl` | `C.method`, `C.inspired_by`, `C.novelty.score` |
| GMRP (condition A) method, novelty 4/5 | same | `A.method`, `A.novelty.score` |
| STAMM-vs-GMRP pairwise novelty/validity | `data/eval/layer12_20260503T172939/pairwise.jsonl` | `(axis, x="C", y="A").winner` resolved via `condition_for_a/b` |
| STAMM-vs-GMRP pairwise coherence | `data/eval/coherence_q23_CA/judgements.jsonl` | `winner` resolved via `condition_for_a/b` |
| STAMM single-pass timing 81 s / $0.07 (paraphraser 29 s, synthesizer 45 s) | hand-measured single-pass run; not stored. Re-measure with `time` if needed. | — |

```bash
# C-vs-A pairwise coherence on Q23 (single judgment)
uv run python scripts/coherence_pairwise_eval.py \
  --ideation-jsonl data/eval/ideation_20260503T131152/evaluations.jsonl \
  --pair C-A --judge-model <JUDGE> \
  --output-dir data/eval/coherence_q23_CA --limit 1
```

## Appendix B — Q23 C-vs-D pairwise (STAMM vs MaskSCM)

| Paper claim | Source | Field (filter `query_id == "Q23"`) |
|---|---|---|
| MaskSCM (condition D) method, novelty 4/5, 67 s / $0.06 | `data/eval/ideation_20260503T131152/evaluations.jsonl` (timing hand-measured) | `D.method`, `D.novelty.score` |
| Pairwise STAMM vs MaskSCM novelty / validity | `data/eval/layer12_20260503T172939/pairwise.jsonl` | `(axis, x="C", y="D").winner` |
| Pairwise STAMM vs MaskSCM coherence | `data/eval/coherence_20260505T171635/judgements.jsonl` | `winner` (Q23 record) |

## Appendix C — Iteration behavior

| Paper claim | Script | Output file | Field |
|---|---|---|---|
| 28/30 (C) and 26/30 (D) converge in round 1 | [`scripts/loop_benchmark.py`](../scripts/loop_benchmark.py) | `data/eval/loop_benchmark_<ts>/aggregate.json` | `c_convergence_round.histogram`, `d_convergence_round.histogram` |
| Mean rounds C 1.07 / D 1.13 | same | same | `c_mean_rounds`, `d_mean_rounds` |
| Mean novelty 4.1/5 (both) | same | same | `c_mean_final_score`, `d_mean_final_score` |
| Loop pairwise novelty 50/50, validity 40/30/30, coherence 47/53 | same | same | `pairwise_novelty`, `pairwise_validity`, `pairwise_coherence` |

## Appendix E — Details

| Paper claim | Source |
|---|---|
| 1,167 seeds / 446 papers / 7 ML domains | `data/library/shard_*/seeds.jsonl` |
| Random seeds: D pool seed = 0, pairwise position-randomization seed = 42 | hardcoded in [`scripts/loop_benchmark.py`](../scripts/loop_benchmark.py) and [`scripts/ideation_eval.py`](../scripts/ideation_eval.py) |
| Compute budget (cost / wall-clock) | `cost_total` field in every `<run-dir>/summary.json` (sum across all eval runs); per-query breakdown in `<RUN>/evaluations.jsonl[*].cost` |
| Lens text per condition | rendered by [`papergym/agents/synthesizer/synthesize.yaml`](../src/papergym/agents/synthesizer/synthesize.yaml); per-condition lens construction in [`eval/ideation/evaluate.py`](../eval/ideation/evaluate.py) (lines 78, 110, 148) |

## End-to-end reproduction

A single command runs all of the above in order, against an existing library and queries file:

```bash
bash scripts/reproduce_paper.sh
```

It will create new timestamped run directories under `data/eval/` and print a final cost summary. Existing released run directories are kept untouched.
