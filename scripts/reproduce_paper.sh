#!/usr/bin/env bash
# Re-run every evaluation behind the paper (Stages 2, 3, and the loop
# ablation). Stage 1 lives in the PaperGym_notool companion repository.
#
# Inputs (must already exist):
#   - data/library/                 cross-domain seed library (build it via
#                                   the pipeline in README, or use the
#                                   released snapshot)
#   - data/queries.yaml             30-problem benchmark
#
# Outputs: new timestamped directories under data/eval/. Each script
# prints its own cost / wall-clock summary; the final block of this
# script prints a grand total.
#
# Required env: provider credentials for both LITELLM_MODEL (generator)
# and JUDGE_MODEL (judge). See .env.examples.
#
# Usage:
#   bash scripts/reproduce_paper.sh                # run everything
#   SKIP_LOOP=1 bash scripts/reproduce_paper.sh    # skip loop ablation
set -euo pipefail

cd "$(dirname "$0")/.."

# Load .env so the shell sees the same keys the Python scripts do.
# Provides LITELLM_MODEL (generator) and JUDGE_MODEL (held-out judge).
if [[ -f .env ]]; then
    set -a
    . .env
    set +a
fi

JUDGE="${JUDGE_MODEL:?set JUDGE_MODEL in .env}"
LIB="data/library"
QUERIES="data/queries.yaml"

run() { echo; echo "▶ $*"; "$@"; }

# --- Stage 2: cross-domain retrieval -----------------------------------
if [[ "${SKIP_STAGE2:-0}" != "1" ]]; then
    run uv run python scripts/retrieval_eval.py \
        --queries "$QUERIES" \
        --library "$LIB" \
        --judge-model "$JUDGE"
fi

# --- Stage 3: ideation + 3 downstream judges ---------------------------
if [[ "${SKIP_STAGE3:-0}" != "1" ]]; then
    run uv run python scripts/ideation_eval.py \
        --queries "$QUERIES" \
        --library "$LIB" \
        --conditions A,B,C,D \
        --judge-model "$JUDGE"
    RUN=$(ls -td data/eval/ideation_* | head -1)
    echo "ideation run dir: $RUN"

    run uv run python scripts/ideation_layers_eval.py \
        --ideation-jsonl "$RUN/evaluations.jsonl" \
        --pairs C-A,C-D,C-B \
        --axes novelty,validity \
        --judge-model "$JUDGE"

    run uv run python scripts/coherence_pairwise_eval.py \
        --ideation-jsonl "$RUN/evaluations.jsonl" \
        --pair C-D \
        --judge-model "$JUDGE"

    run uv run python scripts/coherence_per_condition_eval.py \
        --ideation-jsonl "$RUN/evaluations.jsonl" \
        --conditions A,B,C,D \
        --judge-model "$JUDGE"

    run uv run python scripts/inspired_by_grounding_eval.py \
        --ideation-jsonl "$RUN/evaluations.jsonl" \
        --conditions C,D \
        --judge-model "$JUDGE"
fi

# --- Appendix C: novelty iteration loop --------------------------------
if [[ "${SKIP_LOOP:-0}" != "1" ]]; then
    run uv run python scripts/loop_benchmark.py
fi

# --- Grand total -------------------------------------------------------
echo
echo "── grand total across all summary.json files ──"
uv run python - <<'PY'
import json, glob, sys
total_cost = 0.0
total_wall = 0.0
for path in sorted(glob.glob("data/eval/*/summary.json")
                   + glob.glob("data/eval/*/aggregate.json")):
    try:
        d = json.loads(open(path).read())
    except Exception:
        continue
    ct = d.get("cost_total")
    if isinstance(ct, dict):
        c = ct.get("total_cost_usd", 0.0)
        w = ct.get("wall_clock_s", 0.0)
        total_cost += c
        total_wall += w
        print(f"  {path:60s}  ${c:7.3f}  {w:8.1f} s")
    elif "total_cost_usd" in d:  # loop_benchmark aggregate.json
        c = d["c_total_cost_usd"] + d.get("d_total_cost_usd", 0)
        w = d.get("total_wall_clock_s", 0.0)
        total_cost += c
        total_wall += w
        print(f"  {path:60s}  ${c:7.3f}  {w:8.1f} s")
print()
print(f"  TOTAL: ${total_cost:.2f}, {total_wall/3600:.2f} h")
PY
