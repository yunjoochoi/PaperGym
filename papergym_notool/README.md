# PaperGym — No-Tool Extraction Baseline

This subdirectory holds the **Stage 1 ablation baseline** used in the paper:
a single-pass accumulator that receives the entire paper markdown in its
user message and emits seeds in one completion — no `read` / `grep` / `bash`
tool loop. It exists to isolate what the tool-augmented accumulator
contributes beyond direct prompting.

For the main pipeline (tool-augmented Stage 1, Stages 2–3, judges), see
[`../README.md`](../README.md).

---

## When to use this

Run this baseline when reproducing the Stage 1 comparison in Section 3.2
of the paper:

- *Tool-augmented* extraction → built with the parent project's accumulator.
- *Direct (no-tool)* extraction → built **here**.

Both libraries are scored by the same Stage 1 rubric judges in the parent
project (`../scripts/seed_quality_eval.py`, `../scripts/seed_shuffled.py`).

---

## Setup

This subdirectory is a self-contained Python project (its own
`pyproject.toml` and `uv.lock`). Install in a separate virtualenv from the
parent project:

```bash
cd papergym_notool/
uv venv .venv --python 3.11
uv pip install -e ".[dev]"
source .venv/bin/activate
cp .env.examples .env   # add OPENAI_API_KEY etc.
```

Host requirements: Docker daemon (the baseline still sandboxes each paper
in a fresh container; only the in-container agent loop is replaced).

---

## Reproduce Stage 1 baseline

The procedure mirrors the parent project's bootstrap, but uses the
no-tool accumulator image.

### 1. Build the no-tool accumulator image

```bash
bash scripts/build_image.sh   # → papergym-accumulator-notool:latest
```

### 2. Pick papers (or reuse the parent's sample)

```bash
# Option A: reuse the parent's sampled arxiv_ids (recommended for the
# paired Stage 1 comparison — same 30 papers as the tool-augmented run)
cp ../data/arxiv_ids.jsonl data/arxiv_ids.jsonl

# Option B: resample independently
.venv/bin/python scripts/sample_envs.py --out data/arxiv_ids.jsonl
```

### 3. Run the no-tool accumulator

```bash
.venv/bin/python scripts/run_accumulator.py \
    --arxiv-ids data/arxiv_ids.jsonl \
    --library-root data/library \
    --events-dir data/events
```

Inside each container, `accumulate_one.py` fetches and converts the paper
to markdown, then calls the accumulator agent in single-pass mode: the
full `paper.md` is injected into the user message and the model emits the
`{title, domain, seeds[]}` JSON in one completion. No tool dispatch, no
ReAct loop.

Output library shape matches the parent project's
(`data/library/seeds.jsonl` + `data/library/faiss/<DOMAIN>.index`), so the
parent's judges accept it as-is.

### 4. Score with the parent's Stage 1 rubric judges

From the parent project (`../`), with both libraries on disk:

```bash
cd ..
source .venv/bin/activate
.venv/bin/python scripts/seed_quality_eval.py \
    --tool-library     data/library \
    --notool-library   papergym_notool/data/library \
    --out              data/eval/stage1
.venv/bin/python scripts/seed_shuffled.py \
    --tool-library     data/library \
    --notool-library   papergym_notool/data/library \
    --out              data/eval/stage1_shuffled
```

(Argument names may vary; pass `--help` on each script for the current
flags. The expected outputs reproduce the Stage 1 numbers in Section 3.2
of the paper.)

---

## What differs from the parent project

Conceptually, only Stage 1's *accumulator agent* differs. Concretely:

| File | Difference |
|---|---|
| `src/papergym/agents/accumulator/agent.py` | Single-pass completion; no `read`/`grep`/`bash` binding, no `run_tool_loop`. |
| `src/papergym/agents/accumulator/accumulator.yaml` | Prompt assumes the full paper is in the user message. |
| `scripts/accumulate_one.py` | No `--max-steps` (no loop). |
| `docker/Dockerfile` | Smaller — no tool deps. |

Stages 2–3 (paraphraser, retrieval, synthesizer) are **not** exercised in
this subdirectory; run those from the parent project against whichever
Stage 1 library you want to ablate over.

---

## Tests

```bash
.venv/bin/pytest tests/unit -q
```

All tests mocked (no Docker / network needed).
