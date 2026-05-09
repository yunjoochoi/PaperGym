# PaperGym Redesign — Autonomous Paper-Level Agent

Supersedes `2026-04-23-papergym-design.md` for the target/seed distinction.

## Motivation

The prior design split "target paper" (hand-curated under `targets/<paper>/`) from "seed papers" (`bootstrap/seed_surveys.yaml`). This forced per-paper curation (Dockerfile, singularity.def, target.yaml, benchmark.yaml, run_inference.sh) and modeled bootstrap as "apply seed method to target repo," which misrepresents the loop.

The correct loop is **per-paper self-contained**: for each paper, reproduce its canonical benchmark inside a container, observe problems/mismatches/design flaws during reproduction, propose a fix, measure Δ-bench. Bootstrap and ideation share this loop; they differ only in whether library retrieval is enabled.

## Core model

- No `target` vs `seed` distinction. Input is a list of papers.
- Per paper: fresh container → agent clones repo, fetches PDF, reproduces benchmark, Observer catches problems, Methodologist proposes method, Implementer tests, Scorer measures Δ, Writer+Reviewer, Library.add.
- **Bootstrap mode** = iterate papers with `--disable-library` (retrieval off; seeds still persisted).
- **Ideation mode** = `--disable-library` off; Methodologist sees retrieved seeds from prior runs.

## Inputs

```bash
scripts/run_pipeline.py --paper "<title or arxiv ID>"       # single
scripts/run_pipeline.py --papers bootstrap/seed_surveys.yaml # batch
```

`bootstrap/seed_surveys.yaml`:
```yaml
papers:
  - "Muennighoff et al., Simple test-time scaling, 2025"
  - "Wei et al., Chain-of-thought prompting, 2022"
```

No per-paper hand-curation. No `targets/` folder.

## Container

Single generic base image — `docker/Dockerfile.base` + `docker/singularity.def.base`:
- `nvidia/cuda:12.1.0-devel-ubuntu22.04`
- `uv`, `git`, `curl`, `wget`, `build-essential`, `python3.11`
- Network enabled (clone, PDF fetch, HF download)
- `HF_HOME=/hf_cache` mounted from host `~/.cache/huggingface`
- `/experiment_result` mounted to host `<repo_slug>_experiment_result/`

Built once via `scripts/build_base_image.sh`.

## Agent tools (4)

- `Read(path, offset, limit)` — container file read
- `Write(path, content)` — Read-after-Write enforced
- `Bash(cmd, timeout)` — git clone, uv install, inference run, curl
- `fetch_paper(query)` — arxiv + Semantic Scholar lookup. Downloads PDF and converts to markdown (`pymupdf4llm`). Returns `{title, arxiv_id, pdf_url, github_url, abstract, pdf_path}`. Agent `Read`s `pdf_path`.

Agent's default flow: `fetch_paper(title)` → `Bash("git clone <github_url>")` → `Read(pdf_path)` → run benchmark.

## Score protocol

- Inference script must print `FINAL_SCORE: <float>` on stdout (Implementer prompt enforces).
- Host parses regex `FINAL_SCORE:\s*([0-9.]+)`.
- Benchmark name/metric stored as seed metadata (Δ computed from raw numbers only).
- `--disable-reproduce` uses `reported_score` extracted by agent from PDF, saved to `<repo_slug>_experiment_result/paper_meta.json`.
- Baseline cache: `<repo_slug>_experiment_result/baseline/score.json` with `{score, benchmark_name, reproduced_at}`. No separate `runs/baselines/`.

## Agent I/O changes

- `Observer` drops `target` arg. Receives `paper_title`, `repo_slug`, baseline log path, pdf_path. Outputs `problem_signals` (same).
- `Methodologist` drops `target`/`baseline_score`. Receives `paper_title`, `problem_signals`, optional `retrieved_seeds`. Outputs `MethodSpec`.
- `Implementer` unchanged structurally; context includes `paper_title` + default flow template in system prompt.
- `Writer`/`Reviewer` drop target-specific fields, use `paper_title` + `benchmark_name`.

`papergym/targets.py` removed. New `papergym/paper.py`:
```python
@dataclass
class PaperContext:
    title: str
    arxiv_id: str | None
    repo_url: str | None
    repo_slug: str
    pdf_path: str | None
    benchmark_name: str | None
    reported_score: float | None
```

## File layout

```
src/papergym/
├── config.py, llm.py, log.py, types.py, paper.py, observation.py, pipeline.py, run.py
├── agents/          # Observer/Methodologist/Implementer/Writer/Reviewer (target-free)
├── prompts/
├── envs/            # Deployment (Docker/Singularity) + reproduce (repo_slug keyed)
├── library/         # store + bootstrap (paper-list loop)
└── tools/           # code_tools + fetch_paper + retrieval + scoring

docker/
├── Dockerfile.base
└── singularity.def.base

scripts/
├── build_base_image.sh
└── run_pipeline.py       # --paper / --papers

bootstrap/seed_surveys.yaml
library/                  # global seeds.jsonl + embeddings.pkl
runs/sif/base.sif         # shared base image only
<repo_slug>_experiment_result/
  baseline/      # score.json, stdout.log, paper_meta.json
  ideation/      # per-iter artifacts
  ideation_docs/ # accepted seeds
  events.jsonl   # full run trace (baseline + all iters)
```

## Removals

- `targets/` folder (s1, speculative_actions)
- `scripts/reproduce_baseline.py`, `build_bootstrap.py`, `run_ideation.py`, `run_pipeline.sh`, `build_target_image.sh`, `build_target_sif.sh`
- `src/papergym/targets.py`

## Open risks

- Agent fails to find github_url from title → `fetch_paper` must be robust (fallback: Semantic Scholar → arxiv comments field → Google Scholar scrape)
- Benchmark extraction from PDF flaky → Implementer prompt must instruct "if canonical benchmark ambiguous, halt with error"
- Model weights not cached → `HF_HOME` mount critical; first run downloads, subsequent runs reuse
- Reproduction failure (env mismatch) → agent has autonomy to debug via Bash; max_steps=500 budget
