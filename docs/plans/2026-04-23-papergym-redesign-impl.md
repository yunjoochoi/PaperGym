# PaperGym Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace target/seed curation model with autonomous paper-level pipeline: agent discovers repo/pdf/benchmark from paper title inside a single generic container.

**Architecture:** See `docs/plans/2026-04-23-papergym-redesign.md`. One generic CUDA image, 4 agent tools (Read/Write/Bash/fetch_paper), unified `run_pipeline.py` entry, all artifacts under `<repo_slug>_experiment_result/`.

**Tech Stack:** Python 3.11, uv, Docker/Singularity, OpenAI API, `pymupdf4llm` for PDF→markdown, arxiv + Semantic Scholar APIs.

**Commit policy:** User owns commits. Each task ends at "stage for user commit" — do NOT run `git commit`.

---

## Task 1: Generic base image

**Files:**
- Create: `docker/Dockerfile.base`
- Create: `docker/singularity.def.base`
- Create: `scripts/build_base_image.sh`
- Delete: `scripts/build_target_image.sh`, `scripts/build_target_sif.sh`

**Step 1: Write Dockerfile.base**

```dockerfile
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf_cache \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git curl wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN uv pip install --system --no-cache \
        pymupdf4llm arxiv requests tenacity

WORKDIR /workspace
RUN mkdir -p /workspace/papers /experiment_result /hf_cache
```

**Step 2: Write singularity.def.base**

```
Bootstrap: docker
From: nvidia/cuda:12.1.0-devel-ubuntu22.04

%post
    export DEBIAN_FRONTEND=noninteractive
    apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git curl wget ca-certificates build-essential
    rm -rf /var/lib/apt/lists/*
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ln -s /root/.local/bin/uv /usr/local/bin/uv
    uv pip install --system --no-cache pymupdf4llm arxiv requests tenacity
    mkdir -p /workspace/papers /experiment_result /hf_cache

%environment
    export PYTHONUNBUFFERED=1
    export HF_HOME=/hf_cache
    export UV_LINK_MODE=copy

%runscript
    exec /bin/bash "$@"
```

**Step 3: Write build_base_image.sh**

```bash
#!/usr/bin/env bash
set -euo pipefail

BACKEND=${1:-singularity}
DOCKER_DIR="$(dirname "$0")/../docker"

if [ "$BACKEND" = "docker" ]; then
    docker build -t papergym-base:latest -f "$DOCKER_DIR/Dockerfile.base" "$DOCKER_DIR"
else
    SIF_PATH="$(dirname "$0")/../runs/sif/base.sif"
    mkdir -p "$(dirname "$SIF_PATH")"
    singularity build "$SIF_PATH" "$DOCKER_DIR/singularity.def.base"
fi
```

**Step 4: Delete old build scripts**

```bash
rm scripts/build_target_image.sh scripts/build_target_sif.sh
chmod +x scripts/build_base_image.sh
```

**Step 5: Stage for user commit**

```bash
git add docker/ scripts/build_base_image.sh
git rm scripts/build_target_image.sh scripts/build_target_sif.sh
```

---

## Task 2: `fetch_paper` tool

**Files:**
- Create: `src/papergym/tools/fetch_paper.py`
- Create: `tests/unit/test_fetch_paper.py`
- Modify: `src/papergym/tools/__init__.py`

**Step 1: Write failing test**

```python
# tests/unit/test_fetch_paper.py
from unittest.mock import patch, MagicMock
from papergym.tools.fetch_paper import fetch_paper, PaperMeta


def test_fetch_paper_arxiv_hit():
    mock_result = MagicMock(
        title="Simple test-time scaling",
        entry_id="http://arxiv.org/abs/2501.19393v1",
        pdf_url="http://arxiv.org/pdf/2501.19393v1.pdf",
        summary="We show that...",
        comment="Code: https://github.com/simplescaling/s1",
    )
    with patch("papergym.tools.fetch_paper._arxiv_search", return_value=[mock_result]), \
         patch("papergym.tools.fetch_paper._download_and_convert", return_value="/workspace/papers/2501.19393v1.md"):
        meta = fetch_paper("Simple test-time scaling", pdf_dir="/workspace/papers")

    assert meta.arxiv_id == "2501.19393v1"
    assert meta.github_url == "https://github.com/simplescaling/s1"
    assert meta.pdf_path.endswith(".md")
```

**Step 2: Run test — expect fail**

```bash
pytest tests/unit/test_fetch_paper.py -v
```

Expected: `ModuleNotFoundError: papergym.tools.fetch_paper`

**Step 3: Implement `fetch_paper.py`**

```python
"""Paper metadata lookup + PDF fetch tool."""
import re
from dataclasses import dataclass
from pathlib import Path

import arxiv
import pymupdf4llm
import requests
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class PaperMeta:
    title: str
    arxiv_id: str | None
    pdf_url: str | None
    github_url: str | None
    abstract: str
    pdf_path: str | None


_GITHUB_RE = re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
def _arxiv_search(query: str, max_results: int = 3):
    client = arxiv.Client()
    return list(client.results(arxiv.Search(query=query, max_results=max_results)))


def _extract_github(*texts: str) -> str | None:
    for t in texts:
        if not t:
            continue
        m = _GITHUB_RE.search(t)
        if m:
            return m.group(0).rstrip(".,;)")
    return None


def _download_and_convert(pdf_url: str, out_path: Path) -> str:
    resp = requests.get(pdf_url, timeout=60)
    resp.raise_for_status()
    pdf_bytes = out_path.with_suffix(".pdf")
    pdf_bytes.write_bytes(resp.content)
    md = pymupdf4llm.to_markdown(str(pdf_bytes))
    out_path.write_text(md)
    return str(out_path)


def fetch_paper(query: str, pdf_dir: str = "/workspace/papers") -> PaperMeta:
    results = _arxiv_search(query)
    if not results:
        return PaperMeta(title=query, arxiv_id=None, pdf_url=None,
                         github_url=None, abstract="", pdf_path=None)

    r = results[0]
    arxiv_id = r.entry_id.rsplit("/", 1)[-1]
    github_url = _extract_github(r.comment or "", r.summary)
    pdf_dir_p = Path(pdf_dir)
    pdf_dir_p.mkdir(parents=True, exist_ok=True)
    md_path = pdf_dir_p / f"{arxiv_id}.md"
    pdf_path = _download_and_convert(r.pdf_url, md_path)

    return PaperMeta(title=r.title, arxiv_id=arxiv_id, pdf_url=r.pdf_url,
                     github_url=github_url, abstract=r.summary, pdf_path=pdf_path)
```

**Step 4: Add to `tools/__init__.py` exports**

```python
from .fetch_paper import fetch_paper, PaperMeta
```

**Step 5: Run test — expect pass**

```bash
pytest tests/unit/test_fetch_paper.py -v
```

**Step 6: Add deps to pyproject.toml**

Check `pyproject.toml` has `arxiv`, `pymupdf4llm`, `requests`, `tenacity`. Add any missing.

**Step 7: Stage for user commit**

```bash
git add src/papergym/tools/fetch_paper.py src/papergym/tools/__init__.py \
        tests/unit/test_fetch_paper.py pyproject.toml
```

---

## Task 3: `PaperContext` dataclass replaces `Target`

**Files:**
- Create: `src/papergym/paper.py`
- Delete: `src/papergym/targets.py`

**Step 1: Write `paper.py`**

```python
"""Per-paper runtime context. Replaces targets.py."""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PaperContext:
    title: str
    arxiv_id: str | None = None
    repo_url: str | None = None
    repo_slug: str | None = None
    pdf_path: str | None = None
    benchmark_name: str | None = None
    reported_score: float | None = None
    github_url: str | None = None
    abstract: str = ""

    def experiment_dir(self, root: Path) -> Path:
        if not self.repo_slug:
            raise ValueError("repo_slug not set; clone repo before computing experiment_dir")
        return root / f"{self.repo_slug}_experiment_result"


def derive_repo_slug(repo_url: str) -> str:
    slug = repo_url.rstrip("/").split("/")[-1]
    return slug[:-4] if slug.endswith(".git") else slug
```

**Step 2: Delete `targets.py`**

```bash
git rm src/papergym/targets.py
```

**Step 3: Grep-check for remaining imports**

```bash
grep -rn "from papergym.targets\|papergym\.targets" src/ scripts/ tests/
```

Expected: empty (we clean up imports in Task 4+).

**Step 4: Stage for user commit** (deferred until imports cleaned in later tasks)

---

## Task 4: Rewrite Observer for paper-level input

**Files:**
- Modify: `src/papergym/agents/observer.py`
- Modify: `src/papergym/prompts/observer.yaml`
- Modify: `tests/unit/test_observer.py`

**Step 1: Update Observer input dataclass**

Remove `target: Target` field. Add:
```python
@dataclass
class ObserverInput:
    paper: PaperContext
    baseline_stdout_path: str  # inside container
    pdf_markdown_path: str     # inside container
```

**Step 2: Update observer prompt**

System prompt: "You are observing reproduction of paper '{title}'. Read the paper markdown and baseline stdout log. Identify mismatches between paper's claims and reproduced behavior: wrong metrics, missing implementation details, flaky training, suboptimal hyperparameters."

Remove target.name / target.repo_url refs.

**Step 3: Update existing Observer test**

Replace `target=fake_target()` with `paper=PaperContext(title="...", repo_slug="s1")`.

**Step 4: Run tests**

```bash
pytest tests/unit/test_observer.py -v
```

**Step 5: Stage**

---

## Task 5: Rewrite Methodologist for paper-level input

**Files:**
- Modify: `src/papergym/agents/methodologist.py`
- Modify: `src/papergym/prompts/methodologist.yaml`
- Modify: `tests/unit/test_methodologist.py`

**Step 1: Update input dataclass**

Remove `target` and `baseline_score`. Add `paper: PaperContext` and `baseline_score: float`.

**Step 2: Update prompt** to remove target-specific language.

**Step 3: Update tests.**

**Step 4: Run tests, stage.**

---

## Task 6: Update Implementer — inject paper-aware default flow

**Files:**
- Modify: `src/papergym/agents/implementer.py`
- Modify: `src/papergym/prompts/implementer.yaml`
- Modify: `tests/unit/test_implementer.py`

**Step 1: Update system prompt**

Append:
```
Default flow when starting fresh:
1. fetch_paper("<title>") → note github_url, pdf_path
2. git clone <github_url>
3. Read pdf_path to learn canonical benchmark, metric, inference command
4. Read repo README + entry scripts
5. Install deps (uv pip install -r requirements.txt)
6. Run inference → must print `FINAL_SCORE: <float>` on stdout
7. Halt with final observation
```

**Step 2: Add `fetch_paper` tool to dispatcher**

In `_invoke_tool`:
```python
elif name == "fetch_paper":
    meta = tools.fetch_paper(args["query"], pdf_dir=self.pdf_dir)
    return json.dumps(asdict(meta))
```

Tool schema advertised to LLM:
```json
{"name": "fetch_paper", "params": {"query": "paper title"}}
```

**Step 3: Update tests** — add a fetch_paper call case.

**Step 4: Run tests, stage.**

---

## Task 7: Update Writer + Reviewer

**Files:**
- Modify: `src/papergym/agents/writer.py`, `src/papergym/agents/reviewer.py`
- Modify: prompts, tests

**Step 1:** Replace target refs with `paper.title`, `paper.benchmark_name`.

**Step 2:** `ReviewerInput` keeps `ideation_doc`, `delta_bench`, `benchmark`, `web_search_results` — just ensure `benchmark` is sourced from `paper.benchmark_name` (no target needed).

**Step 3:** Run tests, stage.

---

## Task 8: Unified output layout + baseline cache move

**Files:**
- Modify: `src/papergym/envs/reproduce.py`
- Modify: `src/papergym/log.py` (if events.jsonl path hard-coded)

**Step 1: Update `BaselineCache`**

Change signature:
```python
class BaselineCache:
    def __init__(self, experiment_root: Path):
        self.dir = experiment_root / "baseline"
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self) -> Path:
        return self.dir / "score.json"

    def get(self) -> float | None: ...
    def set(self, score: float, benchmark_name: str) -> None: ...
```

One cache per paper (keyed by directory, not by name).

**Step 2: Update `RunLogger`** — write `events.jsonl` to `<experiment_root>/events.jsonl`.

**Step 3: Update tests**

**Step 4: Run tests, stage.**

---

## Task 9: Update `library/bootstrap.py` — paper list loop

**Files:**
- Modify: `src/papergym/library/bootstrap.py`
- Modify: `bootstrap/seed_surveys.yaml`

**Step 1: Rewrite `SurveyEntry`**

```python
@dataclass
class SurveyEntry:
    paper_title: str  # only required field
    notes: str = ""
```

Remove `paper_id`, `variant_id`, `method`, `hint_files`, `observation`, `problem`, `source_signal`. Agent discovers everything.

**Step 2: Rewrite `build_bootstrap_from_survey`** to accept `List[SurveyEntry]` and call pipeline per entry with `disable_library=True`.

**Step 3: Rewrite `bootstrap/seed_surveys.yaml`**

```yaml
papers:
  - title: "Muennighoff et al., Simple test-time scaling, 2025"
  - title: "Wei et al., Chain-of-thought prompting elicits reasoning in large language models, 2022"
```

**Step 4: Update tests.**

---

## Task 10: `pipeline.py` — accept `PaperContext`, drive baseline+iters

**Files:**
- Modify: `src/papergym/pipeline.py`
- Modify: `src/papergym/run.py`

**Step 1: Pipeline.run() signature**

```python
def run(self, paper: PaperContext, experiment_root: Path,
        max_iters: int, disable_library: bool, ...) -> None:
```

Inside:
1. Baseline: fresh container, Implementer runs "baseline flow" (fetch_paper → clone → reproduce → FINAL_SCORE). Parse score → cache.
2. For iter in 1..max_iters: fresh container, Observer → retrieve → Methodologist → Implementer → Scorer → Writer → Reviewer → Library.add.

**Step 2: Remove `reproduce.py`'s external baseline driver** (or keep helper, but main driving logic in pipeline).

**Step 3: Run unit tests, stage.**

---

## Task 11: New `scripts/run_pipeline.py`

**Files:**
- Create: `scripts/run_pipeline.py`
- Delete: `scripts/reproduce_baseline.py`, `scripts/build_bootstrap.py`, `scripts/run_ideation.py`, `scripts/run_pipeline.sh`

**Step 1: Write `run_pipeline.py`**

```python
"""Unified entry: single paper or batch from YAML."""
import argparse
import yaml
from pathlib import Path

from papergym.config import Config
from papergym.paper import PaperContext
from papergym.pipeline import Pipeline


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--paper", help="Single paper title or arxiv ID")
    g.add_argument("--papers", help="YAML file with list under 'papers' key")
    p.add_argument("--backend", choices=["singularity", "docker"], default="singularity")
    p.add_argument("--gpus", default="auto")
    p.add_argument("--max-iters", type=int, default=1)
    p.add_argument("--implementer-max-steps", type=int, default=500)
    p.add_argument("--disable-library", action="store_true")
    p.add_argument("--disable-reviewer", action="store_true")
    p.add_argument("--disable-reproduce", action="store_true")
    args = p.parse_args()

    cfg = Config.load()
    titles = [args.paper] if args.paper else [
        e["title"] for e in yaml.safe_load(Path(args.papers).read_text())["papers"]
    ]

    pipeline = Pipeline.build(cfg, args)
    for title in titles:
        paper = PaperContext(title=title)
        pipeline.run(paper=paper, experiment_root=cfg.output_root,
                     max_iters=args.max_iters,
                     disable_library=args.disable_library,
                     disable_reviewer=args.disable_reviewer,
                     disable_reproduce=args.disable_reproduce)


if __name__ == "__main__":
    main()
```

**Step 2: Delete old scripts**

```bash
git rm scripts/reproduce_baseline.py scripts/build_bootstrap.py \
       scripts/run_ideation.py scripts/run_pipeline.sh
```

**Step 3: Smoke test** — `.venv/bin/python scripts/run_pipeline.py --paper "test" --help`

**Step 4: Stage.**

---

## Task 12: Remove `targets/` folder

**Step 1:**

```bash
git rm -r targets/
```

**Step 2: Confirm nothing imports from it**

```bash
grep -rn "targets/" src/ scripts/ tests/ | grep -v "\.md"
```

Expected: no code references (only doc/README mentions, fix in Task 14).

**Step 3: Stage.**

---

## Task 13: Update tests

**Files:**
- Modify: `tests/unit/test_pipeline.py`, `tests/integration/test_end_to_end.py`, others

**Step 1:** Replace all `Target`/`load_target` usage with `PaperContext`.

**Step 2:** Mock `fetch_paper` in unit tests (no real network).

**Step 3: Run full suite**

```bash
.venv/bin/pytest tests/unit -v
```

Expected: all pass.

**Step 4: Stage.**

---

## Task 14: Update README

**Files:**
- Modify: `README.md`

**Step 1: Rewrite Stage 1** — remove per-paper curation. New Stage 1 = "Build base image once: `bash scripts/build_base_image.sh`".

**Step 2: Merge Stages 2+3** into single "Run pipeline" section:

```bash
# Bootstrap (disable library retrieval, process paper list)
.venv/bin/python scripts/run_pipeline.py \
    --papers bootstrap/seed_surveys.yaml --disable-library

# Ideation on a specific paper
.venv/bin/python scripts/run_pipeline.py \
    --paper "Simple test-time scaling" --max-iters 3
```

**Step 3: Update Runtime Layout section** — reflect unified `<repo_slug>_experiment_result/`.

**Step 4: Update Layout section** — reflect new `src/papergym/` structure.

**Step 5: Stage.**

---

## Task 15: End-to-end smoke test

**Step 1: Build base image** (user action, slow):

```bash
bash scripts/build_base_image.sh singularity
```

**Step 2: Run on s1**

```bash
.venv/bin/python scripts/run_pipeline.py \
    --paper "Simple test-time scaling" \
    --max-iters 1 \
    --implementer-max-steps 500 \
    --disable-library
```

**Step 3: Verify outputs**

```bash
ls s1_experiment_result/
# Expect: baseline/ (score.json, stdout.log), ideation/, ideation_docs/, events.jsonl
```

**Step 4: If pass, task #32 (prompt tuning) picks up from real events.jsonl.**

---

## Acceptance criteria

- [x] Single generic base image builds (`papergym-base:latest` or `.sif`)
- [x] `scripts/run_pipeline.py --paper "<title>"` runs end-to-end with no per-paper curation
- [x] `<repo_slug>_experiment_result/` contains baseline/score.json, events.jsonl, ideation/*
- [x] All unit tests pass
- [x] `targets/` folder deleted
- [x] Only one design doc remains: `docs/plans/2026-04-23-papergym-redesign.md`
