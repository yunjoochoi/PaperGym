# PaperGym v3 — Cross-domain idea synthesis (ICML AI4Research workshop)

## 1. Goal & workshop fit

ICML AI4Research workshop submission. Core narrative: **cross-domain idea synthesis**.

**Hypothesis.** When a researcher poses a high-level problem (e.g., "long-context efficient inference"), retrieving mechanism seeds from 7 adjacent ML domains and synthesizing across them yields more diverse / stronger method proposals than single-domain retrieval.

**Contributions.**
1. Domain-tagged seed library (~1,000–2,000 seeds across 7 ML domains).
2. Essence-driven paraphrase-per-domain retrieval pipeline.
3. Provenance-tracked synthesis output — each method proposal carries explicit attribution `(seed_id, domain, borrowed_aspect)`.

Section 4 (Experiment) is designed separately by the user and is **out of scope** for this document.

## 2. Pivot from current PaperGym

| | v2 (reproduce-grounded) | **v3 (cross-domain synthesis)** |
|---|---|---|
| Goal | measure Δ-bench by reproducing methods | propose methods by cross-domain synthesis |
| Infra | Docker container per paper, conda envs, GPU benchmarks | local file system, no Docker, no GPU |
| Seed | `(observation, problem, method, Δ-bench)` | `(problem, method, domain, paper_title)` |
| Pipeline | Observer → Methodologist → Implementer → Loop A/B → Writer → Reviewer | **Bootstrap → Accumulator → Library → Retrieval → Synthesizer** |
| Tools | Bash via container exec | Bash on local host (Accumulator only) |

Migration: one-shot cleanup PR. Current state preserved on `legacy` branch. Details deferred to writing-plans.

## 3. Architecture overview

```
scripts/bootstrap_papers.py
    │  (deterministic, no LLM)
    │  7 domains × 9 cells (year × citation tier) → 840 papers
    ▼
data/papers/<DOMAIN>/<arxiv_id>/
    ├── paper.md          (markdown via docling)
    └── repo/             (lazy-cloned by Accumulator if confident)

scripts/run_accumulator.py
    │  (per paper, agentic ReAct: Read / Grep / Bash, max_steps = 100)
    │  → 1–3 seeds per paper
    ▼
data/library/
    ├── seeds.jsonl       (Seed metadata, no embeddings)
    └── faiss/<DOMAIN>.index   (7 IndexFlatIP, embeddings only)

scripts/run_synthesis.py --query "..."
    │  paraphrase-per-domain (1 LLM call) → 7 paraphrases
    │  retrieve top-K=3 per domain (≤ 21 seeds)
    │  synthesize (1 LLM call, no tools) → method JSON with provenance
    ▼
stdout: { method, rationale, inspired_by[] }
```

Domain encoded in directory path. No per-paper `meta.json`. Sampling rationale lives in `bootstrap_log.jsonl`.

## 4. Components

### 4.1 Bootstrap — `scripts/bootstrap_papers.py`

Deterministic. No LLM.

- **Input:** per-domain budget. `{LLM_NLP: 180, MULTIMODAL: 150, CV: 130, RL: 100, IR_REC: 100, SPEECH: 100, ROBOTICS: 80}` = 840 total.
- **Process:**
  1. Map `Domain` → Semantic Scholar `s2FieldsOfStudy` labels, with explicit overrides for known mistags (`Vision and Language → MULTIMODAL`, `Visual Question Answering → MULTIMODAL`, `Image Captioning → MULTIMODAL`).
  2. For each domain, query S2 for papers in 2017–2025 with citation count above noise floor.
  3. Bucket into a 9-cell grid: 3 year cohorts (2017–19, 2020–22, 2023–25) × 3 citation tiers (top 5%, 5–25%, 25–50% **within the same year cohort**). Year-relative percentile so 2025 papers don't get filtered out by absolute citation cutoffs.
  4. Sample budget/9 per cell. If a cell underfills, take what exists.
  5. For each sampled paper: download PDF (`arxiv` lib) → markdown (`docling`) → write to `data/papers/<DOMAIN>/<arxiv_id>/paper.md`.
- **Output:** `data/papers/<DOMAIN>/<arxiv_id>/paper.md` + appended row in `bootstrap_log.jsonl`.
- **Failures:** PDF download failures (paywall, no arxiv id) skipped, logged to `bootstrap_log.jsonl`. No retry.

### 4.2 Accumulator — `accumulator/agent.py`

Agentic ReAct loop. One paper → 1–3 seeds.

- **Tools:** Read, Grep, Bash. `max_steps = 100`.
- **Inputs (initial context):** paper directory path (`data/papers/<DOMAIN>/<arxiv_id>/`), domain (from path), `arxiv_id` (from path).
- **Process:**
  1. Read `paper.md`. Extract title from H1 (first line). Internalize the paper.
  2. Look for github URL in the paper text. If found and **clearly the paper's official repo** (not a baseline / cited work), `Bash git clone <url> repo/`. If unclear or absent, proceed with paper-only.
  3. (If repo cloned) Use Bash / Grep to locate the main contribution's implementation. Read the relevant files.
  4. Extract 1–3 distinct main contributions as seeds. "Distinct" means non-overlapping mechanisms — avoid splitting one idea into multiple sub-points.
  5. Emit each seed as `{problem, method}`. The orchestrator wraps with `seed_id`, `domain`, `paper_title`, `paper_id`.
- **Output JSON schema:**
  ```json
  {"seeds": [{"problem": "...", "method": "..."}, ...]}
  ```
- **Failure:** if extraction fails (paper unintelligible, mechanism unclear), emit `{"seeds": []}` and log to `accumulator_log.jsonl`. No retry.
- **Domain.** Accumulator does **not** re-classify domain. The bootstrap-assigned domain (encoded in the directory path) is the source of truth.

### 4.3 Library — `library/store.py`

Storage:
```
data/library/
├── seeds.jsonl                   # Seed metadata (no embeddings), insertion-ordered
└── faiss/<DOMAIN>.index          # 7 IndexFlatIP, embeddings, insertion-ordered
```

`seeds.jsonl` row order matches `faiss/<DOMAIN>.index` position order **per domain**. The two are synced by append-only operation pairs.

API:
```python
class LibraryStore:
    def add(self, seed: Seed, embedding: np.ndarray) -> None:
        # 1. seeds.jsonl append
        # 2. faiss/<DOMAIN>.index add
        # 3. in-memory cache: seeds_by_domain[seed.domain].append(seed)
        ...

    def retrieve(self, domain: Domain, query_emb: np.ndarray, k: int = 3) -> list[Seed]:
        scores, positions = self.faiss_indices[domain].search(query_emb.reshape(1, -1), k)
        return [self.seeds_by_domain[domain][p] for p in positions[0]]

    def get(self, seed_id: str) -> Seed: ...
```

If FAISS index is lost, rebuild from `seeds.jsonl` by re-embedding each seed's `problem` field (~840 embeddings × OpenAI text-embedding-3-small ≈ negligible cost).

### 4.4 Retrieval — `retrieval/{paraphrase,retrieve}.py`

**Paraphrase (single LLM call).** Essence-driven.

- **Input:** user query string.
- **Process:**
  1. Distill the underlying problem **essence** — the abstract challenge, not the surface vocabulary.
  2. Re-instantiate that essence in each of the 7 domains' natural problem space, using each domain's vocabulary.
- **Prompt principle (verbatim instruction to the model):**
  > For the user's query, first extract the underlying problem essence — the abstract challenge being posed. Then for each of the 7 domains, re-instantiate that essence as a problem statement that domain's researchers would naturally pose, using their vocabulary and concerns. The paraphrase must preserve the *abstract problem*, not just substitute keywords. Push hard — even cross-domain mappings that feel forced at the keyword level often surface useful seeds when essence-aligned. Only emit `null` for a domain when the underlying problem essence genuinely has no analog there.
- **Output JSON schema:**
  ```json
  {
    "essence": "<one-sentence abstract problem>",
    "paraphrases": {
      "LLM_NLP":    "...",
      "MULTIMODAL": "...",
      "CV":         "...",
      "RL":         "...",
      "IR_REC":     "...",
      "SPEECH":     "...",
      "ROBOTICS":   null
    }
  }
  ```
- **Null policy:** partial nulls allowed. Default expectation is 7 non-null paraphrases.

**Retrieve.**

- For each non-null paraphrase, embed (`text-embedding-3-small`, L2-normalized) and call `LibraryStore.retrieve(domain=D, query_emb=e, k=3)`.
- Result: up to 21 `(Seed, source_paraphrase_domain)` tuples. Note: `seed.domain == source_paraphrase_domain` always, since each domain queries its own sub-index.

### 4.5 Synthesizer — `synthesizer/synthesize.py`

Single LLM call. **No tools.**

- **Input:** user query + the up-to-21 retrieved seeds. Each seed is rendered into the prompt as:
  ```
  Seed <seed_id> [<domain>] from "<paper_title>":
    Problem: <problem>
    Method:  <method>
  ```
- **Output JSON schema:**
  ```json
  {
    "method":    "<3–5 paragraph method description>",
    "rationale": "<why this addresses the user query>",
    "inspired_by": [
      {"seed_id": "<sid>", "domain": "RL",     "borrowed_aspect": "credit assignment trick"},
      {"seed_id": "<sid>", "domain": "IR_REC", "borrowed_aspect": "ANN-style approximate retrieval"}
    ]
  }
  ```
- `inspired_by` is the workshop's measurement unit — concrete cross-domain attribution per proposed method.

## 5. Schemas

### Domain (Enum)
```python
class Domain(Enum):
    LLM_NLP    = "LLM_NLP"
    MULTIMODAL = "MULTIMODAL"
    CV         = "CV"
    RL         = "RL"
    IR_REC     = "IR_REC"
    SPEECH     = "SPEECH"
    ROBOTICS   = "ROBOTICS"
```

### Seed
```python
@dataclass
class Seed:
    seed_id:     str        # uuid4 hex[:12]
    problem:     str        # 1–2 sentences
    method:      str        # 3–5 sentences, includes "what's special"
    domain:      Domain
    paper_title: str
    paper_id:    str        # arxiv_id
    # NB: embedding is NOT stored on Seed — FAISS is source of truth.
```

### `bootstrap_log.jsonl` row
```json
{"arxiv_id": "...", "domain": "MULTIMODAL", "year": 2024, "citations": 1234,
 "year_cohort": "2023-25", "citation_tier": "top_5_25", "title": "...",
 "status": "ok" | "skipped_no_pdf" | "skipped_paywall"}
```

## 6. Tools

| Tool   | User           | Purpose                                              |
|---     |---             |---                                                   |
| Read   | Accumulator    | read `paper.md`, repo files                          |
| Grep   | Accumulator    | locate functions / classes in cloned repos           |
| Bash   | Accumulator    | `git clone`, `tree`, `ls`, `find`, `wc`              |
| (none) | Synthesizer    | single LLM call, all context in initial prompt       |

The existing `agents/tool_loop.py` is reused. Tool schemas and prompt templates from v2 (`Observer`, `Implementer`, `done`-with-score) are deleted; replaced with Accumulator-specific schemas and prompts.

## 7. Driver scripts

| Script | When to run | Output |
|---|---|---|
| `scripts/bootstrap_papers.py` | one-time setup | populates `data/papers/`, writes `bootstrap_log.jsonl` |
| `scripts/run_accumulator.py [--max-papers N] [--domain D]` | after bootstrap | populates `data/library/`, writes `accumulator_log.jsonl` |
| `scripts/run_synthesis.py --query "<text>"` | per user query | stdout JSON, appends to `events.jsonl` |

`events.jsonl` keeps the per-message format from v2 (system/user/assistant/tool messages tagged with stage + iter, plus thin lifecycle markers).

## 8. Dependencies

- **Add:** none — `faiss-cpu`, `arxiv`, `docling`, `requests`, `openai` are already declared.
- **Remove:** `docker` (no container runtime). `pydantic` already removed.
- **Note:** `faiss-cpu` is used in `IndexFlatIP` mode — exact brute-force search via FAISS API. At N≈2,000 per domain index, this is sub-millisecond.

## 9. Migration outline

Single cleanup PR. `legacy` branch preserves the v2 state.

**DELETE:**
- `pipeline.py`, `agents/{observer, implementer, scorer, writer, reviewer, analyst}.py`
- `envs/` (Docker deployment, baseline cache)
- `tools/{code_tools, scoring}.py`
- `prompts/{observer, implementer, methodologist*, methodologist_refine, writer, reviewer, analyst}.yaml`
- `tests/unit/test_agent_{observer, implementer, methodologist, writer, reviewer, analyst}.py`
- `tests/unit/test_{pipeline, reproduce, deployment, outer_state, scoring}.py`
- `tests/integration/test_docker_smoke.py`
- `scripts/{run_pipeline, build_base_image}.sh`, `docker/`

**KEEP (and adapt):**
- `agents/{base, tool_loop}.py`
- `library/store.py` (rewritten — FAISS sub-indices, no embeddings on Seed)
- `tools/retrieval.py` (rewritten as `retrieval/`)
- `tools/{fetch_paper, web_search}.py`
- `llm.py`, `log.py`, `config.py`, `paper.py`, `observation.py`, `types.py`

**NEW:**
- `accumulator/agent.py`, `accumulator/prompts.yaml`
- `synthesizer/synthesize.py`, `synthesizer/prompts.yaml`
- `retrieval/paraphrase.py`, `retrieval/retrieve.py`, `retrieval/prompts.yaml`
- `bootstrap/sampler.py` (S2 query, year×citation grid)
- `bootstrap/fetch.py` (PDF download + markdown)
- `domain.py` (Enum + S2 field mapping + override table)
- `scripts/{bootstrap_papers, run_accumulator, run_synthesis}.py`

File-level migration plan, ordering, and test strategy are produced by the writing-plans skill in the next step.

## 10. Out of scope

- Section 4 Experiment design (user-owned).
- Multi-user or concurrent retrieval. Workshop scope = single user, single query.
- API server / live serving. Per-query CLI is sufficient.
- Cross-paper seed deduplication. If two papers propose near-identical mechanisms, both seeds remain — diversity at retrieval time is enforced by the per-domain partition.
- Seed quality filtering / scoring. Accumulator's output is accepted as-is.
