# Paper as Environment — Workshop Framing Notes

> Design notes that the workshop submission text will draw from. Not a
> spec; not a plan. Plain assertions about *why* the system is shaped the
> way it is.

## Position in one sentence

PaperGym treats each ML paper as an **episodic environment** that an
agent enters, explores via constrained tools (Read / Grep / Bash), and
exits with a structured artifact (1–3 mechanism *seeds*) — building a
cross-domain seed library that powers method-synthesis on novel queries.

## Why "environment" is the right abstraction

| Property of an RL environment | Counterpart in PaperGym |
|---|---|
| Episodic boundaries | One paper = one episode (one `--rm` Docker container) |
| Reproducible reset (paper-id keyed)¹ | `PaperEnv.reset()` ⇒ `paper.md` from `arxiv_id` |
| Tool-mediated state transitions | `PaperEnv.step(name, args)` ⇒ Read / Grep / Bash dispatch |
| Isolation between episodes | Per-paper Docker container (untrusted `git clone` + `bash`) |
| Reproducibility from seed | `arxiv_id` is the only input; idempotent re-runs produce the same `paper.md` |

¹ Not deterministic in the strict RL sense: `reset()` calls into the
arxiv API + docling, both of which depend on network state and an
external service. "Reproducible" means the same `arxiv_id` produces
the same `paper.md` modulo arxiv version drift and the pinned docling
version baked into the Docker image. Once `paper.md` exists, `step()`
calls are deterministic functions of the workspace.

The Gym shape is not cosmetic. It maps directly to natural seams in the
problem: paper boundary, fresh sandbox per paper, deterministic
inputs, and a tool surface that's narrow enough to enumerate.

## What we deliberately defer (future work)

The current system implements the **environment + agent + library**
loop. It does not yet close the loop with a reward signal. Specifically:

- **No reward in `step()`** — `step()` returns an observation string,
  not `(obs, reward, done, info)`. Reward would require a downstream
  evaluator on synthesis quality (e.g., another LLM rating method
  proposals, or human ratings) that scores back to seed quality.
- **No policy update** — the Accumulator is an LLM with a fixed
  prompt; there is no training loop adjusting parameters or weighting
  seeds based on past synthesis success.
- **No re-rollout** — once a paper has been processed, its seeds enter
  the library and are immutable. A reward signal would naturally drive
  re-rollout (try a different extraction prompt, prune low-quality
  seeds) but we don't yet route that signal.

These are "Phase 2" extensions. The workshop submission positions Phase 1
(environment + library + synthesis) as a standalone contribution: a
**reusable, isolated, reproducible substrate** for cross-domain idea
synthesis. Adding reward closes the loop; the substrate must work
first.

## The cross-domain claim

Each `PaperEnv` carries a domain label (one of 7) assigned by the
Accumulator from paper content. The library is partitioned by this
label. At synthesis time, a user query is paraphrased into each
domain's vocabulary; we retrieve top-K seeds *per domain* and feed all
of them to a single synthesizer that emits a method proposal with
explicit `inspired_by` provenance.

This forces the synthesizer to consider mechanisms outside the query's
natural domain. The provenance field is the workshop's measurement
unit: every concrete trick in the proposed method must trace back to at
least one seed, and cross-domain inspirations (where the seed's domain
differs from the query's natural domain) are the artifacts the system
exists to produce.

## Why not just retrieval-augmented generation over papers?

Three hypotheses motivate the seed-and-synthesize structure over
document-level RAG. They are properties we *aim for* — verifying them
empirically is part of the evaluation roadmap.

1. **Compression for retrieval, not stuffing.** A 30-page paper
   compresses to 1–3 problem/method pairs (~500 tokens). The point is
   not that the full library fits in one context (it doesn't —
   500 × ~500 = ~250k tokens, and even long-context models couldn't
   usefully retrieve over that as a single prompt); it is that vector
   search over short, mechanism-level units behaves better than over
   30-page documents. Granularity, not capacity.
2. **Hypothesis: cross-domain mechanism similarity.** Vector search
   over a paper's full text retrieves on surface-level vocabulary;
   vector search over a paper's distilled method *should* retrieve on
   mechanism similarity even when the source domain uses different
   language. This is the central empirical claim of the system and the
   evaluation has to test it (e.g., paraphrased queries returning
   relevant out-of-domain seeds at higher rates than full-text RAG
   baselines). Asserted as a goal here, not a result.
3. **Provenance is structural, not enforced.** The synthesizer emits an
   `inspired_by` field listing seed IDs. This makes the proposal
   *auditable* — a human can trace each mechanism back to a seed and
   the seed back to a paper. Note that the system does not currently
   *enforce* provenance: there is no post-hoc validator checking that
   listed seed IDs exist in the library or that the cited mechanism
   actually appears in the seed text. Validation is a near-term
   addition; for now "auditable" means "human-spot-checkable", not
   "machine-verified".

## Reproducibility

- `arxiv_id` is the only paper-side input. `LITELLM_MODEL`,
  `OPENAI_API_BASE`, and the embedding model are read from environment
  variables at runtime and forwarded into containers, but they are
  **not yet snapshotted** into the artifact manifest alongside
  `seeds.jsonl` / `accumulator_log.jsonl`. Recording them per-run is
  a near-term fix.
- A run is reproducible up to provider determinism (LLM sampling,
  embedding model versioning behind a vendor endpoint) and arxiv
  version drift.
- Sample list (`arxiv_ids.jsonl`) is generated by `sample_envs.py` with
  a fixed `--seed`.
- Docker image pins all deps via `pip install -e .`; docling models are
  pre-warmed at build time.
- Library is sharded for safe concurrent writes and merged read-only at
  synthesis time. Re-runs route the same arxiv_id to the same shard
  via stable hash.

## Relation to prior work

PaperGym sits between two lines of existing systems and is positioned
against both:

- **Agent-task environments** (SWE-Bench, WebArena, MLE-Bench, etc.)
  give an agent a sandboxed task and a tool surface. PaperGym borrows
  this shape — `--rm` container per episode, narrow tool surface
  (Read / Grep / Bash), Docker-mediated isolation against untrusted
  `git clone` and `bash` — but the *task* is reading a paper and
  emitting structured seeds, not solving a coding problem. The
  artifact persisted out of the episode is data for a downstream
  retrieval + synthesis stage, not a code patch.
- **Agent-driven literature/idea synthesis** (ResearchAgent, SciAgent,
  IdeaSynth, and the broader "AI-as-co-scientist" line) take user
  queries and propose research directions, often with citations. The
  differences here are (a) a persistent, domain-partitioned seed
  library decouples ingestion from synthesis time, so retrieval is
  fast and the same library powers many queries; (b) the synthesizer
  is forced to consider mechanisms outside the query's natural domain
  via per-domain paraphrase + retrieval, with explicit `inspired_by`
  provenance. Cross-domain attribution is the differentiator.

We do not claim novelty in environment design or in agent-driven
synthesis individually — the contribution is the combination plus the
cross-domain framing.

## Evaluation roadmap

The submission's empirical claims rest on evaluations not all of which
are run yet. The roadmap, in order of priority:

1. **Seed quality vs. synthesis quality.** Human-judged: do high-rated
   seeds (problem and method clearly stated, distinctive method)
   produce better-rated synthesis outputs?
2. **Cross-domain retrieval beats single-domain.** For a held-out set
   of queries, does the per-domain paraphrase retrieval surface
   mechanisms that single-domain retrieval (or full-text RAG) misses?
3. **Provenance correlates with rated novelty.** Do synthesis outputs
   with more cross-domain `inspired_by` entries get higher novelty
   ratings from domain experts than within-domain ones?
4. **OOD graceful failure.** On queries far from any seed cluster
   in embedding space, does the system refuse / express low confidence
   rather than hallucinate?

Items 1 and 2 are the gating ones for the submission; 3 and 4 are
follow-up.
