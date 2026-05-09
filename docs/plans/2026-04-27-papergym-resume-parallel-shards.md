# PaperGym — Resume + Parallel + Sharded Library

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Accumulator runs resumable, parallel across N Docker
containers, and safe under concurrent library writes via per-worker
shards. Disk pressure stays at zero on the host (containers stay `--rm`,
PDF/docling stays inside container — automatic cleanup on exit).

**Non-goals (deferred):**
- Host-side PDF prepare (current `--rm` cleanup makes this less valuable; revisit if disk/network becomes a bottleneck).
- RL-style reward feedback loop.
- Event-logging extension (latency/retry/cost in `events.jsonl`).

**Tech stack:** Python 3.11 stdlib `concurrent.futures.ProcessPoolExecutor`,
existing FAISS/JSONL store, no new deps.

**Current state to be aware of:**
- PR-A landed: `PaperEnv`, `Accumulator(env=)`, litellm env vars + dotenv, Dockerfile docling pre-warm.
- 49 unit tests green.
- `scripts/run_accumulator.py` is a sequential `for arxiv_id in ids: subprocess.run(["docker", "run", "--rm", ...])`. No resume, no parallelism, single library root forwarded.
- `LibraryStore` is a single-writer abstraction (FAISS index per domain + `seeds.jsonl`). Multi-process concurrent writes would race destructively. Hence sharding.

---

## Architecture

### Sharding model

Each parallel worker (= one slot in the pool) owns one shard:

```
<library_root>/
├── shard_0/
│   ├── seeds.jsonl
│   └── faiss/<domain>.index
├── shard_1/...
├── shard_2/...
└── shard_3/...
```

Worker N is assigned `shard_id = worker_idx`. Inside the container,
`accumulate_one.py` constructs `LibraryStore(library_root / f"shard_{N}")`.
No cross-worker writes → no contention.

Synthesis time reads all shards as one merged in-memory view.

### Resume model

`accumulator_log.jsonl` already records `{arxiv_id, status}` per paper.
`status ∈ {ok, skipped, error}`. On run start, load the log, skip any
arxiv_id whose latest entry is `ok` or `skipped`. `error` entries are
retried (treated as not-yet-done). The log is the single source of truth;
shards are derived state.

### Parallelism model

`ProcessPoolExecutor(max_workers=N)`. Each task = one paper. Workers
spawn Docker containers via `subprocess.run`. Workers persist across
tasks (one process handles many papers serially), so no per-task Python
startup cost. Shard assignment: `shard_id = task_index % N` is **wrong**
because the same arxiv_id might land on different shards on re-run (then
we'd skip it via the log, which is fine — but for new failures, retries
should land on the same shard for write locality). Use **stable shard
assignment**: `shard_id = stable_hash(arxiv_id) % N` so re-runs route
identically.

---

## Phase 0 — Pre-flight

### Task 0.0 — Verify PR-A is committed

**Steps:**
1. `git status` — confirm working tree clean (PR-A changes committed).
2. `pytest tests/unit -q` — green baseline (49 tests).
3. Note count for later comparison.

### Task 0.1 — Commit this plan

```bash
git add docs/plans/2026-04-27-papergym-resume-parallel-shards.md
git commit -m "docs: resume + parallel + sharded library plan"
```

---

## Phase 1 — Sharded library writes (container-side)

Goal: each container writes to its own shard subdir. Synthesizer not yet
updated — that's Phase 3.

### Task 1.0 — Add `--shard-id` to `accumulate_one.py`

**Files:**
- `scripts/accumulate_one.py` (edited)

**Steps:**
1. Add CLI arg:
   ```python
   p.add_argument("--shard-id", type=int, default=0,
                   help="Worker shard index; library writes go to library_root/shard_<id>/")
   ```
2. Change library construction:
   ```python
   library = LibraryStore(args.library_root / f"shard_{args.shard_id}")
   ```
3. Log the shard_id in the per-paper log record (debug-friendly):
   ```python
   _log(log_path, arxiv_id=..., status="ok", shard_id=args.shard_id, ...)
   ```
   (the global `accumulator_log.jsonl` is at `args.library_root / "accumulator_log.jsonl"` — single file shared across shards. JSONL append of a single small line is atomic on POSIX, but to be extra safe see Task 2.2.)

**Verify:** `pytest tests/unit/test_scripts_accumulate_one.py -q` may need updates: existing tests that assert `LibraryStore(library_root)` retrieval need to check `library_root / "shard_0"` instead. Update the assertions accordingly.

### Task 1.1 — Test: per-shard isolation

**Files:**
- `tests/unit/test_scripts_accumulate_one.py` (edited)

**Steps:**
1. Add a test that runs `main` twice with different `--shard-id` values for two different arxiv_ids and verifies seeds end up in `shard_0/` and `shard_1/` respectively.
2. Check that `accumulator_log.jsonl` (single global file) records both with their shard_ids.

**Verify:** new test passes; existing tests still green.

---

## Phase 2 — Parallel + resume in `run_accumulator.py`

Goal: spawn N containers in parallel, skip already-done arxiv_ids.

### Task 2.0 — Stable shard assignment helper

**Files:**
- `scripts/run_accumulator.py` (edited)

**Steps:**
1. Add module-level helper:
   ```python
   def _shard_for(arxiv_id: str, n_shards: int) -> int:
       """Stable shard assignment: same arxiv_id → same shard across runs."""
       import hashlib
       h = int(hashlib.sha1(arxiv_id.encode()).hexdigest(), 16)
       return h % n_shards
   ```
2. Pure function, easy to unit test.

### Task 2.1 — Resume helper: load done set

**Files:**
- `scripts/run_accumulator.py` (edited)

**Steps:**
1. Add helper:
   ```python
   def _already_done(library_root: Path) -> set[str]:
       """Return arxiv_ids whose latest log entry is 'ok' or 'skipped'."""
       log_path = library_root / "accumulator_log.jsonl"
       if not log_path.exists():
           return set()
       latest: dict[str, str] = {}
       for line in log_path.read_text().splitlines():
           if not line.strip():
               continue
           rec = json.loads(line)
           latest[rec["arxiv_id"]] = rec.get("status", "")
       return {aid for aid, s in latest.items() if s in ("ok", "skipped")}
   ```
2. In `main`: `done = _already_done(args.library_root); ids = [i for i in ids if i not in done]`.
3. Print summary: `f"resume: skipping {len(done)} already-done papers, processing {len(ids)} new"`.

### Task 2.2 — Parallel spawn via ProcessPoolExecutor

**Files:**
- `scripts/run_accumulator.py` (edited)

**Steps:**
1. Replace the sequential loop with:
   ```python
   from concurrent.futures import ProcessPoolExecutor, as_completed

   def _spawn_one(arxiv_id, *, image, library_root, events_dir,
                   max_steps, n_shards):
       shard_id = _shard_for(arxiv_id, n_shards)
       cmd = [
           "docker", "run", "--rm",
           "-v", f"{library_root.resolve()}:/library:rw",
           "-v", f"{events_dir.resolve()}:/events:rw",
       ]
       for var in FORWARDED_ENV:
           if var in os.environ:
               cmd.extend(["-e", f"{var}={os.environ[var]}"])
       cmd.extend([
           image,
           "--arxiv-id", arxiv_id,
           "--max-steps", str(max_steps),
           "--shard-id", str(shard_id),
       ])
       rc = subprocess.run(cmd).returncode
       return (arxiv_id, rc)

   with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
       futures = [
           pool.submit(_spawn_one, aid, image=args.image,
                       library_root=args.library_root,
                       events_dir=args.events_dir,
                       max_steps=args.max_steps,
                       n_shards=args.max_workers)
           for aid in ids
       ]
       for fut in as_completed(futures):
           aid, rc = fut.result()
           if rc != 0:
               print(f"warn: {aid} exited non-zero ({rc})", file=sys.stderr)
   ```
2. Add CLI arg `--max-workers` (default 4).
3. Set `n_shards = max_workers` so each worker has a dedicated shard.
4. Concurrent writes to `accumulator_log.jsonl` from N container processes: one short JSON line per record, well under PIPE_BUF (4KB on Linux). POSIX guarantees atomic append for `O_APPEND` writes ≤ PIPE_BUF, which Python's text-mode append uses. **No extra locking needed** — verify in Task 2.3 with a stress test.

### Task 2.3 — Stress test for concurrent log writes

**Files:**
- `tests/unit/test_scripts_run_accumulator.py` (edited)

**Steps:**
1. Add a test that fakes `subprocess.run` to write a JSONL line to `accumulator_log.jsonl` (simulating what `accumulate_one.py` does), then runs 20 fake "papers" through `ProcessPoolExecutor` with 4 workers, then verifies all 20 lines parse cleanly (no torn writes). This is a smoke test for the atomicity claim.

**Verify:** test runs in <2s (ProcessPoolExecutor startup is the bottleneck).

### Task 2.4 — Resume + max-papers interaction

**Files:**
- `scripts/run_accumulator.py` (edited)

**Steps:**
1. Decide order: filter resume FIRST, then `--max-papers` slice. So `--max-papers 10` means "process 10 NEW papers", not "process the first 10 from arxiv_ids.jsonl regardless of done state". This is the intuitive UX for repeat runs.
2. Update tests in `test_scripts_run_accumulator.py` if they pin the old order.

---

## Phase 3 — Synthesizer reads all shards

Goal: at synthesis time, present shards as a single merged library to
`retrieve_cross_domain` without changing Synthesizer logic.

### Task 3.0 — `LibraryStore.open_merged` class method

**Files:**
- `src/papergym/library/store.py` (edited)

**Steps:**
1. Add:
   ```python
   @classmethod
   def open_merged(cls, root: Path, embedding_dim: int = EMBEDDING_DIM
                    ) -> "LibraryStore":
       """Open a library that may be sharded into <root>/shard_*/.
       
       If shard_N/ subdirs are found, load all of them into a single
       in-memory store (no on-disk merge). Otherwise treat <root> as a
       plain LibraryStore.
       """
       shard_dirs = sorted(p for p in root.glob("shard_*") if p.is_dir())
       if not shard_dirs:
           return cls(root, embedding_dim=embedding_dim)
       merged = cls(root, embedding_dim=embedding_dim)
       # Wipe any state from the just-constructed merged instance.
       merged._seeds_by_domain = {d: [] for d in Domain}
       merged._faiss = {}
       for shard in shard_dirs:
           sub = cls(shard, embedding_dim=embedding_dim)
           for d in Domain:
               for s in sub._seeds_by_domain[d]:
                   merged._seeds_by_domain[d].append(s)
               if d in sub._faiss and sub._faiss[d].ntotal > 0:
                   if d not in merged._faiss:
                       merged._faiss[d] = faiss.IndexFlatIP(embedding_dim)
                   # Reconstruct vectors from the shard index and append.
                   vecs = sub._faiss[d].reconstruct_n(0, sub._faiss[d].ntotal)
                   merged._faiss[d].add(vecs)
       # IMPORTANT: this merged instance is read-only (its add() would
       # write to root/seeds.jsonl, ignoring shards). Document this.
       merged._read_only = True
       return merged
   ```
2. Override `add()` to raise if `getattr(self, "_read_only", False)` is True. Synthesis path doesn't add; this guards against accidental misuse.

### Task 3.1 — Synthesizer uses merged view

**Files:**
- `scripts/run_synthesis.py` (edited)

**Steps:**
1. Replace `library = LibraryStore(args.library_root)` with `library = LibraryStore.open_merged(args.library_root)`.
2. No other change — `retrieve_cross_domain` works on the merged instance.

### Task 3.2 — Tests for shard merge

**Files:**
- `tests/unit/test_library_store_merge.py` (new)

**Steps:**
1. Build two shards manually with different seeds in different domains, call `open_merged`, verify:
   - All seeds present in merged store
   - FAISS counts match jsonl counts per domain
   - `retrieve` returns top-k across shards correctly
   - `add()` raises on the merged instance
2. Test fallback: empty root (no shards) → behaves like normal `LibraryStore(root)`.

**Verify:** `pytest tests/unit -q` green.

---

## Phase 4 — Cleanup + docs

### Task 4.0 — `bootstrap/` directory cleanup (optional)

`bootstrap/__init__.py` is empty; only `bootstrap/fetch.py` lives there.
We could move `fetch.py` to `env/preparer.py` and delete `bootstrap/`,
but that's churn for no behavior change. **Skip unless touching this
area for another reason.**

### Task 4.1 — README / design doc updates

**Files:**
- `README.md` (or new doc section)

**Steps:**
1. Document the shard layout under `data/library/shard_*/` so users know what they're seeing on disk.
2. Document `--max-workers` flag and default.
3. Document resume behavior: "re-run is safe; only papers without an `ok`/`skipped` log entry are reprocessed".

### Task 4.2 — Final verification

**Steps:**
1. `pytest tests/unit -q` — all green.
2. `grep -rn "LibraryStore(args.library_root)" scripts/ tests/` — should only appear in `accumulate_one.py` (intentional, shard-specific). `run_synthesis.py` should use `open_merged`.
3. Manual smoke (optional): run pipeline against 3-5 papers with `--max-workers 2`, verify two shards get populated and synthesis works against the merged view.

---

## Risks & mitigation

| Risk | Mitigation |
|---|---|
| `accumulator_log.jsonl` torn write under N concurrent appenders | Records are short single JSON lines well under PIPE_BUF (4KB). POSIX guarantees atomic O_APPEND. Validated in Task 2.3 stress test. |
| Re-run lands a paper on a different shard than first run | `_shard_for` is deterministic (sha1-based). Same arxiv_id → same shard always. |
| `error` papers re-run forever if they keep failing | Acceptable for now: error is rare and worth retrying. Future: add `--retry-errors` toggle, default true. Not in scope. |
| Merged view has stale data after a parallel run completes | `open_merged` re-reads from disk every call. Synthesizer is a one-shot script, no cache. Acceptable. |
| Container kill mid-write to a shard's FAISS index | Existing `LibraryStore._load` checks `idx.ntotal != n_seeds` and raises. Fix-up: delete the younger of the two files in that shard, re-run accumulator on the affected paper(s). Document in README. |

## Out-of-scope reminders

- Host-side fetch / prepare (current `--rm` handles cleanup).
- Reward / RL loop.
- Event-logging fields (latency/retry/cost).
- Replacing `bootstrap/` directory.
