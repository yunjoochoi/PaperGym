"""Host: iterate arxiv_ids.jsonl and run the no-tool accumulator locally.

Resumes by skipping arxiv_ids whose latest entry in
<library_root>/accumulator_log.jsonl is `ok` or `skipped`. Errors are
retried.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

import accumulate_one

load_dotenv(override=True)

# Fixed shard count. Changing this after the library is populated would
# re-route arxiv_ids to different shards and produce duplicates.
N_SHARDS = 4


def _read_arxiv_ids(path: Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("arxiv_id"):
            out.append(row["arxiv_id"])
    return out


def _shard_for(arxiv_id: str, n_shards: int = N_SHARDS) -> int:
    """Stable shard assignment: same arxiv_id → same shard across runs."""
    h = int(hashlib.sha1(arxiv_id.encode()).hexdigest(), 16)
    return h % n_shards


def _already_done(library_root: Path) -> set[str]:
    """arxiv_ids whose latest log entry is `ok` or `skipped`. Errors retry."""
    log_path = library_root / "accumulator_log.jsonl"
    if not log_path.exists():
        return set()
    latest: dict[str, str] = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        aid = rec.get("arxiv_id")
        status = rec.get("status", "")
        if aid:
            latest[aid] = status
    return {aid for aid, s in latest.items() if s in ("ok", "skipped")}


def _run_one(arxiv_id: str, *, work_root: Path, library_root: Path,
             events_dir: Path, papers_cache: Path | None) -> tuple[str, int]:
    shard_id = _shard_for(arxiv_id)
    argv = [
        "--arxiv-id", arxiv_id,
        "--work-root", str(work_root),
        "--library-root", str(library_root),
        "--events-dir", str(events_dir),
        "--shard-id", str(shard_id),
    ]
    if papers_cache is not None and papers_cache.exists():
        argv.extend(["--papers-cache", str(papers_cache)])
    try:
        accumulate_one.main(argv=argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        return (arxiv_id, code)
    return (arxiv_id, 0)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arxiv-ids", required=True, type=Path,
                    help="jsonl produced by sample_envs.py")
    p.add_argument("--library-root", required=True, type=Path)
    p.add_argument("--events-dir", required=True, type=Path)
    p.add_argument("--work-root", type=Path, default=Path("data/work"),
                    help="Host directory for paper.md and per-paper traces.")
    p.add_argument("--max-papers", type=int, default=None,
                    help="Process at most N NEW papers (after resume filter).")
    p.add_argument("--max-workers", type=int, default=4,
                    help="Parallel local worker slots. Shard count is fixed at "
                         f"{N_SHARDS}.")
    p.add_argument("--spawn-delay-s", type=float, default=10.0,
                    help="Sleep between paper starts to avoid bursting "
                         "arXiv fetch requests. Set 0 to disable.")
    p.add_argument("--papers-cache", type=Path, default=None,
                    help="Host path to a pre-converted paper.md cache "
                         "(e.g., data/papers_cache). "
                         "When supplied, the Accumulator skips docling "
                         "for any arxiv_id present there.")
    args = p.parse_args(argv)

    if "OPENAI_API_KEY" not in os.environ:
        print("error: OPENAI_API_KEY not set in environment", file=sys.stderr)
        sys.exit(1)

    args.library_root.mkdir(parents=True, exist_ok=True)
    args.events_dir.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)

    all_ids = _read_arxiv_ids(args.arxiv_ids)
    done = _already_done(args.library_root)
    pending = [i for i in all_ids if i not in done]
    if args.max_papers:
        pending = pending[:args.max_papers]

    print(f"resume: {len(done)} done, {len(pending)} pending "
          f"(of {len(all_ids)} total); {args.max_workers} workers, "
          f"{N_SHARDS} shards, spawn_delay={args.spawn_delay_s:g}s",
          file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = []
        for n, aid in enumerate(pending):
            if n > 0 and args.spawn_delay_s > 0:
                time.sleep(args.spawn_delay_s)
            futures.append(
                pool.submit(_run_one, aid,
                            work_root=args.work_root,
                            library_root=args.library_root,
                            events_dir=args.events_dir,
                            papers_cache=args.papers_cache)
            )
        for fut in as_completed(futures):
            aid, rc = fut.result()
            if rc != 0:
                print(f"warn: {aid} exited non-zero ({rc})", file=sys.stderr)


if __name__ == "__main__":
    main()
