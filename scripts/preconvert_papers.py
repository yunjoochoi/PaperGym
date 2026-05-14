"""GPU-server one-shot: download arxiv PDFs and convert to markdown via docling.

Output goes to <cache-root>/<arxiv_id>/paper.md so the PaperGym Accumulator
containers can mount the cache and skip the slow docling step entirely.

Setup on the GPU server (HF must be reachable here):

    git clone <PaperGym repo>
    cd PaperGym
    uv venv .venv && source .venv/bin/activate
    uv pip install -e .
    # Pre-download docling models once
    python -c "from docling.utils.model_downloader import download_models; download_models()"

Run:

    python scripts/preconvert_papers.py \
        --arxiv-ids data/arxiv_ids.jsonl \
        --cache-root /scratch/papers_cache \
        --workers 4

Then copy the cache back to the main host:

    rsync -av gpu-server:/scratch/papers_cache/ \
        ./data/papers_cache/

Notes
-----
* docling auto-uses CUDA if available (5-10x faster than CPU).
* Each worker process loads docling independently (~30s warmup each), so
  worker count = (GPU memory budget // ~3 GB) is a reasonable upper bound.
  Use --workers 1 if the GPU is small.
* Resume-safe: rerunning skips arxiv_ids whose paper.md already exists.
* docling failures fall back to pymupdf4llm inside fetch_paper_to_disk().
"""
import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _convert_one(arxiv_id: str, cache_root: Path) -> tuple[str, bool, str]:
    """Worker: download + convert one paper. Returns (id, ok, message)."""
    from papergym.env.preparer import fetch_paper_to_disk
    try:
        path = fetch_paper_to_disk(arxiv_id=arxiv_id, root=cache_root)
        return arxiv_id, True, str(path)
    except Exception as e:
        return arxiv_id, False, f"{type(e).__name__}: {e}"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arxiv-ids", required=True, type=Path,
                    help="jsonl with one {arxiv_id: ...} per line.")
    p.add_argument("--cache-root", required=True, type=Path,
                    help="Output dir; paper.md is written under "
                         "<cache-root>/<arxiv_id>/paper.md.")
    p.add_argument("--workers", type=int, default=4,
                    help="Parallel workers. Lower if GPU memory is tight.")
    p.add_argument("--limit", type=int, default=None,
                    help="Process at most N papers (after the resume filter).")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.cache_root.mkdir(parents=True, exist_ok=True)

    all_ids = []
    with args.arxiv_ids.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("arxiv_id"):
                all_ids.append(row["arxiv_id"])

    pending = [
        aid for aid in all_ids
        if not (args.cache_root / aid / "paper.md").exists()
    ]
    cached_already = len(all_ids) - len(pending)
    if args.limit:
        pending = pending[:args.limit]

    logging.info(
        "%d total / %d cached / %d pending (workers=%d)",
        len(all_ids), cached_already, len(pending), args.workers,
    )
    if not pending:
        return

    t0 = time.time()
    n_ok = n_fail = 0
    failures: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_convert_one, aid, args.cache_root): aid
            for aid in pending
        }
        for fut in as_completed(futures):
            aid, ok, msg = fut.result()
            if ok:
                n_ok += 1
            else:
                n_fail += 1
                failures.append((aid, msg))
                logging.warning("FAIL %s: %s", aid, msg)
            done = n_ok + n_fail
            if done % 10 == 0 or done == len(pending):
                elapsed = time.time() - t0
                rate_per_min = done / max(elapsed / 60, 1e-6)
                eta_min = (len(pending) - done) / max(rate_per_min, 1e-6)
                logging.info(
                    "%d/%d done (%d ok, %d fail) — %.2f papers/min, "
                    "ETA %.1f min",
                    done, len(pending), n_ok, n_fail,
                    rate_per_min, eta_min,
                )

    elapsed_min = (time.time() - t0) / 60
    logging.info(
        "DONE — %d ok, %d fail in %.1f min", n_ok, n_fail, elapsed_min,
    )
    if failures:
        log_path = args.cache_root / "preconvert_failures.jsonl"
        with log_path.open("a", encoding="utf-8") as f:
            for aid, msg in failures:
                f.write(json.dumps(
                    {"arxiv_id": aid, "error": msg},
                    ensure_ascii=False,
                ) + "\n")
        logging.info("failures appended to %s", log_path)


if __name__ == "__main__":
    main()
