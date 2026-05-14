# papers_cache

Pre-converted markdown for the 446 arxiv papers backing the seed library.
Used by Stage 1 grounding rubrics (`scripts/seed_quality_eval.py`,
`scripts/seed_shuffled.py`).

- Source: arxiv.org preprints
- Conversion: docling (CUDA path) with pymupdf4llm fallback,
  via `scripts/preconvert_papers.py`
- Rights: copyright remains with the original authors; this cache is bundled
  solely to make Stage 1 reproducible without a GPU
- Layout: `<arxiv_id>/paper.md`, one per paper (446 total)
- Regenerate from scratch:
  `python scripts/preconvert_papers.py --arxiv-ids data/arxiv_ids.jsonl --cache-root data/papers_cache`
