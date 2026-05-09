"""Sample arxiv_ids per domain (no PDF fetch — that happens inside the Accumulator container)."""
import argparse
import json
import os
import random
import sys
from pathlib import Path

from dotenv import load_dotenv

from papergym.env.sampler import (PaperSearchClient, is_listed_venue,
                                    sample_paper_grid)
from papergym.domain import Domain, DOMAIN_FIELDS

load_dotenv(override=True)

DEFAULT_BUDGET = {
    Domain.LLM_NLP: 107,
    Domain.MULTIMODAL: 89,
    Domain.CV: 77,
    Domain.RL: 60,
    Domain.IR_REC: 60,
    Domain.SPEECH: 60,
    Domain.ROBOTICS: 47,
}


def _search_for_domain(client: PaperSearchClient, domain: Domain,
                        max_results: int) -> list[dict]:
    """One bulk request per domain: labels OR-combined via S2 boolean syntax."""
    or_query = " | ".join(f'"{lbl}"' for lbl in DOMAIN_FIELDS[domain])
    rows = client.search_papers(query=or_query,
                                  year_range=(2017, 2025),
                                  max_results=max_results)
    seen = set()
    out = []
    for r in rows:
        pid = r.get("paperId")
        if pid and pid not in seen:
            out.append(r); seen.add(pid)
    return out


def _arxiv_id(row: dict) -> str | None:
    eid = row.get("externalIds") or {}
    return eid.get("ArXiv")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path,
                    help="output jsonl with one row per sampled paper")
    p.add_argument("--budget-per-domain", type=int, default=None)
    p.add_argument("--only-domain", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--api-key", default=None,
                    help="Semantic Scholar API key; falls back to S2_API_KEY env var.")
    p.add_argument("--all-venues", action="store_true",
                    help="Disable the venue allowlist filter (default: keep "
                         "only listed venues — NeurIPS/ICML/ICLR/CVPR/...).")
    p.add_argument("--over-fetch", type=int, default=10,
                    help="Multiplier on per-domain budget for S2 over-fetch. "
                         "Server-side publicationTypes=Conference + client-side "
                         "venue allowlist trim less now, so 10x is usually "
                         "enough.")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    api_key = args.api_key or os.environ.get("S2_API_KEY")
    client = PaperSearchClient(api_key=api_key)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    domains = [Domain(args.only_domain)] if args.only_domain else list(Domain)
    total = 0
    with args.out.open("w", encoding="utf-8") as fp:
        for source_domain in domains:
            budget = args.budget_per_domain or DEFAULT_BUDGET[source_domain]
            over_fetch = budget * args.over_fetch
            print(f"[{source_domain.value}] searching (budget={budget}, "
                  f"over-fetch={over_fetch})...",
                  file=sys.stderr, flush=True)
            rows = _search_for_domain(client=client, domain=source_domain,
                                        max_results=over_fetch)
            rows = [r for r in rows if _arxiv_id(r)]
            if not args.all_venues:
                before = len(rows)
                rows = [r for r in rows if is_listed_venue(r)]
                print(f"[{source_domain.value}] venue filter: "
                      f"{before} -> {len(rows)}",
                      file=sys.stderr, flush=True)

            chosen_rows = sample_paper_grid(rows, budget=budget, rng=rng)
            if not chosen_rows and rows:
                chosen_rows = rows[:budget]
            for chosen in chosen_rows:
                fp.write(json.dumps({
                    "arxiv_id": _arxiv_id(chosen),
                    "source_query_domain": source_domain.value,
                    "year": chosen.get("year"),
                    "citations": chosen.get("citationCount"),
                    "title": chosen.get("title"),
                    "venue": chosen.get("venue"),
                }, ensure_ascii=False) + "\n")
            fp.flush()
            total += len(chosen_rows)
            print(f"[{source_domain.value}] kept {len(chosen_rows)} "
                  f"(running total={total})", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
