from __future__ import annotations

import random
import re
import time
from typing import Optional

import requests


YEAR_BUCKETS = [
    ("2017-19", range(2017, 2020)),
    ("2020-22", range(2020, 2023)),
    ("2023-25", range(2023, 2026)),
]
CITATION_TIERS = [("top_25", 0.0, 0.25),
                  ("mid_25_50", 0.25, 0.50),
                  ("rest", 0.50, 1.00)]


def _year_bucket(year: int) -> str | None:
    for name, rng in YEAR_BUCKETS:
        if year in rng:
            return name
    return None


def bucket_grid(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Partition rows into (year_cohort, citation_tier) cells; tiers are year-relative."""
    by_year: dict[str, list[dict]] = {y: [] for y, _ in YEAR_BUCKETS}
    for r in rows:
        y = r.get("year")
        if y is None:
            continue
        b = _year_bucket(y)
        if b is None:
            continue
        by_year[b].append(r)

    grid: dict[tuple[str, str], list[dict]] = {}
    for cohort, cohort_rows in by_year.items():
        if not cohort_rows:
            continue
        cohort_rows.sort(key=lambda r: -r.get("citationCount", 0))
        n = len(cohort_rows)
        for tier_name, lo, hi in CITATION_TIERS:
            start = int(n * lo)
            end = int(n * hi)
            grid[(cohort, tier_name)] = cohort_rows[start:end]
    return grid


def sample_paper_grid(rows: list[dict], budget: int,
                      rng: random.Random) -> list[dict]:
    """Sample up to `budget` rows spread across (year, citation) cells.

    Pool small (<= budget): keep everything.
    Pool large: take ~budget/n_cells per cell, then top up to `budget`
    from leftovers ordered by citation rank.
    """
    if len(rows) <= budget:
        return list(rows)

    grid = bucket_grid(rows)
    n_cells = max(len(grid), 1)
    per_cell = max(budget // n_cells, 1)
    chosen: list[dict] = []
    chosen_ids: set[str] = set()
    for cell_rows in grid.values():
        take = min(per_cell, len(cell_rows))
        for r in rng.sample(cell_rows, take):
            chosen.append(r)
            chosen_ids.add(r.get("paperId") or "")

    if len(chosen) < budget:
        leftovers = [r for r in rows
                     if (r.get("paperId") or "") not in chosen_ids]
        leftovers.sort(key=lambda r: -r.get("citationCount", 0))
        chosen.extend(leftovers[:budget - len(chosen)])
    return chosen[:budget]


_API_BASE = "https://api.semanticscholar.org/graph/v1"
SEARCH_FIELDS = ("paperId,title,year,citationCount,externalIds,"
                 "s2FieldsOfStudy,authors,venue,publicationVenue")

# Substring patterns matched case-insensitively against the S2 `venue` /
# `publicationVenue.name` strings. S2 venue strings are inconsistent
# (sometimes abbrev, sometimes full name, sometimes workshop suffix), so
# we keep both forms and match permissively.
LISTED_VENUE_PATTERNS = [
    "neurips", "neural information processing systems",
    "icml", "international conference on machine learning",
    "iclr", "international conference on learning representations",
    "cvpr", "computer vision and pattern recognition",
    "iccv", "international conference on computer vision",
    "eccv", "european conference on computer vision",
    "acl", "annual meeting of the association for computational linguistics",
    "aaai",
    "emnlp", "empirical methods in natural language processing",
    "sigir",
    "kdd", "knowledge discovery and data mining",
    "interspeech",
    "icra", "international conference on robotics and automation",
    "iros", "intelligent robots and systems",
    "corl", "conference on robot learning",
    "recsys", "conference on recommender systems",
    "icassp",
]


def _venue_text(row: dict) -> str:
    parts = [row.get("venue") or ""]
    pv = row.get("publicationVenue") or {}
    if isinstance(pv, dict):
        parts.append(pv.get("name") or "")
    return " | ".join(parts).lower()


_VENUE_REGEX = re.compile(
    r"(?<![a-z0-9])(?:" + "|".join(re.escape(p) for p in LISTED_VENUE_PATTERNS)
    + r")(?![a-z0-9])",
    re.IGNORECASE,
)


def is_listed_venue(row: dict) -> bool:
    """Match listed venues against the row's venue text on word-ish boundaries.

    The boundary lookarounds reject substring hits like 'kdd' inside
    'ECML/PKDD' or 'acl' inside 'oracle', while still allowing slashes,
    spaces, and punctuation around the abbrev.
    """
    text = _venue_text(row)
    if not text.strip():
        return False
    return bool(_VENUE_REGEX.search(text))


class PaperSearchClient:
    def __init__(self, api_key: Optional[str] = None, sleep_s: float = 2.0):
        self.api_key = api_key
        self.sleep_s = sleep_s

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key} if self.api_key else {}

    def _get_with_backoff(self, url: str, params: dict,
                           max_retries: int = 10) -> dict:
        """Wrap GET with exponential backoff on 429/5xx; respect Retry-After."""
        delay = 5.0
        for attempt in range(max_retries):
            r = requests.get(url, params=params,
                              headers=self._headers(), timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                if attempt == max_retries - 1:
                    r.raise_for_status()
                retry_after = r.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                wait = min(wait, 120.0)
                time.sleep(wait)
                delay = min(delay * 2, 120.0)
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError("unreachable")

    def search_papers(self, *, query: str, year_range: tuple[int, int],
                      max_results: int,
                      publication_types: Optional[str] = "Conference"
                      ) -> list[dict]:
        params: dict = {
            "query": query,
            "fields": SEARCH_FIELDS,
            "year": f"{year_range[0]}-{year_range[1]}",
            "limit": 100,
        }
        if publication_types:
            params["publicationTypes"] = publication_types
        out: list[dict] = []
        token: Optional[str] = None
        while len(out) < max_results:
            if token is not None:
                params["token"] = token
            data = self._get_with_backoff(
                f"{_API_BASE}/paper/search/bulk", params)
            out.extend(data.get("data", []))
            nxt = data.get("next")
            if nxt is None:
                break
            token = nxt if isinstance(nxt, str) else None
            time.sleep(self.sleep_s)
        return out[:max_results]
