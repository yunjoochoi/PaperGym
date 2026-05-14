import random
from papergym.env.sampler import bucket_grid, sample_paper_grid


def _row(paper_id, year, citations):
    return {"paperId": paper_id, "year": year, "citationCount": citations}


def test_bucket_grid_assigns_year_cohort_and_citation_tier():
    rows = (
        [_row(f"a{i}", 2018, 1000 - i) for i in range(20)] +
        [_row(f"b{i}", 2024, 500 - i) for i in range(20)]
    )
    grid = bucket_grid(rows)
    assert ("2017-19", "top_25") in grid
    assert ("2017-19", "mid_25_50") in grid
    assert ("2017-19", "rest") in grid
    assert ("2023-25", "top_25") in grid
    assert all(r["year"] == 2018 for r in grid[("2017-19", "top_25")])


def test_sample_paper_grid_distributes_per_cell(monkeypatch):
    rng = random.Random(0)
    rows = [_row(f"p{i}", 2018, 1000 - i) for i in range(30)]
    sampled = sample_paper_grid(rows, budget=9, rng=rng)
    assert len(sampled) <= 9
