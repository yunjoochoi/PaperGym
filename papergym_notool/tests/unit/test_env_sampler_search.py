from unittest.mock import MagicMock, patch
from papergym.env.sampler import PaperSearchClient


def _resp(payload, status=200):
    r = MagicMock(); r.status_code = status; r.json.return_value = payload
    r.raise_for_status = MagicMock()
    return r


def test_search_papers_paginates_until_budget(monkeypatch):
    pages = [
        _resp({"data": [{"paperId": f"p{i}", "title": f"t{i}",
                          "year": 2024, "citationCount": 100 - i,
                          "externalIds": {"ArXiv": f"2401.000{i}"},
                          "s2FieldsOfStudy": [{"category": "Computer Science"}]}
                         for i in range(50)],
                "next": 50}),
        _resp({"data": [{"paperId": f"q{i}", "title": f"x{i}",
                          "year": 2023, "citationCount": 50 - i,
                          "externalIds": {"ArXiv": f"2301.000{i}"},
                          "s2FieldsOfStudy": [{"category": "Computer Science"}]}
                         for i in range(30)]}),
    ]
    iter_pages = iter(pages)
    fake_get = lambda *a, **kw: next(iter_pages)
    monkeypatch.setattr("papergym.env.sampler.requests.get", fake_get)

    client = PaperSearchClient(api_key=None)
    rows = client.search_papers(query="machine learning", year_range=(2017, 2025), max_results=80)
    assert len(rows) == 80
    assert rows[0]["paperId"] == "p0"
