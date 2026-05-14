from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

import papergym.env.preparer as preparer_mod
from papergym.env.preparer import fetch_paper_to_disk


def test_writes_paper_md_under_arxivid_dir(tmp_path: Path):
    arxiv_id = "2401.12345"
    with patch("papergym.env.preparer._download_pdf",
                return_value=tmp_path / "tmp.pdf") as dl, \
         patch("papergym.env.preparer._pdf_to_markdown",
                return_value="# Title\n\nbody") as conv:
        out = fetch_paper_to_disk(arxiv_id=arxiv_id, root=tmp_path / "papers")
    assert out == tmp_path / "papers" / arxiv_id / "paper.md"
    assert out.read_text().startswith("# Title")
    dl.assert_called_once()
    conv.assert_called_once()


def test_download_pdf_retries_transient_fetch_error(tmp_path: Path, monkeypatch):
    responses = [
        _pdf_response([b"short"], content_length=100),
        _pdf_response([b"%PDF full"], content_length=9),
    ]
    get = MagicMock(side_effect=responses)
    sleeps: list[float] = []

    monkeypatch.setattr(preparer_mod.requests, "get", get)
    monkeypatch.setattr(preparer_mod.random, "uniform", lambda _lo, _hi: 0.0)

    out = preparer_mod._download_pdf(
        "2401.12345",
        tmp_path,
        max_attempts=2,
        base_sleep_s=0.1,
        sleep_fn=sleeps.append,
    )

    assert out == tmp_path / "2401.12345.pdf"
    assert out.read_bytes() == b"%PDF full"
    assert get.call_count == 2
    assert sleeps == [0.1]


def test_download_pdf_does_not_retry_non_retryable_error(
    tmp_path: Path, monkeypatch
):
    response = _pdf_response([b"not found"], status_code=404)
    get = MagicMock(return_value=response)
    monkeypatch.setattr(preparer_mod.requests, "get", get)

    with pytest.raises(requests.HTTPError):
        preparer_mod._download_pdf(
            "bad-id",
            tmp_path,
            max_attempts=3,
            base_sleep_s=0.1,
            sleep_fn=lambda _s: None,
        )

    assert get.call_count == 1


def _pdf_response(
    chunks: list[bytes],
    *,
    status_code: int = 200,
    content_length: int | None = None,
    content_type: str = "application/pdf",
):
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"content-type": content_type}
    if content_length is not None:
        response.headers["content-length"] = str(content_length)
    response.iter_content.return_value = iter(chunks)
    response.__enter__.return_value = response
    response.__exit__.return_value = None

    def raise_for_status():
        if status_code >= 400:
            raise requests.HTTPError(response=response)

    response.raise_for_status.side_effect = raise_for_status
    return response
