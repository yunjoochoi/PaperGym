from __future__ import annotations

import logging
import random
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from urllib.error import HTTPError, URLError

import requests

log = logging.getLogger(__name__)
_docling_converter = None


def _get_docling_converter():
    global _docling_converter
    if _docling_converter is not None:
        return _docling_converter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    _docling_converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
    })
    return _docling_converter


def _pdf_to_markdown(pdf_path: Path) -> str:
    try:
        converter = _get_docling_converter()
        result = converter.convert(str(pdf_path))
        return result.document.export_to_markdown()
    except Exception as exc:
        log.warning("docling failed for %s (%s); falling back to pymupdf4llm",
                    pdf_path, exc)
        import pymupdf4llm
        return pymupdf4llm.to_markdown(str(pdf_path))


def _is_retryable_fetch_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code == 429 or exc.code >= 500
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response is not None else 0
        return status == 429 or status >= 500
    if isinstance(exc, requests.RequestException):
        return True
    if isinstance(exc, (ConnectionError, TimeoutError, URLError)):
        return True
    text = str(exc).lower()
    return (
        "429" in text
        or "timed out" in text
        or "temporary failure" in text
        or "connection" in text
        or "retrieval incomplete" in text
        or "incomplete" in text
    )


def _download_pdf(
    arxiv_id: str,
    dest_dir: Path,
    *,
    max_attempts: int = 6,
    base_sleep_s: float = 10.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    safe_id = arxiv_id.replace("/", "_")
    out_path = dest_dir / f"{safe_id}.pdf"
    tmp_path = out_path.with_suffix(".pdf.tmp")
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    headers = {"User-Agent": "PaperGym/0.1 (research paper ingestion)"}

    for attempt in range(1, max_attempts + 1):
        try:
            with requests.get(
                url, stream=True, timeout=60, headers=headers
            ) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "pdf" not in content_type:
                    raise ValueError(
                        f"URL did not return a PDF: {content_type or 'unknown'}"
                    )
                expected = int(response.headers.get("content-length") or 0)
                got = 0
                with tmp_path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 256):
                        if not chunk:
                            continue
                        got += len(chunk)
                        f.write(chunk)
                if expected and got < expected:
                    raise IOError(
                        f"retrieval incomplete: got only {got} out of "
                        f"{expected} bytes"
                    )
            tmp_path.replace(out_path)
            return out_path
        except Exception as exc:
            last_exc = exc
            tmp_path.unlink(missing_ok=True)
            if attempt == max_attempts or not _is_retryable_fetch_error(exc):
                raise
            wait_s = min(base_sleep_s * (2 ** (attempt - 1)), 180.0)
            wait_s += random.uniform(0.0, min(5.0, wait_s * 0.25))
            log.warning(
                "fetch failed for %s on attempt %s/%s (%s); retrying in %.1fs",
                arxiv_id, attempt, max_attempts, exc, wait_s,
            )
            sleep_fn(wait_s)

    assert last_exc is not None
    raise last_exc


def fetch_paper_to_disk(*, arxiv_id: str, root: Path,
                        cache_root: Path | None = None) -> Path:
    """Fetch one paper; return path to root/arxiv_id/paper.md.

    When cache_root is given and a pre-converted paper.md is already
    present there (e.g. produced on a GPU server), symlink it instead
    of re-downloading and re-converting.
    """
    out_dir = Path(root) / arxiv_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paper_md = out_dir / "paper.md"
    if paper_md.exists():
        return paper_md

    if cache_root is not None:
        cached = Path(cache_root) / arxiv_id / "paper.md"
        if cached.exists():
            paper_md.symlink_to(cached)
            return paper_md

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = _download_pdf(arxiv_id, Path(tmp))
        md = _pdf_to_markdown(pdf_path)
    paper_md.write_text(md, encoding="utf-8")
    return paper_md
