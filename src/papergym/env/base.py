from __future__ import annotations

from pathlib import Path

from .preparer import fetch_paper_to_disk


class PaperEnv:
    """A paper as an environment.

    Holds the per-paper state (arxiv_id, paper_dir) and the paper.md
    preparation lifecycle. Tool implementations live under the tools
    package; the agent owns dispatch and binds paper_dir into tools
    that need a working directory.
    """

    def __init__(self, *, arxiv_id: str, work_root: Path,
                 cache_root: Path | None = None):
        self.arxiv_id = arxiv_id
        self.work_root = Path(work_root)
        self.paper_dir = self.work_root / arxiv_id
        self.cache_root = Path(cache_root) if cache_root is not None else None

    def reset(self) -> None:
        self.paper_dir.mkdir(parents=True, exist_ok=True)
        fetch_paper_to_disk(
            arxiv_id=self.arxiv_id,
            root=self.work_root,
            cache_root=self.cache_root,
        )

    def close(self) -> None:
        pass
