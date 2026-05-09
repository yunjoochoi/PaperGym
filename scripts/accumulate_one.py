"""In-container: fetch one paper, run Accumulator, persist seeds + events to mounts."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from papergym.agents.accumulator import Accumulator
from papergym.agents.base import PromptLoader
from papergym.env import PaperEnv
from papergym.library import LibraryStore, Seed, new_seed_id
from papergym.llm import LLMClient
from papergym.log import RunLogger

load_dotenv(override=True)


def _log(log_path: Path, **fields) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(fields, ensure_ascii=False) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arxiv-id", required=True)
    p.add_argument("--work-root", type=Path, default=Path("/work"))
    p.add_argument("--library-root", type=Path, default=Path("/library"))
    p.add_argument("--events-dir", type=Path, default=Path("/events"))
    p.add_argument("--papers-cache", type=Path, default=None,
                    help="If set, look up pre-converted "
                         "<papers-cache>/<arxiv-id>/paper.md to skip docling.")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--shard-id", type=int, default=0,
                    help="Worker shard index; library writes go to "
                         "library_root/shard_<id>/.")
    args = p.parse_args(argv)

    args.events_dir.mkdir(parents=True, exist_ok=True)
    args.library_root.mkdir(parents=True, exist_ok=True)
    shard_root = args.library_root / f"shard_{args.shard_id}"
    shard_root.mkdir(parents=True, exist_ok=True)

    log_path = args.library_root / "accumulator_log.jsonl"
    env = PaperEnv(arxiv_id=args.arxiv_id, work_root=args.work_root,
                   cache_root=args.papers_cache)
    exit_code = 0

    try:
        try:
            env.reset()
        except Exception as exc:
            # paper_dir was created by PaperEnv.__init__; write a one-line
            # events.jsonl so the host gets *some* trace of fetch failures.
            env.paper_dir.mkdir(parents=True, exist_ok=True)
            with RunLogger(env.paper_dir) as logger:
                logger.event("fetch_error", arxiv_id=args.arxiv_id,
                             error=str(exc))
            _log(log_path, arxiv_id=args.arxiv_id, status="error",
                 stage="fetch", shard_id=args.shard_id, error=str(exc))
            exit_code = 1
            return

        llm = LLMClient()
        prompts = PromptLoader("papergym.agents.accumulator")
        accumulator = Accumulator(llm=llm, prompts=prompts,
                                    max_steps=args.max_steps)
        library = LibraryStore(shard_root)

        try:
            result = accumulator.run(env=env)
        except Exception as exc:
            _log(log_path, arxiv_id=args.arxiv_id, status="error",
                 stage="accumulator", shard_id=args.shard_id, error=str(exc))
            exit_code = 1
            return

        domain = result["domain"]
        if domain is None:
            _log(log_path, arxiv_id=args.arxiv_id, status="skipped",
                 shard_id=args.shard_id,
                 reason="invalid_or_missing_domain")
            return

        title = result["title"] or args.arxiv_id
        n_added = 0
        for entry in result["seeds"][:3]:
            problem = (entry.get("problem") or "").strip()
            method = (entry.get("method") or "").strip()
            if not problem or not method:
                continue
            seed = Seed(seed_id=new_seed_id(), problem=problem,
                        method=method, domain=domain,
                        paper_title=title, paper_id=args.arxiv_id)
            emb = np.array(llm.embed(problem), dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            library.add(seed, emb)
            n_added += 1
        _log(log_path, arxiv_id=args.arxiv_id, status="ok",
             shard_id=args.shard_id,
             domain=domain.value, title=title, n_seeds=n_added)
    finally:
        src_events = env.paper_dir / "events.jsonl"
        if src_events.exists():
            shutil.copy(src_events, args.events_dir / f"{args.arxiv_id}.jsonl")
        env.close()
        if exit_code:
            sys.exit(exit_code)


if __name__ == "__main__":
    main()
