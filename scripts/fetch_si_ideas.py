"""Fetch Si et al. (NoviScl/AI-Researcher) idea proposals + join human
execution scores into data/si_ideas/<idea_id>.json (IdeaSpec dicts).

Idea-proposal archives (Google Drive, from AI-Researcher/README.md):
  Human:     1Z2Nd7WNNks-eCoqUgPzx1_ovYqU8OiPx
  AI:        1AFjSUCTj4wL081R2b17nVuNSs7xvzh8F
  AI_Rerank: 15r3TzFd-6dPXdSMx0shZ4q3ljDOEzp8I
  Execution: 1PpxeTz_-xHHcMXyUwv1Oed1avTaiD5vv
The exact in-archive layout is resolved at runtime by _load_proposals();
adjust if the archive nests differently.
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

DRIVE_IDS = {
    "Human": "1Z2Nd7WNNks-eCoqUgPzx1_ovYqU8OiPx",
    "AI": "1AFjSUCTj4wL081R2b17nVuNSs7xvzh8F",
    "AI_Rerank": "15r3TzFd-6dPXdSMx0shZ4q3ljDOEzp8I",
}
_EXEC_METRICS = ("overall_score", "novelty_score", "excitement_score",
                 "effectiveness_score", "soundness_score")


def _download(file_id: str, dest: Path) -> Path:
    """Download a Google Drive file via gdown. Isolated for test stubbing."""
    import gdown
    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=file_id, output=str(dest), quiet=False)
    return dest


def _mean_exec_scores(exec_data: dict) -> dict:
    """Column-oriented review dump -> {idea_id: {metric: mean}}.
    Skips non-numeric / missing cells. Metric names drop the _score suffix."""
    ids = exec_data["idea_id"]
    by_idea = defaultdict(lambda: defaultdict(list))
    for metric in _EXEC_METRICS:
        col = exec_data.get(metric)
        if not col:
            continue
        for idea_id, val in zip(ids, col):
            try:
                by_idea[str(idea_id).strip()][metric].append(float(val))
            except (TypeError, ValueError):
                continue
    out = {}
    for idea_id, metrics in by_idea.items():
        out[idea_id] = {m.replace("_score", ""): statistics.mean(v)
                        for m, v in metrics.items() if v}
    return out


def _write_ideaspecs(proposals: dict, means: dict, meta: dict,
                     out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for idea_id, text in proposals.items():
        m = meta.get(idea_id, {})
        rec = {
            "idea_id": idea_id,
            "condition": m.get("condition", _condition_of(idea_id)),
            "topic": m.get("topic", _topic_of(idea_id)),
            "title": m.get("title", idea_id),
            "proposal_text": text,
            "human_exec_scores": means.get(idea_id, {}),
        }
        (out_dir / f"{idea_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2))
        n += 1
    return n


def _condition_of(idea_id: str) -> str:
    return "AI" if idea_id.endswith("_AI") else (
        "AI_Rerank" if idea_id.endswith("_AI_Rerank") else "Human")


def _topic_of(idea_id: str) -> str:
    return idea_id.split("_")[0]


def _load_proposals(extract_dir: Path) -> dict:
    """Map idea_id -> proposal text from an extracted archive. Each idea is a
    file whose stem is the idea_id (txt/md/json). JSON files are read for a
    'proposal'/'text'/'idea' field, else dumped whole."""
    proposals = {}
    for p in sorted(extract_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".txt", ".md", ".json"):
            continue
        idea_id = p.stem.strip()
        if p.suffix.lower() == ".json":
            try:
                obj = json.loads(p.read_text())
                text = (obj.get("proposal") or obj.get("text")
                        or obj.get("idea") or json.dumps(obj))
            except json.JSONDecodeError:
                text = p.read_text()
        else:
            text = p.read_text()
        proposals[idea_id] = text
    return proposals


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ai-researcher", type=Path,
                   default=Path("../AI-Researcher"),
                   help="Path to the NoviScl/AI-Researcher checkout.")
    p.add_argument("--out", type=Path, default=Path("data/si_ideas"))
    p.add_argument("--download-dir", type=Path, default=Path("data/si_raw"))
    args = p.parse_args(argv)

    exec_json = (args.ai_researcher / "reviews_execution"
                 / "data_points_all_execution.json")
    means = _mean_exec_scores(json.loads(exec_json.read_text()))

    proposals = {}
    for cond, fid in DRIVE_IDS.items():
        archive = args.download_dir / f"{cond}.zip"
        if not archive.exists():
            _download(fid, archive)
        extract_dir = args.download_dir / cond
        if not extract_dir.exists():
            import shutil
            shutil.unpack_archive(str(archive), str(extract_dir))
        proposals.update(_load_proposals(extract_dir))

    n = _write_ideaspecs(proposals, means, {}, args.out)
    print(f"wrote {n} ideas to {args.out}; "
          f"{len(means)} have human exec scores")


if __name__ == "__main__":
    main()
