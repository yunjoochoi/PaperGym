from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def grep(pattern: str, path: str, *, paper_dir: Path, timeout: int = 120) -> str:
    # `set -o pipefail` so `grep | head` propagates grep's rc=1 (no matches)
    # instead of being masked by head's rc=0.
    cmd = (f"set -o pipefail; grep -rn -- {shlex.quote(pattern)} "
           f"{shlex.quote(path)} | head -200")
    try:
        r = subprocess.run(
            ["bash", "-c", cmd],
            cwd=str(paper_dir),
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "(timeout)"
    if r.returncode == 0:
        return r.stdout
    return r.stdout or r.stderr or "(no matches)"


grep.schema = {
    "type": "function",
    "function": {
        "name": "Grep",
        "description": "Search a regex pattern in files under a path.",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path":    {"type": "string"},
            },
            "required": ["pattern", "path"],
        },
    },
}
