from __future__ import annotations

import subprocess
from pathlib import Path


def bash(command: str, timeout: int = 120, description: str = "",
         *, paper_dir: Path) -> str:
    # description is unused at runtime; RunLogger captures it as the
    # LLM-supplied intent string for each tool call in the trace.
    del description
    try:
        r = subprocess.run(
            ["bash", "-c", command],
            cwd=str(paper_dir),
            capture_output=True, text=True, timeout=timeout,
        )
        return f"exit={r.returncode}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    except subprocess.TimeoutExpired:
        return "exit=124\nstdout:\n\nstderr:\ntimeout"


bash.schema = {
    "type": "function",
    "function": {
        "name": "Bash",
        "description": "Run a bash command. Use for `git clone <url>`, `tree`, `ls`, `find`, `wc`. Do NOT run experiments.",
        "parameters": {
            "type": "object",
            "properties": {
                "command":     {"type": "string"},
                "description": {
                    "type": "string",
                    "description": "Brief reason for this command (recorded in trace).",
                },
                "timeout":     {"type": "integer", "default": 120},
            },
            "required": ["command"],
        },
    },
}
