from __future__ import annotations

from pathlib import Path


def read(file_path: str, offset: int = 0, limit: int = 4000) -> str:
    text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    chunk = lines[offset:offset + limit]
    return "\n".join(f"{i + offset + 1:6d}\t{ln}" for i, ln in enumerate(chunk))


read.schema = {
    "type": "function",
    "function": {
        "name": "Read",
        "description": "Read a file (paper.md, cloned repo files). Returns line-numbered content.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "offset":    {"type": "integer", "default": 0},
                "limit":     {"type": "integer", "default": 4000},
            },
            "required": ["file_path"],
        },
    },
}
