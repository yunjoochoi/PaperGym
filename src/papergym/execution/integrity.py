"""Static leakage / cheating guard for agent-written experiment code."""
from __future__ import annotations

import re

_FORBIDDEN = {
    "load_dataset": re.compile(r"\bload_dataset\b"),
    "datasets-import": re.compile(r"\bimport\s+datasets\b|from\s+datasets\b"),
    "huggingface": re.compile(r"huggingface|hf_hub|datasets\.load"),
    "openai-direct": re.compile(r"\bimport\s+openai\b|from\s+openai\b"),
    "anthropic-direct": re.compile(r"\bimport\s+anthropic\b|from\s+anthropic\b"),
    "litellm-direct": re.compile(r"\bimport\s+litellm\b|from\s+litellm\b"),
    "papergym-llm-direct": re.compile(r"papergym\.llm"),
    "raw-network": re.compile(r"\brequests\.(get|post)\b|urllib\.request"),
    "dynamic-import": re.compile(r"\b__import__\b|\bimportlib\b"),
    "socket-network": re.compile(r"\bsocket\b|\bhttp\.client\b"),
    "subprocess-shell": re.compile(r"\bsubprocess\b|\bos\.system\b|\bpopen\b"),
    "package-install": re.compile(
        r"\bpip\b.*\binstall\b|\binstall\b.*\bpip\b", re.DOTALL),
}


def scan_for_leakage(code: str) -> list[str]:
    """Return human-readable flags; empty == clean. The only blessed LLM path is
    papergym.execution.gym_client.metered_llm_call (allowed)."""
    flags = []
    for name, rx in _FORBIDDEN.items():
        if rx.search(code or ""):
            flags.append(f"forbidden pattern: {name}")
    return flags
