"""Isolated experiment sandboxes for the execution gym.

PaperGym has NO execution sandbox today (its `bash` tool is a bare host
subprocess). LocalSandbox keeps that transport for dev/tests; DockerSandbox
(Task 6) adds real isolation for untrusted agent-written code.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Protocol


class SandboxPathError(ValueError):
    """Raised when a sandbox file path would escape the workspace."""


def _resolve_in(work_root: Path, rel: str) -> Path:
    """Resolve rel against work_root and confirm it stays inside it.
    Rejects absolute paths and parent-traversal."""
    if rel != rel.strip() or rel.startswith(("/", "\\")) or ":" in rel[:3]:
        raise SandboxPathError(f"absolute or suspicious path: {rel!r}")
    root = Path(work_root).resolve()
    target = (root / rel).resolve()
    if root != target and root not in target.parents:
        raise SandboxPathError(f"path escapes workspace: {rel!r}")
    return target


class Sandbox(Protocol):
    def reset(self) -> None: ...
    def close(self) -> None: ...
    def write_file(self, rel: str, content: str) -> None: ...
    def read_file(self, rel: str) -> str: ...
    def run_python(self, rel: str, timeout: int = 600) -> tuple[int, str, str]: ...


class LocalSandbox:
    """Runs in an isolated host tmp workdir. NOT secure isolation — dev/tests
    only. __init__ does no I/O (mirrors PaperEnv); reset() creates the dir."""

    def __init__(self, *, work_root: Path):
        self.work_root = Path(work_root)
        self._ready = False

    def __enter__(self) -> "LocalSandbox":
        self.reset()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def reset(self) -> None:
        self.work_root.mkdir(parents=True, exist_ok=True)
        self._ready = True

    def close(self) -> None:
        self._ready = False  # local dir is left for inspection; runner cleans up

    def _ensure(self) -> None:
        if not self._ready:
            self.reset()

    def write_file(self, rel: str, content: str) -> None:
        self._ensure()
        p = _resolve_in(self.work_root, rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def read_file(self, rel: str) -> str:
        return _resolve_in(self.work_root, rel).read_text(encoding="utf-8")

    def run_python(self, rel: str, timeout: int = 600) -> tuple[int, str, str]:
        self._ensure()
        _resolve_in(self.work_root, rel)          # validate; cwd is work_root
        try:
            r = subprocess.run(["python", rel], cwd=str(self.work_root),
                               capture_output=True, text=True, timeout=timeout,
                               env=_sandboxed_env())
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "timeout"


_FORWARDED_ENV = ("GYM_LLM_URL", "GYM_JOB_TOKEN")   # NEVER provider keys

_PROVIDER_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_BASE",
                  "EMBEDDING_MODEL", "LITELLM_MODEL")


def _sandboxed_env() -> dict:
    """Host env minus provider secrets (keep GYM_LLM_URL/PATH/etc.)."""
    return {k: v for k, v in os.environ.items() if k not in _PROVIDER_KEYS}


class DockerSandbox:
    """Runs agent-written code inside a Docker container. workdir is bind-
    mounted at /work; provider env is forwarded so in-container code can call
    the gym's LLM. __init__ does no I/O; reset() creates the host workdir."""

    def __init__(self, *, work_root: Path, image: str = "papergym-exec:latest",
                 network: str = "bridge"):
        self.work_root = Path(work_root)
        self.image = image
        self.network = network
        self._ready = False

    def __enter__(self) -> "DockerSandbox":
        self.reset(); return self

    def __exit__(self, *exc) -> None:
        self.close()

    def reset(self) -> None:
        self.work_root.mkdir(parents=True, exist_ok=True)
        self._ready = True

    def close(self) -> None:
        self._ready = False

    def write_file(self, rel: str, content: str) -> None:
        if not self._ready:
            self.reset()
        p = _resolve_in(self.work_root, rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")

    def read_file(self, rel: str) -> str:
        return _resolve_in(self.work_root, rel).read_text(encoding="utf-8")

    def _env_flags(self) -> list[str]:
        flags = []
        for k in _FORWARDED_ENV:
            v = os.environ.get(k)
            if not v:
                continue
            if k == "GYM_LLM_URL":
                v = v.replace("127.0.0.1", "host.docker.internal") \
                     .replace("localhost", "host.docker.internal")
            flags += ["-e", f"{k}={v}"]
        return flags

    def run_python(self, rel: str, timeout: int = 600) -> tuple[int, str, str]:
        if not self._ready:
            self.reset()
        _resolve_in(self.work_root, rel)          # validate; cwd is /work
        argv = (["docker", "run", "--rm",
                 "--add-host=host.docker.internal:host-gateway",
                 f"--network={self.network}",
                 "-v", f"{self.work_root}:/work", "-w", "/work"]
                + self._env_flags()
                + [self.image, "python", rel])
        try:
            r = subprocess.run(argv, capture_output=True, text=True,
                               timeout=timeout)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "timeout"
