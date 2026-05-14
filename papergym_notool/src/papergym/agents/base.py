import json
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Optional, Union

import yaml
from jinja2 import Template

from ..llm import LLMClient


class PromptLoader:
    """Load prompt YAMLs from either a filesystem path or an installed package.

    Pass a `Path` (filesystem) or a dotted package name like
    `"papergym.agents.accumulator"`. The package form removes the need to
    hardcode container-vs-host paths.
    """

    def __init__(self, prompts_dir: Union[Path, str]):
        self._fs_dir: Optional[Path] = None
        self._pkg: Optional[str] = None
        if isinstance(prompts_dir, Path):
            self._fs_dir = prompts_dir
        else:
            self._pkg = prompts_dir

    def _read(self, name: str) -> str:
        if self._fs_dir is not None:
            return (self._fs_dir / f"{name}.yaml").read_text()
        return resources.files(self._pkg).joinpath(f"{name}.yaml").read_text()

    def render(self, name: str, **fields) -> list[dict]:
        spec = yaml.safe_load(self._read(name))
        messages = []
        for role in ("system", "user"):
            if role in spec:
                rendered = Template(spec[role]).render(**fields)
                messages.append({"role": role, "content": rendered})
        return messages


MessageHook = Optional[Callable[[dict], None]]


class BaseAgent:
    """Single-call JSON-emitting agent: render → chat → json.loads."""

    def __init__(self, llm: LLMClient, prompts: PromptLoader, prompt_name: str,
                 temperature: float = 0.3):
        self.llm = llm
        self.prompts = prompts
        self.prompt_name = prompt_name
        self.temperature = temperature
        self.on_message_hook: MessageHook = None

    def _emit(self, msg: dict) -> None:
        if self.on_message_hook is None:
            return
        try:
            self.on_message_hook(msg)
        except Exception:
            pass

    def call(self, **prompt_fields) -> dict[str, Any]:
        messages = self.prompts.render(self.prompt_name, **prompt_fields)
        for m in messages:
            self._emit({"role": m["role"], "content": m["content"]})
        raw = self.llm.chat(messages, temperature=self.temperature,
                            response_format={"type": "json_object"})
        self._emit({"role": "assistant", "content": raw})
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"invalid JSON from LLM: {e}\nraw: {raw[:500]}")
