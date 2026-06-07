"""Execution agent: drives run_tool_loop with a write/run/read/finish action
space so an LLM can implement and run an idea's experiment in a sandbox."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from papergym.agents.tool_loop import run_tool_loop
from papergym.llm import LLMClient

from .sandbox import Sandbox
from .task import Task
from .types import IdeaSpec, RunArtifact


# ---- tool functions: each carries a .schema (capitalized name) ----
def _write_file(path: str, content: str, *, sandbox: Sandbox) -> str:
    sandbox.write_file(path, content)
    return f"wrote {path} ({len(content)} chars)"


def _run_python(path: str, *, sandbox: Sandbox, timeout: int = 600) -> str:
    rc, out, err = sandbox.run_python(path, timeout=timeout)
    return f"exit={rc}\nstdout:\n{out[:4000]}\nstderr:\n{err[:2000]}"


def _read_file(path: str, *, sandbox: Sandbox) -> str:
    try:
        return sandbox.read_file(path)[:4000]
    except FileNotFoundError:
        return f"[no such file] {path}"


def _finish(summary: str = "") -> str:
    return f"FINISHED: {summary}"


_write_file.schema = {"type": "function", "function": {
    "name": "WriteFile", "description": "Write a text file in the sandbox.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"]}}}
_run_python.schema = {"type": "function", "function": {
    "name": "RunPython",
    "description": "Run a python file in the sandbox. RUN EXPERIMENTS here.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "timeout": {"type": "integer", "default": 600}},
        "required": ["path"]}}}
_read_file.schema = {"type": "function", "function": {
    "name": "ReadFile", "description": "Read a file from the sandbox.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}}
_finish.schema = {"type": "function", "function": {
    "name": "Finish", "description": "Call when predictions.json is written.",
    "parameters": {"type": "object", "properties": {
        "summary": {"type": "string"}}}}}

EXECUTION_TOOL_FNS = (_write_file, _run_python, _read_file, _finish)
EXECUTION_TOOLS = list(EXECUTION_TOOL_FNS)


_SYSTEM = """You are a research-execution agent. Implement the METHOD from the
idea proposal as `method.py`, run it, and write `predictions.json` in the
format {fmt}. Call the LLM ONLY via:
  from papergym.execution.gym_client import metered_llm_call
  text = metered_llm_call([{{"role":"user","content":"..."}}])
Do NOT import datasets/openai/anthropic/litellm/papergym.llm, do NOT download
any dataset, do NOT read test answers — they do not exist in your sandbox.
Develop on dev.json (has answers); predict test_inputs.json (no answers).
Call Finish when predictions.json exists. NO model training."""


@dataclass
class _Tools:
    sandbox: Sandbox
    written: list = field(default_factory=list)

    def dispatch(self, name: str, args: dict) -> str:
        if name == "WriteFile":
            path = args["path"]
            if str(path).endswith(".py") and path not in self.written:
                self.written.append(path)
            return _write_file(path, args["content"], sandbox=self.sandbox)
        if name == "RunPython":
            return _run_python(args["path"], sandbox=self.sandbox,
                               timeout=int(args.get("timeout", 600)))
        if name == "ReadFile":
            return _read_file(args["path"], sandbox=self.sandbox)
        if name == "Finish":
            return _finish(args.get("summary", ""))
        return f"[unknown tool] {name}"


class ExecutionAgent:
    def __init__(self, *, llm: LLMClient, max_steps: int = 40,
                 temperature: float = 0.2):
        self.llm = llm
        self.max_steps = max_steps
        self.temperature = temperature

    def run(self, *, idea: IdeaSpec, task: Task, sandbox: Sandbox) -> RunArtifact:
        tools = _Tools(sandbox)
        system = _SYSTEM.format(fmt=task.manifest()["predictions_format"])
        sandbox.write_file("test_inputs.json",
                           json.dumps(task.inputs(split="test")))
        sandbox.write_file("dev.json",
                           json.dumps(task.examples(split="dev")))
        user = (f"IDEA PROPOSAL:\n{idea.proposal_text}\n\n"
                f"TASK: {json.dumps(task.manifest())}\n"
                f"`dev.json` (id, question, ANSWER) is for developing/tuning your "
                f"method. `test_inputs.json` (id, question — NO answers) is what "
                f"you must predict. Write predictions.json = [{{'id','pred'}}] for "
                f"every test_inputs row.")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        result = run_tool_loop(
            llm=self.llm, messages=messages,
            tools=[fn.schema for fn in EXECUTION_TOOL_FNS],
            dispatch=tools.dispatch, max_steps=self.max_steps,
            temperature=self.temperature)

        seen, parts = set(), []
        for path in list(tools.written) + ["method.py"]:
            if path in seen:
                continue
            seen.add(path)
            try:
                parts.append(f"# {path}\n" + sandbox.read_file(path))
            except FileNotFoundError:
                continue
        code = "\n\n".join(parts)
        predictions = []
        try:
            predictions = json.loads(sandbox.read_file("predictions.json"))
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return RunArtifact(status=result.status, code=code,
                           predictions=predictions, steps=result.steps,
                           trace=result.trace, error=result.reason)
