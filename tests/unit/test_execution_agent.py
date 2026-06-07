import json
from pathlib import Path
from unittest import mock
from papergym.execution.agent import ExecutionAgent, EXECUTION_TOOLS
from papergym.execution.sandbox import LocalSandbox
from papergym.execution.task import GSM8KAccuracyTask
from papergym.execution.types import IdeaSpec
from papergym.llm import ChatReply, ToolCall


def _idea():
    return IdeaSpec(idea_id="Math_3_Human", condition="Human", topic="Math",
                    title="t", proposal_text="Prefix each question with 'Hint:'")


def test_tools_expose_capitalized_schema_names():
    names = {t.schema["function"]["name"] for t in EXECUTION_TOOLS}
    assert names == {"WriteFile", "RunPython", "ReadFile", "Finish"}


def test_agent_writes_runs_and_collects_predictions(tmp_path):
    task = GSM8KAccuracyTask(n_examples=1)
    task._examples = [{"id": "0", "question": "2+2?", "answer": "4"}]
    method = ("import json;"
              "json.dump([{'id':'0','pred':'4'}], open('predictions.json','w'))")
    calls = [
        ChatReply(content="", tool_calls=[ToolCall(
            id="a", name="WriteFile",
            arguments=json.dumps({"path": "method.py", "content": method}))],
            raw_message={"role": "assistant"}),
        ChatReply(content="", tool_calls=[ToolCall(
            id="b", name="RunPython",
            arguments=json.dumps({"path": "method.py"}))],
            raw_message={"role": "assistant"}),
        ChatReply(content="", tool_calls=[ToolCall(
            id="c", name="Finish", arguments=json.dumps({"summary": "done"}))],
            raw_message={"role": "assistant"}),
        ChatReply(content="ok", tool_calls=[], raw_message={"role": "assistant"}),
    ]
    llm = mock.MagicMock()
    llm.chat_with_tools.side_effect = calls
    with LocalSandbox(work_root=tmp_path / "run") as sb:
        agent = ExecutionAgent(llm=llm, max_steps=10)
        run = agent.run(idea=_idea(), task=task, sandbox=sb)
    assert run.predictions == [{"id": "0", "pred": "4"}]
    assert "method.py" in run.code
