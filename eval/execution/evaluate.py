"""
uv run python scripts/fetch_si_ideas.py --ai-researcher ../AI-Researcher
uv run python scripts/check_executability.py       
docker build -f docker/Dockerfile.exec -t papergym-exec:latest .
uv run python scripts/run_execution_gym.py --limit 1 --n-examples 20   # 아이디어 1개 end-to-end

Orchestrate one idea: baseline -> agent execution -> score -> faithfulness."""



from __future__ import annotations

import time
from pathlib import Path

from eval.common import CostSnapshot, cost_summary
from papergym.execution.agent import ExecutionAgent
from papergym.execution.faithfulness import judge_faithfulness
from papergym.execution.scorer import score_effectiveness
from papergym.execution.sandbox import LocalSandbox, DockerSandbox
from papergym.execution.task import Task, GSM8KAccuracyTask  # noqa: F401 (patch target)
from papergym.execution.types import ExecResult, IdeaSpec
from papergym.llm import LLMClient


def run_one_idea(*, idea: IdeaSpec, task: Task, gen_llm: LLMClient,
                 judge_llm: LLMClient, work_root: Path,
                 use_docker: bool = False, image: str = "papergym-exec:latest",
                 max_steps: int = 40) -> ExecResult:
    t0 = time.time()
    gen_before = CostSnapshot.of(gen_llm)
    judge_before = CostSnapshot.of(judge_llm)

    baseline_metric = task.run_baseline(gen_llm)

    sb = (DockerSandbox(work_root=work_root, image=image) if use_docker
          else LocalSandbox(work_root=work_root))
    with sb:
        run = ExecutionAgent(llm=gen_llm, max_steps=max_steps).run(
            idea=idea, task=task, sandbox=sb)

    method_metric, effectiveness = score_effectiveness(task, run, baseline_metric)
    faith = judge_faithfulness(proposal=idea.proposal_text, code=run.code,
                               judge_llm=judge_llm)

    cost = cost_summary(
        judge_before=judge_before, judge_after=CostSnapshot.of(judge_llm),
        gen_before=gen_before, gen_after=CostSnapshot.of(gen_llm),
        wall_clock_s=time.time() - t0)

    return ExecResult(idea_id=idea.idea_id, task_id=task.task_id,
                      baseline_metric=baseline_metric,
                      method_metric=method_metric, effectiveness=effectiveness,
                      faithfulness_score=faith.score, run=run, cost=cost)
