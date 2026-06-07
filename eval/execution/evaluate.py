"""Orchestrate one idea: baseline -> agent execution -> score -> faithfulness.

The agent's in-sandbox LLM calls go through a host-side metering proxy that
wraps gen_llm, so their tokens are captured in gen_llm's cumulative counters
(and therefore in cost_summary's gen delta). meter.usage() is attached as a
BREAKDOWN of how much of that was sandbox-side; it is NOT re-added to the
total (that would double-count)."""
from __future__ import annotations

import os
import time
from pathlib import Path

from eval.common import CostSnapshot, cost_summary
from papergym.execution.agent import ExecutionAgent
from papergym.execution.faithfulness import judge_faithfulness
from papergym.execution.llm_proxy import run_proxy
from papergym.execution.metering import BudgetedLLM, BudgetExceeded, UsageMeter
from papergym.execution.scorer import score_effectiveness
from papergym.execution.sandbox import LocalSandbox, DockerSandbox
from papergym.execution.task import Task, GSM8KAccuracyTask  # noqa: F401 (patch target)
from papergym.execution.types import ExecResult, IdeaSpec, RunArtifact
from papergym.llm import LLMClient


def run_one_idea(*, idea: IdeaSpec, task: Task, gen_llm: LLMClient,
                 judge_llm: LLMClient, work_root: Path,
                 use_docker: bool = False, image: str = "papergym-exec:latest",
                 max_steps: int = 40, budget_usd: float = 5.0) -> ExecResult:
    t0 = time.time()
    gen_before = CostSnapshot.of(gen_llm)
    judge_before = CostSnapshot.of(judge_llm)

    baseline_metric = task.run_baseline(gen_llm, split="test")

    meter = UsageMeter(gen_llm, budget_usd=budget_usd)
    budgeted_llm = BudgetedLLM(meter)
    server, url, _ = run_proxy(meter)
    prev_url = os.environ.get("GYM_LLM_URL")
    os.environ["GYM_LLM_URL"] = url
    try:
        sb = (DockerSandbox(work_root=work_root, image=image) if use_docker
              else LocalSandbox(work_root=work_root))
        with sb:
            try:
                run = ExecutionAgent(llm=budgeted_llm, max_steps=max_steps).run(
                    idea=idea, task=task, sandbox=sb)
            except BudgetExceeded as exc:
                run = RunArtifact(status="budget_exceeded", error=str(exc))
    finally:
        server.shutdown()
        if prev_url is None:
            os.environ.pop("GYM_LLM_URL", None)
        else:
            os.environ["GYM_LLM_URL"] = prev_url

    method_metric, effectiveness, leakage_flags = score_effectiveness(
        task, run, baseline_metric, split="test")
    faith = judge_faithfulness(proposal=idea.proposal_text, code=run.code,
                               judge_llm=judge_llm)

    cost = cost_summary(
        judge_before=judge_before, judge_after=CostSnapshot.of(judge_llm),
        gen_before=gen_before, gen_after=CostSnapshot.of(gen_llm),
        wall_clock_s=time.time() - t0)
    cost["agent_llm"] = meter.usage()   # breakdown only; already in gen delta

    kind = "docker" if use_docker else "local"
    return ExecResult(idea_id=idea.idea_id, task_id=task.task_id,
                      baseline_metric=baseline_metric,
                      method_metric=method_metric, effectiveness=effectiveness,
                      faithfulness_score=faith.score, run=run, cost=cost,
                      leakage_flags=leakage_flags,
                      sandbox=kind, trustworthy=use_docker)
