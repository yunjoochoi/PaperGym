"""Core datatypes shared across the execution gym."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class IdeaSpec:
    """One research-idea proposal to execute, plus any known human-execution
    outcome (for validation in P2). human_exec_scores maps metric name ->
    mean human score, e.g. {"overall": 3.0, "novelty": 4.1}."""
    idea_id: str
    condition: str            # "AI" | "Human" | "AI_Rerank"
    topic: str
    title: str
    proposal_text: str
    human_exec_scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "IdeaSpec":
        return cls(
            idea_id=d["idea_id"], condition=d["condition"], topic=d["topic"],
            title=d["title"], proposal_text=d["proposal_text"],
            human_exec_scores=d.get("human_exec_scores", {}) or {},
        )


@dataclass
class RunArtifact:
    """Output of one agent execution run inside the sandbox."""
    status: str                                  # mirrors LoopResult.status
    code: str = ""                               # the method.py the agent wrote
    stdout: str = ""
    predictions: list = field(default_factory=list)   # [{"id","pred"}, ...]
    steps: int = 0
    trace: list = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ExecResult:
    """Final per-idea result. effectiveness/method_metric are Optional:
    None means the run failed to produce a scorable artifact (NOT 0.0).
    faithfulness_score uses the parse_axis 1-5 scale with 0 = parse failure."""
    idea_id: str
    task_id: str
    baseline_metric: Optional[float]
    method_metric: Optional[float]
    effectiveness: Optional[float]               # method_metric - baseline_metric
    faithfulness_score: int
    run: RunArtifact
    cost: dict
    leakage_flags: list = field(default_factory=list)
    sandbox: str = "local"
    trustworthy: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        return d
