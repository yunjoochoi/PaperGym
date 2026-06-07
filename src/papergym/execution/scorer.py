"""Objective effectiveness scoring: method metric vs baseline (held-out test)."""
from __future__ import annotations

from typing import Optional

from .integrity import scan_for_leakage
from .task import Task
from .types import RunArtifact


def score_effectiveness(task: Task, run: RunArtifact, baseline_metric: float,
                        split: str = "test"
                        ) -> tuple[Optional[float], Optional[float], list]:
    """Returns (method_metric, effectiveness, leakage_flags). If the agent's
    code trips the leakage guard, the run is marked suspect: metrics are None
    (a leaked 100% is NOT a real score)."""
    flags = scan_for_leakage(run.code)
    flags.extend(task.validate_predictions(run.predictions, split=split))
    if flags or not run.predictions:
        return None, None, flags
    method_metric = task.score(run.predictions, split=split)
    return method_metric, method_metric - baseline_metric, flags
