"""Objective effectiveness scoring: method metric vs baseline."""
from __future__ import annotations

from typing import Optional

from .task import Task
from .types import RunArtifact


def score_effectiveness(task: Task, run: RunArtifact,
                        baseline_metric: float
                        ) -> tuple[Optional[float], Optional[float]]:
    """Returns (method_metric, effectiveness). Both None when the run
    produced no predictions — a failed run is missing data, not a 0 score."""
    if not run.predictions:
        return None, None
    method_metric = task.score(run.predictions)
    return method_metric, method_metric - baseline_metric
