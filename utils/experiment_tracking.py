#!/usr/bin/env python3

"""
Experiment tracking abstraction with a ClearML implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Protocol


class ExperimentTracker(Protocol):
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        ...

    def log_artifact(self, name: str, path: Path) -> None:
        ...

    def log_params(self, params: Dict[str, Any]) -> None:
        ...

    def close(self) -> None:
        ...


class NoOpExperimentTracker:
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        return None

    def log_artifact(self, name: str, path: Path) -> None:
        return None

    def log_params(self, params: Dict[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None


class ClearMLExperimentTracker:
    def __init__(
        self,
        project_name: str,
        task_name: str,
        tags: Optional[Iterable[str]] = None,
    ) -> None:
        from clearml import Task

        self._task = Task.init(
            project_name=project_name,
            task_name=task_name,
            tags=list(tags) if tags else None,
            reuse_last_task_id=False,
            auto_connect_frameworks=False,
        )
        self._logger = self._task.get_logger()
        self._iteration = 0

    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        iteration = step if step is not None else self._iteration
        self._logger.report_scalar(
            title=name,
            series="value",
            value=value,
            iteration=iteration,
        )
        if step is None:
            self._iteration += 1

    def log_artifact(self, name: str, path: Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        self._task.upload_artifact(name=name, artifact_object=str(p))

    def log_params(self, params: Dict[str, Any]) -> None:
        self._task.connect(params)

    def close(self) -> None:
        self._task.close()


def normalize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    def _normalize(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {k: _normalize(v) for k, v in value.items() if v is not None}
        if isinstance(value, (list, tuple)):
            return [_normalize(v) for v in value]
        return value

    return {k: _normalize(v) for k, v in params.items() if v is not None}


def build_experiment_tracker(
    kind: str,
    *,
    project_name: str,
    task_name: str,
    tags: Optional[Iterable[str]] = None,
) -> ExperimentTracker:
    if kind == "clearml":
        return ClearMLExperimentTracker(
            project_name=project_name,
            task_name=task_name,
            tags=tags,
        )
    return NoOpExperimentTracker()
