#!/usr/bin/env python3

"""
Experiment tracking abstraction with a ClearML implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, TYPE_CHECKING, Union
import os
from dotenv import load_dotenv

if TYPE_CHECKING:
    from clearml import Task


class ExperimentTracker(Protocol):
    def log_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        ...

    def log_artifact(self, name: str, path: Path) -> None:
        ...

    def log_params(self, params: Dict[str, Any], name: Optional[str] = None) -> None:
        ...

    def connect_args(self, args: argparse.Namespace, name: Optional[str] = None) -> argparse.Namespace:
        ...

    def log_single_value(self, name: str, value: float) -> None:
        ...

    def log_histogram(
        self,
        title: str,
        series: str,
        values: List[Union[int, float]],
        iteration: int = 0,
        xlabels: Optional[List[str]] = None,
        xaxis: Optional[str] = None,
        yaxis: Optional[str] = None,
    ) -> None:
        ...

    def log_plot(
        self,
        title: str,
        series: str,
        figure: Any,
        iteration: int = 0,
    ) -> None:
        ...

    def close(self) -> None:
        ...


class NoOpExperimentTracker:
    def log_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        return None

    def log_artifact(self, name: str, path: Path) -> None:
        return None

    def log_params(self, params: Dict[str, Any], name: Optional[str] = None) -> None:
        return None

    def connect_args(self, args: argparse.Namespace, name: Optional[str] = None) -> argparse.Namespace:
        return args

    def log_single_value(self, name: str, value: float) -> None:
        return None

    def log_histogram(
        self,
        title: str,
        series: str,
        values: List[Union[int, float]],
        iteration: int = 0,
        xlabels: Optional[List[str]] = None,
        xaxis: Optional[str] = None,
        yaxis: Optional[str] = None,
    ) -> None:
        return None

    def log_plot(
        self,
        title: str,
        series: str,
        figure: Any,
        iteration: int = 0,
    ) -> None:
        return None

    def close(self) -> None:
        return None


class ClearMLExperimentTracker:
    def __init__(
        self,
        project_name: str,
        task_name: str,
        tags: Optional[Iterable[str]] = None,
        model_output_uri: Optional[str] = None,
    ) -> None:
        from clearml import Task

        output_uri: Union[bool, str]
        if model_output_uri:
            output_uri = model_output_uri
        else:
            output_uri = True

        self._task: Task = Task.init(
            project_name=project_name,
            task_name=task_name,
            tags=list(tags) if tags else None,
            reuse_last_task_id=False,
            auto_connect_frameworks={'pytorch': ['*.pt']}, # True for anything not specified (e.g. matplotlib). Only log .pt (TorchScript) files for PyTorch.
            output_uri=output_uri,
        )
        self._logger = self._task.get_logger()

        # get docker image digest from environment variable (set by build_image.sh) and log as metadata
        load_dotenv(".docker_image.env")

    @staticmethod
    def _coerce_like(value: Any, template: Any) -> Any:
        if template is None:
            return value

        # bool must be checked before int (since bool is a subclass of int)
        if isinstance(template, bool):
            if isinstance(value, str):
                v = value.strip().lower()
                if v in ("true", "1", "yes", "y", "t", "on"):
                    return True
                if v in ("false", "0", "no", "n", "f", "off"):
                    return False
            return bool(value)

        if isinstance(template, int):
            if isinstance(value, str):
                try:
                    return int(value)
                except Exception:
                    try:
                        return int(float(value))
                    except Exception:
                        return value
            try:
                return int(value)
            except Exception:
                return value

        if isinstance(template, float):
            if isinstance(value, str):
                try:
                    return float(value)
                except Exception:
                    return value
            try:
                return float(value)
            except Exception:
                return value

        if isinstance(template, list):
            if isinstance(value, str):
                s = value.strip()
                try:
                    parsed = json.loads(s)
                    return parsed if isinstance(parsed, list) else [parsed]
                except Exception:
                    return [x.strip() for x in s.split(",") if x.strip()]
            if isinstance(value, (list, tuple)):
                return list(value)
            return value

        if isinstance(template, dict):
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    return parsed if isinstance(parsed, dict) else value
                except Exception:
                    return value
            return value

        if isinstance(template, str):
            return str(value)

        return value

    def log_scalar(self, title: str, series: str, value: float, iteration: int) -> None:
        self._logger.report_scalar(
            title=title,
            series=series,
            value=value,
            iteration=iteration,
        )

    def log_artifact(self, name: str, path: Path) -> None:
        from clearml import OutputModel
        p = Path(path)
        if not p.exists():
            return
        
        # create the OutputModel and upload the file as its weights/artifact
        
        om = OutputModel(
            task=self._task, 
            name=name, 
            framework='pytorch',
            tags=['candidate'],
        )
        om.update_weights(str(p))

        # also attach useful metadata
        script = self._task.data.script

        # git info
        om.set_metadata("git_repo", getattr(script, "repository", ""))
        om.set_metadata("git_branch", getattr(script, "branch", ""))
        om.set_metadata("git_sha", getattr(script, "version_num", ""))
        
        # docker image info
        om.set_metadata("docker_image_digest", os.getenv("DOCKER_IMAGE_DIGEST", ""))
        om.set_metadata("docker_image_tag", os.getenv("DOCKER_IMAGE_TAG", ""))

    def log_params(self, params: Dict[str, Any], name: Optional[str] = None) -> None:
        self._task.connect(params, name=name)

    def connect_args(self, args: argparse.Namespace, name: Optional[str] = None) -> argparse.Namespace:
        """Connect an argparse.Namespace to ClearML and return the (possibly) updated args.

        This is useful for ClearML remote execution where parameter values might be overridden
        from the server/UI and need to be reflected in the running process.
        """
        original = vars(args).copy()

        # Remove non-serializable argparse metadata if present
        if callable(original.get("func")):
            original.pop("func", None)

        connected = self._task.connect(original, name=name)
        connected_dict: Dict[str, Any]
        if connected is None:
            connected_dict = original
        else:
            try:
                connected_dict = dict(connected)  # type: ignore[arg-type]
            except Exception:
                connected_dict = original

        updated: Dict[str, Any] = vars(args).copy()
        for key, connected_value in connected_dict.items():
            template = updated.get(key, None)
            updated[key] = self._coerce_like(connected_value, template)

        new_args = argparse.Namespace(**updated)
        # Preserve original metadata (even if it was not connected)
        for meta_key in ("command", "func"):
            if hasattr(args, meta_key):
                setattr(new_args, meta_key, getattr(args, meta_key))
        return new_args

    def log_single_value(self, name: str, value: float) -> None:
        self._logger.report_single_value(name=name, value=value)

    def log_histogram(
        self,
        title: str,
        series: str,
        values: List[Union[int, float]],
        iteration: int = 0,
        xlabels: Optional[List[str]] = None,
        xaxis: Optional[str] = None,
        yaxis: Optional[str] = None,
    ) -> None:
        """Log a histogram to ClearML.
        
        Args:
            title: Plot title (shown in ClearML UI)
            series: Series name within the plot
            values: Raw values to histogram
            iteration: Iteration/step number
            xlabels: Optional bin labels
            xaxis: X-axis label
            yaxis: Y-axis label
        """
        import numpy as np
        
        # Convert to numpy array for histogram computation
        values_arr = np.array(values)
        
        # Use ClearML's report_histogram which auto-bins
        self._logger.report_histogram(
            title=title,
            series=series,
            values=values_arr,
            iteration=iteration,
            xlabels=xlabels,
            xaxis=xaxis,
            yaxis=yaxis,
        )

    def log_plot(
        self,
        title: str,
        series: str,
        figure: Any,
        iteration: int = 0,
    ) -> None:
        """Log a matplotlib figure to ClearML.
        
        Args:
            title: Plot title
            series: Series name
            figure: Matplotlib figure object
            iteration: Iteration/step number
        """
        self._logger.report_matplotlib_figure(
            title=title,
            series=series,
            figure=figure,
            iteration=iteration,
            report_interactive=False,  # Static image is more reliable
        )

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
    model_output_uri: Optional[str] = None,
) -> ExperimentTracker:
    if kind == "clearml":
        return ClearMLExperimentTracker(
            project_name=project_name,
            task_name=task_name,
            tags=tags,
            model_output_uri=model_output_uri,
        )
    return NoOpExperimentTracker()
