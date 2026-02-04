#!/usr/bin/env python3

"""
Lightweight pipeline core utilities.

Provides:
- Context: shared run state across stages
- Timestamped stage output directories under <run_dir>/<stage>/<timestamp>/
- Discovery helpers to find existing outputs for a stage
- Module loading helper for stage scripts (loaded by file path)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from typing import Any, Dict, Optional, Callable, List


from ..experiment_tracking import ExperimentTracker, NoOpExperimentTracker

# Resolve repo root as two levels up from this file: utils/pipeline/core.py → repo/
CURR = Path(__file__).resolve()
UTILS_DIR = CURR.parent.parent
ROOT = UTILS_DIR.parent


@dataclass
class Context:
    run_dir: Path
    use_latest: bool = True
    prior_outputs: Dict[str, Optional[Path]] = field(default_factory=dict)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tracker: ExperimentTracker = field(default_factory=NoOpExperimentTracker)

    def record_artifact(self, stage: str, output_dir: Path, extras: Optional[Dict[str, Any]] = None) -> None:
        self.artifacts[stage] = {
            'output_dir': output_dir,
            **(extras or {}),
        }

    def get_artifact_dir(self, stage: str) -> Optional[Path]:
        info = self.artifacts.get(stage)
        return Path(info['output_dir']) if info and info.get('output_dir') else None


def stage_base_dir(run_dir: Path, stage_name: str) -> Path:
    base = Path(run_dir) / stage_name
    base.mkdir(parents=True, exist_ok=True)
    return base


def new_stage_timestamp_dir(run_dir: Path, stage_name: str) -> Path:
    base = stage_base_dir(run_dir, stage_name)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def list_stage_outputs(run_dir: Path, stage_name: str) -> List[Path]:
    base = stage_base_dir(run_dir, stage_name)
    if not base.exists():
        return []
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    # Sort by mtime desc
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs


def select_prior_output(run_dir: Path, stage_name: str, *, use_latest: bool = True, prior_path: Optional[Path] = None) -> Optional[Path]:
    if prior_path is not None:
        p = Path(prior_path)
        return p if p.exists() else None
    if not use_latest:
        return None
    options = list_stage_outputs(run_dir, stage_name)
    if options:
        return options[0]
    # Fallback between enumerated and legacy folder names
    alt = stage_name
    if stage_name.startswith(("01_", "02_", "03_", "04_", "05_", "06_")):
        try:
            alt = stage_name.split("_", 1)[1]
        except Exception:
            alt = stage_name
    else:
        prefix_map = {
            "get_data": "01_get_data",
            "relevel": "03_relevel",
            "split": "04_split",
            "train": "05_train",
            "evaluate": "06_evaluate",
        }
        alt = prefix_map.get(stage_name, stage_name)
    if alt != stage_name:
        alt_opts = list_stage_outputs(run_dir, alt)
        if alt_opts:
            return alt_opts[0]
    return None


def load_run_callable(module_path: Path) -> Callable[[Context, Any], Dict[str, Any]]:
    """Load a stage module from an absolute file path and return its run() function."""
    module_path = Path(module_path).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"Stage module not found: {module_path}")
    spec = spec_from_file_location(module_path.stem, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module at {module_path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    if not hasattr(mod, 'run'):
        raise AttributeError(f"Stage module {module_path} has no run(context, args) function")
    return getattr(mod, 'run')


