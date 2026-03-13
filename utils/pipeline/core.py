#!/usr/bin/env python3

"""
Lightweight pipeline core utilities.

Provides:
- Context: shared run state across stages
- Canonical artifact storage under <artifacts_dir>/<stage>/<stage_run_id>/
- Symlinked pipeline run views under <runs_dir>/<pipeline_run_id>/<stage> -> artifacts
- Discovery helpers to find existing artifacts for a stage
- Module loading helper for stage scripts (loaded by file path)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from typing import Any, Dict, Optional, Callable, List, Iterable
import json
import subprocess
import uuid


from ..experiment_tracking import ExperimentTracker, NoOpExperimentTracker

# Resolve repo root as two levels up from this file: utils/pipeline/core.py → repo/
CURR = Path(__file__).resolve()
UTILS_DIR = CURR.parent.parent
ROOT = UTILS_DIR.parent


RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_RUNS_DIR = ROOT / "runs"


def generate_run_timestamp() -> str:
    # Keep consistent with historical CLI naming (local time, second resolution).
    return datetime.now().strftime(RUN_TIMESTAMP_FORMAT)

def _short_uuid(n: int = 8) -> str:
    return uuid.uuid4().hex[:n]


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _normalize_for_json(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(v) for v in value]
    return value


def _write_yaml_compatible_json(path: Path, data: Dict[str, Any]) -> None:
    # JSON is valid YAML 1.2, so this produces a .yml that is parsable by YAML tooling.
    path.write_text(json.dumps(_normalize_for_json(data), indent=2, sort_keys=True) + "\n")


def _git_sha(repo_root: Path) -> Optional[str]:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        sha = (proc.stdout or "").strip()
        return sha if proc.returncode == 0 and sha else None
    except Exception:
        return None


def _ensure_symlink(link_path: Path, target_path: Path) -> None:
    link_path = Path(link_path)
    target_path = Path(target_path)
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink() or link_path.is_file():
            link_path.unlink()
        else:
            raise RuntimeError(f"Refusing to overwrite non-symlink directory: {link_path}")
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target_path)


def new_pipeline_run_dir(
    runs_dir: Path,
    *,
    base_name: str,
) -> Path:
    runs_dir = Path(runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    candidate = runs_dir / base_name
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    for suffix in range(2, 100):
        c = runs_dir / f"{base_name}_{suffix}"
        if not c.exists():
            c.mkdir(parents=True, exist_ok=True)
            return c
    raise RuntimeError(f"Unable to create unique pipeline run directory under '{runs_dir}' for '{base_name}'")


def ensure_pipeline_run_dir(runs_dir: Path, *, pipeline_run_id: str) -> Path:
    runs_dir = Path(runs_dir).resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    p = runs_dir / pipeline_run_id
    if p.exists() and not p.is_dir():
        raise RuntimeError(f"Pipeline run path exists but is not a directory: {p}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def update_latest_symlink(runs_dir: Path, pipeline_run_dir: Path) -> None:
    runs_dir = Path(runs_dir).resolve()
    pipeline_run_dir = Path(pipeline_run_dir).resolve()
    _ensure_symlink(runs_dir / "latest", pipeline_run_dir)


@dataclass
class Context:
    # Pipeline run "view" directory (contains symlinks to canonical artifacts).
    run_dir: Path
    # Canonical artifact store root.
    artifacts_dir: Path = field(default_factory=lambda: DEFAULT_ARTIFACTS_DIR)
    # Runs root (contains pipeline run views and `runs/latest`).
    runs_dir: Path = field(default_factory=lambda: DEFAULT_RUNS_DIR)
    pipeline_run_id: Optional[str] = None
    run_timestamp: str = field(default_factory=generate_run_timestamp)
    use_latest: bool = True
    prior_outputs: Dict[str, Optional[Path]] = field(default_factory=dict)
    artifacts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tracker: ExperimentTracker = field(default_factory=NoOpExperimentTracker)
    _active_stage_key: Optional[str] = None
    _active_stage_folder: Optional[str] = None
    _active_stage_inputs: Dict[str, str] = field(default_factory=dict)

    def new_stage_dir(self, stage_name: str, tag: str = "") -> Path:
        # Stage outputs are written to the canonical artifact store, not under run_dir.
        return new_stage_artifact_dir(
            artifacts_dir=Path(self.artifacts_dir).resolve(),
            stage_folder=stage_name,
            tag=tag,
        )

    def begin_stage(self, stage_key: str, stage_folder: str) -> None:
        self._active_stage_key = stage_key
        self._active_stage_folder = stage_folder
        self._active_stage_inputs = {}

    def resolve_prior_output(self, stage_folder: str, *, prior_path: Optional[Path] = None) -> Path:
        chosen = select_prior_output(
            artifacts_dir=Path(self.artifacts_dir).resolve(),
            stage_folder=stage_folder,
            use_latest=self.use_latest,
            prior_path=prior_path,
        )
        if chosen is None:
            raise FileNotFoundError(f"Could not find prior outputs for stage folder '{stage_folder}' under {self.artifacts_dir}")
        self._active_stage_inputs[stage_folder] = str(Path(chosen).resolve())
        return chosen

    def finalize_stage(
        self,
        *,
        stage_key: str,
        stage_folder: str,
        output_dir: Path,
        args: Any,
        argv: Optional[Iterable[str]] = None,
    ) -> None:
        output_dir = Path(output_dir).resolve()
        stage_run_id = output_dir.name

        pipeline_run_dir = Path(self.run_dir).resolve()
        pipeline_run_dir.mkdir(parents=True, exist_ok=True)
        _ensure_symlink(pipeline_run_dir / stage_folder, output_dir)

        resolved_config_path = output_dir / "resolved_config.yml"
        args_dict = {k: v for k, v in vars(args).items() if k != "func" and not callable(v)}
        _write_yaml_compatible_json(resolved_config_path, args_dict)

        manifest_path = output_dir / "manifest.yaml"
        manifest: Dict[str, Any] = {
            "stage_key": stage_key,
            "stage_folder": stage_folder,
            "stage_run_id": stage_run_id,
            "pipeline_run_id": self.pipeline_run_id or pipeline_run_dir.name,
            "created_at": datetime.now().isoformat(),
            "git_sha": _git_sha(ROOT),
            "argv": list(argv) if argv is not None else None,
            "inputs": self._active_stage_inputs.copy(),
        }
        _write_yaml_compatible_json(manifest_path, manifest)

        self._update_lineage(
            stage_key=stage_key,
            stage_folder=stage_folder,
            stage_run_id=stage_run_id,
            output_dir=output_dir,
            resolved_config_path=resolved_config_path,
            manifest_path=manifest_path,
        )

    def record_artifact(self, stage: str, output_dir: Path, extras: Optional[Dict[str, Any]] = None) -> None:
        self.artifacts[stage] = {
            'output_dir': output_dir,
            **(extras or {}),
        }

    def get_artifact_dir(self, stage: str) -> Optional[Path]:
        info = self.artifacts.get(stage)
        return Path(info['output_dir']) if info and info.get('output_dir') else None

    def _update_lineage(
        self,
        *,
        stage_key: str,
        stage_folder: str,
        stage_run_id: str,
        output_dir: Path,
        resolved_config_path: Path,
        manifest_path: Path,
    ) -> None:
        lineage_path = Path(self.run_dir).resolve() / "lineage.yaml"
        if lineage_path.exists():
            try:
                lineage: Dict[str, Any] = json.loads(lineage_path.read_text())
            except Exception:
                lineage = {}
        else:
            lineage = {}

        lineage.setdefault("pipeline_run_id", self.pipeline_run_id or Path(self.run_dir).name)
        lineage.setdefault("created_at", datetime.now().isoformat())
        lineage.setdefault("git_sha", _git_sha(ROOT))
        lineage.setdefault("stages", {})
        lineage["stages"][stage_key] = {
            "stage_folder": stage_folder,
            "stage_run_id": stage_run_id,
            "artifact_dir": str(Path(output_dir).resolve()),
            "resolved_config": str(Path(resolved_config_path).resolve()),
            "manifest": str(Path(manifest_path).resolve()),
        }
        _write_yaml_compatible_json(lineage_path, lineage)


def stage_base_dir(artifacts_dir: Path, stage_folder: str) -> Path:
    base = Path(artifacts_dir) / stage_folder
    base.mkdir(parents=True, exist_ok=True)
    return base


def new_stage_artifact_dir(
    artifacts_dir: Path,
    stage_folder: str,
    tag: str = "",
    *,
    timestamp: Optional[str] = None,
) -> Path:
    base = stage_base_dir(artifacts_dir, stage_folder)
    ts = str(timestamp).strip() if timestamp else generate_run_timestamp()
    tag_str = str(tag).strip()
    dirname = f"{ts}_{tag_str}_{_short_uuid()}" if tag_str else f"{ts}_{_short_uuid()}"
    out = base / dirname
    if out.exists():
        for suffix in range(2, 100):
            candidate = base / f"{dirname}_{suffix}"
            if not candidate.exists():
                out = candidate
                break
        else:
            raise RuntimeError(
                f"Unable to create unique stage artifact directory under '{base}' "
                f"for base name '{dirname}' after exhausting suffixes 2-99."
            )
    out.mkdir(parents=True, exist_ok=True)
    return out


def list_stage_outputs(artifacts_dir: Path, stage_folder: str) -> List[Path]:
    base = stage_base_dir(artifacts_dir, stage_folder)
    if not base.exists():
        return []
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    # Sort by mtime desc
    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return subdirs


def select_prior_output(
    *,
    artifacts_dir: Path,
    stage_folder: str,
    use_latest: bool = True,
    prior_path: Optional[Path] = None,
) -> Optional[Path]:
    if prior_path is not None:
        p = Path(prior_path)
        return p if p.exists() else None
    if not use_latest:
        return None
    options = list_stage_outputs(artifacts_dir, stage_folder)
    if options:
        return options[0]
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
