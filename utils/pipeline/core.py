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
from zoneinfo import ZoneInfo


from ..experiment_tracking import ExperimentTracker, NoOpExperimentTracker

# Resolve repo root as two levels up from this file: utils/pipeline/core.py → repo/
CURR = Path(__file__).resolve()
UTILS_DIR = CURR.parent.parent
ROOT = UTILS_DIR.parent


RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
RUN_TIMEZONE_NAME = "America/Los_Angeles"
RUN_TIMEZONE = ZoneInfo(RUN_TIMEZONE_NAME)
DEFAULT_ARTIFACTS_DIR = ROOT / "artifacts"
DEFAULT_RUNS_DIR = ROOT / "runs"
LINEAGE_FILENAME = "lineage.json"
STAGE_MANIFEST_FILENAME = "manifest.json"
STAGE_RESOLVED_CONFIG_FILENAME = "resolved_config.json"


def format_run_timestamp(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=RUN_TIMEZONE)
    else:
        dt = dt.astimezone(RUN_TIMEZONE)
    return dt.strftime(RUN_TIMESTAMP_FORMAT)


def generate_run_timestamp(now: Optional[datetime] = None) -> str:
    # Run and artifact directory labels should always be based on US Pacific time.
    return format_run_timestamp(now or datetime.now(tz=RUN_TIMEZONE))

def _short_uuid(n: int = 8) -> str:
    return uuid.uuid4().hex[:n]


def _normalize_for_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _normalize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(v) for v in value]
    return value


def _write_json(path: Path, data: Dict[str, Any]) -> None:
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

    def new_stage_dir(self, stage_folder: Optional[str] = None, tag: str = "") -> Path:
        """Create a new canonical artifact directory for the active stage.

        If called during `registry.run_stage()`, `Context.begin_stage()` has already set
        `self._active_stage_folder`, so callers can omit `stage_folder` to avoid
        accidentally writing under the wrong artifact folder.
        """
        if stage_folder is None:
            if not self._active_stage_folder:
                raise ValueError("new_stage_dir(stage_folder=None) requires Context.begin_stage() to be called first.")
            stage_folder = self._active_stage_folder
        elif self._active_stage_folder and stage_folder != self._active_stage_folder:
            raise ValueError(
                f"Stage folder mismatch: requested '{stage_folder}' but active stage folder is '{self._active_stage_folder}'."
            )
        # Stage outputs are written to the canonical artifact store, not under run_dir.
        return new_stage_artifact_dir(
            artifacts_dir=Path(self.artifacts_dir).resolve(),
            stage_folder=stage_folder,
            tag=tag,
        )

    def begin_stage(self, stage_key: str, stage_folder: str) -> None:
        self._active_stage_key = stage_key
        self._active_stage_folder = stage_folder
        self._active_stage_inputs = {}

    def record_prior_input(self, stage_folder: str, chosen_path: Path) -> Path:
        chosen_path = Path(chosen_path).resolve()
        self._active_stage_inputs[stage_folder] = str(chosen_path)
        return chosen_path

    def get_active_stage_inputs(self) -> Dict[str, Path]:
        return {
            folder: Path(path).resolve()
            for folder, path in self._active_stage_inputs.items()
        }

    def resolve_prior_output(self, stage_folder: str, *, prior_path: Optional[Path] = None) -> Path:
        chosen = select_prior_output(
            artifacts_dir=Path(self.artifacts_dir).resolve(),
            stage_folder=stage_folder,
            use_latest=self.use_latest,
            prior_path=prior_path,
        )
        if chosen is None:
            raise FileNotFoundError(f"Could not find prior outputs for stage folder '{stage_folder}' under {self.artifacts_dir}")
        return self.record_prior_input(stage_folder, chosen)

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

        resolved_config_path = output_dir / STAGE_RESOLVED_CONFIG_FILENAME
        args_dict = {k: v for k, v in vars(args).items() if k != "func" and not callable(v)}
        _write_json(resolved_config_path, args_dict)

        manifest_path = output_dir / STAGE_MANIFEST_FILENAME
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
        _write_json(manifest_path, manifest)
        self._append_stage_info_inputs(output_dir)

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

    def _append_stage_info_inputs(self, output_dir: Path) -> None:
        stage_info_path = Path(output_dir) / "stage_info.txt"
        prior_inputs = self.get_active_stage_inputs()
        lines = [f"prior_inputs: {len(prior_inputs)}"]
        if prior_inputs:
            for folder, path in sorted(prior_inputs.items()):
                lines.append(f"prior_input_{folder}: {path}")
        else:
            lines.append("prior_input_none: true")

        prefix = ""
        if stage_info_path.exists():
            existing = stage_info_path.read_text()
            prefix = existing if existing.endswith("\n") else existing + "\n"
        stage_info_path.write_text(prefix + "\n".join(lines) + "\n")

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
        run_dir = Path(self.run_dir).resolve()
        lineage_path = run_dir / LINEAGE_FILENAME
        legacy_lineage_path = run_dir / "lineage.yaml"
        read_path = lineage_path if lineage_path.exists() else legacy_lineage_path
        if read_path.exists():
            try:
                lineage: Dict[str, Any] = json.loads(read_path.read_text())
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
        _write_json(lineage_path, lineage)


def new_stage_artifact_dir(
    artifacts_dir: Path,
    stage_folder: str,
    tag: str = "",
    *,
    timestamp: Optional[str] = None,
) -> Path:
    base = (Path(artifacts_dir) / stage_folder).resolve()
    base.mkdir(parents=True, exist_ok=True)
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


def _parse_stage_run_timestamp(stage_run_id: str) -> Optional[datetime]:
    # stage_run_id formats:
    # - <ts>_<uuid>
    # - <ts>_<tag>_<uuid>
    # - legacy: <ts>
    # where <ts> is RUN_TIMESTAMP_FORMAT: YYYYMMDD_HHMMSS
    s = str(stage_run_id).strip()
    if not s:
        return None
    parts = s.split("_")
    if len(parts) >= 2:
        ts_str = f"{parts[0]}_{parts[1]}"
    else:
        ts_str = s
    try:
        return datetime.strptime(ts_str, RUN_TIMESTAMP_FORMAT)
    except Exception:
        return None


def list_stage_outputs(artifacts_dir: Path, stage_folder: str) -> List[Path]:
    base = (Path(artifacts_dir) / stage_folder).resolve()
    if not base.exists() or not base.is_dir():
        return []
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    # Sort by parsed timestamp desc; tie-break by mtime desc.
    def _key(p: Path):
        ts = _parse_stage_run_timestamp(p.name) or datetime.min
        try:
            mtime = float(p.stat().st_mtime)
        except Exception:
            mtime = 0.0
        return (ts, mtime)
    subdirs.sort(key=_key, reverse=True)
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
