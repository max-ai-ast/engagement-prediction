#!/usr/bin/env python3

"""
Helpers for resolving lineage-aligned stage artifacts.

This module keeps the provenance logic out of ``cli.py`` so the CLI stays
focused on argument handling and stage orchestration.

The current pipeline is linear at the artifact-folder level:

01_get_data -> 02_user_history -> 03_train -> 04_evaluate

Later stages depend on all earlier stage folders. We derive that folder order
 from the registry's canonical folder names rather than maintaining a second
 hand-written dependency map in the CLI.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import registry as reg
from .core import Context, STAGE_MANIFEST_FILENAME, list_stage_outputs


def _stage_folder_sort_key(folder: str) -> Tuple[int, str]:
    """Sort stage folders by their numeric prefix, falling back to name.

    Folder names currently follow the convention ``NN_name``. Sorting by the
    prefix keeps dependency inference aligned with the pipeline's stage order
    without requiring another manually-maintained list.
    """
    prefix, _, remainder = str(folder).partition("_")
    try:
        return (int(prefix), remainder)
    except ValueError:
        return (10**9, str(folder))


def get_stage_folder_to_keys() -> Dict[str, Tuple[str, ...]]:
    """Return the stage-key variants that write to each artifact folder.

    This is derived directly from ``registry.STAGE_SPECS`` so if a stage is
    renamed or a new train variant is added, the folder-to-key mapping stays in
    sync automatically.
    """
    folder_to_keys: Dict[str, List[str]] = {}
    for stage_key, (_module_path, folder) in reg.STAGE_SPECS.items():
        folder_to_keys.setdefault(folder, []).append(stage_key)
    return {folder: tuple(keys) for folder, keys in folder_to_keys.items()}


def get_stage_input_folders() -> Dict[str, List[str]]:
    """Return cumulative artifact-folder dependencies for the linear pipeline.

    For example, ``03_train`` depends on ``01_get_data`` and
    ``02_user_history``. The result is derived from
    the registered stage folders sorted by their numeric prefixes, so there is
    no separate hand-maintained dependency table to update.
    """
    folder_order = sorted(get_stage_folder_to_keys().keys(), key=_stage_folder_sort_key)
    return {
        folder: folder_order[:idx]
        for idx, folder in enumerate(folder_order)
    }


def load_stage_manifest(stage_dir: Path) -> Dict[str, Any]:
    """Load and lightly validate a stage ``manifest.json`` file."""
    stage_dir = Path(stage_dir).resolve()
    manifest_path = stage_dir / STAGE_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {STAGE_MANIFEST_FILENAME} in stage artifact directory '{stage_dir}'. "
            "Cannot validate lineage for this prior stage output."
        )

    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse manifest at '{manifest_path}': {exc}") from exc

    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest at '{manifest_path}' must contain a JSON object.")
    inputs = manifest.get("inputs")
    if inputs is None:
        manifest["inputs"] = {}
    elif not isinstance(inputs, dict):
        raise ValueError(f"Manifest at '{manifest_path}' has invalid 'inputs'; expected a JSON object.")
    return manifest


def _format_lineage_mismatch(
    *,
    consumer_stage_folder: str,
    artifact_stage_folder: str,
    artifact_dir: Path,
    input_stage_folder: str,
    expected_path: Path,
    recorded_path: Path,
) -> str:
    return (
        f"Misaligned inputs for stage '{consumer_stage_folder}': selected artifact '{artifact_dir}' "
        f"for stage folder '{artifact_stage_folder}' was created from '{recorded_path}' as its "
        f"'{input_stage_folder}' input, but this run resolved '{input_stage_folder}' to "
        f"'{expected_path}'. Pin aligned artifacts or omit ancestor pins so lineage can be inferred "
        "from downstream manifests."
    )


def _get_context_artifact_dir_for_folder(ctx: Context, stage_folder: str) -> Optional[Path]:
    """Return a same-session artifact dir for ``stage_folder`` if one exists."""
    for stage_key in get_stage_folder_to_keys().get(stage_folder, ()):
        art_dir = ctx.get_artifact_dir(stage_key)
        if art_dir is not None and Path(art_dir).exists():
            return Path(art_dir).resolve()
    return None


def validate_explicit_prior_pin_consistency(ctx: Context) -> None:
    """Fail fast when explicitly pinned prior artifacts disagree on lineage.

    This runs before any stage execution starts. It catches cases like a pinned
    ``02_user_history`` artifact whose manifest says it was built from a
    different ``01_get_data`` artifact than the one also pinned for the run.
    """
    explicit = {
        folder: Path(path).resolve()
        for folder, path in ctx.prior_outputs.items()
        if path is not None
    }
    if not explicit:
        return

    manifest_cache: Dict[Path, Dict[str, Any]] = {}
    stage_input_folders = get_stage_input_folders()
    for stage_folder, artifact_dir in explicit.items():
        parents = stage_input_folders.get(stage_folder, [])
        if not parents:
            continue
        manifest = manifest_cache.get(artifact_dir)
        if manifest is None:
            manifest = load_stage_manifest(artifact_dir)
            manifest_cache[artifact_dir] = manifest
        recorded_inputs = manifest.get("inputs", {})
        for parent_folder in parents:
            expected_path = explicit.get(parent_folder)
            if expected_path is None:
                continue
            recorded_raw = recorded_inputs.get(parent_folder)
            if not recorded_raw:
                raise ValueError(
                    f"Explicit prior pins are inconsistent: pinned artifact '{artifact_dir}' for stage "
                    f"folder '{stage_folder}' is missing recorded input '{parent_folder}' in its manifest."
                )
            recorded_path = Path(recorded_raw).resolve()
            if recorded_path != expected_path:
                raise ValueError(
                    f"Explicit prior pins are inconsistent: pinned artifact '{artifact_dir}' for stage "
                    f"folder '{stage_folder}' was created from '{recorded_path}' as its "
                    f"'{parent_folder}' input, but this run also pinned '{parent_folder}' to "
                    f"'{expected_path}'."
                )


def _apply_manifest_constraints(
    *,
    consumer_stage_folder: str,
    artifact_stage_folder: str,
    artifact_dir: Path,
    resolved: Dict[str, Path],
    manifest_cache: Dict[Path, Dict[str, Any]],
    stage_input_folders: Dict[str, List[str]],
) -> None:
    """Merge a selected artifact's recorded parents into the in-flight chain."""
    artifact_dir = Path(artifact_dir).resolve()
    parents = stage_input_folders.get(artifact_stage_folder, [])
    if not parents:
        return

    manifest = manifest_cache.get(artifact_dir)
    if manifest is None:
        manifest = load_stage_manifest(artifact_dir)
        manifest_cache[artifact_dir] = manifest

    recorded_inputs = manifest.get("inputs", {})
    for parent_folder in parents:
        recorded_raw = recorded_inputs.get(parent_folder)
        if not recorded_raw:
            raise ValueError(
                f"Artifact '{artifact_dir}' for stage folder '{artifact_stage_folder}' is missing "
                f"recorded input '{parent_folder}' in its manifest."
            )
        recorded_path = Path(recorded_raw).resolve()
        existing = resolved.get(parent_folder)
        if existing is None:
            resolved[parent_folder] = recorded_path
            continue
        if existing != recorded_path:
            raise ValueError(
                _format_lineage_mismatch(
                    consumer_stage_folder=consumer_stage_folder,
                    artifact_stage_folder=artifact_stage_folder,
                    artifact_dir=artifact_dir,
                    input_stage_folder=parent_folder,
                    expected_path=existing,
                    recorded_path=recorded_path,
                )
            )


def _select_stage_output_matching_inputs(
    *,
    artifacts_dir: Path,
    stage_folder: str,
    expected_inputs: Dict[str, Path],
    manifest_cache: Dict[Path, Dict[str, Any]],
    stage_input_folders: Dict[str, List[str]],
) -> Optional[Path]:
    """Pick the newest stage output whose manifest matches known parent inputs."""
    options = list_stage_outputs(artifacts_dir=artifacts_dir, stage_folder=stage_folder)
    if not options:
        return None

    parents = stage_input_folders.get(stage_folder, [])
    if not parents:
        return Path(options[0]).resolve()

    for option in options:
        option = Path(option).resolve()
        manifest = manifest_cache.get(option)
        if manifest is None:
            try:
                manifest = load_stage_manifest(option)
            except (FileNotFoundError, ValueError):
                continue
            manifest_cache[option] = manifest
        recorded_inputs = manifest.get("inputs", {})
        matches = True
        for parent_folder, expected_path in expected_inputs.items():
            recorded_raw = recorded_inputs.get(parent_folder)
            if not recorded_raw or Path(recorded_raw).resolve() != Path(expected_path).resolve():
                matches = False
                break
        if matches:
            return option
    return None


def resolve_stage_dependencies_for_run(
    *,
    ctx: Context,
    consumer_stage_folder: str,
) -> Dict[str, Path]:
    """Resolve a lineage-aligned upstream artifact chain for one stage run.

    The resolver works in three passes:
    1. Seed the chain from same-session outputs and any explicit prior pins.
    2. Walk backward through stage folders, selecting the newest artifact whose
       manifest is consistent with the parents already known.
    3. Re-apply manifest constraints over the final chain to ensure there are no
       hidden disagreements.

    The return value maps stage folders like ``01_get_data`` to the concrete
    artifact directories that should be used for this run.
    """
    stage_input_folders = get_stage_input_folders()
    deps = list(stage_input_folders.get(consumer_stage_folder, []))
    if not deps:
        return {}

    artifacts_dir = Path(ctx.artifacts_dir).resolve()
    resolved: Dict[str, Path] = {}
    manifest_cache: Dict[Path, Dict[str, Any]] = {}

    for folder in deps:
        chosen = _get_context_artifact_dir_for_folder(ctx, folder)
        if chosen is not None:
            resolved[folder] = chosen
            continue
        prior = ctx.prior_outputs.get(folder)
        if prior is not None:
            resolved[folder] = Path(prior).resolve()

    progress = True
    while len(resolved) < len(deps) and progress:
        progress = False
        for folder in reversed(deps):
            chosen = resolved.get(folder)
            selected_now = False
            if chosen is None:
                expected_inputs = {
                    parent_folder: resolved[parent_folder]
                    for parent_folder in stage_input_folders.get(folder, [])
                    if parent_folder in resolved
                }
                selected = _select_stage_output_matching_inputs(
                    artifacts_dir=artifacts_dir,
                    stage_folder=folder,
                    expected_inputs=expected_inputs,
                    manifest_cache=manifest_cache,
                    stage_input_folders=stage_input_folders,
                )
                if selected is None:
                    continue
                resolved[folder] = selected
                chosen = selected
                selected_now = True
            before_count = len(resolved)
            _apply_manifest_constraints(
                consumer_stage_folder=consumer_stage_folder,
                artifact_stage_folder=folder,
                artifact_dir=chosen,
                resolved=resolved,
                manifest_cache=manifest_cache,
                stage_input_folders=stage_input_folders,
            )
            if selected_now or len(resolved) > before_count:
                progress = True

    missing = [folder for folder in deps if folder not in resolved]
    if missing:
        details = []
        for folder in missing:
            expected_inputs = {
                parent_folder: str(resolved[parent_folder])
                for parent_folder in stage_input_folders.get(folder, [])
                if parent_folder in resolved
            }
            details.append(f"{folder} expected_inputs={expected_inputs or '{}'}")
        raise FileNotFoundError(
            f"Could not resolve a lineage-aligned prior output chain for stage '{consumer_stage_folder}'. "
            f"Missing selections: {', '.join(details)}."
        )

    for folder in deps:
        _apply_manifest_constraints(
            consumer_stage_folder=consumer_stage_folder,
            artifact_stage_folder=folder,
            artifact_dir=resolved[folder],
            resolved=resolved,
            manifest_cache=manifest_cache,
            stage_input_folders=stage_input_folders,
        )

    return resolved


def pin_lineage_aligned_inputs(ctx: Context, stage_key: str, stage_folder_map: Dict[str, str]) -> None:
    """Resolve and persist the artifact chain that should feed ``stage_key``.

    The resolved paths are written back into ``ctx.prior_outputs`` so the stage
    implementation and any downstream helpers read the same lineage-aligned
    artifact set.
    """
    consumer_stage_folder = stage_folder_map[stage_key]
    resolved = resolve_stage_dependencies_for_run(
        ctx=ctx,
        consumer_stage_folder=consumer_stage_folder,
    )
    for folder, path in resolved.items():
        ctx.prior_outputs[folder] = Path(path).resolve()
