#!/usr/bin/env python3

"""
Pipeline registry: maps stage keys to single-file stage implementations and
their output folder names. Stages are loaded by absolute file path to avoid
Python import constraints on numeric folder names.
"""

from pathlib import Path
from typing import Dict, Tuple, Optional

from .core import ROOT, Context, load_run_callable


# Stage specs: stage_key -> (relative_file_path_from_root, stage_folder_name)
STAGE_SPECS: Dict[str, Tuple[str, str]] = {
    'get_data':      ("utils/01_get_data/stage_get_data.py", "01_get_data"),
    'featurize':     ("utils/02_featurize/stage_featurize.py", "02_featurize"),
    'user_history':  ("utils/02_featurize/stage_generate_user_history.py", "02_featurize"),
    'user_summary':  ("utils/03_user_summary/stage_generate_user_summary.py", "03_relevel"),
    # Optional alternative relevelers (caller can override file path if desired)
    'relevel_simple':("utils/03_relevel/stage_relevel_simple.py", "03_relevel"),
    'relevel_gini':  ("utils/03_relevel/stage_relevel_gini.py", "03_relevel"),
    'split':         ("utils/04_split/stage_split_users.py", "04_split"),
    'train':         ("utils/05_train/stage_train.py", "05_train"),
    # Alternative two-tower training module
    'train_two_tower': ("utils/05_train/stage_train_two_tower.py", "05_train"),
    'evaluate':      ("utils/06_evaluate/stage_evaluate.py", "06_evaluate"),
}


def get_stage_spec(stage_name: str) -> Tuple[Path, str]:
    if stage_name not in STAGE_SPECS:
        raise KeyError(f"Unknown stage '{stage_name}'")
    rel_path, folder = STAGE_SPECS[stage_name]
    return (ROOT / rel_path).resolve(), folder


def run_stage(stage_name: str, context: Context, args) -> Dict[str, object]:
    module_path, folder = get_stage_spec(stage_name)
    run_fn = load_run_callable(module_path)
    # Each stage script is responsible for creating a timestamped subdir under
    # <run_dir>/<folder>/<timestamp>/ and returning its path.
    result = run_fn(context, args)
    # Expect: {'output_dir': Path, 'artifacts': {...}}
    out_dir = result.get('output_dir') if isinstance(result, dict) else None
    if out_dir is None:
        raise RuntimeError(f"Stage '{stage_name}' did not return an output_dir")
    context.record_artifact(stage_name, Path(out_dir), extras=(result.get('artifacts') or {}))
    return result


