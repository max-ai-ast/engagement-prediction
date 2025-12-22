#!/usr/bin/env python3

"""
Stage 1: Get data from DigitalOcean Spaces and persist a compact raw bundle.

Outputs under <run_dir>/get_data/<timestamp>/:
- raw_data_<timestamp>.pkl: {'posts_df','likes_df','metadata_df'}
- summary.json: brief counts and parameters
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from utils.pipeline.core import new_stage_timestamp_dir
from utils.helpers import load_most_recent_raw_data, get_stage_logger, log_operation_start
import time


def run(context, args) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '01_get_data')

    # Initialize logger
    logger = get_stage_logger('STAGE_01_GET_DATA', log_file=out_dir / 'stage.log')

    max_files = int(getattr(args, 'max_files_per_table', 5))
    t0 = time.time()
    
    log_operation_start('Load data from DigitalOcean Spaces', 'STAGE_01_GET_DATA', logger)
    posts_df, likes_df, metadata_df = load_most_recent_raw_data(max_files)

    # Save compact pickle
    log_operation_start('Save raw data bundle', 'STAGE_01_GET_DATA', logger)
    import pickle
    ts_name = out_dir.name
    raw_path = out_dir / f"raw_data_{ts_name}.pkl"
    with open(raw_path, 'wb') as f:
        pickle.dump({'posts_df': posts_df, 'likes_df': likes_df, 'metadata_df': metadata_df}, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Summary
    log_operation_start('Write summary files', 'STAGE_01_GET_DATA', logger)
    summary = {
        'max_files_per_table': max_files,
        'counts': {
            'posts': int(len(posts_df)),
            'likes': int(len(likes_df)),
            'metadata': int(len(metadata_df)) if metadata_df is not None else 0,
        }
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Stage info
    info_lines = [
        f"stage: get_data",
        f"runtime_seconds: {time.time()-t0:.2f}",
        f"settings: max_files_per_table={max_files}",
        f"inputs: none",
        f"N_posts: {len(posts_df)}",
        f"N_likes: {len(likes_df)}",
        f"N_metadata: {len(metadata_df) if metadata_df is not None else 0}",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    return {
        'output_dir': out_dir,
        'artifacts': {
            'raw_data_path': str(raw_path),
        },
    }


