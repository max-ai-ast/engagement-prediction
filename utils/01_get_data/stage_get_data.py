#!/usr/bin/env python3

"""
Stage 1: Get data from DigitalOcean Spaces and persist a compact raw bundle.

Outputs under <run_dir>/get_data/<timestamp>/:
- raw_data_<timestamp>.pkl: {'posts_df','likes_df','metadata_df'}
- summary.json: brief counts and parameters
"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from typing import Dict, Any

from utils.pipeline.core import new_stage_timestamp_dir, Context
from utils.helpers import load_most_recent_raw_data_digital_ocean, get_stage_logger, log_operation_start, load_raw_data_ingex
import time

def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '01_get_data')

    # Initialize logger
    logger = get_stage_logger('STAGE_01_GET_DATA', log_file=out_dir / 'stage.log')

    # Parameters (defaults are set in cli.py)
    data_source = args.data_source

    # Inputs for GreenEarth Ingex version
    gcs_bucket = args.gcs_bucket
    posts_start = args.posts_start
    posts_end = args.posts_end
    likes_start = args.likes_start
    likes_end = args.likes_end
    
    # Inputs for DigitalOcean version
    max_files = int(args.max_files_per_table)
    
    t0 = time.time()
    
    if data_source == 'greenearth':
        log_operation_start('Load data from GreenEarth Ingex', 'STAGE_01_GET_DATA', logger)
        posts_df = load_raw_data_ingex(gcs_bucket, 'bsky_posts', posts_start, posts_end)
        likes_df = load_raw_data_ingex(gcs_bucket, 'bsky_likes', likes_start, likes_end)
        metadata_df = None
    elif data_source == 'digitalocean': 
        log_operation_start('Load data from DigitalOcean Spaces', 'STAGE_01_GET_DATA', logger)
        posts_df, likes_df, metadata_df = load_most_recent_raw_data_digital_ocean(max_files)
    else:
        raise ValueError(f"Unknown data_source: {data_source}")
    
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
        'data_source': data_source,
        'posts_start': posts_start,
        'posts_end': posts_end,
        'likes_start': likes_start,
        'likes_end': likes_end,
        'max_files_per_table': max_files,
        'counts': {
            'posts': int(len(posts_df)),
            'likes': int(len(likes_df)),
            'metadata': int(len(metadata_df)) if metadata_df is not None else 0,
        }
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    n_posts = len(posts_df)
    n_likes = len(likes_df)

    context.tracker.log_single_value(name="get_data/n_posts", value=n_posts)
    context.tracker.log_single_value(name="get_data/n_likes", value=n_likes)

    # Stage info
    info_lines = [
        f"stage: get_data",
        f"runtime_seconds: {time.time()-t0:.2f}",
        f"settings: max_files_per_table={max_files}",
        f"inputs: none",
        f"N_posts: {n_posts}",
        f"N_likes: {n_likes}",
        f"N_metadata: {len(metadata_df) if metadata_df is not None else 0}",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    return {
        'output_dir': out_dir,
        'artifacts': {
            'raw_data_path': str(raw_path),
        },
    }

