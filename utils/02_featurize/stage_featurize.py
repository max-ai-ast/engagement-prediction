#!/usr/bin/env python3

"""
Stage 2: Featurize posts (text + optional image) and build an embedding bundle.

Inputs:
- Prefer raw_data_*.pkl from Stage 1 (auto-detected from Context/prior outputs)
- Else, falls back to loading most recent parquet files directly

Outputs under <run_dir>/featurize/<timestamp>/:
- embedding_bundle_<timestamp>.pkl
- liked_posts_texts_*.parquet (and _by_user variant when available)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import pickle

from utils.pipeline.core import new_stage_timestamp_dir, select_prior_output
from utils.helpers import (
    load_most_recent_raw_data_digital_ocean,
    build_candidate_posts,
    compute_post_feature_frame,
    save_bundle,
    find_text_column,
    find_join_key,
    get_stage_logger,
    log_operation_start,
)
import time


def _load_raw_from_prior(prior_dir: Path) -> tuple:
    # Load the first raw_data_*.pkl found in the given directory
    candidates = sorted(prior_dir.glob('raw_data_*.pkl'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No raw_data_*.pkl found under {prior_dir}")
    with open(candidates[0], 'rb') as f:
        payload = pickle.load(f)
    return payload['posts_df'], payload['likes_df'], payload.get('metadata_df')


def run(context, args) -> Dict[str, Any]:
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '02_featurize')

    # Initialize logger
    logger = get_stage_logger('STAGE_02_FEATURIZE', log_file=out_dir / 'stage.log')

    # Try to use prior get_data output when available
    prior_get = select_prior_output(run_dir, '01_get_data', use_latest=context.use_latest, prior_path=context.prior_outputs.get('01_get_data'))

    if prior_get is not None:
        log_operation_start('Load raw data from prior stage', 'STAGE_02_FEATURIZE', logger)
        posts_df, likes_df, metadata_df = _load_raw_from_prior(prior_get)
    else:
        log_operation_start('Load raw data from DigitalOcean Spaces', 'STAGE_02_FEATURIZE', logger)
        max_files = int(getattr(args, 'max_files_per_table', 5))
        posts_df, likes_df, metadata_df = load_most_recent_raw_data_digital_ocean(max_files)

    join_like, join_post = find_join_key(posts_df, likes_df)
    author_col = 'did' if 'did' in posts_df.columns else None
    if author_col is None:
        raise ValueError("Author column 'did' not found in posts data")

    max_posts_per_author = int(getattr(args, 'max_posts_per_author', 3))
    max_liked_posts_per_user = int(getattr(args, 'max_liked_posts_per_user', 100))
    cap_seed = int(getattr(args, 'cap_random_seed', 42))
    image_mode = str(getattr(args, 'image_mode', 'auto'))

    # Candidate posts
    t0 = time.time()
    log_operation_start('Build candidate posts', 'STAGE_02_FEATURIZE', logger)
    candidate_posts = build_candidate_posts(
        posts_df, likes_df, join_like, join_post, author_col,
        max_posts_per_author=max_posts_per_author,
        max_liked_posts_per_user=max_liked_posts_per_user,
        rng_seed=cap_seed,
    )

    # Persist liked-posts texts (by join keys)
    log_operation_start('Write liked posts texts', 'STAGE_02_FEATURIZE', logger)
    try:
        text_col_raw = find_text_column(posts_df)
    except Exception:
        text_col_raw = 'text' if 'text' in posts_df.columns else posts_df.columns[0]
    posts_df_str = posts_df[[join_post, text_col_raw]].copy()
    posts_df_str[join_post] = posts_df_str[join_post].astype(str)
    posts_df_str = posts_df_str.rename(columns={text_col_raw: 'text'})

    likes_min = (
        likes_df[["did", join_like]]
        .dropna(subset=["did", join_like])
        .astype({"did": str, join_like: str})
        .drop_duplicates()
    )
    liked_texts_df = (
        likes_min.merge(posts_df_str[[join_post, 'text']], left_on=join_like, right_on=join_post, how='inner')
        .dropna(subset=['text'])
        .drop_duplicates(subset=['did', join_post])
        [["did", join_post, "text"]]
    )
    liked_texts_path = out_dir / f"liked_posts_texts_{out_dir.name}.parquet"
    liked_texts_by_user_path = out_dir / f"liked_posts_by_user_texts_{out_dir.name}.parquet"
    liked_texts_df.to_parquet(liked_texts_path, index=False)
    liked_texts_df.rename(columns={join_post: 'post_id'}).to_parquet(liked_texts_by_user_path, index=False)

    # Compute embeddings
    log_operation_start('Compute text embeddings', 'STAGE_02_FEATURIZE', logger)
    if image_mode != 'off':
        logger.info(f"Image mode: {image_mode} - will compute image embeddings")
    data_source = getattr(args, 'data_source')
    model_name = getattr(args, 'embedding_model')
    posts_emb_df, embedding_dim = compute_post_feature_frame(candidate_posts, data_source, model_name, image_mode=image_mode)
    text_col = find_text_column(posts_emb_df)

    log_operation_start('Save embedding bundle', 'STAGE_02_FEATURIZE', logger)
    bundle_path = save_bundle(
        out_dir=out_dir,
        posts_emb_df=posts_emb_df,
        likes_df=likes_df,
        join_like=join_like,
        join_post=join_post,
        text_column=text_col,
        author_column=author_col,
        data_source=data_source,
        embedding_model=model_name,
        embedding_dim=int(embedding_dim),
        image_mode=image_mode,
        extra_meta={
            'run_dir': str(run_dir.resolve()),
            'max_posts_per_author': max_posts_per_author,
            'max_liked_posts_per_user': max_liked_posts_per_user,
        },
        liked_posts_texts_path=str(liked_texts_path),
    )

    # Stage info
    info_lines = [
        f"stage: featurize",
        f"runtime_seconds: {time.time()-t0:.2f}",
        f"settings: image_mode={image_mode}, max_posts_per_author={max_posts_per_author}, max_liked_posts_per_user={max_liked_posts_per_user}",
        f"inputs: posts_df, likes_df",
        f"N_posts_raw: {len(posts_df)}",
        f"N_likes_raw: {len(likes_df)}",
        f"N_posts_candidates: {len(candidate_posts)}",
        f"data_source: {data_source}",
        f"embedding_model: {model_name}",
        f"embedding_dim: {embedding_dim}",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')

    return {
        'output_dir': out_dir,
        'artifacts': {
            'embedding_bundle_path': str(Path(bundle_path).resolve()),
            'liked_posts_texts_path': str(liked_texts_path),
            'liked_posts_by_user_texts_path': str(liked_texts_by_user_path),
        }
    }


