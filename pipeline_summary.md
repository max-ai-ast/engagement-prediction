### Engagement Prediction Pipeline — wills_tinkering_folder version

This report documents only the code under `wills_tinkering_folder`. It covers the 6-stage modular pipeline (get_data → featurize → relevel → split → train → evaluate), the user-feature schemas, data artifacts, and how to run it via the bundled CLI.

---

## Overview (6 stages)

- **Stage 1: Get data** — Load most recent posts/likes from DigitalOcean Spaces and persist a compact raw bundle for reuse.
  - Entrypoint: `utils/01_get_data/stage_get_data.py`
  - Output: `outputs/<run>/get_data/<ts>/raw_data_<ts>.pkl`

- **Stage 2: Featurize** — Build candidate posts (liked + per-author-capped remainder), compute text (and optional image) embeddings, and save an embedding bundle.
  - Entrypoint: `utils/02_featurize/stage_featurize.py`
  - Output: `outputs/<run>/featurize/<ts>/embedding_bundle_<ts>.pkl` and liked-posts texts parquet(s)

- **Stage 3: Relevel** — Discover topics with PCA+MiniBatchKMeans over liked-post embeddings, compute per-user topic mixtures, and optionally apply uniform-mixture-balanced releveling.
  - Entrypoint: `utils/03_relevel/stage_relevel_uniform.py`
  - Output: `outputs/<run>/relevel/<ts>/{user_topic_mixtures.parquet, topic_model.pkl, topic_pca.pkl}` (+ optional retained_users.json)

- **Stage 4: Split** — Write `user_splits.json` (train/val/holdout) using eligible and optionally retained users.
  - Entrypoint: `utils/04_split/stage_split_users.py`
  - Output: `outputs/<run>/split/<ts>/user_splits.json` (+ summary)

- **Stage 4: Train** — Train an engagement classifier using embeddings and user history. Build user features (default: per-user KMeans multi-centroid), construct balanced prediction pairs, enforce strict 50/50 class balance, and train an MLP or Two-Tower model. Save checkpoint plus a `training_config.json` describing the feature schema for evaluation-time parity.
  - Entrypoint: `utils/04_train/stage_train_mlp.py` (MLP) or `utils/04_train/stage_train_two_tower.py` (Two-Tower)
  - Output: `outputs/<run>/04_train/<ts>/{checkpoints,plots,logs}/...` and `training_config.json`

- **Stage 6: Evaluation** — Consolidated evaluator supporting `pairs`, `matrix`, and `global_unliked` modes.
  - Entrypoint: `utils/06_evaluate/stage_evaluate.py`
  - Output: `outputs/<run>/evaluate/<ts>_{mode}/...`

A one-shot wrapper `cli.py` orchestrates the above in sequence (with nohup/backgrounding by default). The historical `run-all` token is optional and kept for backwards compatibility.

---

## Stage 1 — Precompute (embeddings bundle)

- Loads newest parquet shards from Spaces, selects candidate posts as:
  1) all posts liked by any user within time window (guaranteed inclusion), and
  2) a per-author-capped sample of remaining posts.
- Computes text embeddings (`SentenceTransformer`) and optionally image embeddings (`torchvision` ResNet18), then saves everything as a bundle.

Key references:
```47:76:/srv/vox/engagement_prediction/wills_tinkering_folder/src/precompute_embeddings.py
def load_most_recent_raw_data(max_files_per_table: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # list_all_objects / download_parquet_files / load_and_combine_data
```
```80:158:/srv/vox/engagement_prediction/wills_tinkering_folder/src/precompute_embeddings.py
def build_candidate_posts(...):
    # liked posts (joinable) + per-author-capped remainder → union
```
```161:179:/srv/vox/engagement_prediction/wills_tinkering_folder/src/precompute_embeddings.py
def compute_post_feature_frame(posts_df: pd.DataFrame, image_mode: str = "auto") -> Tuple[pd.DataFrame, int]:
    # text embeddings + optional image embeddings, return posts_emb_df and total dim
```
```182:216:/srv/vox/engagement_prediction/wills_tinkering_folder/src/precompute_embeddings.py
def save_bundle(...):  # writes embedding_bundle_<ts>.pkl with posts_emb_df, likes_df, join keys, dims, etc.
```

Artifacts under `outputs/<ts>_run_.../precompute/`:
- `embedding_bundle_<ts>.pkl` (contains `posts_emb_df`, `likes_df`, `join_like`, `join_post`, `embedding_dim`, etc.)
- `liked_posts_texts_<ts>.parquet` (+ DID-preserving variants)
- `run_manifest.json` (bundle path, options)

---

## Stage 2 — Relevel and split (topic mixtures)

- Discovers K global topics from liked-post embeddings (PCA + MiniBatchKMeans) and computes per-user topic mixtures.
- Optional “uniform_mixture_balanced” releveling keeps a subset of users to better match an even mixture across topics, parameterized by `alpha` and `min_users_per_topic`.
- Splits re-leveled users into holdout, then train/val.

Key references:
```41:63:/srv/vox/engagement_prediction/wills_tinkering_folder/src/relevel_and_split.py
def _discover_topics(...):  # PCA → MiniBatchKMeans over like-joined embeddings
```
```66:91:/srv/vox/engagement_prediction/wills_tinkering_folder/src/relevel_and_split.py
def _compute_user_topic_mixtures(...):  # per-user topic probability vectors
```
```93:136:/srv/vox/engagement_prediction/wills_tinkering_folder/src/relevel_and_split.py
def _relevel_uniform_mixture(...):  # select users to approach a uniform topic mixture target
```
Outputs under `outputs/<run>/relevel/`:
- `user_splits.json` (train/val/holdout lists, params)
- optionally `topic_model.pkl`, `topic_pca.pkl`, and `topic_prevalence.pdf`

---

## Stage 3 — Train (pairs and strict balancing)

Two primary data paths:
- Preferred: pass `--embedding-bundle` + `--user-splits`. The trainer will:
  - Allocate, per user, a fixed number of latest liked posts as prediction targets and the remainder for user feature construction (capped by `--max-embedding-posts-per-user`), with zero overlap.
  - Build user features (default `multi_centroid`: per-user KMeans centroids + weights). `schema` can be `multi_centroid`, `topic_mixture` (uses saved topic-mix features), or `mean` (average embedding).
  - Construct balanced prediction pairs (50/50 by class). Optionally `--negatives-liked-only` samples negatives from posts liked by others.
  - Enforce strict 50/50 class balance on train and val splits just before dataset creation.
- Legacy: load preprocessed data (`processed_data_*.pkl`) or “fresh” Spaces data (not recommended here).

Key references:
```522:671:/srv/vox/engagement_prediction/wills_tinkering_folder/src/train.py
# bundle + splits path: build per-user disjoint sets, user features (multi_centroid), balanced pairs
```
```646:675:/srv/vox/engagement_prediction/wills_tinkering_folder/src/train.py
# strict 50/50 balance with _enforce_strict_5050_balance(train/val)
```
```735:753:/srv/vox/engagement_prediction/wills_tinkering_folder/src/train.py
# training_config.json saved to ensure evaluation-time schema parity
```
Checkpoints and logs under `outputs/<run>/train/<ts>/`:
- `checkpoints/engagement_model_<ts>.pth` (+ metadata)
- `plots/` training curves and performance plots
- `logs/` JSON results
- `training_config.json` describing `feature_columns`, schema, K, etc.

---

## Stage 4 — Full-feed similarity evaluation

- Loads the model and either the same `embedding_bundle.pkl` (preferred) or legacy processed data.
- Recreates training-time user features (strict schema parity using `feature_columns` and optionally `training_config.json`).
- Modes:
  - `pairs`: build prediction pairs matching the training disjoint split and compute metrics/plots.
  - `matrix`: compute a user×post probability matrix for selected holdout users; also exports a balanced eval set and metrics.

Key references:
```799:829:/srv/vox/engagement_prediction/wills_tinkering_folder/src/evaluate_full_feed_similarity.py
parser = argparse.ArgumentParser(... --embedding-bundle ... --user-splits ... --schema ...)
```
```108:124:/srv/vox/engagement_prediction/wills_tinkering_folder/src/evaluate_full_feed_similarity.py
# schema detection from feature_columns: topic_mixture | multi_centroid | mean
```
```144:166:/srv/vox/engagement_prediction/wills_tinkering_folder/src/evaluate_full_feed_similarity.py
# build holdout user features to match training-time schema
```
```1716:1756:/srv/vox/engagement_prediction/wills_tinkering_folder/src/evaluate_full_feed_similarity.py
# balanced evaluation construction, metrics, and plots
```
Outputs under `outputs/<run>/train/<ts>/evaluate/<ts>_full_feed_similarity/`:
- In matrix mode: `prob_matrix_*.npz` (probs, user_ids, post_ids, optionally texts/user_like_counts)
- In pairs mode: `pairs_eval_*.npz` (ids/labels)
- `balanced_eval_*.npz`, `balanced_metrics_*.json`, `plots/model_performance_*.png`, `summary_*.json`

---

## CLI (wills_tinkering_folder/cli.py)

Subcommands:
- `preprocess`: alternative preprocessing path (not the same as Stage 1+2)
- `train`: train on preprocessed data
- `evaluate`: evaluate a saved model on preprocessed data
- `train-eval`: train using `embedding_bundle.pkl` + `user_splits.json` then run Stage 4
- `run-all` (optional): full modular pipeline (get_data → … → evaluate), creating a run directory and backgrounding via nohup by default

Examples:
```603:705:/srv/vox/engagement_prediction/wills_tinkering_folder/cli.py
# build_parser() subparsers and options
```
Run all stages (backgrounded):
```bash
python wills_tinkering_folder/cli.py --use-latest \
  --max-files-per-table 5 \
  --image-mode auto \
  --max-posts-per-author 3 \
  --global-topic-k 20 \
  --relevel-strategy uniform_mixture_balanced \
  --min-likes-per-user 10 \
  --epochs 300 \
  --batch-size 256
```
Train using bundle + splits, then evaluate:
```210:366:/srv/vox/engagement_prediction/wills_tinkering_folder/cli.py
cli.py train-eval --run-dir <outputs/<ts>_run_...>  # auto-discovers embedding_bundle + user_splits
```

---

## User-feature schemas supported

- **multi_centroid** (default): per-user MiniBatchKMeans over liked-post embeddings, export centroids and weights in fixed layout (`user_k{i}_d{d}`, `user_k{i}_weight`). Robust to variable like counts; capped by `max_embedding_posts_per_user`.
- **topic_mixture**: use topic mixture features discovered in Stage 2 (requires saving and reusing `topic_model.pkl`/`topic_pca.pkl` and mixture columns).
- **mean**: average of liked-post embeddings (fallback). Simpler but less expressive.

Feature schema parity is enforced at evaluation via `feature_columns` saved in the checkpoint or `training_config.json`.

---

## What’s different vs the non-tinkering pipeline

- Persistent run layout under `outputs/<ts>_run_.../` with subfolders per stage.
- Topic-discovery-driven user releveling (uniform mixture balance) prior to splitting.
- Default user features use per-user KMeans (“multi_centroid”) rather than plain mean embeddings.
- Strict 50/50 class balancing enforced on train/val after pair construction.
- Evaluation can operate in “matrix” mode to export a full user×post probability matrix.

---

## Minimal run recipe

1) Stage 1 — Get data:
```bash
python wills_tinkering_folder/cli.py --foreground --use-latest --max-files-per-table 5 --image-mode auto
```
Subsequent stages are orchestrated automatically by the pipeline runner.

Or, end-to-end:
```bash
python wills_tinkering_folder/cli.py --global-topic-k 20 --relevel-strategy uniform_mixture_balanced
```

---

If you want, I can add a compact “make-run.sh” wrapper in this folder that wires your preferred defaults for repeatable runs.
