# Engagement Prediction

This repo trains and evaluates engagement rankers for Bluesky posts. The active workflow is a four-stage artifact pipeline:

1. `01_get_data`: load Ingex parquet data, split likes into train/validation/holdout windows, write post embeddings and compact core tables.
2. `02_user_history`: build per-user, per-hour history lists for model training.
3. `03_train`: train an MLP, two-tower, or BST ranker.
4. `04_evaluate`: run holdout evaluation modules from ranking-row artifacts written by Stage 3.

The historical featurize/relevel/split workflow is not part of the active branch.

## Setup

Install conda, then create the pinned environment:

```bash
conda-lock install -n eng-pred conda-lock.yml
conda activate eng-pred
python -c "import torch; print(torch.__version__)"
```

If dependencies change, update both `environment.yml` and `environment.ci.yml`, then regenerate lock files:

```bash
conda-lock -f environment.yml -p linux-64 --mamba --lockfile conda-lock.yml
conda-lock -f environment.ci.yml -p linux-64 --mamba --lockfile conda-lock.ci.yml
```

ClearML is the implemented experiment tracker. To use it, run `clearml-init` after activating the environment. For local or test runs without ClearML, pass `--experiment-tracker none`.

## Testing

Run tests from this directory with the project conda environment:

```bash
conda run -n eng-pred pytest -q
```

To keep pytest temporary files inside the repo:

```bash
TMPDIR=$PWD conda run -n eng-pred pytest -q
```

Tests live under `tests/` and use the `test_*.py` naming convention.

## Repository Layout

- `cli.py`: unified pipeline CLI. `run-all` is implicit, so `python cli.py --config config.yml` and `python cli.py run-all --config config.yml` are both accepted.
- `compare.py`: checkpoint-backed ranker comparison CLI.
- `utils/01_get_data/stage_get_data.py`: Stage 1 data ingestion, time-window splits, negative sampling pools, embeddings memmap, author index mapping.
- `utils/02_user_history/stage_generate_user_history.py`: Stage 2 user-hour history directory with prior embedding ids, author ids, and time-delta features.
- `utils/03_train/stage_train_mlp.py`: Stage 3 MLP matrix ranker.
- `utils/03_train/stage_train_two_tower.py`: Stage 3 two-tower matrix ranker.
- `utils/03_train/stage_train_bst_ranker.py`: Stage 3 BST heavy ranker.
- `utils/04_evaluate/stage_evaluate.py`: Stage 4 holdout evaluation from compact ranking-row artifacts.
- `utils/dataloaders.py`: bucketed listwise datasets, samplers, and shared user encoders.
- `utils/matrix_ranking.py`: shared matrix ranking metrics, final metric logging, and ranking-row writers.
- `utils/ranking_adapters.py`: `.pth` checkpoint adapters for compare-rankers.
- `utils/pipeline/{core.py,dependencies.py,registry.py}`: artifact directories, lineage, dependency resolution, and stage registry.

## Running The Pipeline

The CLI merges defaults, an optional YAML/JSON config, and explicit command-line flags. CLI flags win over config values.

```bash
python cli.py --config config.yml
```

For foreground local iteration:

```bash
python cli.py --config config.yml --background false --experiment-tracker none
```

For a small Stage 1 smoke run, see `config_test.yml`.

### Output Layout

By default, outputs are written under `outputs/` in two coordinated views:

- `artifacts/<stage_folder>/<stage_run_id>/`: canonical stage artifacts.
- `runs/<pipeline_run_id>/<stage_folder>`: symlinks to the canonical artifacts for one pipeline run.

Each stage writes `manifest.json`, `resolved_config.json`, `stage.log`, and `stage_info.txt` when it completes.

### Stage 1: Get Data

Stage 1 reads Ingex parquet data from GCS, applies date filters, splits users and likes, and writes compact artifacts used by all downstream stages.

Common config keys:

```yaml
gcs_bucket: "greenearth-471522-ingex-extract-prod"
posts_start: "2026-06-20"
posts_end: "2026-06-24"
likes_start: "2026-06-20"
likes_end: "2026-06-24"
train_start: "2026-06-20"
val_start: "2026-06-22"
holdout_start: "2026-06-23"
holdout_end: null
max_trainval_users: 1000
max_unseen_eval_users: 100
max_likes_per_user: 16
negative_samples_per_hour: 10
min_likes_per_user: 0
memory_check: "skip"
```

Important Stage 1 behavior:

- `max_likes_per_user` applies a deterministic random per-user cap.
- `max_trainval_users` samples users eligible for train/validation/seen-holdout rows.
- `max_unseen_eval_users` samples users used only for unseen validation and holdout.
- `negative_samples_per_hour` controls the same-hour post pool used for matrix ranker training.
- `min_author_support` controls which authors get dedicated author embedding rows when author features are enabled.

Primary artifacts include `likes_core_*.parquet`, `posts_core_*.parquet`, `embeddings_*.npy`, and, when available, `author_idx_*.parquet`.

### Stage 2: User History

Stage 2 creates `history_posts_*.parquet`, keyed by `(did, like_hour_bucket)`.

The history artifact includes:

- `prior_emb_indices`: prior liked post embedding ids, sorted most-recent first.
- `prior_like_age_hours_at_bucket_start`: age of each prior like relative to the target hour bucket, aligned with `prior_emb_indices`.
- `prior_author_indices`: aligned author ids when Stage 1 wrote author metadata.
- `raw_prior_count`: uncapped count before `max_prior_likes`.

Common config:

```yaml
max_prior_likes: 64
```

### Stage 3: Training

All active model types use the same Stage 1/2 artifact contract and bucketed same-hour candidates.

Shared training options:

```yaml
model_type: "two-tower" # mlp, two-tower, or bst-ranker
max_history_len: 64
epochs: 100
batch_size: 128
learning_rate: 3e-4
patience: 10
early_stopping_min_delta: 0.002
metrics_top_ks: [30]
num_dataloader_workers: 4
dataloader_pin_memory: true
dataloader_prefetch_factor: 1
dataloader_persistent_workers: false
```

#### MLP

The MLP path scores the full user-by-candidate matrix for each hour bucket. It supports `summarized`, `full_transformer`, and `cross_attention` user encoders.

```bash
python cli.py --model-type mlp --user-encoder summarized --stop-after train
```

Useful options:

- `--hidden-dims`
- `--dropout-rate-mlp`
- `--user-summarization mean|ema|linear_recency`
- `--ema-alpha`

#### Two-Tower

The two-tower path independently encodes users and candidate posts, then scores with a dot product over the shared embedding space. It supports `full_transformer` and `cross_attention` user encoders.

```bash
python cli.py --model-type two-tower --user-encoder cross_attention --stop-after train
```

Useful options:

- `--shared-dim`
- `--user-hidden-dim`
- `--post-hidden-dim`
- `--num-attention-heads`
- `--num-attention-layers`
- `--l2-normalize-embeddings`
- `--similarity-temperature`
- `--use-author-embedding-table`

Two-tower training writes checkpoint files, `training_config.json`, `training_results.json`, TorchScript tower artifacts, a serving manifest, and holdout ranking rows under `eval/`.

#### BST Ranker

The BST ranker fuses content embeddings, author embeddings, time-delta buckets, and a candidate-aware transformer. It currently requires author embeddings.

```bash
python cli.py --model-type bst-ranker \
  --use-author-embedding-table \
  --prediction-hidden-dims 64 32 16 \
  --stop-after train
```

BST training uses matrix ranking over same-hour candidate sets with additional sampled negatives. It requires `bst_num_transformer_layers: 1` because it uses the optimized one-layer matrix scorer.

Useful options:

- `--bst-additional-batch-negatives`
- `--content-projection-dim`
- `--author-projection-dim`
- `--bst-model-dim`
- `--bst-time-embedding-dim`
- `--bst-num-attention-heads`
- `--bst-num-transformer-layers`
- `--bst-transformer-ff-dim`
- `--bst-dropout-rate`
- `--bst-time-delta-bucket-boundaries-hours`
- `--bst-max-train-batches-per-epoch`

Current branch note: BST training writes train/validation metrics and checkpoints, but Stage 4 evaluation expects holdout ranking-row artifacts. Until BST holdout ranking rows are wired in, use `--stop-after train` for BST runs and compare checkpoints with `compare-rankers`.

### Stage 4: Evaluate

Stage 4 consumes holdout ranking rows from Stage 3:

```text
03_train/<stage_run_id>/eval/holdout_unseen_users_ranking_rows.parquet
03_train/<stage_run_id>/eval/holdout_seen_users_ranking_rows.parquet
```

Run the full pipeline for MLP or two-tower:

```bash
python cli.py --model-type two-tower --user-encoder cross_attention
```

Or evaluate a pinned training output:

```bash
python cli.py --start-from evaluate --prior-03-train 20260620_120000_train_two_tower
```

Useful options:

- `--eval-holdout-type unseen_users|seen_users`
- `--skip-modules cold_start_curves,performance_inequality`
- `--prior-03-train`

## Compare Rankers

`compare-rankers` evaluates saved `.pth` checkpoints on shared bucketed candidate sets without rerunning training.

```bash
python cli.py compare-rankers \
  --output-dir /mnt/data/dave/outputs \
  --prior-01-get-data 20260617_205310_fec862c8 \
  --prior-02-user-history 20260618_095653_14c6b8fc \
  --model tt:two-tower:/path/to/two_tower.pth \
  --model bst:bst-ranker:/path/to/bst_ranker.pth \
  --splits val val_unseen_users holdout_unseen_users \
  --metrics-top-ks 30 \
  --batch-size 256 \
  --device cuda
```

Compare outputs are written under `artifacts/compare_rankers/<stage_run_id>/`:

- `metrics.json`
- `metrics.csv`
- `model_specs.json`
- `stage_info.txt`
- `stage.log`

Current compare-rankers assumptions:

- Model specs use `name:type:path`.
- Supported types are `two-tower` and `bst-ranker`.
- Compared checkpoints must use author embeddings.
- If compared checkpoints use different `max_history_len` values, pass `--max-history-len` to choose the evaluation history length.
- BST checkpoints are scored with the optimized one-layer matrix scorer.

## Selective Reruns And Prior Pins

Use `--start-from`, `--stop-after`, and prior pins to reuse artifacts:

```bash
python cli.py --config config.yml \
  --start-from train \
  --stop-after train \
  --prior-01-get-data 20260617_205310_fec862c8 \
  --prior-02-user-history 20260618_095653_14c6b8fc
```

Accepted stage aliases:

- `get_data`
- `user_history`
- `train`, `train_mlp`, `train_two_tower`, `train_bst_ranker`
- `evaluate`

Prior pins can be stage run ids under `artifacts/<stage_folder>/`, absolute paths, or paths relative to `output_dir`.

## Background Runs

By default, `config.yml` may set `background: true`. In background mode, the CLI writes `run-all.resolved-config.json` and starts a foreground child process with `nohup`.

Run in the foreground while iterating:

```bash
python cli.py --config config.yml --background false
```

## Development Notes

- Keep Stage 1/2 artifact schemas stable when possible; all model types share them.
- Use `utils/matrix_ranking.py` for matrix ranking metrics and ranking-row writes.
- Use `utils/ranking_adapters.py` when adding checkpoint-backed comparison support.
- Avoid adding new training paths without registering them in `utils/pipeline/registry.py` and documenting their artifact contract here.

## Contributing

Interested in contributing? Please join the Discord and introduce yourself first: https://discord.com/invite/8bWEyrkrJC.
