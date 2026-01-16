### Engagement Prediction (Modular Workflow)

This repository implements a six-stage modular pipeline to predict engagement on Bluesky posts with minimal duplication of work and fast iteration for releveling experiments.

New design goals:
- Compute and cache all post embeddings up front (text and images) once per time window.
- Perform releveling and user splitting as a separate, iterative step without recomputing embeddings.
- Reuse the same user-featurization logic in both training and evaluation.
- Support multiple model architectures (MLP and Two-Tower) for comparison.

### Setup instructions
1. Install conda (recommended: Miniforge).  

    We recommend Miniforge, a minimal conda distribution that defaults to conda-forge.  
    Download from: https://github.com/conda-forge/miniforge

    Follow the installer for your platform. Restart your shell after installation

    Verify:

    ```bash
    conda --version
    ```

    You might have to activate conda from the install location e.g.:
    ```bash
    source /opt/miniconda3/bin/activate
    ```

2. Install conda-lock

    Install conda-lock into your base environment:

    ```bash
    conda install -n base conda-lock
    ```
    Verify:
    ```bash
    conda-lock --version
    ```

3. Create the environment from the lock file and activate it

    This installs exactly the pinned dependencies known to work. You can name your environment whatever you like, e.g. "eng-pred":
    ```bash
    conda-lock install -n eng-pred conda-lock.yml
    conda activate eng-pred
    ```

4. Sanity check
    ```bash
    python -c "import torch; print(torch.__version__)"
    ```

If you modify or add new dependencies, please update environment.yml to reflect the change. Also please update environment.ci.yml. (The latter is the environment file for running tests in github actions. It does not include the large CUDA dependencies because the server that runs the tests does not have a GPU, and they would significantly increase the time to run the tests. The github actions CI will fail if environment.yml and environment.ci.yml are not in sync (see `scripts/check_env_sync.py`)). Then regenerate the conda-lock files for both environments:

```bash
conda-lock -f environment.yml -p linux-64 --mamba --lockfile conda-lock.yml
conda-lock -f environment.ci.yml -p linux-64 --mamba --lockfile conda-lock.ci.yml
```

5. Experiment tracking setup
   The only currently implemented experiment tracker is ClearML. If you'd like to use it, make sure you have ClearML installed (it should be installed via the conda-lock above), and run `clearml-init` in the repo. 

### Testing
This repo utilizes `pytest`. To run the tests locally, simply run the `pytest` command. The tests will automatically be run in github actions for all commits to `main` or any pull request (see `.github/workflows/ci.yml`). The default behavior is to store tmp files in the `/tmp/pytest-of-{username}` directory. To use the current directory, instead run:

```bash
TMPDIR=$PWD pytest
```
Tests all reside in the `tests/` directory and should use the file naming convention: `test_*.py`.

### Repository layout (stages under `utils/`)

- `utils/01_get_data/stage_get_data.py`: Stage 1 — Load most recent parquet dumps from Green Earth Ingex or Spaces and save a compact raw bundle.
- `utils/02_featurize/stage_featurize.py`: Stage 2 — Build candidate post set and compute text+image embeddings → save `embedding_bundle_*.pkl`.
- `utils/03_relevel/stage_relevel_uniform.py`: Stage 3 — Discover topics and compute per-user mixtures; optional uniform-mixture-balanced relevel selection.
- `utils/04_split/stage_split_users.py`: Stage 4 — Produce `user_splits.json` (train/val/holdout).
- `utils/05_train/stage_train.py`: Stage 5 (MLP) — Train MLP model using bundle + splits; saves checkpoint and `training_config.json`.
- `utils/05_train/stage_train_two_tower.py`: Stage 5 (Two-Tower) — Train two-tower model with user history attention encoder.
- `utils/06_evaluate/stage_evaluate.py`: Stage 6 — Consolidated evaluation (pairs, matrix, global_unliked).
- `utils/00_helpers/helpers.py`: Minimal cross-stage helpers (re-exported from existing modules).
- `utils/pipeline/{core.py, registry.py}`: Context, timestamped output dirs, and stage registry.

Shared utilities remain:
- `utils/user_features.py`: Shared user-featurization (topic_mixture, multi_centroid, mean)
- `utils/data_utils_with_images.py`, `utils/train_test_helpers.py`, `utils/visual_helpers.py`: Data I/O, modeling, plotting helpers

Legacy scripts (still available, not recommended for the new workflow): `src/preprocess.py` and old CLI flows; the old run-all flow is defunct.

### Model Architectures

#### MLP Model (default)
The original architecture using a multi-layer perceptron:
- User features: Multi-centroid representations (per-user KMeans over liked post embeddings)
- Post features: Pre-computed text + image embeddings
- Training: BCE loss with 50/50 balanced positive/negative pairs

#### Two-Tower Model
A retrieval-optimized architecture with separate user and post encoders:
- **User Tower**: Encodes user preferences from their liked post history using:
  - Self-attention layers to capture interest patterns
  - Learnable positional embeddings (recency-aware)
  - Dual aggregation: attention-weighted pooling + mean pooling
- **Post Tower**: Projects post embeddings to shared space via MLP
- **Training**: BCE loss with explicit positive/negative pairs

### Quick start

Below, replace paths with your actual workspace if different.  

The `run-all` command can be run with command-line args (as in the examples below), or with a YAML config file, e.g.:

```bash
python cli.py --config config.yml run-all
```

Example config file:
```yaml
posts_start: "2026-01-04"
posts_end: "2026-01-04T02:00:00"
likes_start: "2026-01-04"
likes_end: "2026-01-04T02:00:00"
foreground: true
output_dir: "/path/to/outputs"
start_from: "train"
model_type: "two-tower" 
```

1) Stage 1 — Get data (creates a run dir)  
The default behavior is to use data from [Green Earth ingex](https://github.com/greenearth-social/ingex) with date filters. For example:
```bash
python cli.py run-all --foreground --use-latest \
  --posts-start 2026-01-04 --posts-end 2026-01-04T06:00:00 --likes-start 2026-01-04 --likes-end 2026-01-04T06:00:00
```
Note that there is a default GCS Bucket but it can also be overridden using `--gcs-bucket`. To use Digital Ocean Spaces data instead, specify the data source, e.g.:
```bash
python cli.py run-all --foreground --use-latest \
  --data-source digitalocean --max-files-per-table 5 --max-posts-per-author 3 --image-mode auto
```
Creates a run directory like `outputs/<timestamp>_run_d<files>_mppa<cap>/` 
2) Stage 2 - Featurize, saves `featurize/embedding_bundle_<timestamp>.pkl` with:
- `posts_emb_df` (post_emb_* and image_emb_* columns)
- `likes_df`
- `join_like`, `join_post`, `text_column`, `author_column`
- `embedding_dim`, `image_mode`, `embedding_model`, metadata  

If `--data-source` is `greenearth`, embeddings are assumed to already exist and just need to be extracted. If `--data-source` is `digitalocean`, embeddings are computed.

3) Stage 3 — Relevel users (iterate here without recomputing embeddings)
```bash
# via run-all or directly calling the stage script; parameters still accepted
```
Saves under the run directory in `relevel/`:
- `user_topic_mixtures.parquet`, optional `topic_model.pkl` and `topic_pca.pkl`
- optional `retained_users.json` when using uniform-mixture-balanced selection

4) Stage 4 — Split users → `user_splits.json`

5) Stage 5 — Train using the bundle + splits
```bash
# orchestrated via run-all; training dir: <run_dir>/train/<timestamp>/
```
Notes:
- Training allocates each user's liked posts into embedding vs target sets, builds user features from embedding posts, and creates balanced positive/negative target pairs.
- The model checkpoint saves the feature schema for evaluation to match dimensions.

6) Stage 6 — Evaluation (no embedding recompute)
```bash
# via consolidated stage_evaluate.py (modes: pairs | matrix | global_unliked)
```
Outputs:
- Probability matrix `.npz`, optional balanced eval set `.npz`, metrics JSON, and plots under `outputs/.../evaluate/` or `outputs/full_feed_similarity/<timestamp>/`.

### User feature schemas (MLP model only)
- `topic_mixture`: Requires a KMeans topic model (and optional PCA) from Stage 2; user features are per-like topic mixtures.
- `multi_centroid`: Per-user MiniBatchKMeans over embedding-capable liked posts; exports K centroids and weights.
- `mean`: Mean embedding of embedding-capable liked posts.

The shared implementation lives in `utils/user_features.py` and is used consistently in training and evaluation.

### Important flags and tips
- Image embeddings: Ensure the `--image-mode` choice at Stage 1 matches expectations at training/evaluation (dimension consistency is enforced).
- Device: Most scripts accept `--device cuda` when available.
- Reproducibility: Set seeds via `--cap-random-seed` (Stage 1/4) and `--random-seed` (Stage 2/3).
- Outputs:
  - Stage 1: `outputs/precompute/<timestamp>/embedding_bundle_<timestamp>.pkl`
  - Stage 2: `outputs/relevel/<timestamp>/{user_splits.json, topic_model.pkl, topic_pca.pkl}`
  - Stage 3: `outputs/checkpoints/engagement_model_<timestamp>.pth`, plus logs/plots
  - Stage 4: `outputs/.../evaluate/<timestamp>_*` and `outputs/full_feed_similarity/<timestamp>/`

### Legacy workflow
The previous `src/preprocess.py` → `src/train.py` → `src/evaluate_full_feed_similarity.py` flow that recomputed embeddings during evaluation is still supported for compatibility, but the new four-stage workflow above is recommended.

### End-to-end execution

#### MLP Model (default)
Run all six stages with the default MLP architecture:
```bash
python cli.py run-all \
  --max-files-per-table 7 --max-posts-per-author 3 --image-mode auto \
  --global-topic-k 20 --relevel-strategy uniform_mixture_balanced --min-likes-per-user 10 \
  --epochs 300 --batch-size 256 --device cuda
```

#### Two-Tower Model
Run all six stages with the two-tower architecture:
```bash
python cli.py run-all --model-type two-tower \
  --max-files-per-table 7 --max-posts-per-author 3 --image-mode auto \
  --global-topic-k 20 --relevel-strategy uniform_mixture_balanced --min-likes-per-user 10 \
  --epochs 100 --batch-size 256 --device cuda
```

Two-tower specific options:
- `--model-type two-tower`: Use the two-tower architecture instead of MLP
- `--shared-dim 128`: Output embedding dimension for both towers
- `--num-attention-heads 4`: Number of attention heads in user history encoder
- `--num-attention-layers 2`: Number of transformer layers in user history encoder
- `--max-history-len 20`: Maximum number of liked posts per user for history encoding

#### Train-Eval with Two-Tower
Train a two-tower model on an existing run directory:
```bash
python cli.py train-eval --model-type two-tower \
  --run-dir outputs/20231215_run_d7_mppa3/ \
  --epochs 100 --batch-size 256 --device cuda
```

Or with explicit paths:
```bash
python cli.py train-eval --model-type two-tower \
  --embedding-bundle outputs/run/02_featurize/embedding_bundle_*.pkl \
  --user-splits outputs/run/04_split/user_splits.json \
  --shared-dim 256 --num-attention-heads 8 --max-history-len 50
```

#### Standalone Two-Tower Training
The two-tower module can also be run directly:
```bash
python utils/05_train/stage_train_two_tower.py \
  --embedding-bundle path/to/embedding_bundle.pkl \
  --user-splits path/to/user_splits.json \
  --epochs 100 --device cuda
```

By default, `run-all` runs in the background with nohup and writes a log under `outputs/run-all_<ts>.log`, then mirrors it to `<run_dir>/run-all.log` once the run directory is created. Use `--foreground` to run interactively.

### Two-Tower Output Structure

Two-tower training outputs:
```
05_train/<timestamp>/
├── checkpoints/
│   ├── two_tower_<ts>.pth
│   └── two_tower_best.pth
├── plots/
│   ├── training_history_<ts>.png
│   ├── val_performance_<ts>.png
│   └── holdout_performance_<ts>.png
├── holdout_eval/
│   ├── predictions.parquet
│   └── metrics_overall.json
├── training_config.json
└── stage_info.txt
```

### Testing
Use `pytest` for lightweight tests where available.
```bash
pip install -r requirements.txt pytest
pytest -q
```

# Specific Testing Examples

Two-tower model:
```bash
python cli.py run-all --model-type two-tower   --max-files-per-table 14 --max-posts-per-author 5 --image-mode off   --global-topic-k 20 --relevel-strategy uniform_mixture_balanced --min-likes-per-user 10   --epochs 100 --batch-size 256 --device cuda
```

MLP model:
```bash
python cli.py run-all --model-type mlp   --max-files-per-table 14 --max-posts-per-author 5 --image-mode off   --global-topic-k 20 --relevel-strategy uniform_mixture_balanced --min-likes-per-user 10   --epochs 100 --batch-size 256 --device cuda
```