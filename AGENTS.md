# Agent guidelines

This file documents conventions for AI agents and contributors working on this repo.

## Default values: single source of truth

**All default values for pipeline/training parameters live in `cli.py`** in the `DEFAULTS` dict. The CLI merges user config and CLI flags with `DEFAULTS` to produce the final `args` namespace.

- **Do not add default values** to model or encoder `__init__` parameters that are driven by the CLI (e.g. `user_hidden_dim`, `user_output_dim`, `num_attention_heads`, `num_attention_layers`, `max_history_len`, `attention_dropout`, `shared_dim`, `post_hidden_dim`, `dropout_rate`, `user_encoder_type`). Require callers to pass these explicitly.
- **Do not duplicate defaults** in stage scripts, dataloaders, or model classes. The only place to define defaults for run-all parameters is `cli.py`.
- When adding a new CLI-controlled hyperparameter: add it to `DEFAULTS` in `cli.py`, add the corresponding `--flag` to the run-all parser, and pass the value from `args` into the model/encoder constructors without defining a default in those constructors.

This keeps a single source of truth and avoids drift between CLI defaults and in-code defaults.

## Args extraction in stage `run()` functions

In each stage's `run(context, args)` function, **extract all `args.*` values into local variables once** at the top of the function (in a single `# --- hyperparams ---` block), then use only the local variables in all downstream code. Do not scatter `args.*`, `int(args.*)`, or `float(args.*)` throughout the function body.

- This makes it easy to see every parameter at a glance.
- If a value ever needs post-processing (e.g. clamping, conditional override), the change happens in one place and is consistent for all downstream uses.
- Encoder-specific args (e.g. attention params) can be extracted at the top of the relevant branch rather than the universal block.

## Likes caps: three flags with distinct semantics

The pipeline has three different per-user/per-target caps on likes. Don't confuse them:

| Flag | Stage | Semantics |
|------|-------|-----------|
| `--max-likes-per-user` | Stage 1 (get_data) | Random per-user cap applied at GCS ingestion time. Determines what hits `likes_core_*.parquet`. |
| `--effective-likes-cap` | Stages 2 + 3 | **Optional** additional per-user cap re-applied to `likes_core_*.parquet` at training-prep time, *without* re-running Stage 1. Lets cap sweeps reuse a single ingestion. `None` (default) = no additional cap. |
| `--max-prior-likes` | Stage 3 (user_history) | Per-target cap on the *length of the prior-likes history list* shown to each model prediction. Distinct from the per-user volume cap above. |

The first two share the helper `utils/likes_cap.py:apply_per_user_random_cap`, which hashes `(did, subject_uri)` with a fixed seed.  Reusing the same seed across sweep cells guarantees that the result of `cap=N` is a **strict subset** of `cap=M` for `N < M` — see `tests/test_likes_cap.py` for the property tests.  This nesting invariant is what makes cap sweeps interpretable.

`--effective-likes-cap-seed` controls the hash seed for the Stage-2/3 cap; when unset, it inherits `--cap-random-seed` (default 42), which is usually what you want.

The effective cap is recorded in:
- Stage 2 + Stage 3 `stage_info.txt`
- Stage 4 `training_config.json` (so Stage 5 can enforce parity)
- ClearML metrics (so cells can be sliced in the UI by cap)

## Cap × architecture sweep harness

`run_cap_arch_sweep.sh <sweep_config.yml>` orchestrates a full cap × arch × seed sweep.

### Sweep phases and parallelism rules

| Phase | What runs | Parallelism |
|-------|-----------|-------------|
| **A — prep** | Stage 2 (target_posts) + Stage 3 (user_history) per cap level | **Sequential** — one cap at a time |
| **B — MLP train** | Stage 4 training for all MLP × seed cells in this cap | **Parallel ≤ `max_parallel_mlp`** — but only safe when `skip_holdout_pred_during_train: true`; with holdout-pred enabled, each process materialises ~25 GB of holdout data, so parallelism risks OOM |
| **C — TT train** | Stage 4 training for all two-tower × seed cells in this cap | **Sequential** — heavy GPU footprint |
| **C.5 — holdout-pred** | `scripts/run_holdout_pred.py` on every trained checkpoint | **Sequential — never parallel** (high per-process memory footprint) |
| **D — Stage 5 eval** | `cli.py --start-from evaluate` for every trained cell | **Parallel-safe** — lightweight; reads parquets only |

Phase C.5 only runs when `skip_holdout_pred_during_train: true` is set in the sweep YAML (see below).  Stage 5 eval (Phase D) is currently launched by the user after the sweep completes — not wired into the harness yet.

### `--skip-holdout-pred` + Phase C.5

For sweeps with many cells (e.g. sweep 02: 36 cells), enable this pattern to allow parallel MLP training without OOM:

```yaml
max_parallel_mlp: 2
skip_holdout_pred_during_train: true   # Phase C.5 generates holdout parquets sequentially
```

The harness passes `--skip-holdout-pred` to every `cli.py` train invocation.  Each training process exits after saving the model checkpoint and writing train/val predictions, *without* materialising the holdout dataset.  After all training cells finish, Phase C.5 iterates the plan sequentially and calls `scripts/run_holdout_pred.py <cell_dir>` for each cell.  Phase C.5 skips cells that already have `predictions/holdout_unseen_users.parquet` (idempotent — safe to re-run after a partial failure).

### YAML config

- Canonical schema: `sweeps/01_cross_arch_validation.yml`.
- `scripts/_emit_sweep_plan.py` parses the YAML and emits a TSV plan + bash variables for the harness.
- Each cell's `training_config.json` records the effective cap and `skip_holdout_pred` flag, making it fully self-describing for cross-cell aggregation.

### Mid-pipeline launch requirement

Any sweep that starts at `--start-from target_posts` (or later) **must** supply `--train-start`, `--val-start`, `--holdout-start`, and `--holdout-user-fraction` via `extra_cli_args` in the YAML.  These are not inferred automatically from Stage 1's `stage_info.txt`.  Missing any of them raises a `ValueError` from `_get_train_start()`.

## Eval-module ordering footgun

`utils/05_evaluate/evals/__init__.py:discover_modules()` uses `pkgutil.iter_modules`, which sorts by file name alphabetically.  When you add an eval module that depends on another module's output (e.g. `z_bias_summary_export.py` consumes `synthetic_feed.py`'s summary JSON), **name it so it sorts after its dependency**.  Convention: prefix with `z_` for "runs last".  Files prefixed with `_` are skipped entirely (used for shared helpers like `_helpers.py`).
