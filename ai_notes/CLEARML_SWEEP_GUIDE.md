# ClearML Data Sweep Guide

This guide explains how to run hyperparameter sweeps to analyze the impact of data filtering parameters on sample sizes (N) and memory consumption, and how to monitor results in ClearML.

## Quick Start

The sweep functionality is now consolidated in `utils/memory_helpers.py`. Use it via Python:

```python
from utils.memory_helpers import (
    generate_sweep_config,
    run_memory_sweep,
    export_sweep_results_from_clearml,
    fit_memory_model,
    save_model_weights,
)

# Generate a sweep config
config = generate_sweep_config(
    name="data_sweep_260205",
    tag="data-sweep-26-02-05",
    days_options=[7, 14, 21],
    users_options=[10000, 50000, 100000],
)

# Preview what will run (no execution)
run_memory_sweep(config, dry_run=True)

# Run the full sweep
run_memory_sweep(config)

# Run with resume capability (skips completed experiments)
run_memory_sweep(config, resume=True)
```

## Sweep Configuration

Configs can be generated programmatically or loaded from YAML in `utils/memory_helper_artifacts/`. Key sections:

### Fixed Parameters
Parameters that stay constant across all experiments:
```yaml
fixed:
  posts_start: "2026-01-01"
  likes_start: "2026-01-01"
  stop_after: "get_data"  # Only run Stage 1
```

### Sweep Parameters
Parameters that vary (grid search over all combinations):
```yaml
sweep_params:
  data_window_days: [7, 14, 21]      # End dates relative to start
  max_liking_users: [10000, 50000, 100000, 250000]
  max_likes_per_user: [100, 500]
  negative_posts_sample: [10000, 50000, 100000]
```

This creates 72 experiments (3 × 4 × 2 × 3).

## Monitoring in ClearML

### 1. Access the Dashboard

Go to your ClearML server (typically https://app.clear.ml or your self-hosted instance).

### 2. Find Your Experiments

1. Click **Projects** in the left sidebar
2. Select **"Engagement Prediction"** (or your project name)
3. You'll see a list of tasks (experiments)

### 3. Filter Sweep Experiments

Use the tag filter to show only sweep experiments:
1. Click the **Filter** icon
2. Add filter: `Tags` → `contains` → `data-sweep`

### 4. Compare Experiments

This is the most powerful feature for sweep analysis:

1. Select multiple experiments (checkbox on the left)
2. Click the **Compare** button in the toolbar
3. Switch to the **Scalars** tab

You'll see side-by-side comparison of all metrics.

### 5. Key Metrics to Compare

| Metric | Description |
|--------|-------------|
| `get_data/n_users_final_after_join` | Final number of users in output |
| `get_data/n_likes_final_after_join` | Final number of likes in output |
| `get_data/n_posts_core` | Final number of posts in output |
| `get_data/memory_peak_gb` | Peak memory usage during processing |
| `get_data/memory_growth_gb` | Memory growth from start to end |
| `get_data/user_retention_pct` | % of initial users retained |
| `get_data/likes_retention_pct` | % of initial likes retained |
| `get_data/liked_post_match_rate` | % of liked posts found in posts data |

### 6. Create Comparison Charts

In the Compare view:
1. Go to **Scalars** tab
2. Select metrics to plot
3. Use **Parallel Coordinates** for multi-dimensional analysis
4. Export charts as needed

### 7. View Experiment Parameters

Click on any experiment to see:
- **Configuration** tab: All parameters used
- **Scalars** tab: All logged metrics
- **Artifacts** tab: `summary.json` with full statistics

## Understanding the Attrition Pipeline

The filtering stages tracked in ClearML:

```
LIKES PIPELINE:
n_likes_initial       → Initial likes (after time filter)
n_users_initial       → Initial users
    ↓ min-likes pre-filter
n_users_eligible      → Users with >= min_likes_per_user
    ↓ user sampling
n_users_sampled       → Sampled users (if max_liking_users set)
n_likes_after_user_sample → Likes from sampled users
    ↓ per-user cap
n_likes_after_cap     → After max_likes_per_user cap
    ↓ min-likes verification
n_users_final         → Users passing final threshold
n_likes_final         → Likes for those users
    ↓ post-join filter
n_users_final_after_join → Users with matching posts
n_likes_final_after_join → Likes with matching posts (OUTPUT)

POSTS PIPELINE:
n_posts_total         → Posts in time range
n_liked_posts         → Posts that were liked by our users
n_random_sample       → Random sample for negatives
n_posts_core          → Combined output (liked + random)
```

## Customizing the Sweep

### Reduce Experiment Count

If 72 experiments is too many, edit `configs/data_sweep.yml`:

```yaml
sweep_params:
  data_window_days: [14, 21]  # 2 instead of 3
  max_liking_users: [50000, 100000]  # 2 instead of 4
  max_likes_per_user: [100]  # 1 instead of 2
  negative_posts_sample: [50000]  # 1 instead of 3
```

This would give 2 × 2 × 1 × 1 = 4 experiments.

### Add New Parameters

Add any parameter from `cli.py run-all`:

```yaml
sweep_params:
  min_likes_per_user: [2, 5, 10]  # Add new sweep dimension
```

### Change Tags

Tags help organize experiments:

```yaml
sweep:
  tags:
    - "data-sweep"
    - "memory-analysis"
    - "v2"  # Version your sweep runs
```

## Troubleshooting

### ClearML Not Connecting

Check your `~/clearml.conf` file has valid credentials:
```bash
clearml-init  # Re-run setup if needed
```

### Experiment Not Appearing

- Tasks appear after the first metric is logged
- Check the experiment started successfully (watch terminal output)
- Refresh the ClearML dashboard

### Memory Issues

If experiments OOM:
1. Reduce `max_liking_users` values
2. Reduce `data_window_days` values
3. Add more RAM or use a larger instance

### Resume After Failure

The sweep tracks progress in `outputs/sweeps/<sweep_name>/progress.json`.
Use `resume=True` to skip completed experiments:

```python
run_memory_sweep(config, resume=True)
```

## Advanced: ClearML Comparison Tips

### Parallel Coordinates Plot

Best for understanding how multiple parameters affect outcomes:
1. In Compare view, click **Parallel Coordinates**
2. Select parameters and metrics
3. Lines show how each experiment maps parameters to outcomes

### Scatter Plot

Good for seeing relationships:
1. In Compare view, click **Scatter**
2. X-axis: `max_liking_users`
3. Y-axis: `memory_peak_gb`
4. Color by: `data_window_days`

### Export Data

For analysis in Python, use the built-in export function:

```python
from utils.memory_helpers import export_sweep_results_from_clearml

# Export all data from a sweep tag to CSV
data = export_sweep_results_from_clearml(
    tag="data-sweep-26-02-05",
    output_file="sweep_results_260205.csv",
)
```

Results are saved to `utils/memory_helper_artifacts/`.

## Next Steps: Updating the Memory Model

After running a sweep:

```python
from utils.memory_helpers import (
    export_sweep_results_from_clearml,
    fit_memory_model,
    save_model_weights,
)

# 1. Export results from ClearML
data = export_sweep_results_from_clearml(
    tag="data-sweep-26-02-05",
    output_file="sweep_results_260205.csv",
)

# 2. Fit new memory model
weights = fit_memory_model("sweep_results_260205.csv", version="260205")

# 3. Save weights (automatically updates model_weights_latest.json)
save_model_weights(weights, "model_weights_260205.json")
```

The memory estimator will automatically use the new model weights.

## Running Production Pipeline

After finding optimal parameters, run the full pipeline:

```bash
python cli.py run-all \
  --posts-start 2026-01-01 --posts-end 2026-01-15 \
  --max-liking-users 100000 \
  --max-likes-per-user 500 \
  --negative-posts-sample 100000
```
