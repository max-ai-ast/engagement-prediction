# Memory Helper Artifacts

This directory contains configuration files and model weights for the memory prediction system used in Stage 1 (data ingestion).

## Contents

- `model_weights_*.json` - Trained linear regression model coefficients for predicting memory usage based on dataset characteristics
- `sweep_config_*.yml` - Configuration used during the parameter sweep that generated the training data for the memory model

## Purpose

The memory helper system predicts the RAM requirements for processing large datasets before loading them into memory. This allows the pipeline to:

1. Estimate memory usage based on row count, column count, and data types
2. Apply appropriate memory limits to prevent out-of-memory errors
3. Enable safe processing of large datasets with automated memory management

## Usage

These artifacts are loaded automatically by `utils/memory_helpers.py` when predicting memory requirements in Stage 1 (`utils/01_get_data/stage_get_data.py`).

The memory model is used as follows:
```python
from utils.memory_helpers import predict_memory_usage

# Predict memory for a dataframe operation
predicted_mb = predict_memory_usage(
    row_count=1_000_000,
    col_count=20,
    dtypes={'int64': 5, 'float64': 10, 'string': 5}
)
```

## Updating the Model

If you need to retrain the memory prediction model with new data:

1. Run a parameter sweep with `run_training_sweep.sh` or similar, collecting memory usage statistics
2. Use the training scripts in `utils/memory_helpers.py` to fit a new linear model
3. Save the new weights with `save_model_weights(weights, "model_weights_<date>.json")`
4. Update the `model_weights_latest.json` symlink or filename reference

See comments in `utils/memory_helpers.py` for more details on the model training workflow.
