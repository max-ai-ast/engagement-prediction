# Agent guidelines

This file documents conventions for AI agents and contributors working on this repo.

## Default values: single source of truth

**All default values for pipeline/training parameters live in `cli.py`** in the `DEFAULTS` dict. The CLI merges user config and CLI flags with `DEFAULTS` to produce the final `args` namespace.

- **Do not add default values** to model or encoder `__init__` parameters that are driven by the CLI (e.g. `user_hidden_dim`, `user_output_dim`, `num_attention_heads`, `num_attention_layers`, `max_history_len`, `attention_dropout`, `shared_dim`, `post_hidden_dim`, `dropout_rate`, `user_encoder_type`). Require callers to pass these explicitly.
- **Do not duplicate defaults** in stage scripts, dataloaders, or model classes. The only place to define defaults for run-all parameters is `cli.py`.
- When adding a new CLI-controlled hyperparameter: add it to `DEFAULTS` in `cli.py`, add the corresponding `--flag` to the run-all parser, and pass the value from `args` into the model/encoder constructors without defining a default in those constructors.

This keeps a single source of truth and avoids drift between CLI defaults and in-code defaults.
