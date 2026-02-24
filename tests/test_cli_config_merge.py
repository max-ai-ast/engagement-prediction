from pathlib import Path
import textwrap

import pytest

import cli


def test_merge_args_with_config_prioritizes_cli_over_config(tmp_path):
    config_path = Path(tmp_path) / "config.yml"
    config_path.write_text(
        textwrap.dedent(
            """
            epochs: 5
            embedding_model: all_MiniLM_L12_v2
            """
        ).strip()
    )

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--config",
            str(config_path),
            "run-all",
            "--epochs",
            "7",
            "--batch-size",
            "512",
        ]
    )

    merged = cli._merge_args_with_config(args)

    assert merged.epochs == 7  # CLI overrides config
    assert merged.embedding_model == "all_MiniLM_L12_v2"  # Config overrides defaults
    assert merged.batch_size == 512  # CLI overrides default
    assert merged.learning_rate == cli.DEFAULTS["learning_rate"]


def test_background_arg_reconstruction_produces_only_recognized_flags():
    """Ensure that all DEFAULTS keys produce valid CLI flags when reconstructed for the nohup child."""
    parser = cli.build_parser()
    raw_args = parser.parse_args(["run-all"])
    args = cli._merge_args_with_config(raw_args)

    # Simulate the background arg reconstruction (mirrors cmd_run_all logic)
    cli_args = []
    for k, v in vars(args).items():
        if k in ("command", "foreground", "_initial_log", "output_dir", "func"):
            continue
        if v is None or v is False:
            continue
        opt = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            cli_args.append(opt)
        elif isinstance(v, list):
            cli_args.extend([opt] + [str(x) for x in v])
        else:
            cli_args.extend([opt, str(v)])
    cli_args.extend(["--foreground", "--output-dir", "/tmp/test_run"])

    # The child must be able to parse all reconstructed args without error
    child_raw = parser.parse_args(["run-all"] + cli_args)
    child_args = cli._merge_args_with_config(child_raw)

    assert child_args.foreground is True
    assert child_args.use_post_encoder == cli.DEFAULTS["use_post_encoder"]
    assert child_args.experiment_tracker == cli.DEFAULTS["experiment_tracker"]


def test_background_arg_reconstruction_preserves_use_post_encoder():
    """Confirm --use-post-encoder and --no-use-post-encoder are both accepted."""
    parser = cli.build_parser()

    args_enabled = cli._merge_args_with_config(
        parser.parse_args(["run-all", "--use-post-encoder"])
    )
    assert args_enabled.use_post_encoder is True

    args_disabled = cli._merge_args_with_config(
        parser.parse_args(["run-all", "--no-use-post-encoder"])
    )
    assert args_disabled.use_post_encoder is False


def test_merge_args_with_config_rejects_unknown_keys(tmp_path):
    config_path = Path(tmp_path) / "invalid.yml"
    config_path.write_text("unknown_flag: true\n")

    parser = cli.build_parser()
    args = parser.parse_args(["--config", str(config_path), "run-all"])

    with pytest.raises(ValueError):
        cli._merge_args_with_config(args)
