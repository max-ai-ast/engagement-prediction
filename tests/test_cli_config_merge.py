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


def test_merge_args_with_config_rejects_unknown_keys(tmp_path):
    config_path = Path(tmp_path) / "invalid.yml"
    config_path.write_text("unknown_flag: true\n")

    parser = cli.build_parser()
    args = parser.parse_args(["--config", str(config_path)])

    with pytest.raises(ValueError):
        cli._merge_args_with_config(args)
