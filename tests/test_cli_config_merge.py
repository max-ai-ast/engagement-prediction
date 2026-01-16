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
            data_source: digitalocean
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
            "--relevel-method",
            "simple",
        ]
    )

    merged = cli._merge_args_with_config(args)

    assert merged.epochs == 7  # CLI overrides config
    assert merged.data_source == "digitalocean"  # Config overrides defaults
    assert merged.embedding_model == "all_MiniLM_L12_v2"
    assert merged.relevel_method == "simple"
    assert merged.batch_size == cli.DEFAULTS["batch_size"]


def test_merge_args_with_config_rejects_unknown_keys(tmp_path):
    config_path = Path(tmp_path) / "invalid.yml"
    config_path.write_text("unknown_flag: true\n")

    parser = cli.build_parser()
    args = parser.parse_args(["--config", str(config_path), "run-all"])

    with pytest.raises(ValueError):
        cli._merge_args_with_config(args)
