from pathlib import Path
import textwrap

import pytest

import cli


@pytest.mark.parametrize(
    "argv",
    [
        # New behavior: `run-all` is optional.
        ["--config", "{config}", "--epochs", "7", "--batch-size", "512"],
        # Backwards compatible: still accepts `run-all`.
        ["--config", "{config}", "run-all", "--epochs", "7", "--batch-size", "512"],
    ],
)
def test_merge_args_with_config_prioritizes_cli_over_config(tmp_path, argv):
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
    args = parser.parse_args([a.format(config=str(config_path)) for a in argv])

    merged = cli._merge_args_with_config(args)

    assert merged.epochs == 7  # CLI overrides config
    assert merged.embedding_model == "all_MiniLM_L12_v2"  # Config overrides defaults
    assert merged.batch_size == 512  # CLI overrides default
    assert merged.learning_rate == cli.DEFAULTS["learning_rate"]


@pytest.mark.parametrize(
    "argv",
    [
        ["--config", "{config}"],
        ["--config", "{config}", "run-all"],
    ],
)
def test_merge_args_with_config_rejects_unknown_keys(tmp_path, argv):
    config_path = Path(tmp_path) / "invalid.yml"
    config_path.write_text("unknown_flag: true\n")

    parser = cli.build_parser()
    args = parser.parse_args([a.format(config=str(config_path)) for a in argv])

    with pytest.raises(ValueError):
        cli._merge_args_with_config(args)


def test_background_effective_config_preserves_no_post_encoder(tmp_path):
    parser = cli.build_parser()
    raw = parser.parse_args(["--no-post-encoder"])
    merged = cli._merge_args_with_config(raw)

    run_dir = Path(tmp_path) / "run"
    initial_log = run_dir / "run-all.log"
    cfg = cli._build_effective_config_for_background_run(
        merged, run_dir=run_dir, initial_log=initial_log
    )

    assert cfg["use_post_encoder"] is False
    assert cfg["foreground"] is True
    assert cfg["output_dir"] == str(run_dir.resolve())
    assert cfg["_initial_log"] == str(initial_log)


def test_background_effective_config_allows_cli_to_override_config_to_default(tmp_path):
    # Config disables post encoder, CLI re-enables it (even though True is the DEFAULTS value).
    config_path = Path(tmp_path) / "config.yml"
    config_path.write_text("use_post_encoder: false\n")

    parser = cli.build_parser()
    raw = parser.parse_args(["--config", str(config_path), "--post-encoder"])
    merged = cli._merge_args_with_config(raw)

    run_dir = Path(tmp_path) / "run"
    initial_log = run_dir / "run-all.log"
    cfg = cli._build_effective_config_for_background_run(
        merged, run_dir=run_dir, initial_log=initial_log
    )

    assert cfg["use_post_encoder"] is True
