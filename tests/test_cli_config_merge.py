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

    output_root = Path(tmp_path) / "out"
    run_dir = output_root / "runs" / "run"
    initial_log = run_dir / "run-all.log"
    cfg = cli._build_effective_config_for_background_run(
        merged, output_root=output_root, initial_log=initial_log
    )

    assert cfg["use_post_encoder"] is False
    assert cfg["background"] is False
    assert cfg["output_dir"] == str(output_root.resolve())
    assert cfg["_initial_log"] == str(initial_log)


def test_merge_args_with_config_defaults_l2_normalize_embeddings_to_false():
    parser = cli.build_parser()
    raw = parser.parse_args([])
    merged = cli._merge_args_with_config(raw)

    assert merged.l2_normalize_embeddings is False


def test_mlp_allows_cross_attention_user_encoder():
    parser = cli.build_parser()
    raw = parser.parse_args(["--model-type", "mlp", "--user-encoder", "cross_attention"])
    merged = cli._merge_args_with_config(raw)

    assert merged.user_encoder in cli.VALID_USER_ENCODERS_BY_MODEL_TYPE[merged.model_type]


def test_background_effective_config_preserves_no_l2_normalize_embeddings(tmp_path):
    parser = cli.build_parser()
    raw = parser.parse_args(["--no-l2-normalize-embeddings"])
    merged = cli._merge_args_with_config(raw)

    output_root = Path(tmp_path) / "out"
    run_dir = output_root / "runs" / "run"
    initial_log = run_dir / "run-all.log"
    cfg = cli._build_effective_config_for_background_run(
        merged, output_root=output_root, initial_log=initial_log
    )

    assert cfg["l2_normalize_embeddings"] is False


def test_background_effective_config_allows_cli_to_override_config_to_default(tmp_path):
    # Config disables post encoder, CLI re-enables it (even though True is the DEFAULTS value).
    config_path = Path(tmp_path) / "config.yml"
    config_path.write_text("use_post_encoder: false\n")

    parser = cli.build_parser()
    raw = parser.parse_args(["--config", str(config_path), "--post-encoder"])
    merged = cli._merge_args_with_config(raw)

    output_root = Path(tmp_path) / "out"
    run_dir = output_root / "runs" / "run"
    initial_log = run_dir / "run-all.log"
    cfg = cli._build_effective_config_for_background_run(
        merged, output_root=output_root, initial_log=initial_log
    )

    assert cfg["use_post_encoder"] is True


def test_background_effective_config_allows_cli_to_override_config_to_default_l2_normalization(tmp_path):
    config_path = Path(tmp_path) / "config.yml"
    config_path.write_text("l2_normalize_embeddings: false\n")

    parser = cli.build_parser()
    raw = parser.parse_args(["--config", str(config_path), "--l2-normalize-embeddings"])
    merged = cli._merge_args_with_config(raw)

    output_root = Path(tmp_path) / "out"
    run_dir = output_root / "runs" / "run"
    initial_log = run_dir / "run-all.log"
    cfg = cli._build_effective_config_for_background_run(
        merged, output_root=output_root, initial_log=initial_log
    )

    assert cfg["l2_normalize_embeddings"] is True


@pytest.mark.parametrize(
    ("config_key", "disable_flag"),
    [
        ("dataloader_pin_memory", "--no-dataloader-pin-memory"),
        ("dataloader_persistent_workers", "--no-dataloader-persistent-workers"),
        ("background", "--no-background"),
    ],
)
def test_merge_args_with_config_allows_cli_to_disable_true_config_bool(tmp_path, config_key, disable_flag):
    config_path = Path(tmp_path) / "config.yml"
    config_path.write_text(f"{config_key}: true\n")

    parser = cli.build_parser()
    raw = parser.parse_args(["--config", str(config_path), disable_flag])
    merged = cli._merge_args_with_config(raw)

    assert getattr(merged, config_key) is False


def test_merge_args_with_config_accepts_prior_pins(tmp_path):
    config_path = Path(tmp_path) / "config.yml"
    config_path.write_text(
        textwrap.dedent(
            """
            prior_01_get_data: 20260101_000000_deadbeef
            prior_02_target_posts: 20260102_000000_cafebabe
            """
        ).strip()
        + "\n"
    )

    parser = cli.build_parser()
    raw = parser.parse_args(["--config", str(config_path)])
    merged = cli._merge_args_with_config(raw)

    assert merged.prior_01_get_data == "20260101_000000_deadbeef"
    assert merged.prior_02_target_posts == "20260102_000000_cafebabe"
