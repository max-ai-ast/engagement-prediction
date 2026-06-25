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


def test_negative_samples_per_hour_replaces_old_negative_posts_sample(tmp_path):
    parser = cli.build_parser()
    raw = parser.parse_args(["--negative-samples-per-hour", "123"])
    merged = cli._merge_args_with_config(raw)

    assert merged.negative_samples_per_hour == 123
    assert cli.DEFAULTS["negative_samples_per_hour"] == 1000
    assert "negative_posts_sample" not in cli.DEFAULTS

    with pytest.raises(SystemExit):
        parser.parse_args(["--negative-posts-sample", "123"])

    config_path = Path(tmp_path) / "old.yml"
    config_path.write_text("negative_posts_sample: 123\n")
    raw = parser.parse_args(["--config", str(config_path)])
    with pytest.raises(ValueError):
        cli._merge_args_with_config(raw)


def test_negative_sampling_popularity_args_merge_from_cli_and_config(tmp_path):
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--negative-sampling-alpha", "0.25",
        "--min-likes-per-negative-post", "12",
    ])
    merged = cli._merge_args_with_config(raw)

    assert merged.negative_sampling_alpha == 0.25
    assert merged.min_likes_per_negative_post == 12
    assert cli.DEFAULTS["negative_sampling_alpha"] == 0.5
    assert cli.DEFAULTS["min_likes_per_negative_post"] == 50

    config_path = Path(tmp_path) / "negative_sampling.yml"
    config_path.write_text("negative_sampling_alpha: 0.75\nmin_likes_per_negative_post: 80\n")
    raw = parser.parse_args(["--config", str(config_path), "--negative-sampling-alpha", "0.4"])
    merged = cli._merge_args_with_config(raw)

    assert merged.negative_sampling_alpha == 0.4
    assert merged.min_likes_per_negative_post == 80


def test_user_sampling_args_replace_old_names(tmp_path):
    parser = cli.build_parser()
    raw = parser.parse_args(["--max-trainval-users", "123", "--max-unseen-eval-users", "45"])
    merged = cli._merge_args_with_config(raw)

    assert merged.max_trainval_users == 123
    assert merged.max_unseen_eval_users == 45
    assert "max_liking_users" not in cli.DEFAULTS
    assert "holdout_user_fraction" not in cli.DEFAULTS

    with pytest.raises(SystemExit):
        parser.parse_args(["--max-liking-users", "123"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--holdout-user-fraction", "0.2"])

    config_path = Path(tmp_path) / "old_sampling.yml"
    config_path.write_text("max_liking_users: 123\nholdout_user_fraction: 0.2\n")
    raw = parser.parse_args(["--config", str(config_path)])
    with pytest.raises(ValueError):
        cli._merge_args_with_config(raw)


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


def test_two_tower_rejects_summarized_user_encoder_before_running_stages(tmp_path):
    parser = cli.build_parser()
    raw = parser.parse_args(["--model-type", "two-tower", "--user-encoder", "summarized"])
    merged = cli._merge_args_with_config(raw)
    merged.output_dir = str(tmp_path)
    ctx = cli.Context(
        run_dir=Path(tmp_path) / "runs" / "run",
        artifacts_dir=Path(tmp_path) / "artifacts",
        runs_dir=Path(tmp_path) / "runs",
        pipeline_run_id="run",
    )

    with pytest.raises(ValueError, match="user-encoder 'summarized'"):
        cli.cmd__run_all_exec(merged, ctx)


def test_mlp_allows_author_embedding_table():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "mlp",
        "--user-encoder", "summarized",
        "--use-author-embedding-table",
        "--author-embedding-dim", "8",
    ])
    merged = cli._merge_args_with_config(raw)

    assert merged.use_author_embedding_table is True
    assert merged.model_type == "mlp"


def test_bst_ranker_model_type_maps_train_alias():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--start-from", "train",
        "--stop-after", "train",
        "--use-author-embedding-table",
        "--prediction-hidden-dims", "144", "72",
    ])
    merged = cli._merge_args_with_config(raw)

    train_key = cli._get_train_key(merged.model_type)
    stage_order = cli._get_stage_order_for_model_type(train_key)
    start_idx, stop_idx, includes_train = cli._get_stage_folder_and_start_stop_indices(
        stage_order,
        merged.start_from,
        merged.stop_after,
        train_key,
    )

    assert train_key == "train_bst_ranker"
    assert stage_order[start_idx] == "train_bst_ranker"
    assert stage_order[stop_idx] == "train_bst_ranker"
    assert includes_train is True
    assert merged.bst_num_transformer_layers == 1


def test_bst_ranker_explicit_train_stage_names_parse():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--start-from", "train_bst_ranker",
        "--stop-after", "train_bst_ranker",
        "--use-author-embedding-table",
        "--prediction-hidden-dims", "144", "72",
    ])
    merged = cli._merge_args_with_config(raw)

    assert merged.start_from == "train_bst_ranker"
    assert merged.stop_after == "train_bst_ranker"


def test_merge_args_with_config_accepts_bst_ranker_keys(tmp_path):
    config_path = Path(tmp_path) / "bst.yml"
    config_path.write_text(
        textwrap.dedent(
            """
            model_type: bst-ranker
            use_author_embedding_table: true
            bst_model_dim: 96
            content_projection_dim: 80
            author_projection_dim: 24
            bst_time_embedding_dim: 32
            bst_num_attention_heads: 8
            bst_num_transformer_layers: 1
            bst_transformer_ff_dim: 384
            bst_dropout_rate: 0.2
            bst_norm_first: true
            bst_time_delta_bucket_boundaries_hours: [1, 2, 4]
            prediction_hidden_dims: [128, 64]
            bst_weight_decay: 0.02
            bst_additional_batch_negatives: 32
            batch_size: 16
            bst_max_train_batches_per_epoch: 5
            """
        ).strip()
        + "\n"
    )

    parser = cli.build_parser()
    raw = parser.parse_args(["--config", str(config_path)])
    merged = cli._merge_args_with_config(raw)

    assert merged.model_type == "bst-ranker"
    assert merged.bst_model_dim == 96
    assert merged.content_projection_dim == 80
    assert merged.author_projection_dim == 24
    assert merged.bst_time_embedding_dim == 32
    assert merged.bst_num_attention_heads == 8
    assert merged.prediction_hidden_dims == [128, 64]
    assert merged.bst_additional_batch_negatives == 32
    assert merged.batch_size == 16
    assert merged.bst_max_train_batches_per_epoch == 5
    cli._validate_bst_config(merged)


def test_bst_ranker_training_defaults():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--use-author-embedding-table",
    ])
    merged = cli._merge_args_with_config(raw)

    assert merged.bst_additional_batch_negatives == 64
    assert merged.batch_size == cli.DEFAULTS["batch_size"]
    assert merged.bst_max_train_batches_per_epoch is None
    cli._validate_bst_config(merged)


def test_bst_ranker_requires_one_transformer_layer():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--use-author-embedding-table",
        "--bst-num-transformer-layers", "2",
    ])
    merged = cli._merge_args_with_config(raw)

    with pytest.raises(ValueError, match="requires --bst-num-transformer-layers=1"):
        cli._validate_bst_config(merged)


@pytest.mark.parametrize(
    ("flag", "message"),
    [
        ("--bst-additional-batch-negatives", "bst-additional-batch-negatives"),
        ("--batch-size", "batch-size"),
        ("--bst-max-train-batches-per-epoch", "bst-max-train-batches-per-epoch"),
    ],
)
def test_bst_ranker_rejects_non_positive_listwise_training_controls(flag, message):
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--use-author-embedding-table",
        flag, "0",
    ])
    merged = cli._merge_args_with_config(raw)

    with pytest.raises(ValueError, match=message):
        cli._validate_bst_config(merged)


def test_bst_ranker_requires_author_embedding_table():
    parser = cli.build_parser()
    raw = parser.parse_args(["--model-type", "bst-ranker", "--prediction-hidden-dims", "144", "72"])
    merged = cli._merge_args_with_config(raw)

    with pytest.raises(ValueError, match="use-author-embedding-table"):
        cli._validate_bst_config(merged)


def test_bst_ranker_requires_prediction_hidden_dims():
    parser = cli.build_parser()
    raw = parser.parse_args(["--model-type", "bst-ranker", "--use-author-embedding-table"])
    merged = cli._merge_args_with_config(raw)
    merged.prediction_hidden_dims = None

    with pytest.raises(ValueError, match="prediction-hidden-dims"):
        cli._validate_bst_config(merged)


def test_bst_ranker_accepts_explicit_empty_prediction_hidden_dims():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--use-author-embedding-table",
        "--prediction-hidden-dims",
    ])
    merged = cli._merge_args_with_config(raw)

    assert merged.prediction_hidden_dims == []
    cli._validate_bst_config(merged)


@pytest.mark.parametrize(
    ("arg_name", "error_match"),
    [
        ("content_projection_dim", "content-projection-dim"),
        ("author_projection_dim", "author-projection-dim"),
    ],
)
def test_bst_ranker_validates_projection_dims(arg_name, error_match):
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--use-author-embedding-table",
        "--prediction-hidden-dims", "144", "72",
    ])
    merged = cli._merge_args_with_config(raw)
    setattr(merged, arg_name, 0)

    with pytest.raises(ValueError, match=error_match):
        cli._validate_bst_config(merged)


def test_bst_ranker_validates_transformer_head_divisibility():
    parser = cli.build_parser()
    raw = parser.parse_args([
        "--model-type", "bst-ranker",
        "--use-author-embedding-table",
        "--bst-time-embedding-dim", "15",
        "--prediction-hidden-dims", "144", "72",
    ])
    merged = cli._merge_args_with_config(raw)

    with pytest.raises(ValueError, match="divisible"):
        cli._validate_bst_config(merged)


def test_min_author_support_must_be_positive_even_without_author_table(tmp_path):
    parser = cli.build_parser()
    raw = parser.parse_args(["--min-author-support", "0", "--stop-after", "get_data"])
    merged = cli._merge_args_with_config(raw)
    merged.output_dir = str(tmp_path)
    ctx = cli.Context(
        run_dir=Path(tmp_path) / "runs" / "run",
        artifacts_dir=Path(tmp_path) / "artifacts",
        runs_dir=Path(tmp_path) / "runs",
        pipeline_run_id="run",
    )

    with pytest.raises(ValueError, match="min-author-support"):
        cli.cmd__run_all_exec(merged, ctx)


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
            prior_02_user_history: 20260102_000000_cafebabe
            """
        ).strip()
        + "\n"
    )

    parser = cli.build_parser()
    raw = parser.parse_args(["--config", str(config_path)])
    merged = cli._merge_args_with_config(raw)

    assert merged.prior_01_get_data == "20260101_000000_deadbeef"
    assert merged.prior_02_user_history == "20260102_000000_cafebabe"
