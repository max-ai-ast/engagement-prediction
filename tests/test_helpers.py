from datetime import datetime, timezone

import numpy as np
import pandas as pd
import polars as pl
import pytest

from utils.helpers import (
    find_join_key,
    find_text_column,
    parse_one_ts,
    validate_dataframe_schema,
    compute_memory_model_features,
    predict_memory_gb,
    MEMORY_MODEL_COEFFICIENTS,
    MEMORY_MODEL_FEATURE_NAMES,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2024-02-10T13:45:00+0000", datetime(2024, 2, 10, 13, 45, tzinfo=timezone.utc)),
        ("2024-02-10T13:45:00", datetime(2024, 2, 10, 13, 45, tzinfo=timezone.utc)),
        ("2024-02-10", datetime(2024, 2, 10, 0, 0, tzinfo=timezone.utc)),
    ],
)
def test_parse_one_ts_accepts_known_formats(raw: str, expected: datetime):
    assert parse_one_ts(raw) == expected


def test_parse_one_ts_returns_none_for_missing():
    assert parse_one_ts(None) is None


def test_parse_one_ts_rejects_unrecognized_format():
    with pytest.raises(ValueError):
        parse_one_ts("10-02-2024 13:45")


def test_find_join_key_prefers_known_columns():
    posts = pd.DataFrame({"commit_cid": ["c1", "c2"], "other": [1, 2]})
    likes = pd.DataFrame({"subject_cid": ["c1", "c3"], "other": [2, 3]})

    assert find_join_key(posts, likes) == ("subject_cid", "commit_cid")


def test_find_join_key_falls_back_to_common_overlap():
    posts = pd.DataFrame({"post_id": ["p1", "p2"], "text": ["foo", "bar"]})
    likes = pd.DataFrame({"post_id": ["p2", "p3"], "user": ["u1", "u2"]})

    assert find_join_key(posts, likes) == ("post_id", "post_id")


def test_find_text_column_prefers_record_text_when_present():
    posts = pd.DataFrame({"record_text": ["body"], "titleText": ["t1"]})

    assert find_text_column(posts) == "record_text"


def test_find_text_column_falls_back_to_first_text_column():
    posts = pd.DataFrame({"titleText": ["t1"], "some_text_field": ["body"]})

    assert find_text_column(posts) == "titleText"


def test_validate_dataframe_schema_accepts_matching_schema():
    df = pd.DataFrame(
        {
            "did": ["u1", "u2"],
            "liked": [1, 0],
            "created_at": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    schema = {"did": str, "liked": int, "created_at": "datetime64[ns]"}

    validate_dataframe_schema(df, schema, allow_extra_columns=True)


def test_validate_dataframe_schema_accepts_polars_lazyframe():
    lf = pl.DataFrame(
        {
            "did": ["u1", "u2"],
            "liked": [1, 0],
            "created_at": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
        }
    ).lazy()
    schema = {"did": str, "liked": int, "created_at": "datetime64[ns]"}

    validate_dataframe_schema(lf, schema, allow_extra_columns=True)


def test_validate_dataframe_schema_rejects_missing_columns():
    df = pd.DataFrame({"did": ["u1"]})
    schema = {"did": str, "liked": int}

    with pytest.raises(ValueError, match="Missing columns"):
        validate_dataframe_schema(df, schema)


def test_validate_dataframe_schema_rejects_missing_columns_polars():
    df = pl.DataFrame({"did": ["u1"]})
    schema = {"did": str, "liked": int}

    with pytest.raises(ValueError, match="Missing columns"):
        validate_dataframe_schema(df, schema)


def test_validate_dataframe_schema_rejects_dtype_mismatch():
    df = pd.DataFrame(
        {
            "liked": ["yes", "no"],
            "created_at": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    schema = {"liked": int, "created_at": "datetime64[ns]"}

    with pytest.raises(ValueError, match="Dtype mismatches"):
        validate_dataframe_schema(df, schema)


def test_validate_dataframe_schema_rejects_dtype_mismatch_polars():
    df = pl.DataFrame({"liked": ["yes", "no"]})
    schema = {"liked": int}

    with pytest.raises(ValueError, match="Dtype mismatches"):
        validate_dataframe_schema(df, schema)


def test_validate_dataframe_schema_rejects_unexpected_columns_when_strict():
    df = pd.DataFrame({"did": ["u1"], "liked": [1], "extra": [True]})
    schema = {"did": str, "liked": int}

    with pytest.raises(ValueError, match="Unexpected columns"):
        validate_dataframe_schema(df, schema, allow_extra_columns=False)


def test_validate_dataframe_schema_rejects_unexpected_columns_when_strict_polars():
    df = pl.DataFrame({"did": ["u1"], "liked": [1], "extra": [True]})
    schema = {"did": str, "liked": int}

    with pytest.raises(ValueError, match="Unexpected columns"):
        validate_dataframe_schema(df, schema, allow_extra_columns=False)


# =============================================================================
# Memory Estimation Model Tests
# =============================================================================

class TestMemoryModelFeatures:
    """Tests for compute_memory_model_features function."""

    def test_computes_all_features(self):
        """All expected feature names are present in output."""
        features = compute_memory_model_features(
            data_window_days=7,
            max_liking_users=10000,
            max_likes_per_user=100,
            negative_posts_sample=50000,
            likes_initial=50_000_000,
        )
        
        for name in MEMORY_MODEL_FEATURE_NAMES:
            assert name in features, f"Missing feature: {name}"

    def test_feature_scaling(self):
        """Features are scaled as expected."""
        features = compute_memory_model_features(
            data_window_days=14,
            max_liking_users=50000,
            max_likes_per_user=500,
            negative_posts_sample=100000,
            likes_initial=100_000_000,
        )
        
        # Check scaling factors
        assert features['data_window_days'] == 14
        assert features['max_liking_users_10k'] == 5.0  # 50000 / 10000
        assert features['max_likes_per_user_100'] == 5.0  # 500 / 100
        assert features['negative_posts_sample_10k'] == 10.0  # 100000 / 10000
        assert features['log_max_liking_users'] == pytest.approx(np.log10(50000), rel=1e-6)
        assert features['sqrt_likes_initial_1e6'] == pytest.approx(np.sqrt(100_000_000) / 1000, rel=1e-6)

    def test_interaction_terms(self):
        """Interaction terms are computed correctly."""
        features = compute_memory_model_features(
            data_window_days=7,
            max_liking_users=20000,
            max_likes_per_user=100,
            negative_posts_sample=10000,
            likes_initial=50_000_000,
        )
        
        # days_x_users_10k = 7 * 20000 / 10000 = 14
        assert features['days_x_users_10k'] == pytest.approx(14.0, rel=1e-6)
        
        # users_x_log_users = (20000 / 10000) * log10(20000) = 2 * 4.301 = 8.602
        expected = 2.0 * np.log10(20000)
        assert features['users_x_log_users'] == pytest.approx(expected, rel=1e-6)


class TestPredictMemoryGB:
    """Tests for predict_memory_gb function."""

    def test_returns_positive_value(self):
        """Prediction is always positive."""
        features = compute_memory_model_features(
            data_window_days=7,
            max_liking_users=10000,
            max_likes_per_user=100,
            negative_posts_sample=10000,
            likes_initial=50_000_000,
        )
        
        prediction = predict_memory_gb(features)
        assert prediction >= 1.0

    def test_prediction_in_reasonable_range(self):
        """Prediction is in expected range for typical configurations."""
        # Small config (7 days, 10k users)
        features_small = compute_memory_model_features(
            data_window_days=7,
            max_liking_users=10000,
            max_likes_per_user=100,
            negative_posts_sample=10000,
            likes_initial=50_000_000,
        )
        pred_small = predict_memory_gb(features_small)
        
        # Large config (21 days, 500k users)
        features_large = compute_memory_model_features(
            data_window_days=21,
            max_liking_users=500000,
            max_likes_per_user=1000,
            negative_posts_sample=100000,
            likes_initial=170_000_000,
        )
        pred_large = predict_memory_gb(features_large)
        
        # Small config should be in ~20-40 GB range
        assert 15 < pred_small < 50, f"Small config prediction {pred_small} out of expected range"
        
        # Large config should be in ~60-120 GB range
        assert 50 < pred_large < 150, f"Large config prediction {pred_large} out of expected range"
        
        # Large should be significantly more than small
        assert pred_large > pred_small * 1.5

    def test_custom_coefficients(self):
        """Can use custom coefficients."""
        features = compute_memory_model_features(
            data_window_days=7,
            max_liking_users=10000,
            max_likes_per_user=100,
            negative_posts_sample=10000,
            likes_initial=50_000_000,
        )
        
        # Use simple coefficients for testing
        simple_coeffs = {'intercept': 10.0}
        for name in MEMORY_MODEL_FEATURE_NAMES:
            simple_coeffs[name] = 0.0
        
        prediction = predict_memory_gb(features, coefficients=simple_coeffs)
        assert prediction == 10.0

    def test_coefficients_exist_for_all_features(self):
        """Model coefficients exist for all feature names."""
        assert 'intercept' in MEMORY_MODEL_COEFFICIENTS
        for name in MEMORY_MODEL_FEATURE_NAMES:
            assert name in MEMORY_MODEL_COEFFICIENTS, f"Missing coefficient for: {name}"


class TestMemoryModelAccuracy:
    """Tests verifying model accuracy against known sweep results."""

    # Sample sweep results for validation (from sweep_results.csv)
    SWEEP_SAMPLES = [
        # (data_window_days, max_liking_users, max_likes_per_user, negative_posts_sample, likes_initial, actual_peak_gb)
        (7, 10000, 100, 10000, 48682613, 27.35),
        (14, 10000, 100, 10000, 136957557, 45.40),
        (21, 10000, 100, 10000, 172152261, 50.31),
        (7, 50000, 100, 50000, 48682613, 31.06),
        (14, 100000, 100, 50000, 136957557, 51.49),
        (21, 250000, 100, 50000, 172152261, 69.68),
    ]

    @pytest.mark.parametrize("sample", SWEEP_SAMPLES)
    def test_prediction_within_25_percent(self, sample):
        """Model predictions are within 25% of actual values from sweep."""
        days, users, likes_cap, neg_sample, likes_init, actual_gb = sample
        
        features = compute_memory_model_features(
            data_window_days=days,
            max_liking_users=users,
            max_likes_per_user=likes_cap,
            negative_posts_sample=neg_sample,
            likes_initial=likes_init,
        )
        
        predicted_gb = predict_memory_gb(features)
        error_pct = abs(predicted_gb - actual_gb) / actual_gb * 100
        
        assert error_pct < 25, (
            f"Prediction error {error_pct:.1f}% exceeds 25% threshold. "
            f"Predicted: {predicted_gb:.2f} GB, Actual: {actual_gb:.2f} GB"
        )
