from datetime import datetime, timezone

import pandas as pd
import pytest

from utils.helpers import (
    find_join_key,
    find_text_column,
    parse_one_ts,
    validate_dataframe_schema,
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


def test_validate_dataframe_schema_rejects_missing_columns():
    df = pd.DataFrame({"did": ["u1"]})
    schema = {"did": str, "liked": int}

    with pytest.raises(ValueError, match="Missing columns"):
        validate_dataframe_schema(df, schema)


def test_validate_dataframe_schema_rejects_dtype_mismatch():
    df = pd.DataFrame({"liked": ["yes", "no"]})
    schema = {"liked": int}

    with pytest.raises(ValueError, match="Dtype mismatches"):
        validate_dataframe_schema(df, schema)


def test_validate_dataframe_schema_rejects_unexpected_columns_when_strict():
    df = pd.DataFrame({"did": ["u1"], "liked": [1], "extra": [True]})
    schema = {"did": str, "liked": int}

    with pytest.raises(ValueError, match="Unexpected columns"):
        validate_dataframe_schema(df, schema, allow_extra_columns=False)
