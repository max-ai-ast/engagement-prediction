import importlib

import pandas as pd


stage_evaluate = importlib.import_module("utils.04_evaluate.stage_evaluate")
EvalContext = stage_evaluate.EvalContext
ColdStartCurvesModule = importlib.import_module("utils._evals.cold_start_curves").ColdStartCurvesModule
PerformanceInequalityModule = importlib.import_module("utils._evals.performance_inequality").PerformanceInequalityModule


def _ranking_rows_df():
    return pd.DataFrame({
        "did": ["u1", "u1", "u2"],
        "split": ["holdout_unseen_users", "holdout_unseen_users", "holdout_unseen_users"],
        "like_hour_bucket": ["2026-05-01T00:00:00Z", "2026-05-01T01:00:00Z", "2026-05-01T00:00:00Z"],
        "num_embedding_likes": [1, 3, 1],
        "num_total_likes": [5, 5, 2],
        "candidate_count": [4, 5, 4],
        "positive_count": [1, 2, 1],
        "positive_rank_mean": [1.0, 2.0, 3.0],
        "ndcg@1": [1.0, 0.5, 0.0],
        "recall@1": [1.0, 0.5, 0.0],
        "average_precision": [1.0, 0.75, 0.5],
        "auc_roc": [1.0, 0.8, 0.4],
    })


def _ctx(tmp_path):
    ranking_rows_df = _ranking_rows_df()
    user_metadata_df = stage_evaluate.compute_user_metadata_from_ranking_rows(ranking_rows_df)
    return EvalContext(
        predictions_df=pd.DataFrame(columns=["did", "post_id", "y_true", "y_pred_proba"]),
        user_metadata_df=user_metadata_df,
        output_dir=tmp_path,
        timestamp="20260527_000000",
        config={},
        ranking_rows_df=ranking_rows_df,
    )


def test_compute_user_metadata_from_ranking_rows_uses_user_maxima():
    metadata = stage_evaluate.compute_user_metadata_from_ranking_rows(_ranking_rows_df())

    assert metadata.set_index("did").loc["u1", "num_embedding_likes"] == 3
    assert metadata.set_index("did").loc["u1", "num_total_likes"] == 5
    assert metadata.set_index("did").loc["u2", "num_embedding_likes"] == 1


def test_cold_start_curves_accepts_ranking_rows(tmp_path):
    result = ColdStartCurvesModule().run(_ctx(tmp_path))

    assert result["total_ranking_rows_analyzed"] == 3
    assert result["total_users_analyzed"] == 2
    assert (tmp_path / "cold_start_curves" / "binned_metrics.csv").exists()
    assert (tmp_path / "cold_start_curves" / "ndcg_at_1_cold_start.png").exists()


def test_performance_inequality_accepts_ranking_rows(tmp_path):
    result = PerformanceInequalityModule().run(_ctx(tmp_path))

    assert result["total_users"] == 2
    assert result["total_ranking_rows"] == 3
    assert "gini_ndcg@1" in result
    assert (tmp_path / "performance_inequality" / "per_user_metrics.csv").exists()
    assert (tmp_path / "performance_inequality" / "lorenz_ndcg_at_1.png").exists()