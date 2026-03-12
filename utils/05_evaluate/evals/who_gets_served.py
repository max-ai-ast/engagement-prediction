#!/usr/bin/env python3

"""
Who-Gets-Well-Served Evaluation Module

Tests the hypothesis that users whose content preferences align with the
population-level signal ("mainstream" tastes) receive better model
performance, while users with distinctive preferences are underserved.

Approach:
1. Build a per-user trait-preference vector (mean trait value of liked posts).
2. Compute "mainstreamness" = cosine similarity between each user's
   preference vector and the population-mean preference vector.
3. Correlate mainstreamness with per-user performance metrics (recall, F1,
   precision).

Outputs (under who_gets_served/):
- mainstreamness_vs_<metric>.png:              scatter per metric
- performance_by_mainstreamness_quintile.png:  grouped bar chart
- who_gets_served_summary.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import spearmanr

from . import EvalContext, EvalModule, compute_per_user_metrics
from .trait_corrs import _load_inferences, _unnest_text_inferences
from .trait_amplification import MIN_USER_POSTS, _filter_eligible_users

MIN_FINITE_TRAITS = 10
METRICS = ["recall", "f1", "precision"]

_METRIC_COLORS = {
    "recall": "#ff7f0e",
    "f1": "#9467bd",
    "precision": "#2ca02c",
}


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _build_preference_vectors(
    user_ids: np.ndarray,
    user_to_rows: Dict[Any, np.ndarray],
    y_true: np.ndarray,
    trait_arrays: Dict[str, np.ndarray],
    finite_masks: Dict[str, np.ndarray],
    trait_keys: List[str],
) -> Dict[Any, np.ndarray]:
    """Per-user preference vector: mean trait value among liked posts.

    Returns {did: np.array of length n_traits} for users with enough data.
    """
    n_traits = len(trait_keys)
    vectors: Dict[Any, np.ndarray] = {}

    for uid in user_ids:
        rows = user_to_rows[uid]
        liked = rows[y_true[rows] == 1]
        if len(liked) < MIN_FINITE_TRAITS:
            continue

        vec = np.full(n_traits, np.nan)
        n_valid = 0
        for ti, key in enumerate(trait_keys):
            vals = trait_arrays[key][liked]
            mask = finite_masks[key][liked]
            if mask.sum() >= 3:
                vec[ti] = float(np.mean(vals[mask]))
                n_valid += 1

        if n_valid < max(3, n_traits // 3):
            continue
        np.nan_to_num(vec, copy=False, nan=0.0)
        vectors[uid] = vec

    return vectors


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_scatter(
    mainstreamness: np.ndarray,
    metric_values: np.ndarray,
    metric_name: str,
    rho: float,
    p_value: float,
    out_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    color = _METRIC_COLORS.get(metric_name, "#333333")

    ax.scatter(mainstreamness, metric_values, s=8, alpha=0.4, color=color,
               edgecolors="none")

    z = np.polyfit(mainstreamness, metric_values, 1)
    x_line = np.linspace(float(mainstreamness.min()),
                         float(mainstreamness.max()), 100)
    ax.plot(x_line, np.polyval(z, x_line), color=color, linewidth=1.5,
            linestyle="--", alpha=0.8)

    ax.set_xlabel("Mainstreamness (cosine sim to population mean)")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
    ax.set_title(f"Mainstreamness vs {metric_name.title()}   "
                 f"(ρ = {rho:.3f}, {sig})")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    path = out_dir / f"mainstreamness_vs_{metric_name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_quintile_bars(
    merged: pd.DataFrame,
    out_dir: Path,
) -> Path:
    merged = merged.copy()
    merged["quintile"] = pd.qcut(
        merged["mainstreamness"], 5, labels=["Q1\n(least)", "Q2", "Q3", "Q4", "Q5\n(most)"],
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    quintile_order = ["Q1\n(least)", "Q2", "Q3", "Q4", "Q5\n(most)"]
    bar_width = 0.22
    x = np.arange(len(quintile_order))

    for mi, metric in enumerate(METRICS):
        means = [
            merged.loc[merged["quintile"] == q, metric].mean()
            for q in quintile_order
        ]
        offset = (mi - 1) * bar_width
        color = _METRIC_COLORS.get(metric, "#333333")
        ax.bar(x + offset, means, bar_width, color=color, edgecolor="white",
               linewidth=0.5, label=metric.title())

    ax.set_xticks(x)
    ax.set_xticklabels(quintile_order, fontsize=9)
    ax.set_xlabel("Mainstreamness quintile")
    ax.set_ylabel("Mean metric value")
    ax.set_title("Model Performance by Taste Mainstreamness")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()

    path = out_dir / "performance_by_mainstreamness_quintile.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class WhoGetsServedModule(EvalModule):
    name = "who_gets_served"
    description = (
        "Correlates model performance with taste mainstreamness to show "
        "who benefits most from engagement optimisation"
    )

    def run(self, ctx: EvalContext) -> Dict[str, Any]:
        out_dir = self.get_output_dir(ctx)
        run_dir = ctx.config.get("run_dir")
        if run_dir is None:
            return {"skipped": True, "reason": "run_dir not in eval config"}

        try:
            inferences_lf = _load_inferences(Path(run_dir))
        except FileNotFoundError as e:
            return {"skipped": True, "reason": str(e)}

        # --- Per-user performance metrics ---
        per_user_perf = compute_per_user_metrics(ctx.predictions_df)

        # --- Filter eligible users, join inferences ---
        preds = pl.from_pandas(ctx.predictions_df)
        preds = _filter_eligible_users(preds)
        n_users_eligible = preds["did"].n_unique()
        if n_users_eligible < 10:
            return {"skipped": True, "reason": f"only {n_users_eligible} eligible users"}

        joined = (
            inferences_lf
            .join(preds.lazy(), left_on="at_uri", right_on="post_id",
                  how="inner")
            .collect()
        )
        if len(joined) < 50:
            return {"skipped": True, "reason": f"only {len(joined)} posts matched inferences"}

        flat, group_names = _unnest_text_inferences(joined)

        y_true = flat["y_true"].to_numpy().astype(float)
        user_col = flat["did"].to_numpy()
        user_ids = np.unique(user_col)
        user_to_rows: Dict[Any, np.ndarray] = {
            u: np.where(user_col == u)[0] for u in user_ids
        }

        # --- Flatten all trait arrays ---
        trait_arrays: Dict[str, np.ndarray] = {}
        finite_masks: Dict[str, np.ndarray] = {}
        all_trait_keys: List[str] = []

        for gname in group_names:
            gdf = flat.select(gname).unnest(gname)
            for col in gdf.columns:
                key = f"{gname}::{col}"
                arr = gdf[col].to_numpy().astype(float)
                trait_arrays[key] = arr
                finite_masks[key] = np.isfinite(arr)
                all_trait_keys.append(key)

        if not all_trait_keys:
            return {"skipped": True, "reason": "no trait columns found"}

        # --- Build per-user preference vectors ---
        pref_vectors = _build_preference_vectors(
            user_ids, user_to_rows, y_true,
            trait_arrays, finite_masks, all_trait_keys,
        )
        if len(pref_vectors) < 10:
            return {"skipped": True, "reason": f"only {len(pref_vectors)} users with valid preference vectors"}

        # --- Population-mean preference vector ---
        all_vecs = np.stack(list(pref_vectors.values()))
        pop_mean = all_vecs.mean(axis=0)

        # --- Mainstreamness per user ---
        mainstream_scores: Dict[Any, float] = {
            uid: _cosine_similarity(vec, pop_mean)
            for uid, vec in pref_vectors.items()
        }

        # --- Merge with performance ---
        ms_df = pd.DataFrame([
            {"did": str(uid), "mainstreamness": score}
            for uid, score in mainstream_scores.items()
        ])
        merged = ms_df.merge(per_user_perf, on="did", how="inner")
        if len(merged) < 10:
            return {"skipped": True, "reason": f"only {len(merged)} users after merge"}

        mainstreamness = merged["mainstreamness"].to_numpy()

        # --- Scatter plots and correlations ---
        plot_paths: list[str] = []
        corr_results: Dict[str, Any] = {}

        for metric in METRICS:
            if metric not in merged.columns:
                continue
            vals = merged[metric].to_numpy()
            finite = np.isfinite(vals)
            if finite.sum() < 10:
                continue
            rho, pval = spearmanr(mainstreamness[finite], vals[finite])
            corr_results[metric] = {
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "n_users": int(finite.sum()),
            }
            plot_paths.append(str(_plot_scatter(
                mainstreamness[finite], vals[finite],
                metric, float(rho), float(pval), out_dir,
            )))

        # --- Quintile bar chart ---
        plot_paths.append(str(_plot_quintile_bars(merged, out_dir)))

        # --- Quintile-level summary ---
        merged_q = merged.copy()
        merged_q["quintile"] = pd.qcut(
            merged_q["mainstreamness"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        )
        quintile_summary: Dict[str, Any] = {}
        for q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            qdf = merged_q[merged_q["quintile"] == q]
            quintile_summary[q] = {
                "n_users": len(qdf),
                "mean_mainstreamness": float(qdf["mainstreamness"].mean()),
            }
            for metric in METRICS:
                if metric in qdf.columns:
                    quintile_summary[q][f"mean_{metric}"] = float(qdf[metric].mean())

        summary = {
            "n_users_with_vectors": len(pref_vectors),
            "n_users_merged": len(merged),
            "n_traits": len(all_trait_keys),
            "correlations": corr_results,
            "quintiles": quintile_summary,
        }
        self.save_json(summary, out_dir / "who_gets_served_summary.json")

        return {
            "n_users_merged": len(merged),
            "correlations": corr_results,
            "plot_paths": plot_paths,
        }
