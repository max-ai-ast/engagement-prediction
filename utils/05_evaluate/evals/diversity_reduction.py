#!/usr/bin/env python3

"""
Diversity Reduction Evaluation Module

For each inference group (topic, emotion_sentiment, sentiment, toxicity, …)
and each user, computes the Shannon entropy of the mean trait vector across
two post sets:

- actual likes:     posts where y_true == 1
- model top picks:  top-quartile posts by y_pred_proba

An entropy ratio H(predicted) / H(actual) < 1 means the model concentrates
content into fewer categories than the user's organic behaviour.

Outputs (under diversity_reduction/):
- <group>_entropy.png:          histogram of per-user entropy ratios
- <group>_category_shift.png:   grouped bar chart of mean category shares
                                (actual vs predicted)
- diversity_summary_bars.png:   one bar per group showing median entropy ratio
- diversity_reduction_summary.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, NamedTuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from . import EvalContext, EvalModule, scaled_figsize
from .trait_corrs import _load_inferences, _unnest_text_inferences
from .trait_amplification import MIN_USER_POSTS, _filter_eligible_users

MIN_LIKED_POSTS = 10
MIN_USERS_PER_GROUP = 30

_GROUP_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _shannon_entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability vector (zeros handled safely)."""
    p = p[p > 0]
    return -float(np.sum(p * np.log(p)))


class GroupEntropyResult(NamedTuple):
    H_actual: np.ndarray
    H_predicted: np.ndarray
    user_ids: np.ndarray
    mean_share_actual: np.ndarray
    mean_share_predicted: np.ndarray


def _compute_entropy_ratios(
    user_ids: np.ndarray,
    user_to_rows: Dict[Any, np.ndarray],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trait_arrays: Dict[str, np.ndarray],
    finite_masks: Dict[str, np.ndarray],
    trait_labels: List[str],
) -> GroupEntropyResult | None:
    """Per-user entropy of mean trait vector: actual likes vs model top picks.

    For each eligible user, builds the mean trait vector across all traits in
    the group for liked posts and top-quartile predicted posts, normalises each
    to a probability distribution, and computes Shannon entropy.

    Also accumulates population-level mean share vectors for the category-shift
    plot.
    """
    n_traits = len(trait_labels)
    if n_traits < 2:
        return None

    keys = [trait_labels[i] for i in range(n_traits)]

    h_actual_list: List[float] = []
    h_pred_list: List[float] = []
    uid_list: List[Any] = []
    share_actual_accum = np.zeros(n_traits, dtype=np.float64)
    share_pred_accum = np.zeros(n_traits, dtype=np.float64)

    for uid in user_ids:
        rows = user_to_rows[uid]
        yt = y_true[rows]
        yp = y_pred[rows]

        liked_idx = rows[yt == 1]
        if len(liked_idx) < MIN_LIKED_POSTS:
            continue

        n_top = max(1, len(rows) // 4)
        top_idx = rows[np.argsort(yp)[-n_top:]]

        vec_actual = np.empty(n_traits)
        vec_pred = np.empty(n_traits)
        valid = True

        for ti, key in enumerate(keys):
            a_vals = trait_arrays[key][liked_idx]
            a_mask = finite_masks[key][liked_idx]
            if a_mask.sum() < 3:
                valid = False
                break
            vec_actual[ti] = float(np.mean(a_vals[a_mask]))

            p_vals = trait_arrays[key][top_idx]
            p_mask = finite_masks[key][top_idx]
            if p_mask.sum() < 3:
                valid = False
                break
            vec_pred[ti] = float(np.mean(p_vals[p_mask]))

        if not valid:
            continue

        np.clip(vec_actual, 0, None, out=vec_actual)
        np.clip(vec_pred, 0, None, out=vec_pred)

        s_actual = vec_actual.sum()
        s_pred = vec_pred.sum()
        if s_actual == 0 or s_pred == 0:
            continue

        p_actual = vec_actual / s_actual
        p_pred = vec_pred / s_pred

        h_a = _shannon_entropy(p_actual)
        h_p = _shannon_entropy(p_pred)
        if h_a == 0:
            continue

        h_actual_list.append(h_a)
        h_pred_list.append(h_p)
        uid_list.append(uid)
        share_actual_accum += p_actual
        share_pred_accum += p_pred

    if len(h_actual_list) < MIN_USERS_PER_GROUP:
        return None

    n = len(h_actual_list)
    mean_share_actual = share_actual_accum / n
    mean_share_pred = share_pred_accum / n

    return GroupEntropyResult(
        H_actual=np.array(h_actual_list),
        H_predicted=np.array(h_pred_list),
        user_ids=np.array(uid_list),
        mean_share_actual=mean_share_actual,
        mean_share_predicted=mean_share_pred,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_entropy_histogram(
    group_name: str,
    result: GroupEntropyResult,
    out_dir: Path,
) -> Path:
    """Histogram of per-user H(predicted) / H(actual) for one group."""
    ratios = result.H_predicted / result.H_actual

    x_upper = max(float(np.percentile(ratios, 95)), 1.5)
    n_bins = 40
    bins = np.linspace(0, x_upper, n_bins + 1)
    n_clipped = int((ratios > x_upper).sum())
    clipped = ratios[ratios <= x_upper]

    fig, ax = plt.subplots(figsize=scaled_figsize(7, 4))
    _, _, patches = ax.hist(clipped, bins=bins, density=True,
                            edgecolor="white", linewidth=0.4)
    for patch in patches:
        center = patch.get_x() + patch.get_width() / 2
        if center < 1.0:
            patch.set_facecolor("#D65F5F")
            patch.set_alpha(0.6)
        else:
            patch.set_facecolor("#4878CF")
            patch.set_alpha(0.5)

    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8,
               label="no change")
    med = float(np.median(ratios))
    ax.axvline(med, color="#d62728", linewidth=1.0,
               label=f"median = {med:.3f}")

    frac_below = float((ratios < 1.0).mean())
    title = (
        f"{group_name.replace('_', ' ').title()} — Entropy Ratio  "
        f"({frac_below:.0%} of users narrowed)"
    )
    if n_clipped > 0:
        title += f"  [{n_clipped} clipped]"
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("H(predicted) / H(actual)", fontsize=9)
    ax.set_ylabel("density", fontsize=9)
    ax.set_xlim(0, x_upper)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
    plt.tight_layout()

    path = out_dir / f"{group_name}_entropy.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_category_shift(
    group_name: str,
    trait_labels: List[str],
    result: GroupEntropyResult,
    out_dir: Path,
) -> Path:
    """Grouped horizontal bar chart: actual vs predicted mean category share."""
    actual = result.mean_share_actual
    predicted = result.mean_share_predicted
    shift = predicted - actual

    order = np.argsort(shift)
    labels = [trait_labels[i] for i in order]
    actual_s = actual[order]
    predicted_s = predicted[order]

    y = np.arange(len(labels))
    bar_h = 0.35

    fig, ax = plt.subplots(figsize=scaled_figsize(8, max(3, 0.45 * len(labels))))
    ax.barh(y - bar_h / 2, actual_s, bar_h, label="actual likes",
            color="#4878CF", edgecolor="white", linewidth=0.4)
    ax.barh(y + bar_h / 2, predicted_s, bar_h, label="model top picks",
            color="#D65F5F", edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("mean share (normalised)", fontsize=9)
    ax.set_title(
        f"{group_name.replace('_', ' ').title()} — Category Shares: "
        "Actual Likes vs Model Top Picks",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="lower right", framealpha=0.7)
    plt.tight_layout()

    path = out_dir / f"{group_name}_category_shift.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_summary_bars(
    group_medians: Dict[str, float],
    group_ratios: Dict[str, np.ndarray],
    group_color_map: Dict[str, str],
    out_dir: Path,
) -> Path:
    """Median bar + jittered user-level points per group."""
    sorted_groups = sorted(group_medians, key=group_medians.get)  # type: ignore[arg-type]
    labels = [g.replace("_", " ") for g in sorted_groups]
    medians = [group_medians[g] for g in sorted_groups]
    colors = [group_color_map.get(g, "#999999") for g in sorted_groups]

    fig, ax = plt.subplots(figsize=scaled_figsize(8, max(3, 0.7 * len(labels))))
    y_pos = np.arange(len(labels))

    rng = np.random.default_rng(42)
    for i, gname in enumerate(sorted_groups):
        vals = group_ratios.get(gname, np.array([]))
        if len(vals) == 0:
            continue
        jitter = rng.uniform(-0.25, 0.25, size=len(vals))
        ax.scatter(vals, i + jitter, s=3, alpha=0.15, color=colors[i],
                   edgecolors="none", rasterized=True)

    ax.barh(y_pos, medians, height=0.5, color=colors, edgecolor="white",
            linewidth=0.5, alpha=0.45, zorder=3)
    for i, med in enumerate(medians):
        ax.plot(med, i, marker="|", color="black", markersize=14,
                markeredgewidth=1.8, zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Entropy ratio  H(predicted) / H(actual)", fontsize=9)
    ax.set_title("Diversity Reduction Summary (< 1 = model narrows diversity)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()

    path = out_dir / "diversity_summary_bars.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class DiversityReductionModule(EvalModule):
    name = "diversity_reduction"
    description = (
        "Measures whether the model's high-confidence predictions narrow "
        "trait diversity relative to actual user likes"
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

        preds = pl.from_pandas(ctx.predictions_df)
        preds = _filter_eligible_users(preds)
        n_users_eligible = preds["did"].n_unique()
        if n_users_eligible < 5:
            return {"skipped": True, "reason": f"only {n_users_eligible} eligible users"}

        joined = (
            inferences_lf
            .join(preds.lazy(), left_on="at_uri", right_on="post_id",
                  how="inner")
            .collect()
        )
        n_posts_matched = len(joined)
        if n_posts_matched < 50:
            return {"skipped": True, "reason": f"only {n_posts_matched} posts matched inferences"}

        flat, group_names = _unnest_text_inferences(joined)

        y_true = flat["y_true"].to_numpy().astype(float)
        y_pred = flat["y_pred_proba"].to_numpy()
        user_col = flat["did"].to_numpy()

        user_ids = np.unique(user_col)
        user_to_rows: Dict[Any, np.ndarray] = {
            u: np.where(user_col == u)[0] for u in user_ids
        }

        # --- Per-group trait arrays ---
        group_trait_arrays: Dict[str, Dict[str, np.ndarray]] = {}
        group_finite_masks: Dict[str, Dict[str, np.ndarray]] = {}
        group_labels: Dict[str, List[str]] = {}

        for gname in group_names:
            gdf = flat.select(gname).unnest(gname)
            cols = gdf.columns
            group_labels[gname] = cols
            g_arrays: Dict[str, np.ndarray] = {}
            g_masks: Dict[str, np.ndarray] = {}
            for col in cols:
                arr = gdf[col].to_numpy().astype(float)
                g_arrays[col] = arr
                g_masks[col] = np.isfinite(arr)
            group_trait_arrays[gname] = g_arrays
            group_finite_masks[gname] = g_masks

        # --- Entropy computation + plots per group ---
        group_color_map = {
            g: _GROUP_COLORS[i % len(_GROUP_COLORS)]
            for i, g in enumerate(group_names)
        }
        plot_paths: List[str] = []
        group_medians: Dict[str, float] = {}
        group_ratios: Dict[str, np.ndarray] = {}
        groups_json: Dict[str, Any] = {}

        for gname in group_names:
            labels = group_labels[gname]
            result = _compute_entropy_ratios(
                user_ids, user_to_rows, y_true, y_pred,
                group_trait_arrays[gname],
                group_finite_masks[gname],
                labels,
            )
            if result is None:
                continue

            ratios = result.H_predicted / result.H_actual
            med = float(np.median(ratios))
            group_medians[gname] = med
            group_ratios[gname] = ratios

            plot_paths.append(
                str(_plot_entropy_histogram(gname, result, out_dir)))
            plot_paths.append(
                str(_plot_category_shift(gname, labels, result, out_dir)))

            groups_json[gname] = {
                "median_entropy_ratio": med,
                "mean_entropy_ratio": float(np.mean(ratios)),
                "frac_below_1": float((ratios < 1.0).mean()),
                "median_H_actual": float(np.median(result.H_actual)),
                "median_H_predicted": float(np.median(result.H_predicted)),
                "n_users_computed": len(ratios),
                "category_shares_actual": {
                    labels[i]: float(result.mean_share_actual[i])
                    for i in range(len(labels))
                },
                "category_shares_predicted": {
                    labels[i]: float(result.mean_share_predicted[i])
                    for i in range(len(labels))
                },
            }

        if group_medians:
            plot_paths.insert(
                0, str(_plot_summary_bars(group_medians, group_ratios,
                                          group_color_map, out_dir)))

        summary = {
            "n_users_eligible": n_users_eligible,
            "n_posts_matched": n_posts_matched,
            "min_user_posts": MIN_USER_POSTS,
            "groups": groups_json,
        }
        self.save_json(summary, out_dir / "diversity_reduction_summary.json")

        return {
            "n_users_eligible": n_users_eligible,
            "n_posts_matched": n_posts_matched,
            "groups_plotted": len(groups_json),
            "plot_paths": plot_paths,
        }
