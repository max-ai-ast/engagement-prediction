#!/usr/bin/env python3

"""
Diversity Reduction Evaluation Module

Tests whether the model's high-confidence predictions are less
trait-diverse than a user's actual likes.  For each user and each NLP
trait, compares:

- actual diversity:    variance of the trait among liked posts (y_true == 1)
- predicted diversity: variance of the trait among the user's top-quartile
                       posts by y_pred_proba

A diversity ratio < 1 means the model narrows the range of that trait the
user would be exposed to ("homogenisation").

Outputs (under diversity_reduction/):
- <group>_diversity.png:       small-multiples KDE of per-user diversity ratios
- diversity_summary_bars.png:  horizontal bar chart of median ratios across traits
- diversity_reduction_summary.json
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

from . import EvalContext, EvalModule
from .trait_corrs import _load_inferences, _unnest_text_inferences
from .trait_amplification import MIN_USER_POSTS, _filter_eligible_users

MIN_LIKED_POSTS_PER_TRAIT = 10
MIN_USERS_PER_TRAIT = 30

_GROUP_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _compute_diversity_ratios(
    user_ids: np.ndarray,
    user_to_rows: Dict[Any, np.ndarray],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trait_arrays: Dict[str, np.ndarray],
    finite_masks: Dict[str, np.ndarray],
    valid_keys: List[str],
) -> Dict[str, List[float]]:
    """Per-user diversity ratio (predicted variance / actual variance) per trait.

    Returns {key: list_of_ratios} with one ratio per eligible user.
    """
    ratios: Dict[str, List[float]] = defaultdict(list)

    for uid in user_ids:
        rows = user_to_rows[uid]
        yt = y_true[rows]
        yp = y_pred[rows]

        liked_idx = rows[yt == 1]
        if len(liked_idx) < MIN_LIKED_POSTS_PER_TRAIT:
            continue

        n_top = max(1, len(rows) // 4)
        top_idx = rows[np.argsort(yp)[-n_top:]]

        for key in valid_keys:
            t_liked = trait_arrays[key][liked_idx]
            m_liked = finite_masks[key][liked_idx]
            if m_liked.sum() < MIN_LIKED_POSTS_PER_TRAIT:
                continue
            var_actual = float(np.var(t_liked[m_liked]))
            if var_actual == 0:
                continue

            t_top = trait_arrays[key][top_idx]
            m_top = finite_masks[key][top_idx]
            if m_top.sum() < 3:
                continue
            var_pred = float(np.var(t_top[m_top]))

            ratios[key].append(var_pred / var_actual)

    return {k: v for k, v in ratios.items() if len(v) >= MIN_USERS_PER_TRAIT}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _safe_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    try:
        return gaussian_kde(values)(grid)
    except Exception:
        return None


def _plot_group_diversity(
    group_name: str,
    trait_ratios: Dict[str, np.ndarray],
    out_dir: Path,
) -> Path:
    labels = sorted(trait_ratios.keys())
    n = len(labels)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for idx, label in enumerate(labels):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        vals = trait_ratios[label]

        lo = max(0, float(np.percentile(vals, 1)) - 0.1)
        hi = float(np.percentile(vals, 99)) + 0.1
        grid = np.linspace(lo, hi, 300)

        density = _safe_kde(vals, grid)
        if density is not None:
            ax.fill_between(grid, density, where=(grid < 1.0),
                            alpha=0.3, color="#D65F5F")
            ax.fill_between(grid, density, where=(grid >= 1.0),
                            alpha=0.15, color="#4878CF")
            ax.plot(grid, density, color="#333333", linewidth=0.9)
        else:
            ax.hist(vals, bins=30, density=True, alpha=0.4, color="#999999")

        ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8,
                   label="no change")
        med = float(np.median(vals))
        ax.axvline(med, color="#d62728", linewidth=1.0,
                   label=f"median = {med:.3f}")

        frac_below = float((vals < 1.0).mean())
        ax.set_title(f"{label}  ({frac_below:.0%} < 1)",
                     fontsize=8, fontweight="bold")
        ax.set_xlabel("diversity ratio (pred / actual)", fontsize=7)
        ax.set_ylabel("density", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.7)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"{group_name.replace('_', ' ').title()} — Diversity Ratio "
        "(top-quartile predicted variance / actual-liked variance)",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = out_dir / f"{group_name}_diversity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_summary_bars(
    all_medians: Dict[str, Tuple[str, float]],
    group_color_map: Dict[str, str],
    out_dir: Path,
) -> Path:
    from matplotlib.patches import Patch

    sorted_keys = sorted(all_medians,
                         key=lambda k: all_medians[k][1])
    labels = [k.split("::")[-1] for k in sorted_keys]
    medians = [all_medians[k][1] for k in sorted_keys]
    colors = [group_color_map.get(all_medians[k][0], "#999999")
              for k in sorted_keys]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(labels))))
    y_pos = np.arange(len(labels))

    ax.barh(y_pos, medians, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Median diversity ratio (pred / actual)")
    ax.set_title("Diversity Reduction Summary (< 1 = model narrows diversity)")

    seen: set[str] = set()
    handles = []
    for k in sorted_keys:
        g = all_medians[k][0]
        if g not in seen:
            seen.add(g)
            handles.append(Patch(facecolor=group_color_map.get(g, "#999999"),
                                 label=g.replace("_", " ")))
    ax.legend(handles=handles, fontsize=6, loc="lower right", framealpha=0.8)
    plt.tight_layout()

    path = out_dir / "diversity_summary_bars.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
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

        # --- Full holdout, filter to eligible users ---
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

        trait_arrays: Dict[str, np.ndarray] = {}
        finite_masks: Dict[str, np.ndarray] = {}
        group_labels: Dict[str, List[str]] = {}

        for gname in group_names:
            gdf = flat.select(gname).unnest(gname)
            cols = gdf.columns
            group_labels[gname] = cols
            for col in cols:
                key = f"{gname}::{col}"
                arr = gdf[col].to_numpy().astype(float)
                trait_arrays[key] = arr
                finite_masks[key] = np.isfinite(arr)

        all_keys = list(trait_arrays.keys())

        # --- Diversity ratios ---
        ratios = _compute_diversity_ratios(
            user_ids, user_to_rows, y_true, y_pred,
            trait_arrays, finite_masks, all_keys,
        )

        # --- Plots ---
        group_color_map = {
            g: _GROUP_COLORS[i % len(_GROUP_COLORS)]
            for i, g in enumerate(group_names)
        }
        plot_paths: list[str] = []
        all_medians: Dict[str, Tuple[str, float]] = {}

        for gname in group_names:
            group_traits: Dict[str, np.ndarray] = {}
            for label in group_labels.get(gname, []):
                key = f"{gname}::{label}"
                if key not in ratios:
                    continue
                arr = np.array(ratios[key])
                group_traits[label] = arr
                all_medians[key] = (gname, float(np.median(arr)))
            if not group_traits:
                continue
            plot_paths.append(
                str(_plot_group_diversity(gname, group_traits, out_dir)))

        if all_medians:
            plot_paths.insert(
                0, str(_plot_summary_bars(all_medians, group_color_map, out_dir)))

        # --- Summary JSON ---
        groups_json: Dict[str, Any] = {}
        for gname in group_names:
            gdict: Dict[str, Any] = {}
            for label in group_labels.get(gname, []):
                key = f"{gname}::{label}"
                if key not in ratios:
                    continue
                arr = np.array(ratios[key])
                gdict[label] = {
                    "median_ratio": float(np.median(arr)),
                    "mean_ratio": float(np.mean(arr)),
                    "frac_below_1": float((arr < 1.0).mean()),
                    "n_users_computed": len(arr),
                }
            if gdict:
                groups_json[gname] = gdict

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
