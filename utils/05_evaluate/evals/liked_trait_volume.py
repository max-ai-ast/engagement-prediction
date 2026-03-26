#!/usr/bin/env python3

"""
Liked-Trait Volume Evaluation Module

For each holdout user, computes the mean of each NLP content trait across
their liked posts.  Users are split into high-volume vs low-volume likers
using the "givers-of-half-the-likes" definition (the smallest set of users
whose likes account for >= 50 % of total like volume).  Points plots compare
the two groups' per-trait means with 95 % CIs and FDR-corrected significance
stars.

Two plot variants are produced per inference group:

- **raw**: Y-axis in original trait units (classifier probabilities).
- **std**: Y-axis standardised as (user mean - holdout mean) / holdout SD,
  comparable to the synthetic-feed user_pref metric.

Outputs (under liked_trait_volume/):
- {group}_liked_trait_volume_raw.png
- {group}_liked_trait_volume_std.png
- liked_trait_volume_summary.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from . import EvalContext, EvalModule, scaled_figsize
from .trait_corrs import _load_inferences, _unnest_text_inferences

MIN_LIKED_POSTS = 5

_VOLUME_COLORS = {"lo": "#6baed6", "hi": "#08519c"}

# Traits to exclude from specific groups (group_name -> set of col names).
_EXCLUDED_TRAITS: Dict[str, set] = {
    "moderation": {"OK"},
}


# ---------------------------------------------------------------------------
# Volume split
# ---------------------------------------------------------------------------

def _givers_of_half_the_likes(
    user_ids: np.ndarray,
    did_to_likes: Dict[Any, int],
) -> Tuple[Dict[Any, bool], float, str]:
    """Return a mapping uid -> is_high for the 50 %-of-volume split."""
    total = sum(did_to_likes.get(u, 0) for u in user_ids)
    if total == 0:
        return {u: False for u in user_ids}, 0.0, "0.0"

    sorted_uids = sorted(user_ids, key=lambda u: did_to_likes.get(u, 0),
                          reverse=True)
    cum = 0
    high_set: set = set()
    for uid in sorted_uids:
        cum += did_to_likes.get(uid, 0)
        high_set.add(uid)
        if cum >= total * 0.5:
            break

    pct_hi = 100.0 * len(high_set) / len(user_ids)
    is_high = {u: u in high_set for u in user_ids}
    return is_high, pct_hi, f"{pct_hi:.1f}"


# ---------------------------------------------------------------------------
# Per-user trait means
# ---------------------------------------------------------------------------

def _per_user_trait_means(
    flat: pl.DataFrame,
    group_names: list[str],
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, list[str]]]:
    """Compute per-user mean of each trait column across liked posts.

    Returns ``(trait_data, group_labels)`` where
    ``trait_data[group::col]`` is a dict ``{did: mean_trait_value}`` keyed
    by user, and ``group_labels[group]`` lists the trait columns.
    """
    trait_data: Dict[str, Dict[str, np.ndarray]] = {}
    group_labels: Dict[str, list[str]] = {}

    for gname in group_names:
        base = flat.select("did", gname).unnest(gname)
        cols = [c for c in base.columns if c != "did"]
        group_labels[gname] = cols

        excluded = _EXCLUDED_TRAITS.get(gname, set())
        for col in cols:
            if col in excluded:
                continue
            work = base.select("did", col)
            if work[col].dtype not in (pl.Float32, pl.Float64):
                work = work.with_columns(pl.col(col).cast(pl.Float64))

            finite_vals = work.filter(pl.col(col).is_finite())
            per_user = (
                finite_vals
                .group_by("did")
                .agg(
                    pl.col(col).mean().alias("trait_mean"),
                    pl.len().alias("n"),
                )
                .filter(pl.col("n") >= MIN_LIKED_POSTS)
            )
            if len(per_user) < 10:
                continue

            key = f"{gname}::{col}"
            dids = per_user["did"].to_numpy()
            means = per_user["trait_mean"].to_numpy()
            post_mean = float(finite_vals[col].mean())
            trait_data[key] = {"dids": dids, "means": means, "post_mean": post_mean}

    return trait_data, group_labels


def _holdout_baselines(
    flat: pl.DataFrame,
    group_names: list[str],
) -> Dict[str, Tuple[float, float]]:
    """Mean and SD of each trait across all holdout posts (likes + negatives)."""
    baselines: Dict[str, Tuple[float, float]] = {}
    for gname in group_names:
        base = flat.select(gname).unnest(gname)
        cols = base.columns
        for col in cols:
            vals = base[col].to_numpy().astype(np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) < 20:
                continue
            baselines[f"{gname}::{col}"] = (
                float(np.mean(vals)),
                float(np.std(vals, ddof=1)),
            )
    return baselines


# ---------------------------------------------------------------------------
# Points plot
# ---------------------------------------------------------------------------

def _plot_volume_points(
    group_name: str,
    trait_data: Dict[str, Dict[str, np.ndarray]],
    group_labels: list[str],
    is_high: Dict[Any, bool],
    pct_hi_str: str,
    baselines: Dict[str, Tuple[float, float]] | None,
    standardize: bool,
    out_dir: Path,
) -> Tuple[Path, Dict[str, Dict[str, Any]], list[str]]:
    """Single-panel mean +/- 95 % CI point plot, high-likers vs low-likers."""
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    z95 = 1.96
    tag = "std" if standardize else "raw"

    raw_stats: Dict[str, Dict[str, Any]] = {}
    for col in group_labels:
        key = f"{group_name}::{col}"
        td = trait_data.get(key)
        if td is None:
            continue
        dids = td["dids"]
        means = td["means"].copy()

        post_mean = td.get("post_mean", float("nan"))

        if standardize and baselines is not None:
            bl = baselines.get(key)
            if bl is None or bl[1] < 1e-12:
                continue
            means = (means - bl[0]) / bl[1]
            post_mean = (post_mean - bl[0]) / bl[1]

        mask_hi = np.array([is_high.get(uid, False) for uid in dids])
        lo_vals = means[~mask_hi]
        hi_vals = means[mask_hi]

        n_lo, n_hi = len(lo_vals), len(hi_vals)
        if n_lo < 5 or n_hi < 5:
            continue

        mean_lo = float(np.mean(lo_vals))
        mean_hi = float(np.mean(hi_vals))
        ci_lo = z95 * float(np.std(lo_vals, ddof=1)) / np.sqrt(n_lo)
        ci_hi = z95 * float(np.std(hi_vals, ddof=1)) / np.sqrt(n_hi)

        _, p_val = ttest_ind(hi_vals, lo_vals, equal_var=False)
        mean_all_users = float(np.mean(means))

        raw_stats[col] = {
            "mean_lo": mean_lo, "ci_lo": ci_lo, "n_lo": n_lo,
            "mean_hi": mean_hi, "ci_hi": ci_hi, "n_hi": n_hi,
            "mean_all_users": mean_all_users,
            "mean_all_posts": post_mean,
            "diff": mean_hi - mean_lo,
            "p_val": float(p_val),
        }

    labels = sorted(raw_stats, key=lambda k: raw_stats[k]["diff"])
    n = len(labels)
    fname = f"{group_name}_liked_trait_volume_{tag}.png"

    if n == 0:
        path = out_dir / fname
        fig, ax = plt.subplots(figsize=scaled_figsize(4, 3))
        ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#999999")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        return path, {}, []

    x = np.arange(n)
    dodge = 0.15

    m_lo = [raw_stats[l]["mean_lo"] for l in labels]
    c_lo = [raw_stats[l]["ci_lo"]   for l in labels]
    m_hi = [raw_stats[l]["mean_hi"] for l in labels]
    c_hi = [raw_stats[l]["ci_hi"]   for l in labels]
    m_all_users = [raw_stats[l]["mean_all_users"] for l in labels]
    m_all_posts = [raw_stats[l]["mean_all_posts"] for l in labels]

    PLOT_AREA_HEIGHT = 4.0
    BOTTOM_MARGIN = 2.0
    total_h = PLOT_AREA_HEIGHT + BOTTOM_MARGIN
    fig_w, fig_h = scaled_figsize(max(6, 0.55 * n), total_h)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.errorbar(
        x - dodge, m_lo, yerr=c_lo,
        fmt="o", color=_VOLUME_COLORS["lo"], markersize=5,
        capsize=3, capthick=0.8, linewidth=0.8,
        label=f"remaining users",
    )
    ax.errorbar(
        x + dodge, m_hi, yerr=c_hi,
        fmt="o", color=_VOLUME_COLORS["hi"], markersize=5,
        capsize=3, capthick=0.8, linewidth=0.8,
        label=f"top {pct_hi_str}% of users (50% vol)",
    )
    ax.scatter(x, m_all_users, marker="D", s=28, color="#666666", zorder=5,
               label="overall user mean")
    ax.scatter(x, m_all_posts, marker="s", s=22, color="#aaaaaa", zorder=4,
               label="overall post mean")

    p_vals = np.array([raw_stats[l]["p_val"] for l in labels])
    if len(p_vals) >= 1:
        _, q_vals, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")
    else:
        q_vals = p_vals

    y_top = max(
        max(m + c for m, c in zip(m_lo, c_lo)),
        max(m + c for m, c in zip(m_hi, c_hi)),
        max(m_all_users),
        max(m_all_posts),
    )
    star_y = y_top + (0.02 if not standardize else 0.05)
    for i, q in enumerate(q_vals):
        if q < 0.05:
            ax.text(i, star_y, "*", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#333333")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    if standardize:
        ylabel = "Mean trait in liked posts  [(user − holdout mean) / SD]"
    else:
        ylabel = "Mean trait in liked posts  [raw, mean ± 95% CI]"
    ax.set_ylabel(ylabel, fontsize=8)

    ax.set_title(
        f"{group_name.replace('_', ' ').title()}"
        f" — Liked-Trait Prevalence by Volume"
        f"  (top {pct_hi_str}% = 50% of likes)",
        fontsize=10, fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=7)
    fig.subplots_adjust(bottom=BOTTOM_MARGIN / total_h, top=0.92, left=0.10,
                        right=0.97)

    path = out_dir / fname
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path, raw_stats, labels


# ---------------------------------------------------------------------------
# Standalone legend
# ---------------------------------------------------------------------------

def _save_legend(pct_hi_str: str, out_dir: Path) -> Path:
    """Save a standalone legend PNG matching the volume-points plots."""
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=_VOLUME_COLORS["lo"],
            markersize=7, label="remaining users",
        ),
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=_VOLUME_COLORS["hi"],
            markersize=7, label=f"top {pct_hi_str}% of users (50% vol)",
        ),
        plt.Line2D(
            [0], [0], marker="D", color="w", markerfacecolor="#666666",
            markersize=7, label="overall user mean",
        ),
        plt.Line2D(
            [0], [0], marker="s", color="w", markerfacecolor="#aaaaaa",
            markersize=6, label="overall post mean",
        ),
    ]
    fig, ax = plt.subplots(figsize=(1, 1))
    legend = ax.legend(handles=handles, fontsize=8, framealpha=0.9,
                       loc="center", frameon=True)
    ax.set_axis_off()
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted()
    )
    path = out_dir / "liked_trait_volume_legend.png"
    fig.savefig(path, dpi=300, bbox_inches=bbox)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class LikedTraitVolumeModule(EvalModule):
    name = "liked_trait_volume"
    description = (
        "Compares mean NLP-trait prevalence in liked posts between "
        "high-volume and low-volume likers (50%-of-likes split)"
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

        # --- Liked posts joined to inferences ---
        liked = preds.filter(pl.col("y_true") == 1).select("did", "post_id")
        liked_joined = (
            inferences_lf
            .join(liked.lazy(), left_on="at_uri", right_on="post_id",
                  how="inner")
            .collect()
        )
        n_liked_matched = len(liked_joined)
        if n_liked_matched < 50:
            return {
                "skipped": True,
                "reason": f"only {n_liked_matched} liked posts matched inferences",
            }

        liked_flat, group_names = _unnest_text_inferences(liked_joined)

        # --- Per-user trait means (from liked posts only) ---
        trait_data, group_labels = _per_user_trait_means(liked_flat, group_names)

        # --- Baselines from all holdout posts for standardisation ---
        all_joined = (
            inferences_lf
            .join(preds.lazy().select("post_id"), left_on="at_uri",
                  right_on="post_id", how="inner")
            .collect()
        )
        all_flat, _ = _unnest_text_inferences(all_joined)
        baselines = _holdout_baselines(all_flat, group_names)

        # --- User like counts & volume split ---
        all_user_ids = np.unique(liked_flat["did"].to_numpy())
        meta = ctx.user_metadata_df
        did_to_likes: Dict[Any, int] = dict(
            zip(meta["did"], meta["num_total_likes"])
        )
        is_high, pct_hi, pct_hi_str = _givers_of_half_the_likes(
            all_user_ids, did_to_likes,
        )

        n_hi = sum(1 for v in is_high.values() if v)
        n_lo = len(is_high) - n_hi

        # --- Standalone legend ---
        legend_path = _save_legend(pct_hi_str, out_dir)

        # --- Plots & summary ---
        plot_paths: list[str] = []
        groups_json: Dict[str, Any] = {}

        for gname in group_names:
            cols = group_labels.get(gname, [])
            gname_traits = {
                k: v for k, v in trait_data.items()
                if k.startswith(f"{gname}::")
            }
            if not gname_traits:
                continue

            gdict: Dict[str, Any] = {}

            for standardize in (False, True):
                result = _plot_volume_points(
                    gname, trait_data, cols, is_high, pct_hi_str,
                    baselines, standardize, out_dir,
                )
                path, stats, ordered_labels = result
                plot_paths.append(str(path))

                tag = "std" if standardize else "raw"
                for label in ordered_labels:
                    s = stats[label]
                    entry = gdict.setdefault(label, {})
                    entry[f"mean_{tag}_hi"] = s["mean_hi"]
                    entry[f"mean_{tag}_lo"] = s["mean_lo"]
                    entry[f"diff_{tag}"] = s["diff"]
                    entry[f"p_val_{tag}"] = s["p_val"]
                    entry["n_hi"] = s["n_hi"]
                    entry["n_lo"] = s["n_lo"]

            if gdict:
                groups_json[gname] = gdict

        summary = {
            "n_liked_matched": n_liked_matched,
            "n_users": len(all_user_ids),
            "n_hi": n_hi,
            "n_lo": n_lo,
            "pct_hi": pct_hi,
            "groups": groups_json,
        }
        self.save_json(summary, out_dir / "liked_trait_volume_summary.json")

        return {
            "n_liked_matched": n_liked_matched,
            "n_users": len(all_user_ids),
            "groups_plotted": len(groups_json),
            "plot_paths": plot_paths,
            "legend_path": str(legend_path),
        }
