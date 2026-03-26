#!/usr/bin/env python3

"""
Trait Over-Serving Evaluation Module

Measures content over-serving in **natural trait units** rather than
correlation space, producing results that support direct verbal statements
such as "the model over-serves content that is 0.12 SD more negative in
sentiment than what users actually engage with."

For each user *u* and trait *t* (using the balanced holdout with 50/50
liked / not-liked per user):

    actual_pref   = mean(trait | liked) − mean(trait | not-liked)
    model_pref    = mean(trait | model-top-half) − mean(trait | model-bottom-half)
    over_serving  = model_pref − actual_pref        (raw trait units)
    std_over_serv = over_serving / global_sd(trait)  (standardised, cross-trait)

The model-top-half is determined by splitting each user's posts at the
within-user median y_pred_proba, giving equal-sized halves that mirror the
50/50 actual split.

Outputs (under trait_over_serving/):
- <group>_over_serving.png:     small-multiples KDE of std_over_serving
- over_serving_summary_bar.png: headline bar chart across all traits
- trait_over_serving_summary.json
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde, ttest_1samp

from . import EvalContext, EvalModule, scaled_figsize
from .trait_corrs import _load_inferences, _unnest_text_inferences
from .trait_amplification import MIN_USER_POSTS, _filter_eligible_users

MIN_POSTS_PER_HALF = 8

_GROUP_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TraitOverServingResult(NamedTuple):
    over_serving_raw: np.ndarray
    over_serving_std: np.ndarray
    global_sd: float
    n_users: int
    cohen_d: float
    t_stat: float
    p_value: float


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _compute_over_serving(
    flat: pl.DataFrame,
    group_names: list[str],
) -> Tuple[
    Dict[str, TraitOverServingResult],
    Dict[str, list[str]],
]:
    """Per-user over-serving for every trait.

    Returns ``(trait_results, group_labels)``.
    """
    trait_results: Dict[str, TraitOverServingResult] = {}
    group_labels: Dict[str, list[str]] = {}

    y_true = flat["y_true"].to_numpy()
    y_pred = flat["y_pred_proba"].to_numpy()
    dids = flat["did"].to_numpy()

    unique_dids = np.unique(dids)
    did_to_rows: Dict[Any, np.ndarray] = {
        d: np.where(dids == d)[0] for d in unique_dids
    }

    for gname in group_names:
        gdf = flat.select(gname).unnest(gname)
        cols = gdf.columns
        group_labels[gname] = cols

        for col in cols:
            key = f"{gname}::{col}"
            trait_vals = gdf[col].to_numpy().astype(np.float64)
            finite_mask = np.isfinite(trait_vals)

            finite_vals = trait_vals[finite_mask]
            if len(finite_vals) < 20:
                continue
            global_sd = float(np.std(finite_vals, ddof=1))
            if global_sd < 1e-12:
                continue

            os_raw_list: List[float] = []

            for did in unique_dids:
                rows = did_to_rows[did]
                fm = finite_mask[rows]
                valid_rows = rows[fm]
                yt = y_true[valid_rows]
                yp = y_pred[valid_rows]
                tv = trait_vals[valid_rows]

                if np.std(tv) < 1e-6:
                    continue

                liked = yt == 1
                not_liked = ~liked
                if liked.sum() < MIN_POSTS_PER_HALF or not_liked.sum() < MIN_POSTS_PER_HALF:
                    continue

                actual_pref = float(tv[liked].mean() - tv[not_liked].mean())

                median_pred = np.median(yp)
                top_half = yp >= median_pred
                bot_half = ~top_half
                if top_half.sum() < MIN_POSTS_PER_HALF or bot_half.sum() < MIN_POSTS_PER_HALF:
                    continue

                model_pref = float(tv[top_half].mean() - tv[bot_half].mean())
                os_raw_list.append(model_pref - actual_pref)

            if len(os_raw_list) < 10:
                continue

            os_raw = np.array(os_raw_list)
            os_std = os_raw / global_sd

            sd_std = float(np.std(os_std, ddof=1))
            cohen_d = float(np.mean(os_std)) / sd_std if sd_std > 1e-12 else 0.0
            t_res = ttest_1samp(os_std, 0.0)

            trait_results[key] = TraitOverServingResult(
                over_serving_raw=os_raw,
                over_serving_std=os_std,
                global_sd=global_sd,
                n_users=len(os_raw),
                cohen_d=cohen_d,
                t_stat=float(t_res.statistic),
                p_value=float(t_res.pvalue),
            )

    return trait_results, group_labels


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _fmt_pvalue(p: float) -> str:
    if p < 0.001:
        return "p < .001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.2f}"


def _safe_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    try:
        return gaussian_kde(values)(grid)
    except Exception:
        return None


def _plot_group_over_serving(
    group_name: str,
    trait_results: Dict[str, TraitOverServingResult],
    out_dir: Path,
) -> Path:
    """Small-multiples KDE of standardised over-serving per trait."""
    labels = sorted(trait_results.keys())
    n = len(labels)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=scaled_figsize(5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for idx, label in enumerate(labels):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        tr = trait_results[label]
        vals = tr.over_serving_std

        lo = float(vals.min()) - 0.1
        hi = float(vals.max()) + 0.1
        grid = np.linspace(lo, hi, 300)

        density = _safe_kde(vals, grid)
        if density is not None:
            color_fill = "#e06000" if float(np.mean(vals)) >= 0 else "#4878CF"
            ax.fill_between(grid, density, alpha=0.3, color=color_fill)
            ax.plot(grid, density, color=color_fill, linewidth=1.0)
        else:
            ax.hist(vals, bins=30, density=True, alpha=0.3, color="#e06000")

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        mean_v = float(np.mean(vals))
        median_v = float(np.median(vals))
        pct_over = float(np.mean(vals > 0) * 100)

        ax.axvline(mean_v, color="#c04000", linewidth=1.0,
                   label=f"mean = {mean_v:+.3f} SD")
        ax.axvline(median_v, color="#c04000", linewidth=0.8, linestyle=":",
                   label=f"median = {median_v:+.3f} SD")

        p_str = _fmt_pvalue(tr.p_value)
        ax.set_title(f"{label}  (d = {tr.cohen_d:+.3f}, {p_str})",
                     fontsize=8, fontweight="bold")
        ax.set_xlabel("Over-serving (SD of trait)", fontsize=7)
        ax.set_ylabel("density", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(
            title=f"{pct_over:.0f}% > 0  (N={tr.n_users})",
            title_fontsize=5, fontsize=5, loc="upper right", framealpha=0.7,
        )

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"{group_name.replace('_', ' ').title()}"
        " — Content Over-Serving (model − actual preference)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = out_dir / f"{group_name}_over_serving.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_summary_bar(
    all_results: Dict[str, TraitOverServingResult],
    group_color_map: Dict[str, str],
    out_dir: Path,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Path:
    """Horizontal bar chart of mean standardised over-serving with bootstrap CIs."""
    from matplotlib.patches import Patch

    rng = np.random.default_rng(seed)

    _BarEntry = Tuple[str, float, float, float, float, float]
    bar_data: Dict[str, _BarEntry] = {}
    for key, tr in all_results.items():
        gname = key.split("::")[0]
        vals = tr.over_serving_std
        mean_v = float(np.mean(vals))

        boot_means = np.empty(n_bootstrap)
        n = len(vals)
        for b in range(n_bootstrap):
            boot_means[b] = vals[rng.integers(0, n, size=n)].mean()
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))

        bar_data[key] = (gname, mean_v, ci_lo, ci_hi, tr.cohen_d, tr.p_value)

    sorted_keys = sorted(bar_data, key=lambda k: abs(bar_data[k][1]), reverse=True)
    labels = [k.split("::")[-1] for k in sorted_keys]
    means = [bar_data[k][1] for k in sorted_keys]
    ci_los = [bar_data[k][2] for k in sorted_keys]
    ci_his = [bar_data[k][3] for k in sorted_keys]
    cohen_ds = [bar_data[k][4] for k in sorted_keys]
    p_vals = [bar_data[k][5] for k in sorted_keys]
    colors = [group_color_map.get(bar_data[k][0], "#999999") for k in sorted_keys]

    xerr_neg = [m - lo for m, lo in zip(means, ci_los)]
    xerr_pos = [hi - m for m, hi in zip(means, ci_his)]

    fig, ax = plt.subplots(figsize=scaled_figsize(9, max(3, 0.35 * len(labels))))
    y_pos = np.arange(len(labels))

    ax.barh(y_pos, means, color=colors, edgecolor="white", linewidth=0.5,
            xerr=[xerr_neg, xerr_pos],
            error_kw=dict(ecolor="#333333", capsize=2, linewidth=0.7))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Mean over-serving (SD of trait)  [95% bootstrap CI]")
    ax.set_title("Content Over-Serving Summary")

    x_max = max(abs(m) + xe for m, xe in zip(means, xerr_pos))
    for i, (d, p) in enumerate(zip(cohen_ds, p_vals)):
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        ann = f"d={d:+.2f}{sig}"
        ax.text(
            x_max * 1.02, y_pos[i], ann,
            va="center", ha="left", fontsize=5.5, color="#333333",
        )

    seen: set[str] = set()
    handles = []
    for k in sorted_keys:
        g = bar_data[k][0]
        if g not in seen:
            seen.add(g)
            handles.append(Patch(facecolor=group_color_map.get(g, "#999999"),
                                 label=g.replace("_", " ")))
    ax.legend(handles=handles, fontsize=6, loc="lower right", framealpha=0.8)
    plt.tight_layout()

    path = out_dir / "over_serving_summary_bar.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Headline text
# ---------------------------------------------------------------------------

def _generate_headline(
    all_results: Dict[str, TraitOverServingResult],
) -> List[str]:
    """Auto-generate verbal summary sentences for the largest over-serving effects."""
    if not all_results:
        return ["No traits had sufficient data for over-serving analysis."]

    biggest_key = max(
        all_results,
        key=lambda k: abs(all_results[k].cohen_d),
    )
    tr = all_results[biggest_key]
    trait_name = biggest_key.split("::")[-1]
    mean_std = float(np.mean(tr.over_serving_std))
    direction = "over" if mean_std > 0 else "under"
    p_str = _fmt_pvalue(tr.p_value)

    sentences = [
        f"The model {direction}-serves {trait_name} by "
        f"{abs(mean_std):.3f} SD on average "
        f"(d = {tr.cohen_d:+.3f}, {p_str}, "
        f"N = {tr.n_users} users with trait variation).",
    ]

    sig_keys = [
        k for k in all_results if all_results[k].p_value < 0.05
    ]
    n_sig = len(sig_keys)
    n_total = len(all_results)
    if n_sig > 0:
        n_over = sum(1 for k in sig_keys if all_results[k].cohen_d > 0)
        n_under = n_sig - n_over
        sentences.append(
            f"{n_sig}/{n_total} traits show statistically significant "
            f"over-serving effects (p < .05): "
            f"{n_over} over-served, {n_under} under-served."
        )

    return sentences


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class TraitOverServingModule(EvalModule):
    name = "trait_over_serving"
    description = (
        "Measures content over-serving in natural trait units: "
        "how much more/less of each trait the model serves relative to "
        "actual user preferences"
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
        n_users_total = preds["did"].n_unique()
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
            return {
                "skipped": True,
                "reason": f"only {n_posts_matched} posts matched inferences",
            }

        flat, group_names = _unnest_text_inferences(joined)

        trait_results, group_labels = _compute_over_serving(flat, group_names)

        # --- Plots ---
        group_color_map = {
            g: _GROUP_COLORS[i % len(_GROUP_COLORS)]
            for i, g in enumerate(group_names)
        }
        plot_paths: list[str] = []

        for gname in group_names:
            group_traits: Dict[str, TraitOverServingResult] = {}
            for label in group_labels.get(gname, []):
                key = f"{gname}::{label}"
                if key in trait_results:
                    group_traits[label] = trait_results[key]
            if not group_traits:
                continue
            plot_paths.append(
                str(_plot_group_over_serving(gname, group_traits, out_dir)))

        if trait_results:
            plot_paths.insert(
                0, str(_plot_summary_bar(trait_results, group_color_map,
                                         out_dir)))

        # --- Headline ---
        headline_sentences = _generate_headline(trait_results)

        # --- Summary JSON ---
        groups_json: Dict[str, Any] = {}
        for gname in group_names:
            gdict: Dict[str, Any] = {}
            for label in group_labels.get(gname, []):
                key = f"{gname}::{label}"
                if key not in trait_results:
                    continue
                tr = trait_results[key]
                mean_raw = float(np.mean(tr.over_serving_raw))
                mean_std = float(np.mean(tr.over_serving_std))
                gdict[label] = {
                    "mean_over_serving_raw": mean_raw,
                    "median_over_serving_raw": float(np.median(tr.over_serving_raw)),
                    "mean_over_serving_std": mean_std,
                    "median_over_serving_std": float(np.median(tr.over_serving_std)),
                    "sd_over_serving_std": float(np.std(tr.over_serving_std, ddof=1)),
                    "cohen_d": tr.cohen_d,
                    "t_stat": tr.t_stat,
                    "p_value": tr.p_value,
                    "pct_users_over_served": float(np.mean(tr.over_serving_raw > 0) * 100),
                    "global_trait_sd": tr.global_sd,
                    "n_users": tr.n_users,
                }
            if gdict:
                groups_json[gname] = gdict

        summary = {
            "headline": headline_sentences,
            "n_users_total": n_users_total,
            "n_users_eligible": n_users_eligible,
            "min_user_posts": MIN_USER_POSTS,
            "min_posts_per_half": MIN_POSTS_PER_HALF,
            "n_posts_matched": n_posts_matched,
            "groups": groups_json,
        }
        self.save_json(summary, out_dir / "trait_over_serving_summary.json")

        return {
            "headline": headline_sentences,
            "n_users_eligible": n_users_eligible,
            "n_posts_matched": n_posts_matched,
            "groups_plotted": len(groups_json),
            "plot_paths": plot_paths,
        }
