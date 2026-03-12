#!/usr/bin/env python3

"""
Trait Ecological Evaluation Module

Surfaces ecological fallacy / prolific-liker bias by comparing three
quantities for each NLP content trait:

(1) Tweet-level rho: Spearman correlation between the trait and y_true
    across all rows, ignoring user identity.  Because each row contributes
    equally, users who generate more likes dominate this signal.

(2) User-level true-preference rho distribution: for each user, the
    within-user Spearman correlation between the trait and y_true.
    Gives each user equal voice regardless of volume.

(3) User-level model-preference rho distribution: same as (2) but using
    y_pred_proba instead of y_true.  Reveals whether the model inherits
    the ecological distortion.

Additionally computes per-user rho_pred - rho_true differences to surface
systematic model bias at the individual level.

Outputs (under trait_ecological/):
- <group>_ecological.png:           composite KDE plot per inference group
- <group>_ecological_diff.png:      difference distribution per inference group
- <group>_ecological_quintiles.png: true-pref rho by like-volume quintile
- ecological_gap_summary.png:       headline bar chart of ecological gaps
- trait_ecological_summary.json
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde, spearmanr

from . import EvalContext, EvalModule
from .trait_corrs import _load_inferences, _unnest_text_inferences
from .trait_amplification import MIN_USER_POSTS, _filter_eligible_users

MIN_USERS_PER_TRAIT = 30
MIN_POSTS_PER_USER_TRAIT = 10

_GROUP_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TraitEcoResult(NamedTuple):
    rho_tweet: float
    user_rho_true: np.ndarray
    user_rho_pred: np.ndarray
    user_ids: np.ndarray


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _compute_tweet_level_rhos(
    y_true: np.ndarray,
    trait_arrays: Dict[str, np.ndarray],
    finite_masks: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Tweet-level Spearman rho(trait, y_true) ignoring user identity."""
    results: Dict[str, float] = {}
    for key, vals in trait_arrays.items():
        mask = finite_masks[key]
        if mask.sum() < 10:
            continue
        rho, _ = spearmanr(y_true[mask], vals[mask])
        results[key] = float(rho)
    return results


def _compute_per_user_rhos(
    user_ids: np.ndarray,
    user_to_rows: Dict[Any, np.ndarray],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trait_arrays: Dict[str, np.ndarray],
    finite_masks: Dict[str, np.ndarray],
    valid_keys: set[str],
) -> Dict[str, Tuple[List[float], List[float], List[Any]]]:
    """Within-user rho(trait, y_true) and rho(trait, y_pred_proba) per trait.

    Returns {key: (list_of_rho_true, list_of_rho_pred, list_of_user_ids)}.
    Only includes users with >= MIN_POSTS_PER_USER_TRAIT finite trait values
    and non-zero variance in both the outcome and trait columns.
    """
    rho_true_lists: Dict[str, List[float]] = defaultdict(list)
    rho_pred_lists: Dict[str, List[float]] = defaultdict(list)
    uid_lists: Dict[str, List[Any]] = defaultdict(list)

    for uid in user_ids:
        rows = user_to_rows[uid]
        yt = y_true[rows]
        yp = y_pred[rows]

        for key in valid_keys:
            tv = trait_arrays[key][rows]
            m = finite_masks[key][rows]
            if int(m.sum()) < MIN_POSTS_PER_USER_TRAIT:
                continue
            yt_m, yp_m, tv_m = yt[m], yp[m], tv[m]
            if yt_m.std() == 0 or tv_m.std() == 0 or yp_m.std() == 0:
                continue
            rt, _ = spearmanr(yt_m, tv_m)
            rp, _ = spearmanr(yp_m, tv_m)
            rho_true_lists[key].append(float(rt))
            rho_pred_lists[key].append(float(rp))
            uid_lists[key].append(uid)

    return {
        key: (rho_true_lists[key], rho_pred_lists[key], uid_lists[key])
        for key in valid_keys
        if len(rho_true_lists[key]) >= MIN_USERS_PER_TRAIT
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _safe_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    """Gaussian KDE evaluated on *grid*, or None on failure."""
    try:
        return gaussian_kde(values)(grid)
    except Exception:
        return None


def _plot_group_ecological(
    group_name: str,
    trait_results: Dict[str, TraitEcoResult],
    out_dir: Path,
) -> Path:
    """Small-multiples: user-level KDEs + tweet-level vertical line."""
    labels = sorted(trait_results.keys())
    n = len(labels)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for idx, label in enumerate(labels):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        tr = trait_results[label]
        grid = np.linspace(-1, 1, 300)

        density_true = _safe_kde(tr.user_rho_true, grid)
        if density_true is not None:
            ax.fill_between(grid, density_true, alpha=0.3, color="#999999",
                            label="user ρ(trait, liked)")
            ax.plot(grid, density_true, color="#666666", linewidth=0.8)
        else:
            ax.hist(tr.user_rho_true, bins=30, density=True, alpha=0.3,
                    color="#999999", label="user ρ(trait, liked)")

        density_pred = _safe_kde(tr.user_rho_pred, grid)
        if density_pred is not None:
            ax.plot(grid, density_pred, color="#4878CF", linewidth=1.3,
                    label="user ρ(trait, pred)")
        else:
            ax.hist(tr.user_rho_pred, bins=30, density=True, alpha=0.3,
                    color="#4878CF", label="user ρ(trait, pred)")

        ax.axvline(tr.rho_tweet, color="#D65F5F", linestyle="--",
                   linewidth=1.3,
                   label=f"tweet-level ρ = {tr.rho_tweet:.3f}")

        mean_true = float(np.mean(tr.user_rho_true))
        mean_pred = float(np.mean(tr.user_rho_pred))
        ax.axvline(mean_true, color="#666666", linewidth=0.9, alpha=0.7,
                   label=f"mean user true = {mean_true:.3f}")
        ax.axvline(mean_pred, color="#4878CF", linewidth=0.9, alpha=0.7,
                   label=f"mean user pred = {mean_pred:.3f}")

        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.set_xlabel("Spearman ρ", fontsize=7)
        ax.set_ylabel("density", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, loc="upper right", framealpha=0.7)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"{group_name.replace('_', ' ').title()} — Ecological Validity",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = out_dir / f"{group_name}_ecological.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_group_diff(
    group_name: str,
    trait_results: Dict[str, TraitEcoResult],
    out_dir: Path,
) -> Path:
    """Small-multiples: per-user rho_pred - rho_true distribution."""
    labels = sorted(trait_results.keys())
    n = len(labels)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)

    for idx, label in enumerate(labels):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        tr = trait_results[label]
        diff = tr.user_rho_pred - tr.user_rho_true

        lo = float(diff.min()) - 0.05
        hi = float(diff.max()) + 0.05
        grid = np.linspace(lo, hi, 300)

        density = _safe_kde(diff, grid)
        if density is not None:
            ax.fill_between(grid, density, alpha=0.35, color="#ff7f0e")
            ax.plot(grid, density, color="#e06000", linewidth=1.0)
        else:
            ax.hist(diff, bins=30, density=True, alpha=0.35, color="#ff7f0e")

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        mean_d = float(np.mean(diff))
        median_d = float(np.median(diff))
        ax.axvline(mean_d, color="#e06000", linewidth=1.0,
                   label=f"mean = {mean_d:+.4f}")

        ax.set_title(f"{label}  (med = {median_d:+.4f})",
                     fontsize=8, fontweight="bold")
        ax.set_xlabel("ρ_pred − ρ_true  (per user)", fontsize=7)
        ax.set_ylabel("density", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.7)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"{group_name.replace('_', ' ').title()}"
        " — Model Bias per User  (ρ_pred − ρ_true)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = out_dir / f"{group_name}_ecological_diff.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


_QUINTILE_COLORS = ["#c6dbef", "#6baed6", "#3182bd", "#08519c", "#08306b"]
_QUINTILE_LABELS = ["Q1 (fewest)", "Q2", "Q3", "Q4", "Q5 (most)"]


def _plot_group_quintiles(
    group_name: str,
    trait_results: Dict[str, TraitEcoResult],
    did_to_likes: Dict[Any, int],
    quintile_edges: np.ndarray,
    out_dir: Path,
) -> Path:
    """Small-multiples: user-level rho_true KDEs stratified by like-volume quintile.

    Each panel shows one trait with five overlaid KDE curves (one per quintile
    of num_total_likes) plus the tweet-level rho as a dashed vertical line.
    """
    labels = sorted(trait_results.keys())
    n = len(labels)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5 * ncols, 3.5 * nrows),
                             squeeze=False)
    grid = np.linspace(-1, 1, 300)

    for idx, label in enumerate(labels):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        tr = trait_results[label]

        likes = np.array([did_to_likes.get(uid, 0) for uid in tr.user_ids])
        q_assign = np.digitize(likes, quintile_edges[1:-1])

        for qi in range(5):
            mask = q_assign == qi
            if mask.sum() < 5:
                continue
            subset = tr.user_rho_true[mask]
            density = _safe_kde(subset, grid)
            if density is not None:
                ax.plot(grid, density, color=_QUINTILE_COLORS[qi],
                        linewidth=1.2, label=f"{_QUINTILE_LABELS[qi]} (n={mask.sum()})")
            else:
                ax.hist(subset, bins=25, density=True, alpha=0.25,
                        color=_QUINTILE_COLORS[qi],
                        label=f"{_QUINTILE_LABELS[qi]} (n={mask.sum()})")

        ax.axvline(tr.rho_tweet, color="#D65F5F", linestyle="--",
                   linewidth=1.3,
                   label=f"tweet-level ρ = {tr.rho_tweet:.3f}")

        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.set_xlabel("Spearman ρ(trait, liked)", fontsize=7)
        ax.set_ylabel("density", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, loc="upper right", framealpha=0.7)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"{group_name.replace('_', ' ').title()}"
        " — True-Preference ρ by Like-Volume Quintile",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = out_dir / f"{group_name}_ecological_quintiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_gap_summary(
    all_gaps: Dict[str, Tuple[str, float]],
    group_color_map: Dict[str, str],
    out_dir: Path,
) -> Path:
    """Horizontal bar chart of ecological gap per trait across all groups."""
    from matplotlib.patches import Patch

    sorted_keys = sorted(all_gaps,
                         key=lambda k: abs(all_gaps[k][1]), reverse=True)
    labels = [k.split("::")[-1] for k in sorted_keys]
    gaps = [all_gaps[k][1] for k in sorted_keys]
    colors = [group_color_map.get(all_gaps[k][0], "#999999")
              for k in sorted_keys]

    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(labels))))
    y_pos = np.arange(len(labels))

    ax.barh(y_pos, gaps, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Ecological gap  (tweet-level ρ − mean user-level ρ)")
    ax.set_title("Ecological Gap Summary: Prolific-Liker Distortion per Trait")

    seen: set[str] = set()
    handles = []
    for k in sorted_keys:
        g = all_gaps[k][0]
        if g not in seen:
            seen.add(g)
            handles.append(Patch(facecolor=group_color_map.get(g, "#999999"),
                                 label=g.replace("_", " ")))
    ax.legend(handles=handles, fontsize=6, loc="lower right", framealpha=0.8)
    plt.tight_layout()

    path = out_dir / "ecological_gap_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class TraitEcologicalModule(EvalModule):
    name = "trait_ecological"
    description = (
        "Surfaces ecological fallacy / prolific-liker bias by comparing "
        "tweet-level vs user-level trait-engagement correlations"
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
        n_users_total = preds["did"].n_unique()
        preds = _filter_eligible_users(preds)
        n_users_eligible = preds["did"].n_unique()
        if n_users_eligible < 5:
            return {"skipped": True, "reason": f"only {n_users_eligible} eligible users"}

        # --- Join to inferences, unnest ---
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

        # --- Extract numpy arrays ---
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

        # --- (1) Tweet-level rhos ---
        tweet_rhos = _compute_tweet_level_rhos(
            y_true, trait_arrays, finite_masks,
        )

        # --- (2) & (3) Per-user rhos ---
        per_user = _compute_per_user_rhos(
            user_ids, user_to_rows, y_true, y_pred,
            trait_arrays, finite_masks,
            valid_keys=set(tweet_rhos.keys()),
        )

        # --- Assemble per-trait result tuples ---
        trait_results: Dict[str, TraitEcoResult] = {}
        for key, (rt_list, rp_list, uid_list) in per_user.items():
            trait_results[key] = TraitEcoResult(
                rho_tweet=tweet_rhos[key],
                user_rho_true=np.array(rt_list),
                user_rho_pred=np.array(rp_list),
                user_ids=np.array(uid_list),
            )

        # --- Like-volume quintiles from user metadata ---
        meta = ctx.user_metadata_df
        did_to_likes: Dict[Any, int] = dict(
            zip(meta["did"], meta["num_total_likes"])
        )
        eligible_likes = np.array([
            did_to_likes.get(uid, 0) for uid in user_ids
        ])
        quintile_edges = np.quantile(
            eligible_likes, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        )

        # --- Plots ---
        group_color_map = {
            g: _GROUP_COLORS[i % len(_GROUP_COLORS)]
            for i, g in enumerate(group_names)
        }
        plot_paths: list[str] = []
        all_gaps: Dict[str, Tuple[str, float]] = {}

        for gname in group_names:
            group_traits: Dict[str, TraitEcoResult] = {}
            for label in group_labels.get(gname, []):
                key = f"{gname}::{label}"
                if key not in trait_results:
                    continue
                group_traits[label] = trait_results[key]
                gap = (trait_results[key].rho_tweet
                       - float(np.mean(trait_results[key].user_rho_true)))
                all_gaps[key] = (gname, gap)
            if not group_traits:
                continue

            plot_paths.append(
                str(_plot_group_ecological(gname, group_traits, out_dir)))
            plot_paths.append(
                str(_plot_group_diff(gname, group_traits, out_dir)))
            plot_paths.append(
                str(_plot_group_quintiles(gname, group_traits,
                                         did_to_likes, quintile_edges,
                                         out_dir)))

        if all_gaps:
            plot_paths.insert(
                0, str(_plot_gap_summary(all_gaps, group_color_map, out_dir)))

        # --- Summary JSON ---
        groups_json: Dict[str, Any] = {}
        all_abs_gaps: list[float] = []

        for gname in group_names:
            gdict: Dict[str, Any] = {}
            for label in group_labels.get(gname, []):
                key = f"{gname}::{label}"
                if key not in trait_results:
                    continue
                tr = trait_results[key]
                diff = tr.user_rho_pred - tr.user_rho_true
                eco_gap = tr.rho_tweet - float(np.mean(tr.user_rho_true))
                all_abs_gaps.append(abs(eco_gap))
                gdict[label] = {
                    "rho_tweet_level": tr.rho_tweet,
                    "rho_user_true_mean": float(np.mean(tr.user_rho_true)),
                    "rho_user_true_median": float(np.median(tr.user_rho_true)),
                    "rho_user_pred_mean": float(np.mean(tr.user_rho_pred)),
                    "rho_user_pred_median": float(np.median(tr.user_rho_pred)),
                    "ecological_gap": eco_gap,
                    "diff_mean": float(np.mean(diff)),
                    "diff_median": float(np.median(diff)),
                    "n_users_computed": len(tr.user_rho_true),
                }
            if gdict:
                groups_json[gname] = gdict

        summary = {
            "n_users_total": n_users_total,
            "n_users_eligible": n_users_eligible,
            "min_user_posts": MIN_USER_POSTS,
            "n_posts_matched": n_posts_matched,
            "mean_abs_ecological_gap": (
                float(np.mean(all_abs_gaps)) if all_abs_gaps else 0.0
            ),
            "groups": groups_json,
        }
        self.save_json(summary, out_dir / "trait_ecological_summary.json")

        return {
            "n_users_eligible": n_users_eligible,
            "n_posts_matched": n_posts_matched,
            "groups_plotted": len(groups_json),
            "mean_abs_ecological_gap": summary["mean_abs_ecological_gap"],
            "plot_paths": plot_paths,
        }
