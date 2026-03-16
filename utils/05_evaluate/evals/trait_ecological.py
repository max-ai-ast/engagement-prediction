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
- <group>_ecological_p90.png:       true-pref rho split at 90th-percentile like volume
- <group>_ecological_p90_diff.png:  model bias split at 90th-percentile like volume
- ecological_gap_summary.png:       headline bar chart of ecological gaps
- trait_ecological_summary.json
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, NamedTuple, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

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

def _compute_all_rhos(
    flat: pl.DataFrame,
    group_names: list[str],
) -> tuple[Dict[str, float], Dict[str, TraitEcoResult], Dict[str, list[str]]]:
    """Tweet-level and per-user Spearman rhos via native Polars groupby.

    Replaces separate tweet-level and per-user Python loops with a single pass
    per trait column using Polars' Rust-native ``pl.corr(..., method="spearman")``
    inside ``group_by().agg()``.

    Returns ``(tweet_rhos, trait_results, group_labels)``.
    """
    tweet_rhos: Dict[str, float] = {}
    trait_results: Dict[str, TraitEcoResult] = {}
    group_labels: Dict[str, list[str]] = {}

    for gname in group_names:
        base = flat.select("did", "y_true", "y_pred_proba", gname).unnest(gname)
        cols = [c for c in base.columns if c not in ("did", "y_true", "y_pred_proba")]
        group_labels[gname] = cols

        for col in cols:
            key = f"{gname}::{col}"
            work = base.select("did", "y_true", "y_pred_proba", col)
            if work[col].dtype not in (pl.Float32, pl.Float64):
                work = work.with_columns(pl.col(col).cast(pl.Float64))
            valid = work.filter(pl.col(col).is_finite())

            if len(valid) < 10:
                continue

            rho_tweet = valid.select(
                pl.corr("y_true", col, method="spearman")
            ).item()
            if rho_tweet is None or not np.isfinite(rho_tweet):
                continue
            tweet_rhos[key] = float(rho_tweet)

            per_user = (
                valid
                .group_by("did")
                .agg(
                    pl.len().alias("n"),
                    pl.corr("y_true", col, method="spearman").alias("rho_true"),
                    pl.corr("y_pred_proba", col, method="spearman").alias("rho_pred"),
                )
                .filter(
                    (pl.col("n") >= MIN_POSTS_PER_USER_TRAIT)
                    & pl.col("rho_true").is_finite()
                    & pl.col("rho_pred").is_finite()
                )
            )

            if len(per_user) < MIN_USERS_PER_TRAIT:
                continue

            trait_results[key] = TraitEcoResult(
                rho_tweet=tweet_rhos[key],
                user_rho_true=per_user["rho_true"].to_numpy(),
                user_rho_pred=per_user["rho_pred"].to_numpy(),
                user_ids=per_user["did"].to_numpy(),
            )

    return tweet_rhos, trait_results, group_labels


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


_P90_COLORS = {"below": "#6baed6", "above": "#08519c"}
_P90_LABELS = {"below": "< p90", "above": "≥ p90"}


def _plot_group_p90(
    group_name: str,
    trait_results: Dict[str, TraitEcoResult],
    did_to_likes: Dict[Any, int],
    p90_threshold: float,
    out_dir: Path,
) -> Path:
    """Small-multiples: user-level rho_true KDEs split at the 90th-percentile like volume."""
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
        above = likes >= p90_threshold

        for key, mask in [("below", ~above), ("above", above)]:
            if mask.sum() < 5:
                continue
            subset = tr.user_rho_true[mask]
            density = _safe_kde(subset, grid)
            lbl = f"{_P90_LABELS[key]} (n={mask.sum()})"
            if density is not None:
                ax.fill_between(grid, density, alpha=0.25,
                                color=_P90_COLORS[key])
                ax.plot(grid, density, color=_P90_COLORS[key],
                        linewidth=1.2, label=lbl)
            else:
                ax.hist(subset, bins=25, density=True, alpha=0.25,
                        color=_P90_COLORS[key], label=lbl)

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
        f" — True-Preference ρ by Like Volume (p90 = {p90_threshold:.0f})",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = out_dir / f"{group_name}_ecological_p90.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_group_p90_diff(
    group_name: str,
    trait_results: Dict[str, TraitEcoResult],
    did_to_likes: Dict[Any, int],
    p90_threshold: float,
    out_dir: Path,
) -> Path:
    """Small-multiples: per-user (rho_pred - rho_true) split at 90th-percentile like volume."""
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

        likes = np.array([did_to_likes.get(uid, 0) for uid in tr.user_ids])
        above = likes >= p90_threshold

        all_lo, all_hi = float(diff.min()) - 0.05, float(diff.max()) + 0.05
        grid = np.linspace(all_lo, all_hi, 300)

        for key, mask in [("below", ~above), ("above", above)]:
            if mask.sum() < 5:
                continue
            subset = diff[mask]
            density = _safe_kde(subset, grid)
            lbl = f"{_P90_LABELS[key]} (n={mask.sum()})"
            if density is not None:
                ax.fill_between(grid, density, alpha=0.25,
                                color=_P90_COLORS[key])
                ax.plot(grid, density, color=_P90_COLORS[key],
                        linewidth=1.2, label=lbl)
            else:
                ax.hist(subset, bins=25, density=True, alpha=0.25,
                        color=_P90_COLORS[key], label=lbl)

        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        mean_d = float(np.mean(diff))
        ax.axvline(mean_d, color="#e06000", linewidth=0.9,
                   label=f"overall mean = {mean_d:+.4f}")

        ax.set_title(label, fontsize=8, fontweight="bold")
        ax.set_xlabel("ρ_pred − ρ_true  (per user)", fontsize=7)
        ax.set_ylabel("density", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=5, loc="upper right", framealpha=0.7)

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"{group_name.replace('_', ' ').title()}"
        f" — Model Bias (ρ_pred − ρ_true) by Like Volume (p90 = {p90_threshold:.0f})",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    path = out_dir / f"{group_name}_ecological_p90_diff.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


class _SplitSpec(NamedTuple):
    """Defines a high/low user split for the points plot."""
    did_is_high: Dict[Any, bool]
    tag: str            # file-name suffix, e.g. "p90", "p50", "likes50pct"
    title: str          # human-readable subtitle
    label_lo: str       # legend label for low group
    label_hi: str       # legend label for high group


def _compute_user_splits(
    did_to_likes: Dict[Any, int],
    eligible_likes: np.ndarray,
    user_ids: np.ndarray,
) -> list[_SplitSpec]:
    """Build the set of high/low user splits to visualise."""
    splits: list[_SplitSpec] = []

    for quantile, tag in [(0.9, "p90"), (0.5, "p50")]:
        thresh = float(np.quantile(eligible_likes, quantile))
        is_high = {uid: did_to_likes.get(uid, 0) >= thresh for uid in user_ids}
        pct_label = f"p{int(quantile * 100)}"
        splits.append(_SplitSpec(
            did_is_high=is_high,
            tag=tag,
            title=f"Like-count {pct_label} = {thresh:.0f}",
            label_lo=f"< {pct_label}",
            label_hi=f"≥ {pct_label}",
        ))

    sorted_uids = sorted(user_ids, key=lambda u: did_to_likes.get(u, 0),
                          reverse=True)
    total_likes = sum(did_to_likes.get(u, 0) for u in user_ids)
    if total_likes > 0:
        cumulative = 0
        high_set: set = set()
        for uid in sorted_uids:
            cumulative += did_to_likes.get(uid, 0)
            high_set.add(uid)
            if cumulative >= total_likes * 0.5:
                break
        is_high_vol = {uid: uid in high_set for uid in user_ids}
        n_hi = len(high_set)
        splits.append(_SplitSpec(
            did_is_high=is_high_vol,
            tag="likes50pct",
            title=f"Top-50%-of-likes ({n_hi} users = 50% of volume)",
            label_lo=f"remaining users",
            label_hi=f"top {n_hi} users (50% vol)",
        ))

    return splits


def _plot_group_split_points(
    group_name: str,
    trait_results: Dict[str, TraitEcoResult],
    split: _SplitSpec,
    out_dir: Path,
) -> Path:
    """Single-panel mean +/- 95% CI point plot of user-level rho_true for a given split.

    Traits are ordered left-to-right by ascending (high-group mean - low-group mean).
    Significance is assessed via Welch t-tests with Benjamini-Hochberg FDR
    correction within the group; traits surviving q < 0.05 are marked with *.
    """
    from scipy.stats import ttest_ind
    from statsmodels.stats.multitest import multipletests

    z95 = 1.96

    raw: Dict[str, Dict[str, Any]] = {}
    for label in trait_results:
        tr = trait_results[label]
        mask_hi = np.array([split.did_is_high.get(uid, False)
                            for uid in tr.user_ids])
        below_vals = tr.user_rho_true[~mask_hi]
        above_vals = tr.user_rho_true[mask_hi]

        n_lo, n_hi = len(below_vals), len(above_vals)
        if n_lo < 5 or n_hi < 5:
            continue

        mean_lo = float(np.mean(below_vals))
        mean_hi = float(np.mean(above_vals))
        ci_lo = z95 * float(np.std(below_vals, ddof=1)) / np.sqrt(n_lo)
        ci_hi = z95 * float(np.std(above_vals, ddof=1)) / np.sqrt(n_hi)

        _, p_val = ttest_ind(above_vals, below_vals, equal_var=False)

        raw[label] = {
            "mean_lo": mean_lo, "ci_lo": ci_lo, "n_lo": n_lo,
            "mean_hi": mean_hi, "ci_hi": ci_hi, "n_hi": n_hi,
            "diff": mean_hi - mean_lo,
            "p_val": float(p_val),
        }

    labels = sorted(raw, key=lambda k: raw[k]["diff"])
    n = len(labels)
    fname = f"{group_name}_ecological_{split.tag}_points.png"
    if n == 0:
        path = out_dir / fname
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                ha="center", va="center", fontsize=9, color="#999999")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    x = np.arange(n)
    dodge = 0.15

    means_lo = [raw[l]["mean_lo"] for l in labels]
    cis_lo   = [raw[l]["ci_lo"]   for l in labels]
    means_hi = [raw[l]["mean_hi"] for l in labels]
    cis_hi   = [raw[l]["ci_hi"]   for l in labels]

    fig, ax = plt.subplots(figsize=(max(6, 0.55 * n), 5))

    ax.errorbar(
        x - dodge, means_lo, yerr=cis_lo,
        fmt="o", color=_P90_COLORS["below"], markersize=5,
        capsize=3, capthick=0.8, linewidth=0.8,
        label=split.label_lo,
    )
    ax.errorbar(
        x + dodge, means_hi, yerr=cis_hi,
        fmt="o", color=_P90_COLORS["above"], markersize=5,
        capsize=3, capthick=0.8, linewidth=0.8,
        label=split.label_hi,
    )

    p_vals = np.array([raw[l]["p_val"] for l in labels])
    _, q_vals, _, _ = multipletests(p_vals, alpha=0.05, method="fdr_bh")

    y_top = max(
        max(m + c for m, c in zip(means_lo, cis_lo)),
        max(m + c for m, c in zip(means_hi, cis_hi)),
    )
    star_y = y_top + 0.02
    for i, q in enumerate(q_vals):
        if q < 0.05:
            ax.text(i, star_y, "*", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color="#333333")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("User-level Spearman ρ(trait, liked)  [mean ± 95% CI]",
                   fontsize=8)
    ax.set_title(
        f"{group_name.replace('_', ' ').title()}"
        f" — True-Preference ρ  ({split.title})",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=7, loc="best", framealpha=0.8)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()

    path = out_dir / fname
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

        # --- Compute tweet-level & per-user rhos via Polars groupby ---
        tweet_rhos, trait_results, group_labels = _compute_all_rhos(
            flat, group_names,
        )

        user_ids = flat["did"].unique().to_numpy()

        # --- Like-volume thresholds from user metadata ---
        meta = ctx.user_metadata_df
        did_to_likes: Dict[Any, int] = dict(
            zip(meta["did"], meta["num_total_likes"])
        )
        eligible_likes = np.array([
            did_to_likes.get(uid, 0) for uid in user_ids
        ])
        p90_threshold = float(np.quantile(eligible_likes, 0.9))

        user_splits = _compute_user_splits(did_to_likes, eligible_likes,
                                           user_ids)

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
                str(_plot_group_p90(gname, group_traits,
                                   did_to_likes, p90_threshold,
                                   out_dir)))
            plot_paths.append(
                str(_plot_group_p90_diff(gname, group_traits,
                                        did_to_likes, p90_threshold,
                                        out_dir)))
            for sp in user_splits:
                plot_paths.append(
                    str(_plot_group_split_points(gname, group_traits,
                                                sp, out_dir)))

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
