#!/usr/bin/env python3

"""
Trait Amplification Evaluation Module

Compares the model's association between predicted engagement and each NLP
content trait against the *actual* association (based on y_true).  The
difference -- "amplification" -- reveals traits the model over- or
under-weights relative to real user preferences.

Uses the full holdout set (positives + random negatives).  Because the holdout
has a balanced 1:1 pos/neg ratio per user, within-user demeaning of y_true is a
no-op (every user's mean is 0.5).  Demeaning of y_pred_proba is meaningful
since the model's per-user base-rate varies.  Trait values are intentionally
*not* demeaned: with random negatives the per-user trait mean is a meaningless
mix of the user's engaged-content signal and the platform average.

Two correlation perspectives are reported:
- **Pooled (post-weighted):** a single Spearman rho across all posts (each
  post contributes equally, so prolific users have more influence).
- **User-averaged:** mean of per-user Spearman rho values (each user
  contributes equally, directly capturing "the average user's preference").

Bootstrap resampling over users provides CIs for both perspectives.

Outputs (under trait_amplification/):
- amplification_scatter.png: headline scatter of rho_true vs rho_pred (pooled)
- <group>_amplification.png: paired-bar detail per inference group (pooled)
- amplification_scatter_user_avg.png: headline scatter (user-averaged)
- <group>_user_avg_amplification.png: paired-bar detail (user-averaged)
- <group>_per_user_scatter.png: small-multiples scatter of per-user rho_pred
  vs rho_true for each trait in the group
- trait_amplification_summary.json: all correlations, deltas, CIs
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr
from tqdm import tqdm

from . import EvalContext, EvalModule, scaled_figsize
from .trait_corrs import _load_inferences, _unnest_text_inferences, eb_shrink

MIN_USER_POSTS = 20
N_BOOTSTRAP = 500
ALPHA = 0.05
PER_USER_MIN_POSTS = 4  # minimum for Fisher-z variance 1/(n-3) to be finite

_GROUP_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _filter_eligible_users(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only users with >= MIN_USER_POSTS posts and at least 1 pos + 1 neg."""
    eligible = (
        df.group_by("did")
        .agg(pl.len().alias("n"), pl.col("y_true").sum().alias("n_pos"))
        .filter(
            (pl.col("n") >= MIN_USER_POSTS)
            & (pl.col("n_pos") >= 1)
            & (pl.col("n_pos") < pl.col("n"))
        )
        .select("did")
    )
    return df.join(eligible, on="did", how="semi")


def _demean_within_user(df: pl.DataFrame) -> pl.DataFrame:
    # y_true demeaning is a no-op with balanced 1:1 pos/neg per user (constant
    # 0.5 shift); kept for forward-compatibility if the sampling ratio changes.
    # y_pred_proba demeaning is meaningful (model base-rate varies per user).
    return df.with_columns(
        (pl.col("y_true") - pl.col("y_true").mean().over("did")).alias("y_true_c"),
        (pl.col("y_pred_proba") - pl.col("y_pred_proba").mean().over("did")).alias("y_pred_c"),
    )


# key = "group::label", value = (rho_true, rho_pred, delta)
CorrelationResults = Dict[str, Tuple[float, float, float]]


def _compute_all_correlations(
    y_true_c: np.ndarray,
    y_pred_c: np.ndarray,
    trait_arrays: Dict[str, np.ndarray],
    finite_masks: Dict[str, np.ndarray],
) -> CorrelationResults:
    """Point estimates for every trait (vectorised Pearson-of-ranks)."""
    keys = list(trait_arrays.keys())
    dense_keys = [k for k in keys if finite_masks[k].all()]
    sparse_keys = [k for k in keys if not finite_masks[k].all()]
    results: CorrelationResults = {}

    if dense_keys:
        yt_r = rankdata(y_true_c).astype(np.float64)
        yp_r = rankdata(y_pred_c).astype(np.float64)
        trait_rank_mat = np.column_stack(
            [rankdata(trait_arrays[k]).astype(np.float64) for k in dense_keys]
        )
        T_c = trait_rank_mat - trait_rank_mat.mean(axis=0)
        yt_c = yt_r - yt_r.mean()
        yp_c = yp_r - yp_r.mean()
        rt_all = _col_pearson(T_c, yt_c)
        rp_all = _col_pearson(T_c, yp_c)
        for i, k in enumerate(dense_keys):
            results[k] = (float(rt_all[i]), float(rp_all[i]), float(rp_all[i] - rt_all[i]))

    for k in sparse_keys:
        mask = finite_masks[k]
        if mask.sum() < 10:
            continue
        yt_sub = rankdata(y_true_c[mask]).astype(np.float64)
        yp_sub = rankdata(y_pred_c[mask]).astype(np.float64)
        tv_sub = rankdata(trait_arrays[k][mask]).astype(np.float64)
        rt = _pearson_1d(yt_sub, tv_sub)
        rp = _pearson_1d(yp_sub, tv_sub)
        results[k] = (float(rt), float(rp), float(rp - rt))

    return results


def _col_pearson(Xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """Pearson r of each column of centred *Xc* with centred *yc*."""
    x_norms = np.sqrt((Xc * Xc).sum(axis=0))
    y_norm = np.sqrt(yc @ yc)
    denom = x_norms * y_norm
    denom = np.where(denom < 1e-30, 1.0, denom)
    return (Xc.T @ yc) / denom


def _pearson_1d(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r of two 1-D arrays (no NaN handling)."""
    xc = x - x.mean()
    yc = y - y.mean()
    denom = np.sqrt((xc @ xc) * (yc @ yc))
    return float((xc @ yc) / denom) if denom > 1e-30 else 0.0


def _rank_columns_no_ties(X: np.ndarray) -> np.ndarray:
    """Rank each column of X via argsort (no tie correction).

    Valid for continuous float scores where ties are negligible.
    """
    n = X.shape[0]
    order = np.argsort(X, axis=0)
    ranks = np.empty_like(X, dtype=np.float64)
    np.put_along_axis(
        ranks, order,
        np.arange(1, n + 1, dtype=np.float64).reshape(-1, 1),
        axis=0,
    )
    return ranks


def _bootstrap_cis(
    user_ids: np.ndarray,
    user_to_rows: Dict[Any, np.ndarray],
    y_true_c: np.ndarray,
    y_pred_c: np.ndarray,
    trait_arrays: Dict[str, np.ndarray],
    finite_masks: Dict[str, np.ndarray],
    valid_keys: set[str],
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float, float, float, float]]:
    """Bootstrap CIs via per-user sufficient statistics.

    Dense traits: pre-aggregate per-user sums / cross-products of pre-ranked
    arrays so each bootstrap iteration only sums O(n_users) vectors instead
    of re-indexing O(n_posts) rows.  Sparse traits fall back to the
    per-iteration index approach.

    Returns {key: (rt_lo, rt_hi, rp_lo, rp_hi, delta_lo, delta_hi)}.
    """
    rng = np.random.default_rng(seed)
    n_users = len(user_ids)
    keys = sorted(valid_keys)

    dense_keys = [k for k in keys if finite_masks[k].all()]
    sparse_keys = [k for k in keys if not finite_masks[k].all()]
    n_dense = len(dense_keys)

    # --- pre-rank all arrays once ----------------------------------------
    yt_r = rankdata(y_true_c).astype(np.float64)
    yp_r = rankdata(y_pred_c).astype(np.float64)

    dense_rank_mat: np.ndarray | None = None
    if n_dense > 0:
        dense_rank_mat = np.column_stack(
            [rankdata(trait_arrays[k]).astype(np.float64) for k in dense_keys]
        )

    # --- per-user sufficient statistics for dense traits -----------------
    pu_n = np.empty(n_users, dtype=np.float64)
    pu_syt = np.empty(n_users)
    pu_syp = np.empty(n_users)
    pu_syt2 = np.empty(n_users)
    pu_syp2 = np.empty(n_users)
    pu_st: np.ndarray | None = None
    pu_st2: np.ndarray | None = None
    pu_syt_t: np.ndarray | None = None
    pu_syp_t: np.ndarray | None = None

    if n_dense > 0:
        pu_st = np.empty((n_users, n_dense))
        pu_st2 = np.empty((n_users, n_dense))
        pu_syt_t = np.empty((n_users, n_dense))
        pu_syp_t = np.empty((n_users, n_dense))

    for i, u in enumerate(user_ids):
        rows = user_to_rows[u]
        yt_u = yt_r[rows]
        yp_u = yp_r[rows]
        pu_n[i] = len(rows)
        pu_syt[i] = yt_u.sum()
        pu_syp[i] = yp_u.sum()
        pu_syt2[i] = (yt_u * yt_u).sum()
        pu_syp2[i] = (yp_u * yp_u).sum()
        if dense_rank_mat is not None:
            t_u = dense_rank_mat[rows]
            pu_st[i] = t_u.sum(axis=0)
            pu_st2[i] = (t_u * t_u).sum(axis=0)
            pu_syt_t[i] = (yt_u[:, None] * t_u).sum(axis=0)
            pu_syp_t[i] = (yp_u[:, None] * t_u).sum(axis=0)

    # --- sparse: pre-rank + index machinery (only when needed) -----------
    sparse_trait_ranks: Dict[str, np.ndarray] = {}
    all_rows: np.ndarray | None = None
    offsets: np.ndarray | None = None

    for k in sparse_keys:
        m = finite_masks[k]
        ranked = np.full(len(y_true_c), np.nan, dtype=np.float64)
        ranked[m] = rankdata(trait_arrays[k][m])
        sparse_trait_ranks[k] = ranked

    if sparse_keys:
        row_arrays = [user_to_rows[u] for u in user_ids]
        all_rows = np.concatenate(row_arrays)
        offsets = np.empty(n_users + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum([len(r) for r in row_arrays], out=offsets[1:])

    # --- accumulators ----------------------------------------------------
    boot_rt: Dict[str, List[float]] = defaultdict(list)
    boot_rp: Dict[str, List[float]] = defaultdict(list)
    boot_d: Dict[str, List[float]] = defaultdict(list)

    for _b in tqdm(range(n_bootstrap), desc="      bootstrap", leave=False):
        sel = rng.integers(0, n_users, size=n_users)

        # --- dense: sufficient statistics --------------------------------
        if n_dense > 0:
            N = pu_n[sel].sum()
            SYt = pu_syt[sel].sum()
            SYp = pu_syp[sel].sum()
            SYt2 = pu_syt2[sel].sum()
            SYp2 = pu_syp2[sel].sum()
            ST = pu_st[sel].sum(axis=0)
            ST2 = pu_st2[sel].sum(axis=0)
            SYtT = pu_syt_t[sel].sum(axis=0)
            SYpT = pu_syp_t[sel].sum(axis=0)

            var_t = N * ST2 - ST * ST
            var_yt = N * SYt2 - SYt * SYt
            denom_t = np.sqrt(np.maximum(var_yt * var_t, 0.0))
            rt_all = np.where(
                denom_t > 1e-30,
                (N * SYtT - SYt * ST) / denom_t,
                0.0,
            )

            var_yp = N * SYp2 - SYp * SYp
            denom_p = np.sqrt(np.maximum(var_yp * var_t, 0.0))
            rp_all = np.where(
                denom_p > 1e-30,
                (N * SYpT - SYp * ST) / denom_p,
                0.0,
            )

            for i, k in enumerate(dense_keys):
                boot_rt[k].append(float(rt_all[i]))
                boot_rp[k].append(float(rp_all[i]))
                boot_d[k].append(float(rp_all[i] - rt_all[i]))

        # --- sparse: post-level index approach ---------------------------
        if sparse_keys:
            sel_starts = offsets[sel]
            sel_lens = offsets[sel + 1] - sel_starts
            total = int(sel_lens.sum())
            rep_starts = np.repeat(sel_starts, sel_lens)
            cum = np.zeros(n_users + 1, dtype=np.int64)
            np.cumsum(sel_lens, out=cum[1:])
            within = np.arange(total, dtype=np.int64) - np.repeat(cum[:-1], sel_lens)
            idx = all_rows[rep_starts + within]

            yt_b = yt_r[idx]
            yp_b = yp_r[idx]

            for k in sparse_keys:
                tv_b = sparse_trait_ranks[k][idx]
                m = np.isfinite(tv_b)
                if m.sum() < 10:
                    continue
                rt = _pearson_1d(yt_b[m], tv_b[m])
                rp = _pearson_1d(yp_b[m], tv_b[m])
                boot_rt[k].append(rt)
                boot_rp[k].append(rp)
                boot_d[k].append(rp - rt)

    lo_q = alpha / 2 * 100
    hi_q = (1 - alpha / 2) * 100
    cis = {}
    for key in keys:
        if key not in boot_d or len(boot_d[key]) < 10:
            continue
        cis[key] = (
            float(np.percentile(boot_rt[key], lo_q)),
            float(np.percentile(boot_rt[key], hi_q)),
            float(np.percentile(boot_rp[key], lo_q)),
            float(np.percentile(boot_rp[key], hi_q)),
            float(np.percentile(boot_d[key], lo_q)),
            float(np.percentile(boot_d[key], hi_q)),
        )
    return cis


def _compute_per_user_rhos(
    user_ids: np.ndarray,
    user_to_rows: Dict[Any, np.ndarray],
    y_true_c: np.ndarray,
    y_pred_c: np.ndarray,
    trait_arrays: Dict[str, np.ndarray],
    finite_masks: Dict[str, np.ndarray],
    valid_keys: set[str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Per-user Spearman rho for each trait, with EB shrinkage.

    Iterates users (outer) with dense-trait batching so y_true / y_pred are
    ranked only once per user instead of once per (user, trait).
    """
    keys = sorted(valid_keys)
    dense_keys = [k for k in keys if finite_masks[k].all()]
    sparse_keys = [k for k in keys if not finite_masks[k].all()]

    dense_mat: np.ndarray | None = None
    if dense_keys:
        dense_mat = np.column_stack([trait_arrays[k] for k in dense_keys])

    acc_rt: Dict[str, List[float]] = defaultdict(list)
    acc_rp: Dict[str, List[float]] = defaultdict(list)
    acc_ns: Dict[str, List[int]] = defaultdict(list)

    for u in tqdm(user_ids, desc="      per-user rho", leave=False):
        rows = user_to_rows[u]
        n_rows = len(rows)
        if n_rows < PER_USER_MIN_POSTS:
            continue

        yt = y_true_c[rows]
        yp = y_pred_c[rows]
        if np.std(yt) < 1e-12 or np.std(yp) < 1e-12:
            continue

        yt_r = rankdata(yt)
        yp_r = rankdata(yp)

        if dense_mat is not None:
            tv_all = dense_mat[rows]
            stds = tv_all.std(axis=0)
            valid_mask = stds > 1e-12

            if valid_mask.any():
                tv_valid = tv_all[:, valid_mask]
                tv_ranks = _rank_columns_no_ties(tv_valid)
                tv_c = tv_ranks - tv_ranks.mean(axis=0)
                yt_c = yt_r - yt_r.mean()
                yp_c = yp_r - yp_r.mean()

                rt_vals = _col_pearson(tv_c, yt_c)
                rp_vals = _col_pearson(tv_c, yp_c)

                valid_indices = np.where(valid_mask)[0]
                for j, col_idx in enumerate(valid_indices):
                    rt_v, rp_v = float(rt_vals[j]), float(rp_vals[j])
                    if np.isfinite(rt_v) and np.isfinite(rp_v):
                        k = dense_keys[col_idx]
                        acc_rt[k].append(rt_v)
                        acc_rp[k].append(rp_v)
                        acc_ns[k].append(n_rows)

        for k in sparse_keys:
            row_finite = finite_masks[k][rows]
            vr = rows[row_finite]
            if len(vr) < PER_USER_MIN_POSTS:
                continue
            tv = trait_arrays[k][vr]
            if np.std(tv) < 1e-12:
                continue
            yt_sub = y_true_c[vr]
            yp_sub = y_pred_c[vr]
            if np.std(yt_sub) < 1e-12 or np.std(yp_sub) < 1e-12:
                continue
            tv_r = rankdata(tv)
            rt = _pearson_1d(rankdata(yt_sub), tv_r)
            rp = _pearson_1d(rankdata(yp_sub), tv_r)
            if np.isfinite(rt) and np.isfinite(rp):
                acc_rt[k].append(float(rt))
                acc_rp[k].append(float(rp))
                acc_ns[k].append(len(vr))

    results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for k in keys:
        if len(acc_rt[k]) >= 5:
            ns = np.array(acc_ns[k])
            shrunk_true, _ = eb_shrink(np.array(acc_rt[k]), ns)
            shrunk_pred, _ = eb_shrink(np.array(acc_rp[k]), ns)
            results[k] = (shrunk_true, shrunk_pred)

    return results


def _user_avg_correlations_and_cis(
    per_user_rhos: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
    seed: int = 42,
) -> Tuple[CorrelationResults, Dict[str, Tuple[float, float, float, float, float, float]]]:
    """Mean-of-per-user rho with bootstrap CIs (resampling users).

    Returns (point_estimates, cis) in the same formats as the pooled helpers
    so the plotting functions can be reused directly.
    """
    rng = np.random.default_rng(seed)

    point: CorrelationResults = {}
    cis: Dict[str, Tuple[float, float, float, float, float, float]] = {}

    for key, (rt_arr, rp_arr) in per_user_rhos.items():
        n = len(rt_arr)
        rt_mean = float(rt_arr.mean())
        rp_mean = float(rp_arr.mean())
        point[key] = (rt_mean, rp_mean, rp_mean - rt_mean)

        boot_rt = np.empty(n_bootstrap)
        boot_rp = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot_rt[b] = rt_arr[idx].mean()
            boot_rp[b] = rp_arr[idx].mean()
        boot_d = boot_rp - boot_rt

        lo_q = alpha / 2 * 100
        hi_q = (1 - alpha / 2) * 100
        cis[key] = (
            float(np.percentile(boot_rt, lo_q)),
            float(np.percentile(boot_rt, hi_q)),
            float(np.percentile(boot_rp, lo_q)),
            float(np.percentile(boot_rp, hi_q)),
            float(np.percentile(boot_d, lo_q)),
            float(np.percentile(boot_d, hi_q)),
        )

    return point, cis


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_scatter(
    group_results: Dict[str, Dict[str, Tuple[float, float, float]]],
    cis: Dict[str, Tuple],
    group_color_map: Dict[str, str],
    out_dir: Path,
    suffix: str = "",
) -> Path:
    """rho_true (x) vs rho_pred (y), one dot per trait, colored by group."""
    fig, ax = plt.subplots(figsize=scaled_figsize(7, 7))

    for gname, traits in group_results.items():
        color = group_color_map[gname]
        xs, ys = [], []
        xel, xeh, yel, yeh = [], [], [], []
        for label, (rt, rp, _) in traits.items():
            key = f"{gname}::{label}"
            if key not in cis:
                continue
            ci = cis[key]
            xs.append(rt); ys.append(rp)
            xel.append(rt - ci[0]); xeh.append(ci[1] - rt)
            yel.append(rp - ci[2]); yeh.append(ci[3] - rp)

        if not xs:
            continue
        ax.errorbar(
            xs, ys, xerr=[xel, xeh], yerr=[yel, yeh],
            fmt="o", ms=4, color=color, ecolor=color, elinewidth=0.5,
            capsize=0, alpha=0.7, label=gname.replace("_", " "),
        )

    lims = list(ax.get_xlim()) + list(ax.get_ylim())
    lo, hi = min(lims), max(lims)
    margin = (hi - lo) * 0.05
    lo -= margin; hi += margin
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.7, alpha=0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("ρ(actual preference, trait)  [within-user]")
    ax.set_ylabel("ρ(predicted preference, trait)  [within-user]")
    title_suffix = f"  {suffix.replace('_', ' ').strip()}" if suffix else ""
    ax.set_title(f"Trait Amplification: Predicted vs Actual{title_suffix}")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.3)
    ax.axvline(0, color="gray", linewidth=0.3)
    plt.tight_layout()

    path = out_dir / f"amplification_scatter{suffix}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_group_bars(
    group_name: str,
    traits: Dict[str, Tuple[float, float, float]],
    cis: Dict[str, Tuple],
    out_dir: Path,
    suffix: str = "",
) -> Path:
    """Paired horizontal bars: rho_true (gray) vs rho_pred (colored)."""
    labels = sorted(traits, key=lambda k: abs(traits[k][2]), reverse=True)
    n = len(labels)

    rt_vals = [traits[k][0] for k in labels]
    rp_vals = [traits[k][1] for k in labels]

    def _err(label_list, pos, val_list):
        lo_list, hi_list = [], []
        for lab, v in zip(label_list, val_list):
            key = f"{group_name}::{lab}"
            if key in cis:
                ci = cis[key]
                lo_list.append(v - ci[pos])
                hi_list.append(ci[pos + 1] - v)
            else:
                lo_list.append(0); hi_list.append(0)
        return [lo_list, hi_list]

    bar_h = 0.35
    fig, ax = plt.subplots(figsize=scaled_figsize(8, max(3, 0.5 * n)))
    y_pos = np.arange(n)

    ax.barh(
        y_pos - bar_h / 2, rt_vals, height=bar_h,
        color="#999999", edgecolor="white", linewidth=0.5, label="actual (ρ_true)",
        xerr=_err(labels, 0, rt_vals),
        error_kw=dict(ecolor="#555555", capsize=1.5, linewidth=0.6),
    )
    pred_colors = ["#4878CF" if v >= 0 else "#D65F5F" for v in rp_vals]
    ax.barh(
        y_pos + bar_h / 2, rp_vals, height=bar_h,
        color=pred_colors, edgecolor="white", linewidth=0.5, label="predicted (ρ_pred)",
        xerr=_err(labels, 2, rp_vals),
        error_kw=dict(ecolor="#333333", capsize=1.5, linewidth=0.6),
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)
    xlabel_detail = "user-averaged" if suffix else "pooled across posts"
    ax.set_xlabel(f"Spearman ρ  [{xlabel_detail}]")
    mean_abs_d = np.mean([abs(traits[k][2]) for k in labels])
    title_suffix = f"  {suffix.replace('_', ' ').strip()}" if suffix else ""
    ax.set_title(f"{group_name.replace('_', ' ').title()}{title_suffix}   (mean |δ| = {mean_abs_d:.4f})")
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()

    path = out_dir / f"{group_name}{suffix}_amplification.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_per_trait_scatter(
    group_name: str,
    labels: List[str],
    per_user_rhos: Dict[str, Tuple[np.ndarray, np.ndarray]],
    pooled_corrs: Dict[str, Tuple[float, float, float]],
    out_dir: Path,
) -> Path | None:
    """Small multiples of per-user rho_true (x) vs rho_pred (y), one subplot per trait."""
    valid_labels = [
        l for l in labels if f"{group_name}::{l}" in per_user_rhos
    ]
    if not valid_labels:
        return None

    valid_labels.sort(
        key=lambda l: abs(pooled_corrs.get(l, (0, 0, 0))[0]),
        reverse=True,
    )

    n = len(valid_labels)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=scaled_figsize(3.5 * ncols, 3.5 * nrows),
        squeeze=False,
    )

    for i, label in enumerate(valid_labels):
        ax = axes[i // ncols][i % ncols]
        key = f"{group_name}::{label}"
        rho_true, rho_pred = per_user_rhos[key]

        colors = np.where(
            rho_pred > rho_true, "#2ca02c", "#d62728",
        )
        ax.scatter(rho_true, rho_pred, s=5, alpha=0.25, edgecolors="none", c=colors)

        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        margin = (hi - lo) * 0.05
        lo -= margin
        hi += margin
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.5, alpha=0.4)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

        ax.set_xlabel("ρ_true  (user)", fontsize=7)
        ax.set_ylabel("ρ_pred  (user)", fontsize=7)
        ax.set_title(label, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.axhline(0, color="gray", linewidth=0.3)
        ax.axvline(0, color="gray", linewidth=0.3)

        r, _ = spearmanr(rho_true, rho_pred)
        ax.text(
            0.05, 0.95,
            f"r={r:.2f}\nn={len(rho_true)}",
            transform=ax.transAxes, fontsize=6, va="top",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1),
        )

    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)

    fig.suptitle(
        f"{group_name.replace('_', ' ').title()} — Per-User Trait ρ",
        fontsize=11,
    )
    plt.tight_layout()

    path = out_dir / f"{group_name}_per_user_scatter.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class TraitAmplificationModule(EvalModule):
    name = "trait_amplification"
    description = "Measures model amplification/suppression of NLP content traits vs actual user preferences"

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

        # --- Join to inferences, demean, unnest ---
        joined = (
            inferences_lf
            .join(preds.lazy(), left_on="at_uri", right_on="post_id", how="inner")
            .collect()
        )
        n_posts_matched = len(joined)
        if n_posts_matched < 50:
            return {"skipped": True, "reason": f"only {n_posts_matched} posts matched inferences"}

        joined = _demean_within_user(joined)
        flat, group_names = _unnest_text_inferences(joined)

        # --- Pre-extract all numpy arrays (keyed as "group::label") ---
        y_true_c = flat["y_true_c"].to_numpy()
        y_pred_c = flat["y_pred_c"].to_numpy()
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
                arr = gdf[col].to_numpy()
                trait_arrays[key] = arr
                finite_masks[key] = np.isfinite(arr)

        n_traits = len(trait_arrays)
        print(f"    {n_users_eligible} users, {n_posts_matched} posts, "
              f"{n_traits} traits across {len(group_names)} groups")

        # --- Point estimates ---
        t0 = time.time()
        all_corrs = _compute_all_correlations(y_true_c, y_pred_c, trait_arrays, finite_masks)
        print(f"    pooled correlations: {len(all_corrs)} traits ({time.time()-t0:.1f}s)")

        # --- Single bootstrap pass across all traits ---
        t0 = time.time()
        all_cis = _bootstrap_cis(
            user_ids, user_to_rows, y_true_c, y_pred_c,
            trait_arrays, finite_masks,
            valid_keys=set(all_corrs.keys()),
        )
        print(f"    pooled bootstrap CIs: {N_BOOTSTRAP} iterations ({time.time()-t0:.1f}s)")

        # --- Per-user rhos for per-trait scatter plots + user-averaged ---
        t0 = time.time()
        per_user_rhos = _compute_per_user_rhos(
            user_ids, user_to_rows, y_true_c, y_pred_c,
            trait_arrays, finite_masks,
            valid_keys=set(all_corrs.keys()),
        )
        print(f"    per-user rhos: {len(per_user_rhos)} traits ({time.time()-t0:.1f}s)")

        t0 = time.time()
        ua_corrs, ua_cis = _user_avg_correlations_and_cis(per_user_rhos)
        print(f"    user-avg bootstrap CIs: {len(ua_corrs)} traits ({time.time()-t0:.1f}s)")

        # --- Partition results back by group and plot ---
        t0 = time.time()
        group_color_map = {g: _GROUP_COLORS[i % len(_GROUP_COLORS)] for i, g in enumerate(group_names)}
        group_results: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
        ua_group_results: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
        plot_paths: list[str] = []

        for gname in group_names:
            traits: Dict[str, Tuple[float, float, float]] = {}
            ua_traits: Dict[str, Tuple[float, float, float]] = {}
            for label in group_labels.get(gname, []):
                key = f"{gname}::{label}"
                if key in all_corrs:
                    traits[label] = all_corrs[key]
                if key in ua_corrs:
                    ua_traits[label] = ua_corrs[key]
            if traits:
                group_results[gname] = traits
                path = _plot_group_bars(gname, traits, all_cis, out_dir)
                plot_paths.append(str(path))
                scatter_path = _plot_per_trait_scatter(
                    gname, list(traits.keys()), per_user_rhos, traits, out_dir,
                )
                if scatter_path is not None:
                    plot_paths.append(str(scatter_path))
            if ua_traits:
                ua_group_results[gname] = ua_traits
                path = _plot_group_bars(gname, ua_traits, ua_cis, out_dir, suffix="_user_avg")
                plot_paths.append(str(path))

        if group_results:
            scatter_path = _plot_scatter(group_results, all_cis, group_color_map, out_dir)
            plot_paths.insert(0, str(scatter_path))
        if ua_group_results:
            scatter_path = _plot_scatter(ua_group_results, ua_cis, group_color_map, out_dir, suffix="_user_avg")
            plot_paths.append(str(scatter_path))
        print(f"    plots: {len(plot_paths)} files ({time.time()-t0:.1f}s)")

        # --- Summary JSON (pooled) ---
        groups_json: Dict[str, Any] = {}
        all_abs_deltas: list[float] = []
        for gname, traits in group_results.items():
            gdict: Dict[str, Any] = {}
            for label, (rt, rp, delta) in traits.items():
                all_abs_deltas.append(abs(delta))
                entry: Dict[str, float] = {"rho_true": rt, "rho_pred": rp, "delta": delta}
                key = f"{gname}::{label}"
                if key in all_cis:
                    ci = all_cis[key]
                    entry.update({
                        "rho_true_ci_lo": ci[0], "rho_true_ci_hi": ci[1],
                        "rho_pred_ci_lo": ci[2], "rho_pred_ci_hi": ci[3],
                        "delta_ci_lo": ci[4], "delta_ci_hi": ci[5],
                    })
                gdict[label] = entry
            groups_json[gname] = gdict

        # --- Summary JSON (user-averaged) ---
        ua_groups_json: Dict[str, Any] = {}
        ua_abs_deltas: list[float] = []
        for gname, traits in ua_group_results.items():
            gdict_ua: Dict[str, Any] = {}
            for label, (rt, rp, delta) in traits.items():
                ua_abs_deltas.append(abs(delta))
                entry_ua: Dict[str, float] = {"rho_true": rt, "rho_pred": rp, "delta": delta}
                key = f"{gname}::{label}"
                if key in ua_cis:
                    ci = ua_cis[key]
                    entry_ua.update({
                        "rho_true_ci_lo": ci[0], "rho_true_ci_hi": ci[1],
                        "rho_pred_ci_lo": ci[2], "rho_pred_ci_hi": ci[3],
                        "delta_ci_lo": ci[4], "delta_ci_hi": ci[5],
                    })
                gdict_ua[label] = entry_ua
            ua_groups_json[gname] = gdict_ua

        summary = {
            "n_users_total": n_users_total,
            "n_users_eligible": n_users_eligible,
            "min_user_posts": MIN_USER_POSTS,
            "n_posts_matched": n_posts_matched,
            "n_bootstrap": N_BOOTSTRAP,
            "mean_abs_amplification": float(np.mean(all_abs_deltas)) if all_abs_deltas else 0.0,
            "groups": groups_json,
            "mean_abs_amplification_user_avg": float(np.mean(ua_abs_deltas)) if ua_abs_deltas else 0.0,
            "groups_user_avg": ua_groups_json,
        }
        self.save_json(summary, out_dir / "trait_amplification_summary.json")

        return {
            "n_users_eligible": n_users_eligible,
            "n_posts_matched": n_posts_matched,
            "groups_plotted": len(group_results),
            "mean_abs_amplification": summary["mean_abs_amplification"],
            "mean_abs_amplification_user_avg": summary["mean_abs_amplification_user_avg"],
            "plot_paths": plot_paths,
        }
