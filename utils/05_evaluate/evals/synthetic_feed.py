#!/usr/bin/env python3

"""
Synthetic Feed Evaluation Module

Builds a synthetic feed for each holdout user by scoring a large random post
pool with the trained model, then decomposes over-serving into user preference
and model amplification using NLP trait comparisons.

For each holdout user *u* and trait *t*:

    pool_mean(t)   = mean trait across all random-pool posts
    user_actual(t) = mean trait across user u's liked holdout posts
    model_feed(t)  = mean trait across user u's top-K synthetic-feed posts

    user_pref(u)     = user_actual(u)  - pool_mean      (user preference)
    model_amp(u)     = model_feed(u)   - user_actual(u)  (model amplification)
    model_excess(u)  = model_feed(u)   - pool_mean       (total excess)

All quantities are computed both **standardized** (divided by pool SD, useful
for cross-trait comparison) and **absolute** (raw trait-probability units,
useful for gauging real-world magnitude).  The additive identity
model_excess = user_pref + model_amp  holds exactly in both scales.

Note: standardized values can inflate rare traits with small pool SDs.
The absolute-scale plots provide a complementary view.

Outputs (under synthetic_feed/):
- synthetic_feed_topk.json:                per-user top-K pool indices (saved
                                           immediately after scoring)
- synthetic_feed_summary.json:             numerical results (saved before plots)
- synthetic_feed_decomposition.png:        headline bar chart (standardized)
- synthetic_feed_decomposition_abs.png:    headline bar chart (absolute)
- <group>_synthetic_feed.png:              per-group bars (standardized)
- <group>_synthetic_feed_abs.png:          per-group bars (absolute)
- <group>_synthetic_feed_prevalence.png:   per-group vertical bars showing
                                           pool / liked / feed prevalence
                                           with 95% CIs and bias arrows
- <group>_synthetic_feed_prevalence_detail.png:
                                           same as above with hi/lo-volume
                                           liker points overlaid on the
                                           liked bar
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import ttest_1samp

from . import EvalContext, EvalModule, scaled_figsize
from .trait_corrs import _load_inferences, _unnest_text_inferences

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_POOL = 10_000
TOP_K = 100
BATCH_SIZE = 2048
MIN_USER_LIKES = 5

_EXCLUDED_TRAITS: Dict[str, set] = {
    "moderation": {"OK"},
    "emotion_sentiment": {"neutral"},
}

_SUBSET_TRAITS: Dict[str, Tuple[str, set]] = {
    "text_arbitrary": (
        "text_arbitrary_selected",
        {
            "News & Media", "Politics", "Programming",
            "Arts & Creative", "Gaming", "Music", "Food & Lifestyle",
        },
    ),
    "sentiment": (
        "sentiment_negative",
        {"Negative"},
    ),
}

# Groups for which to also produce a top-N-by-pool-prevalence subset plot.
# Maps group name -> (subset plot name, N).
_TOP_N_SUBSETS: Dict[str, Tuple[str, int]] = {
    "topic": ("topic_top10", 10),
}

# --- decomposition bar plots (stacked / grouped) ---
DECOMP_USER_PREF_COLOR = "#4878CF"
DECOMP_MODEL_AMP_COLOR = "#D65F5F"
DECOMP_TOTAL_EXCESS_COLOR = "#2ca02c"

# --- prevalence bar plots (pool / liked / feed) ---
PREV_POOL_COLOR = "#aaaaaa"
PREV_LIKED_COLOR = "#4fa16a"###
PREV_FEED_COLOR = "#4878CF"
PREV_ARROW_COLOR = "#4878CF"
PREV_ANNOTATION_COLOR = "#333333"

# --- prevalence detail: volume-split point overlays ---
PREV_LIKED_LO_VOL_COLOR = "#774BD2"
PREV_LIKED_HI_VOL_COLOR = "#D2784B"

# --- per-group color cycle for headline decomposition chart ---
_GROUP_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TraitDecompResult(NamedTuple):
    user_pref_std: np.ndarray
    model_amp_std: np.ndarray
    model_excess_std: np.ndarray
    pool_sd: float
    pool_mean: float
    n_users: int
    cohen_d_pref: float
    cohen_d_amp: float
    cohen_d_excess: float
    p_pref: float
    p_amp: float
    p_excess: float
    n_pool_finite: int
    user_dids: List[str]


# ---------------------------------------------------------------------------
# Data loading (delegates to existing helpers)
# ---------------------------------------------------------------------------

def _load_random_pool(run_dir: Path) -> Tuple[pl.LazyFrame, np.ndarray]:
    """Lazy-scan random-sample posts and load embeddings memmap from 01_get_data.

    Returns a *LazyFrame* (at_uri, emb_idx) so callers can push filters
    (e.g. inference-availability) before collecting.
    """
    from utils.pipeline.core import select_prior_output
    from utils.helpers import load_parquet_from_prior

    get_data_dir = select_prior_output(run_dir, "01_get_data")
    if get_data_dir is None:
        raise FileNotFoundError("No 01_get_data output found")

    random_posts_lf = (
        load_parquet_from_prior(get_data_dir, "posts_core_")
        .filter(pl.col("in_random_sample"))
        .select("at_uri", "emb_idx")
    )

    emb_candidates = sorted(
        get_data_dir.glob("embeddings_*.npy"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not emb_candidates:
        raise FileNotFoundError(f"No embeddings_*.npy found under {get_data_dir}")
    embeddings_mmap = np.load(str(emb_candidates[0]), mmap_mode="r")

    return random_posts_lf, embeddings_mmap


def _load_user_histories(run_dir: Path) -> pl.LazyFrame:
    """Lazy-scan user history from 03_user_history.

    Returns a LazyFrame so callers can push filters (e.g. holdout-only)
    before collecting.
    """
    from utils.pipeline.core import select_prior_output
    from utils.helpers import load_parquet_from_prior

    history_dir = select_prior_output(run_dir, "03_user_history")
    if history_dir is None:
        raise FileNotFoundError("No 03_user_history output found")
    return load_parquet_from_prior(history_dir, "history_posts_")


def _find_checkpoint(ctx: EvalContext) -> Path:
    """Locate the model checkpoint .pth inside the training output.

    ctx.output_dir is ``<train_dir>/evals/<timestamp>``, so train_dir is
    two levels up.
    """
    train_dir = ctx.output_dir.parent.parent
    ckpt_dir = train_dir / "checkpoints"
    candidates = sorted(
        ckpt_dir.glob("engagement_model_*.pth"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    full_ckpts = [
        c for c in candidates
        if "_best" not in c.stem and "_weights" not in c.stem
    ]
    if full_ckpts:
        return full_ckpts[0]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No engagement_model_*.pth under {ckpt_dir}")


# ---------------------------------------------------------------------------
# Model loading and scoring
# ---------------------------------------------------------------------------

def _load_model(ckpt_path: Path, device: str):
    """Reconstruct MLPModel from a saved checkpoint."""
    import torch

    stage_mod = importlib.import_module("utils.04_train.stage_train_mlp")
    MLPModel = stage_mod.MLPModel

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = MLPModel(
        post_embedding_dim=ckpt["embed_dim"],
        hidden_dims=ckpt["hidden_dims"],
        dropout_rate=ckpt["dropout_rate"],
        user_hidden_dim=ckpt["user_hidden_dim"],
        user_output_dim=ckpt["user_output_dim"],
        num_attention_heads=ckpt["num_attention_heads"],
        num_attention_layers=ckpt["num_attention_layers"],
        max_history_len=ckpt["max_history_len"],
        attention_dropout=ckpt["attention_dropout"],
        user_encoder_type=ckpt["user_encoder"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def _compute_user_summaries(
    holdout_dids: List[str],
    history_lf: pl.LazyFrame,
    embeddings_mmap: np.ndarray,
    ckpt: dict,
) -> Dict[str, np.ndarray]:
    """Compute user summary embeddings using the checkpoint's summarizer.

    Accepts a *LazyFrame* so the holdout-user filter is pushed into the scan
    and only the needed rows are materialised.
    """
    from utils.dataloaders import get_summarizer

    summarizer_name = ckpt.get("user_summarization", "mean")
    ema_alpha = ckpt.get("ema_alpha", 0.1)
    summarizer = get_summarizer(summarizer_name, ema_alpha=ema_alpha)

    best_rows = (
        history_lf
        .filter(pl.col("target_did").is_in(holdout_dids))
        .with_columns(pl.col("prior_emb_indices").list.len().alias("_hist_len"))
        .sort("_hist_len", descending=True)
        .group_by("target_did")
        .first()
        .collect()
    )

    user_summaries: Dict[str, np.ndarray] = {}
    for row in best_rows.iter_rows(named=True):
        indices = row["prior_emb_indices"]
        if indices is None or len(indices) == 0:
            continue
        embs = embeddings_mmap[np.array(indices, dtype=np.int64)]
        user_summaries[row["target_did"]] = summarizer.summarize(embs)

    return user_summaries


def _score_pool_for_users(
    model,
    user_summaries: Dict[str, np.ndarray],
    pool_embeddings: np.ndarray,
    device: str,
) -> Dict[str, np.ndarray]:
    """Score all pool posts for every eligible user.  Returns {did: scores}."""
    import torch

    pool_t = torch.tensor(pool_embeddings, dtype=torch.float32, device=device)
    n_pool = pool_t.shape[0]

    user_scores: Dict[str, np.ndarray] = {}

    with torch.no_grad():
        for did, summary in user_summaries.items():
            user_t = torch.tensor(
                summary, dtype=torch.float32, device=device,
            )
            chunks: List[np.ndarray] = []
            for start in range(0, n_pool, BATCH_SIZE):
                batch_posts = pool_t[start : start + BATCH_SIZE]
                bs = batch_posts.shape[0]
                hist = user_t.unsqueeze(0).unsqueeze(0).expand(bs, 1, -1)
                mask = torch.ones(bs, 1, dtype=torch.bool, device=device)
                preds = model(hist, mask, batch_posts).squeeze(-1)
                chunks.append(preds.cpu().numpy())
            user_scores[did] = np.concatenate(chunks)

    return user_scores


# ---------------------------------------------------------------------------
# Trait comparison
# ---------------------------------------------------------------------------

def _compute_trait_decomposition(
    pool_trait_vals: np.ndarray,
    user_actual_traits: Dict[str, np.ndarray],
    user_feed_traits: Dict[str, np.ndarray],
    eligible_dids: List[str],
) -> Optional[TraitDecompResult]:
    """Three-way decomposition for a single trait."""
    finite_pool = pool_trait_vals[np.isfinite(pool_trait_vals)]
    if len(finite_pool) < 20:
        return None
    pool_mean = float(np.mean(finite_pool))
    pool_sd = float(np.std(finite_pool, ddof=1))
    if pool_sd < 1e-12:
        return None

    up_list: List[float] = []
    ma_list: List[float] = []
    me_list: List[float] = []
    did_list: List[str] = []

    for did in eligible_dids:
        actual = user_actual_traits.get(did)
        feed = user_feed_traits.get(did)
        if actual is None or feed is None:
            continue
        if len(actual) < MIN_USER_LIKES:
            continue

        actual_mean = float(np.nanmean(actual))
        feed_mean = float(np.nanmean(feed))

        up_list.append((actual_mean - pool_mean) / pool_sd)
        ma_list.append((feed_mean - actual_mean) / pool_sd)
        me_list.append((feed_mean - pool_mean) / pool_sd)
        did_list.append(did)

    if len(up_list) < 10:
        return None

    up = np.array(up_list)
    ma = np.array(ma_list)
    me = np.array(me_list)

    def _effect(arr: np.ndarray) -> Tuple[float, float]:
        sd = float(np.std(arr, ddof=1))
        d = float(np.mean(arr)) / sd if sd > 1e-12 else 0.0
        return d, float(ttest_1samp(arr, 0.0).pvalue)

    d_pref, p_pref = _effect(up)
    d_amp, p_amp = _effect(ma)
    d_excess, p_excess = _effect(me)

    return TraitDecompResult(
        user_pref_std=up,
        model_amp_std=ma,
        model_excess_std=me,
        pool_sd=pool_sd,
        pool_mean=pool_mean,
        n_users=len(up),
        cohen_d_pref=d_pref,
        cohen_d_amp=d_amp,
        cohen_d_excess=d_excess,
        p_pref=p_pref,
        p_amp=p_amp,
        p_excess=p_excess,
        n_pool_finite=len(finite_pool),
        user_dids=did_list,
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _fmt_pvalue(p: float) -> str:
    if p < 0.001:
        return "p < .001"
    if p < 0.01:
        return f"p = {p:.3f}"
    return f"p = {p:.2f}"


def _significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _plot_decomposition_bar(
    trait_results: Dict[str, TraitDecompResult],
    group_color_map: Dict[str, str],
    out_dir: Path,
    *,
    standardize: bool = True,
) -> Path:
    """Headline horizontal stacked bar: user_pref + model_amp = model_excess.

    When *standardize* is False the bars are in raw trait-probability units
    instead of pool-SD units, giving a sense of absolute magnitude.
    """
    tag = "std" if standardize else "abs"

    def _scale(k: str, arr_name: str) -> float:
        tr = trait_results[k]
        v = float(np.mean(getattr(tr, arr_name)))
        return v if standardize else v * tr.pool_sd

    sorted_keys = sorted(
        trait_results,
        key=lambda k: abs(_scale(k, "model_excess_std")),
        reverse=True,
    )
    n = len(sorted_keys)
    fname = "synthetic_feed_decomposition.png" if standardize else \
            "synthetic_feed_decomposition_abs.png"
    path = out_dir / fname
    if n == 0:
        return path

    fig, ax = plt.subplots(figsize=scaled_figsize(10, max(3, 0.38 * n)))
    y = np.arange(n)
    short = [k.split("::")[-1] for k in sorted_keys]

    pref = [_scale(k, "user_pref_std") for k in sorted_keys]
    amp = [_scale(k, "model_amp_std") for k in sorted_keys]

    ax.barh(y, pref, height=0.6, color=DECOMP_USER_PREF_COLOR, alpha=0.85,
            label="User preference (likes \u2212 pool)")
    ax.barh(y, amp, height=0.6, left=pref, color=DECOMP_MODEL_AMP_COLOR,
            alpha=0.85, label="Model amplification (feed \u2212 likes)")

    fmt = "+.3f" if standardize else "+.4f"
    for i, k in enumerate(sorted_keys):
        tr = trait_results[k]
        excess = _scale(k, "model_excess_std")
        stars = _significance_stars(tr.p_excess)
        offset = 0.01 * (1 if standardize else tr.pool_sd)
        if excess < 0:
            offset = -offset
        ha = "left" if excess >= 0 else "right"
        ax.text(excess + offset, i, f"{excess:{fmt}}{stars}",
                va="center", ha=ha, fontsize=6, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(short, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)

    if standardize:
        ax.set_xlabel("Mean over-serving (pool SDs)", fontsize=9)
        subtitle = "(standardized by pool SD)"
    else:
        ax.set_xlabel("Mean over-serving (raw trait units)", fontsize=9)
        subtitle = "(absolute, raw trait units)"
    ax.set_title(
        f"Synthetic feed: trait decomposition {subtitle}\n"
        "model_excess = user_preference + model_amplification",
        fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=7, loc="lower right", framealpha=0.8)
    plt.tight_layout()
    fig.savefig(path, dpi=360, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_group_decomposition(
    group_name: str,
    group_traits: Dict[str, TraitDecompResult],
    out_dir: Path,
    *,
    standardize: bool = True,
) -> Path:
    """Per-group grouped bar chart: three bars per trait.

    When *standardize* is False the bars are in raw trait-probability units.
    """

    def _scale(k: str, arr_name: str) -> float:
        tr = group_traits[k]
        v = float(np.mean(getattr(tr, arr_name)))
        return v if standardize else v * tr.pool_sd

    labels = sorted(
        group_traits,
        key=lambda k: abs(_scale(k, "model_excess_std")),
        reverse=True,
    )
    n = len(labels)
    suffix = "" if standardize else "_abs"
    path = out_dir / f"{group_name}_synthetic_feed{suffix}.png"
    if n == 0:
        return path

    fig, ax = plt.subplots(figsize=scaled_figsize(8, max(3, 0.5 * n)))
    y = np.arange(n)
    h = 0.25

    pref = [_scale(k, "user_pref_std") for k in labels]
    amp = [_scale(k, "model_amp_std") for k in labels]
    excess = [_scale(k, "model_excess_std") for k in labels]

    ax.barh(y - h, pref, height=h, color=DECOMP_USER_PREF_COLOR, alpha=0.85,
            label="User pref")
    ax.barh(y, amp, height=h, color=DECOMP_MODEL_AMP_COLOR, alpha=0.85,
            label="Model amp")
    ax.barh(y + h, excess, height=h, color=DECOMP_TOTAL_EXCESS_COLOR,
            alpha=0.85, label="Total excess")

    for i, k in enumerate(labels):
        tr = group_traits[k]
        stars = _significance_stars(tr.p_excess)
        ex = excess[i]
        offset = 0.005 if ex >= 0 else -0.005
        ha = "left" if ex >= 0 else "right"
        ax.text(ex + offset, i + h, f"d={tr.cohen_d_excess:+.2f}{stars}",
                va="center", ha=ha, fontsize=6)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.5)

    if standardize:
        ax.set_xlabel("Mean (pool SDs)", fontsize=8)
    else:
        ax.set_xlabel("Mean (raw trait units)", fontsize=8)
    title = group_name.replace("_", " ").title()
    scale_tag = "standardized" if standardize else "absolute"
    ax.set_title(f"{title}: synthetic feed decomposition ({scale_tag})",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, loc="lower right", framealpha=0.8)
    plt.tight_layout()
    fig.savefig(path, dpi=360, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_group_prevalence(
    group_name: str,
    group_traits: Dict[str, TraitDecompResult],
    out_dir: Path,
    *,
    ylim: Optional[Tuple[float, float]] = None,
) -> Tuple[Path, Tuple[float, float]]:
    """Per-group vertical bar chart: pool / liked / feed prevalence with 95% CIs.

    For each trait, three bars show the raw trait prevalence in the random pool,
    in users' liked posts (averaged across users), and in the model's top-K
    synthetic feed (averaged across users).  A directed arrow from the liked
    estimate to the feed estimate visualises the model's bias.

    Returns ``(path, ylim)`` so a companion plot can reuse the same axis range.
    """
    z95 = 1.96
    excluded = _EXCLUDED_TRAITS.get(group_name, set())

    records: Dict[str, Dict[str, float]] = {}
    for label, tr in group_traits.items():
        if label in excluded:
            continue
        user_actual = tr.pool_mean + tr.user_pref_std * tr.pool_sd
        user_feed = tr.pool_mean + tr.model_excess_std * tr.pool_sd

        pool_prev = tr.pool_mean
        pool_ci = z95 * tr.pool_sd / np.sqrt(tr.n_pool_finite)
        liked_prev = float(np.mean(user_actual))
        liked_ci = z95 * float(np.std(user_actual, ddof=1)) / np.sqrt(tr.n_users)
        feed_prev = float(np.mean(user_feed))
        feed_ci = z95 * float(np.std(user_feed, ddof=1)) / np.sqrt(tr.n_users)

        raw_shift = float(np.mean(tr.model_amp_std)) * tr.pool_sd

        records[label] = {
            "pool": pool_prev, "pool_ci": pool_ci,
            "liked": liked_prev, "liked_ci": liked_ci,
            "feed": feed_prev, "feed_ci": feed_ci,
            "raw_shift": raw_shift,
            "cohen_d_amp": tr.cohen_d_amp,
            "p_amp": tr.p_amp,
        }

    labels = sorted(
        records,
        key=lambda k: abs(records[k]["feed"] - records[k]["liked"]),
        reverse=True,
    )
    n = len(labels)
    path = out_dir / f"{group_name}_synthetic_feed_prevalence.png"
    if n == 0:
        return path, (0.0, 1.0)

    x = np.arange(n)
    w = 0.25

    pool_vals = [records[l]["pool"] for l in labels]
    pool_cis = [records[l]["pool_ci"] for l in labels]
    liked_vals = [records[l]["liked"] for l in labels]
    liked_cis = [records[l]["liked_ci"] for l in labels]
    feed_vals = [records[l]["feed"] for l in labels]
    feed_cis = [records[l]["feed_ci"] for l in labels]

    PLOT_H = 4.5
    BOTTOM = 1.8
    total_h = PLOT_H + BOTTOM
    fig_w, fig_h = scaled_figsize(max(6, 0.7 * n), total_h)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Four columns per trait: pool, liked, feed, arrow.  Centre the group on x.
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * w

    ax.bar(
        x + offsets[0], pool_vals, width=w, yerr=pool_cis,
        color=PREV_POOL_COLOR, alpha=0.85, capsize=3,
        error_kw=dict(linewidth=0.8), label="Pool",
    )
    ax.bar(
        x + offsets[1], liked_vals, width=w, yerr=liked_cis,
        color=PREV_LIKED_COLOR, alpha=0.85, capsize=3,
        error_kw=dict(linewidth=0.8), label="Liked (user avg)",
    )
    ax.bar(
        x + offsets[2], feed_vals, width=w, yerr=feed_cis,
        color=PREV_FEED_COLOR, alpha=0.85, capsize=3,
        error_kw=dict(linewidth=0.8), label="Feed (user avg)",
    )

    arrow_x = offsets[3]
    annotation_pad = w * 0.15
    for i in range(n):
        lv, fv = liked_vals[i], feed_vals[i]
        ax_x = x[i] + arrow_x
        ax.annotate(
            "",
            xy=(ax_x, fv), xytext=(ax_x, lv),
            arrowprops=dict(
                arrowstyle="->,head_width=0.25,head_length=0.15",
                color=PREV_ARROW_COLOR, lw=1.5,
                shrinkA=0, shrinkB=0,
            ),
        )

        r = records[labels[i]]
        stars = _significance_stars(r["p_amp"])
        mid_y = (lv + fv) / 2
        ax.text(
            ax_x + annotation_pad, mid_y,
            f"{r['raw_shift']:+.4f}\nd={r['cohen_d_amp']:+.2f}{stars}",
            fontsize=5, va="center", ha="left", color=PREV_ANNOTATION_COLOR,
        )

    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_xlim(x[0] + offsets[0] - w, x[-1] + offsets[3] + 3 * w)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Trait prevalence (raw units)", fontsize=8)

    title = group_name.replace("_", " ").title()
    ax.set_title(
        f"{title}: pool / liked / feed prevalence",
        fontsize=10, fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=7)
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.subplots_adjust(
        bottom=BOTTOM / total_h, top=0.92, left=0.10, right=0.97,
    )
    actual_ylim = ax.get_ylim()
    fig.savefig(path, dpi=360, bbox_inches="tight")
    plt.close(fig)
    return path, actual_ylim


def _plot_group_prevalence_detail(
    group_name: str,
    group_traits: Dict[str, TraitDecompResult],
    is_high: Dict[str, bool],
    pct_hi_str: str,
    out_dir: Path,
    *,
    ylim: Optional[Tuple[float, float]] = None,
) -> Tuple[Path, Tuple[float, float]]:
    """Like _plot_group_prevalence but overlays hi/lo-volume liker points on
    the Liked bar column.

    Returns ``(path, ylim)`` so a companion plot can reuse the same axis range.
    """
    z95 = 1.96
    excluded = _EXCLUDED_TRAITS.get(group_name, set())

    records: Dict[str, Dict[str, float]] = {}
    for label, tr in group_traits.items():
        if label in excluded:
            continue
        user_actual = tr.pool_mean + tr.user_pref_std * tr.pool_sd
        user_feed = tr.pool_mean + tr.model_excess_std * tr.pool_sd

        pool_prev = tr.pool_mean
        pool_ci = z95 * tr.pool_sd / np.sqrt(tr.n_pool_finite)
        liked_prev = float(np.mean(user_actual))
        liked_ci = z95 * float(np.std(user_actual, ddof=1)) / np.sqrt(tr.n_users)
        feed_prev = float(np.mean(user_feed))
        feed_ci = z95 * float(np.std(user_feed, ddof=1)) / np.sqrt(tr.n_users)

        raw_shift = float(np.mean(tr.model_amp_std)) * tr.pool_sd

        mask_hi = np.array([is_high.get(d, False) for d in tr.user_dids])
        lo_actual = user_actual[~mask_hi]
        hi_actual = user_actual[mask_hi]

        lo_prev = float(np.mean(lo_actual)) if len(lo_actual) >= 5 else float("nan")
        lo_ci = (
            z95 * float(np.std(lo_actual, ddof=1)) / np.sqrt(len(lo_actual))
            if len(lo_actual) >= 5 else 0.0
        )
        hi_prev = float(np.mean(hi_actual)) if len(hi_actual) >= 5 else float("nan")
        hi_ci = (
            z95 * float(np.std(hi_actual, ddof=1)) / np.sqrt(len(hi_actual))
            if len(hi_actual) >= 5 else 0.0
        )

        records[label] = {
            "pool": pool_prev, "pool_ci": pool_ci,
            "liked": liked_prev, "liked_ci": liked_ci,
            "feed": feed_prev, "feed_ci": feed_ci,
            "raw_shift": raw_shift,
            "cohen_d_amp": tr.cohen_d_amp,
            "p_amp": tr.p_amp,
            "lo_prev": lo_prev, "lo_ci": lo_ci,
            "hi_prev": hi_prev, "hi_ci": hi_ci,
        }

    labels = sorted(
        records,
        key=lambda k: abs(records[k]["feed"] - records[k]["liked"]),
        reverse=True,
    )
    n = len(labels)
    path = out_dir / f"{group_name}_synthetic_feed_prevalence_detail.png"
    if n == 0:
        return path, (0.0, 1.0)

    x = np.arange(n)
    w = 0.25

    pool_vals = [records[l]["pool"] for l in labels]
    pool_cis = [records[l]["pool_ci"] for l in labels]
    liked_vals = [records[l]["liked"] for l in labels]
    liked_cis = [records[l]["liked_ci"] for l in labels]
    feed_vals = [records[l]["feed"] for l in labels]
    feed_cis = [records[l]["feed_ci"] for l in labels]

    PLOT_H = 4.5
    BOTTOM = 1.8
    total_h = PLOT_H + BOTTOM
    fig_w, fig_h = scaled_figsize(max(6, 0.7 * n), total_h)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * w

    ax.bar(
        x + offsets[0], pool_vals, width=w, yerr=pool_cis,
        color=PREV_POOL_COLOR, alpha=0.85, capsize=3,
        error_kw=dict(linewidth=0.8), label="Pool",
    )
    ax.bar(
        x + offsets[1], liked_vals, width=w, yerr=liked_cis,
        color=PREV_LIKED_COLOR, alpha=0.85, capsize=3,
        error_kw=dict(linewidth=0.8), label="Liked (user avg)",
    )
    ax.bar(
        x + offsets[2], feed_vals, width=w, yerr=feed_cis,
        color=PREV_FEED_COLOR, alpha=0.85, capsize=3,
        error_kw=dict(linewidth=0.8), label="Feed (user avg)",
    )

    # hi/lo volume points on the liked column
    dodge = w * 0.15
    lo_vals = [records[l]["lo_prev"] for l in labels]
    lo_cis = [records[l]["lo_ci"] for l in labels]
    hi_vals_pts = [records[l]["hi_prev"] for l in labels]
    hi_cis = [records[l]["hi_ci"] for l in labels]

    ax.errorbar(
        x + offsets[1] - dodge, lo_vals, yerr=lo_cis,
        fmt="o", color=PREV_LIKED_LO_VOL_COLOR, markersize=4, capsize=2,
        capthick=0.7, linewidth=0.7, zorder=5, label="Liked lo-vol",
    )
    ax.errorbar(
        x + offsets[1] + dodge, hi_vals_pts, yerr=hi_cis,
        fmt="o", color=PREV_LIKED_HI_VOL_COLOR, markersize=4, capsize=2,
        capthick=0.7, linewidth=0.7, zorder=5,
        label=f"Liked hi-vol (top {pct_hi_str}%)",
    )

    arrow_x = offsets[3]
    annotation_pad = w * 0.15
    for i in range(n):
        lv, fv = liked_vals[i], feed_vals[i]
        ax_x = x[i] + arrow_x
        ax.annotate(
            "",
            xy=(ax_x, fv), xytext=(ax_x, lv),
            arrowprops=dict(
                arrowstyle="->,head_width=0.25,head_length=0.15",
                color=PREV_ARROW_COLOR, lw=1.5,
                shrinkA=0, shrinkB=0,
            ),
        )

        r = records[labels[i]]
        stars = _significance_stars(r["p_amp"])
        mid_y = (lv + fv) / 2
        ax.text(
            ax_x + annotation_pad, mid_y,
            f"{r['raw_shift']:+.4f}\nd={r['cohen_d_amp']:+.2f}{stars}",
            fontsize=5, va="center", ha="left", color=PREV_ANNOTATION_COLOR,
        )

    ax.axhline(0, color="black", linewidth=0.4)
    ax.set_xlim(x[0] + offsets[0] - w, x[-1] + offsets[3] + 3 * w)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Trait prevalence (raw units)", fontsize=8)

    title = group_name.replace("_", " ").title()
    ax.set_title(
        f"{title}: pool / liked / feed prevalence (detail)",
        fontsize=10, fontweight="bold",
    )
    ax.tick_params(axis="y", labelsize=7)
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.subplots_adjust(
        bottom=BOTTOM / total_h, top=0.92, left=0.10, right=0.97,
    )
    actual_ylim = ax.get_ylim()
    fig.savefig(path, dpi=360, bbox_inches="tight")
    plt.close(fig)
    return path, actual_ylim


def _save_prevalence_legends(pct_hi_str: str, out_dir: Path) -> List[str]:
    """Save standalone legend PNGs for the prevalence and prevalence-detail plots."""
    paths: List[str] = []

    arrow_proxy = plt.Line2D(
        [0, 0], [0, 1], color=PREV_ARROW_COLOR, lw=1.5,
        marker="^", markersize=6, markevery=[1],
    )

    base_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=PREV_POOL_COLOR, alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, fc=PREV_LIKED_COLOR, alpha=0.85),
        plt.Rectangle((0, 0), 1, 1, fc=PREV_FEED_COLOR, alpha=0.85),
        arrow_proxy,
    ]
    base_labels = [
        "Pool", "Liked (user avg)", "Feed (user avg)",
        "Model bias (F\u2212L)",
    ]

    for suffix, extra_handles, extra_labels in [
        ("prevalence", [], []),
        (
            "prevalence_detail",
            [
                plt.Line2D(
                    [0], [0], marker="o", color="w",
                    markerfacecolor=PREV_LIKED_LO_VOL_COLOR, markersize=7,
                    label="Liked lo-vol",
                ),
                plt.Line2D(
                    [0], [0], marker="o", color="w",
                    markerfacecolor=PREV_LIKED_HI_VOL_COLOR, markersize=7,
                    label=f"Liked hi-vol (top {pct_hi_str}%)",
                ),
            ],
            [
                "Liked lo-vol",
                f"Liked hi-vol (top {pct_hi_str}%)",
            ],
        ),
    ]:
        fig_leg, ax_leg = plt.subplots(figsize=(1, 1))
        legend = ax_leg.legend(
            base_handles + extra_handles,
            base_labels + extra_labels,
            fontsize=8, framealpha=0.9, loc="center", frameon=True,
        )
        ax_leg.set_axis_off()
        fig_leg.canvas.draw()
        bbox = legend.get_window_extent().transformed(
            fig_leg.dpi_scale_trans.inverted(),
        )
        p = out_dir / f"synthetic_feed_{suffix}_legend.png"
        fig_leg.savefig(p, dpi=300, bbox_inches=bbox)
        plt.close(fig_leg)
        paths.append(str(p))

    return paths


# ---------------------------------------------------------------------------
# Headline generation
# ---------------------------------------------------------------------------

def _generate_headline(
    trait_results: Dict[str, TraitDecompResult],
) -> List[str]:
    if not trait_results:
        return ["No traits had enough data for synthetic feed analysis."]

    by_excess_std = sorted(
        trait_results.items(),
        key=lambda kv: abs(float(np.mean(kv[1].model_excess_std))),
        reverse=True,
    )
    by_excess_abs = sorted(
        trait_results.items(),
        key=lambda kv: abs(
            float(np.mean(kv[1].model_excess_std)) * kv[1].pool_sd
        ),
        reverse=True,
    )

    sig_excess = [k for k, tr in trait_results.items() if tr.p_excess < 0.05]
    sig_amp = [k for k, tr in trait_results.items() if tr.p_amp < 0.05]

    sentences: List[str] = []
    sentences.append(
        f"Across {len(trait_results)} traits, "
        f"{len(sig_excess)} show significant model excess "
        f"(feed \u2260 pool, p < .05) and "
        f"{len(sig_amp)} show significant model amplification "
        f"(feed \u2260 likes, p < .05)."
    )

    top_k, top_tr = by_excess_std[0]
    top_label = top_k.split("::")[-1]
    top_excess = float(np.mean(top_tr.model_excess_std))
    top_pref = float(np.mean(top_tr.user_pref_std))
    top_amp = float(np.mean(top_tr.model_amp_std))
    direction = "over" if top_excess > 0 else "under"

    sentences.append(
        f"Largest effect (standardized): {top_label} is "
        f"{direction}-represented by "
        f"{abs(top_excess):.3f} SD in the synthetic feed vs the random pool. "
        f"Of this, {abs(top_pref):.3f} SD reflects user preference and "
        f"{abs(top_amp):.3f} SD is model amplification "
        f"(d_excess = {top_tr.cohen_d_excess:+.2f}, "
        f"{_fmt_pvalue(top_tr.p_excess)})."
    )

    abs_k, abs_tr = by_excess_abs[0]
    abs_label = abs_k.split("::")[-1]
    abs_excess = float(np.mean(abs_tr.model_excess_std)) * abs_tr.pool_sd
    abs_direction = "over" if abs_excess > 0 else "under"
    if abs_k != top_k:
        sentences.append(
            f"Largest effect (absolute): {abs_label} is "
            f"{abs_direction}-represented by "
            f"{abs(abs_excess):.4f} raw units in the synthetic feed vs the "
            f"random pool (pool_mean={abs_tr.pool_mean:.4f}, "
            f"pool_sd={abs_tr.pool_sd:.4f}). "
            f"This differs from the standardized ranking because "
            f"SD-normalization inflates rare traits with small pool SDs."
        )

    by_amp = sorted(
        trait_results.items(),
        key=lambda kv: abs(float(np.mean(kv[1].model_amp_std))),
        reverse=True,
    )
    amp_k, amp_tr = by_amp[0]
    if amp_k != top_k:
        amp_label = amp_k.split("::")[-1]
        amp_val = float(np.mean(amp_tr.model_amp_std))
        amp_dir = "amplifies" if amp_val > 0 else "suppresses"
        sentences.append(
            f"Largest model amplification: the model {amp_dir} {amp_label} "
            f"by {abs(amp_val):.3f} SD beyond user preference "
            f"(d_amp = {amp_tr.cohen_d_amp:+.2f}, "
            f"{_fmt_pvalue(amp_tr.p_amp)})."
        )

    return sentences


# ---------------------------------------------------------------------------
# Module class
# ---------------------------------------------------------------------------

class SyntheticFeedModule(EvalModule):
    name = "synthetic_feed"
    description = (
        "Scores a random post pool to build synthetic feeds per user, "
        "decomposes trait over-serving into user preference vs model "
        "amplification"
    )

    def run(self, ctx: EvalContext) -> Dict[str, Any]:
        out_dir = self.get_output_dir(ctx)
        run_dir = ctx.config.get("run_dir")
        if run_dir is None:
            return {"skipped": True, "reason": "run_dir not in eval config"}
        run_dir = Path(run_dir)

        from utils.helpers import get_device

        # ---- load data ----
        print("    [synthetic_feed] Loading random pool + embeddings...",
              flush=True)
        try:
            random_posts_lf, embeddings_mmap = _load_random_pool(run_dir)
        except FileNotFoundError as e:
            return {"skipped": True, "reason": str(e)}

        try:
            inferences_lf = _load_inferences(run_dir)
        except FileNotFoundError as e:
            return {"skipped": True, "reason": str(e)}

        print("    [synthetic_feed] Loading user histories...", flush=True)
        try:
            history_lf = _load_user_histories(run_dir)
        except FileNotFoundError as e:
            return {"skipped": True, "reason": str(e)}

        # ---- filter to inference-available posts, then subsample ----
        # Semi-join first so we only sample from posts that have inferences
        # (only recent posts have NLP inferences).
        inference_uris = inferences_lf.select("at_uri")
        inference_available = (
            random_posts_lf
            .join(inference_uris, on="at_uri", how="semi")
            .collect()
        )
        n_with_inf = len(inference_available)
        print(
            f"    [synthetic_feed] Random posts with inferences: "
            f"{n_with_inf:,}",
            flush=True,
        )
        if n_with_inf < 100:
            return {
                "skipped": True,
                "reason": f"only {n_with_inf} random posts have inferences",
            }

        n_sample = min(N_POOL, n_with_inf)
        pool_sample = inference_available.sample(n=n_sample, seed=42)

        pool_combined = (
            pool_sample.lazy()
            .join(inferences_lf, on="at_uri", how="inner")
            .collect()
        )
        n_pool = len(pool_combined)

        pool_emb_indices = pool_combined["emb_idx"].to_numpy()
        pool_flat, group_names = _unnest_text_inferences(pool_combined)
        print(f"    [synthetic_feed] Pool: {n_pool} posts sampled for scoring",
              flush=True)

        # ---- load model ----
        print("    [synthetic_feed] Loading model checkpoint...", flush=True)
        try:
            ckpt_path = _find_checkpoint(ctx)
        except FileNotFoundError as e:
            return {"skipped": True, "reason": str(e)}

        device = get_device(None)
        model, ckpt = _load_model(ckpt_path, device)

        # ---- compute user summaries ----
        print("    [synthetic_feed] Computing user summaries...", flush=True)
        preds_pl = pl.from_pandas(ctx.predictions_df)
        holdout_dids = preds_pl["did"].unique().to_list()

        user_summaries = _compute_user_summaries(
            holdout_dids, history_lf, embeddings_mmap, ckpt,
        )

        user_likes_count = (
            preds_pl.filter(pl.col("y_true") == 1)
            .group_by("did")
            .agg(pl.len().alias("n_likes"))
            .filter(pl.col("n_likes") >= MIN_USER_LIKES)
        )
        eligible_dids = [
            d for d in user_likes_count["did"].to_list()
            if d in user_summaries
        ]
        if len(eligible_dids) < 5:
            return {
                "skipped": True,
                "reason": f"only {len(eligible_dids)} eligible users",
            }

        # ---- score pool for each user ----
        print(
            f"    [synthetic_feed] Scoring {n_pool} pool posts for "
            f"{len(eligible_dids)} users...",
            flush=True,
        )
        pool_embeddings = embeddings_mmap[pool_emb_indices].copy()
        eligible_summaries = {d: user_summaries[d] for d in eligible_dids}
        user_scores = _score_pool_for_users(
            model, eligible_summaries, pool_embeddings, device,
        )

        # ---- top-K selection ----
        k = min(TOP_K, n_pool)
        user_topk: Dict[str, np.ndarray] = {}
        for did, scores in user_scores.items():
            user_topk[did] = np.argsort(scores)[-k:][::-1]

        # ---- persist scoring outputs early ----
        self.save_json(
            {
                "n_pool": n_pool,
                "top_k": k,
                "n_users_scored": len(user_scores),
                "user_topk": {
                    did: idx.tolist() for did, idx in user_topk.items()
                },
            },
            out_dir / "synthetic_feed_topk.json",
        )
        print("    [synthetic_feed] Saved top-K indices checkpoint",
              flush=True)

        # ---- trait decomposition ----
        print("    [synthetic_feed] Joining liked posts to inferences...",
              flush=True)

        liked_posts = preds_pl.filter(pl.col("y_true") == 1).select(
            "did", "post_id",
        )
        liked_combined = (
            inferences_lf
            .join(liked_posts.lazy(), left_on="at_uri", right_on="post_id",
                  how="inner")
            .collect()
        )
        print(f"    [synthetic_feed] Matched {len(liked_combined):,} liked "
              f"posts to inferences", flush=True)

        liked_flat, _ = _unnest_text_inferences(liked_combined)
        liked_dids = liked_flat["did"].to_numpy()

        print(f"    [synthetic_feed] Building per-user row index for "
              f"{len(eligible_dids):,} users...", flush=True)
        eligible_set = set(eligible_dids)
        did_to_rows_all: Dict[str, List[int]] = {}
        for i, d in enumerate(liked_dids):
            if d in eligible_set:
                did_to_rows_all.setdefault(d, []).append(i)
        liked_did_to_rows: Dict[str, np.ndarray] = {
            d: np.array(rows)
            for d, rows in did_to_rows_all.items()
            if len(rows) >= MIN_USER_LIKES
        }
        print(f"    [synthetic_feed] {len(liked_did_to_rows):,} users have "
              f">= {MIN_USER_LIKES} liked posts with inferences", flush=True)

        print("    [synthetic_feed] Computing trait decomposition...",
              flush=True)
        trait_results: Dict[str, TraitDecompResult] = {}
        group_labels: Dict[str, List[str]] = {}
        total_traits = 0

        for gi, gname in enumerate(group_names, 1):
            pool_gdf = pool_flat.select(gname).unnest(gname)
            liked_gdf = (
                liked_flat.select(gname).unnest(gname)
                if gname in liked_flat.columns
                else None
            )
            cols = pool_gdf.columns
            group_labels[gname] = cols
            print(
                f"    [synthetic_feed]   group {gi}/{len(group_names)}: "
                f"{gname} ({len(cols)} traits)",
                flush=True,
            )

            for col in cols:
                pool_vals = pool_gdf[col].to_numpy().astype(np.float64)

                user_actual: Dict[str, np.ndarray] = {}
                if liked_gdf is not None and col in liked_gdf.columns:
                    liked_vals = liked_gdf[col].to_numpy().astype(np.float64)
                    for did, rows in liked_did_to_rows.items():
                        v = liked_vals[rows]
                        v = v[np.isfinite(v)]
                        if len(v) >= MIN_USER_LIKES:
                            user_actual[did] = v

                user_feed: Dict[str, np.ndarray] = {}
                for did in eligible_dids:
                    if did not in user_topk:
                        continue
                    fv = pool_vals[user_topk[did]]
                    fv = fv[np.isfinite(fv)]
                    if len(fv) > 0:
                        user_feed[did] = fv

                result = _compute_trait_decomposition(
                    pool_vals, user_actual, user_feed, eligible_dids,
                )
                total_traits += 1
                if result is not None:
                    trait_results[f"{gname}::{col}"] = result

        print(
            f"    [synthetic_feed] Trait decomposition complete: "
            f"{len(trait_results)}/{total_traits} traits had sufficient data",
            flush=True,
        )

        # ---- headline + JSON (saved before plots so a plot crash
        #      doesn't lose the numerical results) ----
        headline = _generate_headline(trait_results)

        groups_json: Dict[str, Any] = {}
        for gname in group_names:
            gdict: Dict[str, Any] = {}
            for label in group_labels.get(gname, []):
                key = f"{gname}::{label}"
                if key not in trait_results:
                    continue
                tr = trait_results[key]
                gdict[label] = {
                    "mean_user_pref_std": float(np.mean(tr.user_pref_std)),
                    "mean_model_amp_std": float(np.mean(tr.model_amp_std)),
                    "mean_model_excess_std": float(
                        np.mean(tr.model_excess_std),
                    ),
                    "mean_user_pref_abs": float(np.mean(tr.user_pref_std)) * tr.pool_sd,
                    "mean_model_amp_abs": float(np.mean(tr.model_amp_std)) * tr.pool_sd,
                    "mean_model_excess_abs": float(np.mean(tr.model_excess_std)) * tr.pool_sd,
                    "pool_mean": tr.pool_mean,
                    "pool_sd": tr.pool_sd,
                    "n_users": tr.n_users,
                    "cohen_d_pref": tr.cohen_d_pref,
                    "cohen_d_amp": tr.cohen_d_amp,
                    "cohen_d_excess": tr.cohen_d_excess,
                    "p_pref": tr.p_pref,
                    "p_amp": tr.p_amp,
                    "p_excess": tr.p_excess,
                }
            if gdict:
                groups_json[gname] = gdict

        summary = {
            "headline": headline,
            "n_pool_posts": n_pool,
            "top_k": k,
            "n_users_eligible": len(eligible_dids),
            "n_users_scored": len(user_scores),
            "min_user_likes": MIN_USER_LIKES,
            "groups": groups_json,
        }
        self.save_json(summary, out_dir / "synthetic_feed_summary.json")
        print("    [synthetic_feed] Saved summary JSON", flush=True)

        # ---- volume split for detail plots ----
        from .liked_trait_volume import _givers_of_half_the_likes

        meta = ctx.user_metadata_df
        did_to_likes: Dict[str, int] = dict(
            zip(meta["did"], meta["num_total_likes"])
        )
        eligible_arr = np.array(eligible_dids)
        is_high, _pct_hi, pct_hi_str = _givers_of_half_the_likes(
            eligible_arr, did_to_likes,
        )

        # ---- plots ----
        print(
            f"    [synthetic_feed] Generating plots for "
            f"{len(trait_results)} traits...",
            flush=True,
        )
        group_color_map = {
            g: _GROUP_COLORS[i % len(_GROUP_COLORS)]
            for i, g in enumerate(group_names)
        }
        plot_paths: List[str] = []
        plot_paths.extend(_save_prevalence_legends(pct_hi_str, out_dir))

        if trait_results:
            plot_paths.append(
                str(_plot_decomposition_bar(
                    trait_results, group_color_map, out_dir,
                    standardize=True,
                ))
            )
            plot_paths.append(
                str(_plot_decomposition_bar(
                    trait_results, group_color_map, out_dir,
                    standardize=False,
                ))
            )

        for gname in group_names:
            gt: Dict[str, TraitDecompResult] = {}
            for label in group_labels.get(gname, []):
                key = f"{gname}::{label}"
                if key in trait_results:
                    gt[label] = trait_results[key]
            if gt:
                plot_paths.append(
                    str(_plot_group_decomposition(
                        gname, gt, out_dir, standardize=True,
                    ))
                )
                plot_paths.append(
                    str(_plot_group_decomposition(
                        gname, gt, out_dir, standardize=False,
                    ))
                )
                detail_path, shared_ylim = _plot_group_prevalence_detail(
                    gname, gt, is_high, pct_hi_str, out_dir,
                )
                prev_path, _ = _plot_group_prevalence(
                    gname, gt, out_dir, ylim=shared_ylim,
                )
                plot_paths.append(str(prev_path))
                plot_paths.append(str(detail_path))

                if gname in _SUBSET_TRAITS:
                    sub_name, sub_keep = _SUBSET_TRAITS[gname]
                    sub_gt = {k: v for k, v in gt.items() if k in sub_keep}
                    if sub_gt:
                        sub_detail, sub_ylim = _plot_group_prevalence_detail(
                            sub_name, sub_gt, is_high, pct_hi_str, out_dir,
                        )
                        sub_prev, _ = _plot_group_prevalence(
                            sub_name, sub_gt, out_dir, ylim=sub_ylim,
                        )
                        plot_paths.append(str(sub_prev))
                        plot_paths.append(str(sub_detail))

                if gname in _TOP_N_SUBSETS:
                    sub_name, top_n = _TOP_N_SUBSETS[gname]
                    top_keys = sorted(
                        gt.keys(), key=lambda t: gt[t].pool_mean, reverse=True
                    )[:top_n]
                    sub_gt = {k: gt[k] for k in top_keys}
                    if sub_gt:
                        sub_detail, sub_ylim = _plot_group_prevalence_detail(
                            sub_name, sub_gt, is_high, pct_hi_str, out_dir,
                        )
                        sub_prev, _ = _plot_group_prevalence(
                            sub_name, sub_gt, out_dir, ylim=sub_ylim,
                        )
                        plot_paths.append(str(sub_prev))
                        plot_paths.append(str(sub_detail))

        return {
            "headline": headline,
            "n_pool_posts": n_pool,
            "n_users_scored": len(user_scores),
            "groups_plotted": len(groups_json),
            "plot_paths": plot_paths,
        }
