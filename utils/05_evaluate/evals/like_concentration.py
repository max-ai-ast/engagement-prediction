#!/usr/bin/env python3

"""
Like-Giving Concentration Evaluation Module

Characterises the long-tailed distribution of like-giving across users.
This is the "setup" framing for ecological-bias analyses: if a small
fraction of users account for most of the likes, the aggregate training
signal is dominated by their preferences.

Outputs (under like_concentration/):
- lorenz_likes.png:              Lorenz curve with Gini and percentile annotations
- likes_distribution.png:        log-scaled histogram of num_total_likes
- like_concentration_summary.json
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import (
    EvalContext,
    EvalModule,
    compute_gini_coefficient,
    compute_lorenz_curve,
    scaled_figsize,
)


def _top_share(sorted_values: np.ndarray, top_frac: float) -> float:
    """Share of total accounted for by the top *top_frac* of the population."""
    n = len(sorted_values)
    k = max(1, int(np.ceil(n * top_frac)))
    return float(sorted_values[-k:].sum() / sorted_values.sum())


def _plot_lorenz(
    likes: np.ndarray,
    gini: float,
    out_dir: Path,
) -> Path:
    cum_pop, cum_val = compute_lorenz_curve(likes)

    fig, ax = plt.subplots(figsize=scaled_figsize(7, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5,
            label="perfect equality")
    ax.plot(cum_pop, cum_val, color="#1f77b4", linewidth=2,
            label=f"likes  (Gini = {gini:.3f})")
    ax.fill_between(cum_pop, cum_val, cum_pop, alpha=0.15, color="#1f77b4")

    half_x = float(np.interp(0.5, cum_val, cum_pop))
    top_pct_for_half = round((1.0 - half_x) * 100)
    half_y = 0.5
    ax.annotate(
        f"top {top_pct_for_half}% → 50% of likes",
        xy=(half_x, half_y),
        xytext=(half_x - 0.25, half_y + 0.15),
        fontsize=8, color="black",
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

    ax.set_xlabel("Cumulative share of users (sorted by likes given)")
    ax.set_ylabel("Cumulative share of total likes")
    ax.set_title("Like-Giving Concentration")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    path = out_dir / "lorenz_likes.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_distribution(
    likes: np.ndarray,
    out_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=scaled_figsize(8, 4.5))

    positive = likes[likes > 0]
    if len(positive) > 0:
        log_bins = np.logspace(
            np.log10(max(1, positive.min())),
            np.log10(positive.max()),
            50,
        )
        ax.hist(positive, bins=log_bins, color="#1f77b4", edgecolor="white",
                linewidth=0.4, alpha=0.8)
        ax.set_xscale("log")

    n_zero = int((likes == 0).sum())
    mean_val = float(np.mean(likes))
    median_val = float(np.median(likes))

    ax.axvline(mean_val, color="#d62728", linestyle="--", linewidth=1.5,
               label=f"mean = {mean_val:,.0f}")
    ax.axvline(median_val, color="#2ca02c", linestyle=":", linewidth=1.5,
               label=f"median = {median_val:,.0f}")

    ax.set_xlabel("Total likes given (log scale)")
    ax.set_ylabel("Number of users")
    ax.set_title(f"Distribution of Like-Giving Volume  (n={len(likes)}, "
                 f"{n_zero} with 0 likes)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()

    path = out_dir / "likes_distribution.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


class LikeConcentrationModule(EvalModule):
    name = "like_concentration"
    description = (
        "Lorenz curve and Gini coefficient for the distribution of "
        "like-giving across users"
    )

    def run(self, ctx: EvalContext) -> Dict[str, Any]:
        out_dir = self.get_output_dir(ctx)

        likes = ctx.user_metadata_df["num_total_likes"].to_numpy().astype(float)
        likes = likes[~np.isnan(likes)]
        n_users = len(likes)

        if n_users < 5:
            return {"skipped": True, "reason": f"only {n_users} users with like data"}

        gini = compute_gini_coefficient(likes)
        sorted_likes = np.sort(likes)

        plot_paths = [
            str(_plot_lorenz(likes, gini, out_dir)),
            str(_plot_distribution(likes, out_dir)),
        ]

        summary = {
            "n_users": n_users,
            "gini": gini,
            "mean": float(np.mean(likes)),
            "median": float(np.median(likes)),
            "std": float(np.std(likes)),
            "percentiles": {
                "p50": float(np.percentile(likes, 50)),
                "p75": float(np.percentile(likes, 75)),
                "p90": float(np.percentile(likes, 90)),
                "p95": float(np.percentile(likes, 95)),
                "p99": float(np.percentile(likes, 99)),
            },
            "top_10pct_share": _top_share(sorted_likes, 0.10),
            "top_5pct_share": _top_share(sorted_likes, 0.05),
            "top_1pct_share": _top_share(sorted_likes, 0.01),
            "n_zero_likes": int((likes == 0).sum()),
        }
        self.save_json(summary, out_dir / "like_concentration_summary.json")

        return {
            "gini": gini,
            "top_10pct_share": summary["top_10pct_share"],
            "top_1pct_share": summary["top_1pct_share"],
            "n_users": n_users,
            "plot_paths": plot_paths,
        }
