#!/usr/bin/env python3

"""
Trait Prevalence Evaluation Module

Compares the mean NLP trait score in positive (y_true == 1) vs negative
(y_true == 0) tweets, grouped by inference group (same grouping as
trait_corrs).  One horizontal grouped bar chart per inference group.

Outputs (under trait_prevalence/):
- <group>_prevalence.png: grouped bar chart per inference group
- trait_prevalence_summary.json: per-trait means and sample sizes
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from . import EvalContext, EvalModule, scaled_figsize
from .trait_corrs import _load_inferences, _unnest_text_inferences


def _means_for_group(
    group_df: pl.DataFrame,
) -> Dict[str, float]:
    """Mean of each column, ignoring nulls/NaNs."""
    means: Dict[str, float] = {}
    for col in group_df.columns:
        vals = group_df[col].to_numpy()
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            continue
        means[col] = float(np.mean(finite))
    return means


def _plot_group(
    group_name: str,
    pos_means: Dict[str, float],
    neg_means: Dict[str, float],
    out_dir: Path,
) -> Path:
    all_labels = sorted(
        set(pos_means) | set(neg_means),
        key=lambda k: abs(pos_means.get(k, 0.0) - neg_means.get(k, 0.0)),
        reverse=True,
    )

    pos_vals = [pos_means.get(k, 0.0) for k in all_labels]
    neg_vals = [neg_means.get(k, 0.0) for k in all_labels]

    bar_height = 0.35
    y = np.arange(len(all_labels))

    fig, ax = plt.subplots(figsize=scaled_figsize(7, max(2.5, 0.45 * len(all_labels))))
    ax.barh(y - bar_height / 2, pos_vals, bar_height,
            label="positives (liked)", color="#4878CF",
            edgecolor="white", linewidth=0.5)
    ax.barh(y + bar_height / 2, neg_vals, bar_height,
            label="negatives (not liked)", color="#D65F5F",
            edgecolor="white", linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels(all_labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean trait score")
    title = group_name.replace("_", " ").title()
    ax.set_title(f"{title} — Trait Prevalence (pos vs neg)")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.8)
    plt.tight_layout()

    path = out_dir / f"{group_name}_prevalence.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


class TraitPrevalenceModule(EvalModule):
    name = "trait_prevalence"
    description = "Mean NLP trait scores in positive vs negative tweets"

    def run(self, ctx: EvalContext) -> Dict[str, Any]:
        out_dir = self.get_output_dir(ctx)
        run_dir = ctx.config.get("run_dir")
        if run_dir is None:
            return {"skipped": True, "reason": "run_dir not in eval config"}

        try:
            inferences_lf = _load_inferences(Path(run_dir))
        except FileNotFoundError as e:
            return {"skipped": True, "reason": str(e)}

        all_preds = pl.from_pandas(ctx.predictions_df)
        joined = (
            inferences_lf
            .join(
                all_preds.select("post_id", "y_true").lazy(),
                left_on="at_uri",
                right_on="post_id",
                how="inner",
            )
            .collect()
        )
        if len(joined) < 30:
            return {"skipped": True, "reason": f"only {len(joined)} posts matched inferences"}

        flat, group_names = _unnest_text_inferences(joined)
        pos_mask = flat["y_true"] == 1
        neg_mask = flat["y_true"] == 0
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())

        plot_paths: List[str] = []
        groups_json: Dict[str, Any] = {}

        for gname in group_names:
            group_df = flat.select(gname).unnest(gname)
            pos_means = _means_for_group(group_df.filter(pos_mask))
            neg_means = _means_for_group(group_df.filter(neg_mask))
            if not pos_means and not neg_means:
                continue

            path = _plot_group(gname, pos_means, neg_means, out_dir)
            plot_paths.append(str(path))

            groups_json[gname] = {
                trait: {
                    "mean_pos": pos_means.get(trait),
                    "mean_neg": neg_means.get(trait),
                }
                for trait in sorted(set(pos_means) | set(neg_means))
            }

        summary = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_matched": len(joined),
            "groups": groups_json,
        }
        self.save_json(summary, out_dir / "trait_prevalence_summary.json")

        return {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "groups_plotted": len(groups_json),
            "plot_paths": plot_paths,
        }
