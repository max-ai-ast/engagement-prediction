#!/usr/bin/env python3

"""
Trait Correlations Evaluation Module

Measures Spearman rank correlations between the model's predicted engagement
probability and NLP content traits (topic, sentiment, toxicity, etc.) on
holdout samples.  Separate analyses are run for negatives (y_true == 0) and
positives (y_true == 1) to keep interpretation clean.

For each inference group (e.g. emotion_sentiment, topic, moderation) a
horizontal bar chart of correlations is saved for each subset, plus a JSON
summary.

Outputs (under trait_corrs/):
- <group>_corr_neg.png: bar chart per group, negative samples
- <group>_corr_pos.png: bar chart per group, positive samples
- trait_corrs_summary.json: all correlations and metadata for both subsets
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr

from . import EvalContext, EvalModule, scaled_figsize

STRUCT_PREFIX = "message.commit.record.text"

_SUBSET_LABELS = {"neg": "negatives", "pos": "positives"}


def _load_inferences(run_dir: Path) -> pl.LazyFrame:
    """Locate and scan inferences_core from the 01_get_data stage output."""
    from utils.pipeline.core import select_prior_output
    from utils.helpers import load_parquet_from_prior

    prior = select_prior_output(run_dir, "01_get_data")
    if prior is None:
        raise FileNotFoundError("No 01_get_data output found")
    return load_parquet_from_prior(prior, "inferences_core_")


def _unnest_text_inferences(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Unnest inferences -> text -> message.commit.record.text into top-level group columns.

    Returns the flattened DataFrame and the list of inference group column names
    (only the struct fields from the text-body path, not sibling structs).
    """
    partially = (
        df
        .unnest("inferences")
        .unnest("text")
        .rename({STRUCT_PREFIX: "_text_inf"})
    )
    group_names = [
        f.name for f in partially.schema["_text_inf"].fields
        if isinstance(f.dtype, pl.Struct)
    ]
    return partially.unnest("_text_inf"), group_names


def eb_shrink(
    rhos: np.ndarray,
    ns: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Empirical-Bayes shrinkage of Spearman correlations via Fisher z-transform.

    Each per-user rho_i (estimated from n_i observations) is shrunk toward the
    grand mean, with shrinkage strength inversely proportional to n_i.

    Steps:
      1. Fisher z-transform: z_i = arctanh(rho_i), with rho clipped to
         +/-0.999 to avoid divergence.
      2. Sampling variance: var_i = 1 / (n_i - 3), the standard large-sample
         approximation for Spearman (Bonett & Wright 2000).
      3. Prior via method-of-moments:
           mu   = mean(z_i)
           tau² = max(0,  var(z_i) - mean(var_i))
         where tau² estimates the true between-user variance after removing
         expected sampling noise.
      4. Posterior mean (James–Stein / normal–normal EB):
           z_shrunk_i = mu + B_i * (z_i - mu)
         with reliability B_i = tau² / (tau² + var_i).
      5. Back-transform: rho_shrunk_i = tanh(z_shrunk_i).

    Returns (shrunk_rhos, tau_sq).  When tau² = 0 (no detectable heterogeneity)
    all rhos collapse to tanh(mu) ≈ mean(rho).
    """
    rhos_clipped = np.clip(rhos, -0.999, 0.999)
    z = np.arctanh(rhos_clipped)
    var_i = 1.0 / (ns.astype(np.float64) - 3.0)

    mu = float(np.mean(z))
    tau_sq = max(0.0, float(np.var(z, ddof=0)) - float(np.mean(var_i)))

    if tau_sq == 0.0:
        return np.full_like(rhos, np.tanh(mu)), tau_sq

    B = tau_sq / (tau_sq + var_i)
    z_shrunk = mu + B * (z - mu)
    return np.tanh(z_shrunk), tau_sq


def _correlations_for_group(
    y_pred: np.ndarray,
    group_df: pl.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, tuple[float, float, float]]:
    """Spearman correlation + Fisher-z CI between y_pred and each column.

    Returns {label: (rho, ci_lo, ci_hi)} for each column with enough data.
    """
    from scipy.stats import norm

    z_crit = norm.ppf(1 - alpha / 2)
    corrs: Dict[str, tuple[float, float, float]] = {}
    for col in group_df.columns:
        vals = group_df[col].to_numpy()
        mask = np.isfinite(vals)
        n = int(mask.sum())
        if n < 10:
            continue
        rho, _ = spearmanr(y_pred[mask], vals[mask])
        se = 1.0 / np.sqrt(n - 3)
        z = np.arctanh(rho)
        ci_lo = float(np.tanh(z - z_crit * se))
        ci_hi = float(np.tanh(z + z_crit * se))
        corrs[col] = (float(rho), ci_lo, ci_hi)
    return corrs


def _plot_group(
    group_name: str,
    corrs: Dict[str, tuple[float, float, float]],
    out_dir: Path,
    suffix: str,
) -> Path:
    labels = sorted(corrs, key=lambda k: abs(corrs[k][0]), reverse=True)
    rhos = [corrs[k][0] for k in labels]
    ci_lo = [corrs[k][1] for k in labels]
    ci_hi = [corrs[k][2] for k in labels]
    xerr_neg = [r - lo for r, lo in zip(rhos, ci_lo)]
    xerr_pos = [hi - r for r, hi in zip(rhos, ci_hi)]
    colors = ["#4878CF" if v >= 0 else "#D65F5F" for v in rhos]

    fig, ax = plt.subplots(figsize=scaled_figsize(7, max(2.5, 0.35 * len(labels))))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, rhos, color=colors, edgecolor="white", linewidth=0.5,
            xerr=[xerr_neg, xerr_pos], error_kw=dict(ecolor="#333333", capsize=2, linewidth=0.8))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Spearman ρ with predicted P(engagement)")
    title = group_name.replace("_", " ").title()
    ax.set_title(f"{title}  ({_SUBSET_LABELS[suffix]})")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()

    path = out_dir / f"{group_name}_corr_{suffix}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


class TraitCorrsModule(EvalModule):
    name = "trait_corrs"
    description = "Spearman correlations between predicted engagement and NLP traits (pos & neg subsets)"

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
        subsets = [("neg", 0), ("pos", 1)]
        subset_summaries: Dict[str, Any] = {}
        plot_paths: list[str] = []
        total_groups_plotted = 0

        for suffix, label_val in subsets:
            preds = (
                all_preds
                .filter(pl.col("y_true") == label_val)
                .select("post_id", "y_pred_proba")
            )
            n_samples = len(preds)

            joined = (
                inferences_lf
                .join(preds.lazy(), left_on="at_uri", right_on="post_id", how="inner")
                .collect()
            )
            n_matched = len(joined)
            if n_matched < 30:
                subset_summaries[suffix] = {
                    "skipped": True,
                    "reason": f"only {n_matched} {_SUBSET_LABELS[suffix]} matched inferences",
                    "n_samples": n_samples,
                    "n_matched": n_matched,
                }
                continue

            flat, group_names = _unnest_text_inferences(joined)
            y_pred = flat["y_pred_proba"].to_numpy()

            all_corrs: Dict[str, Dict[str, tuple[float, float, float]]] = {}

            for gname in group_names:
                group_df = flat.select(gname).unnest(gname)
                corrs = _correlations_for_group(y_pred, group_df)
                if not corrs:
                    continue
                all_corrs[gname] = corrs
                path = _plot_group(gname, corrs, out_dir, suffix=suffix)
                plot_paths.append(str(path))

            total_groups_plotted += len(all_corrs)

            corrs_for_json = {
                g: {label: {"rho": r, "ci_lo": lo, "ci_hi": hi}
                    for label, (r, lo, hi) in labels.items()}
                for g, labels in all_corrs.items()
            }
            subset_summaries[suffix] = {
                "n_samples": n_samples,
                "n_matched": n_matched,
                "coverage_pct": round(100.0 * n_matched / n_samples, 2) if n_samples else 0,
                "groups": list(all_corrs.keys()),
                "correlations": corrs_for_json,
            }

        self.save_json(subset_summaries, out_dir / "trait_corrs_summary.json")

        return {
            "subsets": list(subset_summaries.keys()),
            "groups_plotted": total_groups_plotted,
            "plot_paths": plot_paths,
        }
