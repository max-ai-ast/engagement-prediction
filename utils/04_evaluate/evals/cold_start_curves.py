#!/usr/bin/env python3

"""
Cold Start Curves Evaluation Module

This module analyzes how model performance varies with the amount of user
history available at the time each prediction was made (the "cold start" problem).

Each row in predictions_df has its own ``num_embedding_likes`` value representing
how many prior likes were in the user embedding when that specific prediction was
made.  Predictions are binned by that per-row value so the analysis operates at
the post level, not the user level.

For each performance metric one plot is produced showing:
- One bold aggregate curve computed across all predictions in each bin

Only precision, recall, and F1 are plotted.  Accuracy and AUC-ROC are
excluded because our negative samples are imperfect (we don't know for
certain that the user wouldn't have liked the negative post), which
makes metrics that treat the negative class as ground truth unreliable.
Precision, recall, and F1 depend only on real likes (positives) and are
robust to noisy negatives. Arguably, only recall is cleanly interpretable in this context.

Outputs:
- cold_start_summary.json: Summary statistics and bin-level metrics
- <metric>_cold_start.png: One cold-start curve plot per metric
- binned_metrics.csv: Full per-bin aggregate metrics table
- user_distribution_by_bin.png: Distribution of predictions across bins
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import (
    EvalContext,
    EvalModule,
    compute_classification_metrics,
)


class ColdStartCurvesModule(EvalModule):
    """
    Evaluation module for analyzing cold start behavior at the post level.

    Each prediction carries its own ``num_embedding_likes`` value (the history
    length at the time that specific prediction was made).  Predictions are
    binned by that value and performance metrics are computed per bin to
    understand how model quality varies with available user history.
    """

    name = "cold_start_curves"
    description = "Analyzes model performance as a function of per-prediction history length (embedding likes count)"

    # Default bin edges for number of embedding likes.
    # Integer boundaries with right=False (left-closed, right-open [low, high))
    # ensure each integer 0–10 falls cleanly in its own bin, while later
    # edges define wider ranges like [10, 20), [20, 50), etc.
    DEFAULT_BIN_EDGES = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        20, 50, 100, 500, float('inf'),
    ]

    # Metrics to plot (one plot each).  Only metrics robust to noisy
    # negatives: precision, recall, F1.  AUC-ROC and accuracy are excluded
    # because our negative samples are imperfect.
    METRICS = ['precision', 'recall', 'f1']

    # Plot styling
    FIGURE_SIZE = (10, 6)
    DPI = 150

    METRIC_COLORS = {
        'precision': '#2ca02c',
        'recall':    '#ff7f0e',
        'f1':        '#9467bd',
    }

    def run(self, ctx: EvalContext) -> Dict[str, Any]:
        """
        Run cold start analysis.

        Args:
            ctx: EvalContext with predictions (must include ``num_embedding_likes``
                 column from stage_evaluate enrichment) and output directory.

        Returns:
            Dict with binned metrics and artifact paths.
        """
        out_dir = self.get_output_dir(ctx)

        # --- hyperparams ---
        bin_edges = ctx.config.get('cold_start_bin_edges', self.DEFAULT_BIN_EDGES)

        if ctx.has_ranking_rows:
            return self._run_ranking_rows(ctx, bin_edges)

        # Require per-prediction history length
        if 'num_embedding_likes' not in ctx.predictions_df.columns:
            print("    Warning: num_embedding_likes not in predictions_df, skipping cold start analysis")
            return {
                'status': 'skipped',
                'reason': 'num_embedding_likes not available in predictions_df',
            }

        predictions_df = ctx.predictions_df.copy()
        predictions_df['num_embedding_likes'] = predictions_df['num_embedding_likes'].fillna(0).astype(int)

        n_posts = len(predictions_df)
        n_users = predictions_df['did'].nunique()
        print(f"    Cold start analysis: {n_posts} predictions across {n_users} users...")

        # Assign bin labels to each prediction row
        bin_labels = self._make_bin_labels(bin_edges)
        predictions_df['likes_bin'] = pd.cut(
            predictions_df['num_embedding_likes'],
            bins=bin_edges,
            labels=bin_labels,
            right=False,  # left-closed intervals [low, high)
        )

        # Diagnostics
        self._log_data_health(predictions_df)
        self._log_bin_summary(predictions_df)

        # Aggregate metrics across all predictions per bin
        print("    Computing post-level binned metrics...")
        binned_metrics = self._compute_binned_metrics_post_level(predictions_df, bin_labels)

        binned_path = out_dir / "binned_metrics.csv"
        binned_metrics.to_csv(binned_path, index=False)

        # One plot per metric
        print("    Generating cold start curve plots (one per metric)...")
        plot_paths = {}
        for metric in self.METRICS:
            if metric not in binned_metrics.columns:
                continue
            if binned_metrics[f'{metric}_n'].sum() == 0:
                continue

            plot_path = out_dir / f"{metric}_cold_start.png"
            self._plot_cold_start_per_metric(
                metric=metric,
                binned_metrics=binned_metrics,
                save_path=plot_path,
            )
            plot_paths[f"{metric}_plot_path"] = str(plot_path)

        # User/prediction distribution by bin
        dist_path = out_dir / "user_distribution_by_bin.png"
        self._plot_distribution(predictions_df, dist_path)
        plot_paths['distribution_plot_path'] = str(dist_path)

        # Summary statistics
        summary = self._compute_summary(predictions_df, binned_metrics)
        summary.update(plot_paths)
        summary['binned_metrics_path'] = str(binned_path)
        summary['bin_edges'] = [float(e) if e != float('inf') else 'inf' for e in bin_edges]

        summary_path = out_dir / "cold_start_summary.json"
        self.save_json(summary, summary_path)

        return summary

    def _run_ranking_rows(self, ctx: EvalContext, bin_edges: List[float]) -> Dict[str, Any]:
        out_dir = self.get_output_dir(ctx)
        ranking_rows_df = ctx.ranking_rows_df.copy()
        ranking_rows_df['num_embedding_likes'] = ranking_rows_df['num_embedding_likes'].fillna(0).astype(int)

        bin_labels = self._make_bin_labels(bin_edges)
        ranking_rows_df['likes_bin'] = pd.cut(
            ranking_rows_df['num_embedding_likes'],
            bins=bin_edges,
            labels=bin_labels,
            right=False,
        )

        metric_cols = [
            col for col in ranking_rows_df.columns
            if col.startswith('ndcg@') or col.startswith('recall@') or col in ('average_precision', 'auc_roc')
        ]
        rows = []
        for bin_label in bin_labels:
            bin_data = ranking_rows_df[ranking_rows_df['likes_bin'] == bin_label]
            if len(bin_data) == 0:
                continue
            row: Dict[str, Any] = {
                'bin': str(bin_label),
                'n_predictions': len(bin_data),
                'n_ranking_rows': len(bin_data),
                'n_users': bin_data['did'].nunique(),
                'n_positive': int(bin_data['positive_count'].sum()),
                'mean_embedding_likes': float(bin_data['num_embedding_likes'].mean()),
                'mean_candidate_count': float(bin_data['candidate_count'].mean()),
            }
            for metric in metric_cols:
                values = bin_data[metric].dropna()
                row[metric] = float(values.mean()) if len(values) else float('nan')
                row[f'{metric}_n'] = len(values)
            rows.append(row)

        binned_metrics = pd.DataFrame(rows)
        binned_path = out_dir / "binned_metrics.csv"
        binned_metrics.to_csv(binned_path, index=False)

        plot_paths = {}
        for metric in metric_cols:
            n_col = f'{metric}_n'
            if n_col not in binned_metrics.columns or binned_metrics[n_col].sum() == 0:
                continue
            plot_path = out_dir / f"{metric.replace('@', '_at_')}_cold_start.png"
            self._plot_cold_start_per_metric(
                metric=metric,
                binned_metrics=binned_metrics,
                save_path=plot_path,
            )
            plot_paths[f"{metric}_plot_path"] = str(plot_path)

        dist_path = out_dir / "user_distribution_by_bin.png"
        self._plot_distribution(ranking_rows_df, dist_path)
        plot_paths['distribution_plot_path'] = str(dist_path)

        summary = self._compute_summary_for_ranking_rows(ranking_rows_df, binned_metrics, metric_cols)
        summary.update(plot_paths)
        summary['binned_metrics_path'] = str(binned_path)
        summary['bin_edges'] = [float(e) if e != float('inf') else 'inf' for e in bin_edges]

        summary_path = out_dir / "cold_start_summary.json"
        self.save_json(summary, summary_path)
        return summary

    # ------------------------------------------------------------------
    # Bin helpers
    # ------------------------------------------------------------------

    def _make_bin_labels(self, bin_edges: List[float]) -> List[str]:
        """Create human-readable bin labels for left-closed, right-open [low, high) intervals.

        Expects integer-valued edges.  For a single-integer bin (width 1) the
        label is just that integer; for wider bins the label is ``low-high_inclusive``;
        for the final open-ended bin the label is ``low+``.
        """
        labels = []
        for low, high in zip(bin_edges[:-1], bin_edges[1:]):
            low_int = int(low)
            if high == float('inf'):
                labels.append(f"{low_int}+")
            elif int(high) - low_int == 1:
                labels.append(str(low_int))
            else:
                labels.append(f"{low_int}-{int(high) - 1}")
        return labels

    # ------------------------------------------------------------------
    # Validation / logging
    # ------------------------------------------------------------------

    def _log_data_health(self, predictions_df: pd.DataFrame) -> None:
        """Print diagnostic checks so anomalies are immediately visible."""
        n = len(predictions_df)
        n_users = predictions_df['did'].nunique()
        n_pos = int((predictions_df['y_true'] == 1).sum())
        n_neg = n - n_pos

        print(f"    [data health] {n} predictions ({n_pos} pos, {n_neg} neg) "
              f"across {n_users} users")

        # Pair parity: every bin should have an even count because each
        # target row produces one positive and one negative with the same
        # history length.
        bin_counts = predictions_df['likes_bin'].value_counts()
        odd_bins = [str(b) for b, c in bin_counts.items() if c % 2 != 0]
        if odd_bins:
            print(f"    [data health] WARNING: odd prediction counts in bins "
                  f"{odd_bins} — pos/neg pairing may be broken")
        else:
            print(f"    [data health] All bin counts are even (pos/neg pairing OK)")

        # Zero-history breakdown
        zero_mask = predictions_df['num_embedding_likes'] == 0
        n_zero = int(zero_mask.sum())
        if n_zero > 0:
            z_pos = int((predictions_df.loc[zero_mask, 'y_true'] == 1).sum())
            z_neg = n_zero - z_pos
            print(f"    [data health] {n_zero} zero-history predictions "
                  f"({z_pos} pos, {z_neg} neg)")
        else:
            print(f"    [data health] No zero-history predictions")

        # Coverage
        combos = predictions_df.groupby('did')['num_embedding_likes'].nunique()
        print(f"    [data health] Users span a median of "
              f"{combos.median():.0f} distinct history-length bins "
              f"(min {combos.min()}, max {combos.max()})")
        print(f"    [data health] Note: not all users span all bins — "
              f"low-history moments may fall in train/val splits")

    def _log_bin_summary(self, predictions_df: pd.DataFrame) -> None:
        """Print a compact per-bin summary table."""
        header = (f"    {'bin':>8s}  {'n_pred':>7s}  {'n_user':>7s}  "
                  f"{'n_pos':>6s}  {'n_neg':>6s}  {'frac_pos':>8s}")
        print(header)
        print(f"    {'—' * len(header.strip())}")
        for bin_label in predictions_df['likes_bin'].cat.categories:
            bdf = predictions_df[predictions_df['likes_bin'] == bin_label]
            if len(bdf) == 0:
                continue
            n_pred = len(bdf)
            n_user = bdf['did'].nunique()
            n_pos = int((bdf['y_true'] == 1).sum())
            n_neg = n_pred - n_pos
            frac = n_pos / n_pred if n_pred else 0
            print(f"    {str(bin_label):>8s}  {n_pred:>7d}  {n_user:>7d}  "
                  f"{n_pos:>6d}  {n_neg:>6d}  {frac:>8.3f}")

    # ------------------------------------------------------------------
    # Metric computation
    # ------------------------------------------------------------------

    def _compute_binned_metrics_post_level(
        self,
        predictions_df: pd.DataFrame,
        bin_labels: List[str],
    ) -> pd.DataFrame:
        """Compute aggregate metrics per bin pooling ALL predictions in that bin
        (post-level aggregation, not user-level).
        """
        rows = []
        for bin_label in bin_labels:
            bin_data = predictions_df[predictions_df['likes_bin'] == bin_label]
            if len(bin_data) == 0:
                continue

            y_true = bin_data['y_true'].to_numpy()
            y_pred_proba = bin_data['y_pred_proba'].to_numpy()
            metrics = compute_classification_metrics(y_true, y_pred_proba)

            row: Dict[str, Any] = {
                'bin': str(bin_label),
                'n_predictions': len(bin_data),
                'n_users': bin_data['did'].nunique(),
                'n_positive': int(y_true.sum()),
                'mean_embedding_likes': float(bin_data['num_embedding_likes'].mean()),
            }
            for metric, value in metrics.items():
                row[metric] = value
                row[f'{metric}_n'] = 0 if np.isnan(value) else len(y_true)

            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _plot_cold_start_per_metric(
        self,
        metric: str,
        binned_metrics: pd.DataFrame,
        save_path: Path,
    ) -> None:
        """Plot one cold-start curve: post-level aggregate across all predictions."""
        agg_df = binned_metrics[binned_metrics[f'{metric}_n'] > 0].copy()
        if len(agg_df) == 0:
            return

        bin_order = list(agg_df['bin'])
        agg_x = list(range(len(agg_df)))
        agg_y = agg_df[metric].to_numpy()
        color = self.METRIC_COLORS.get(metric, '#333333')

        fig, ax = plt.subplots(figsize=self.FIGURE_SIZE)

        ax.plot(
            agg_x, agg_y,
            color=color, linewidth=2.5, marker='o', markersize=7,
            label='Aggregate (all posts)',
        )

        for xi, yi, n in zip(agg_x, agg_y, agg_df['n_predictions'].to_numpy()):
            if not np.isnan(yi):
                ax.annotate(
                    f'n={n}', (xi, yi),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=8, alpha=0.75,
                )

        ax.set_xticks(agg_x)
        ax.set_xticklabels(bin_order, rotation=45, ha='right')
        ax.set_xlabel('Number of Embedding Likes at Prediction Time', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        ax.set_title(
            f'Cold Start: {metric.replace("_", " ").title()} vs. History Length',
            fontsize=13,
        )
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)

    def _plot_distribution(
        self,
        predictions_df: pd.DataFrame,
        save_path: Path,
    ) -> None:
        """Plot distribution of predictions and users across bins."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        bin_pred_counts = predictions_df['likes_bin'].value_counts().sort_index()
        bin_user_counts = (
            predictions_df.groupby('likes_bin')['did']
            .nunique()
            .reindex(bin_pred_counts.index)
        )

        ax1.bar(range(len(bin_pred_counts)), bin_pred_counts.to_numpy(),
                color='steelblue', edgecolor='black')
        ax1.set_xticks(range(len(bin_pred_counts)))
        ax1.set_xticklabels(bin_pred_counts.index, rotation=45, ha='right')
        ax1.set_xlabel('Embedding Likes Bin', fontsize=11)
        ax1.set_ylabel('Number of Predictions', fontsize=11)
        ax1.set_title('Predictions Distribution Across History Length Bins', fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(bin_pred_counts.to_numpy()):
            ax1.text(i, v + 0.5, str(v), ha='center', fontsize=9)

        ax2.bar(range(len(bin_user_counts)), bin_user_counts.to_numpy(),
                color='darkorange', edgecolor='black')
        ax2.set_xticks(range(len(bin_user_counts)))
        ax2.set_xticklabels(bin_user_counts.index, rotation=45, ha='right')
        ax2.set_xlabel('Embedding Likes Bin', fontsize=11)
        ax2.set_ylabel('Number of Users', fontsize=11)
        ax2.set_title('Unique Users per History Length Bin', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(bin_user_counts.to_numpy()):
            ax2.text(i, v + 0.5, str(v), ha='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _compute_summary(
        self,
        predictions_df: pd.DataFrame,
        binned_metrics: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Compute summary statistics for the cold start analysis."""
        summary: Dict[str, Any] = {
            'total_predictions_analyzed': len(predictions_df),
            'total_users_analyzed': int(predictions_df['did'].nunique()),
            'num_bins': len(binned_metrics),
            'embedding_likes_stats': {
                'mean': float(predictions_df['num_embedding_likes'].mean()),
                'median': float(predictions_df['num_embedding_likes'].median()),
                'std': float(predictions_df['num_embedding_likes'].std()),
                'min': int(predictions_df['num_embedding_likes'].min()),
                'max': int(predictions_df['num_embedding_likes'].max()),
            },
        }

        for metric in self.METRICS:
            if metric not in binned_metrics.columns:
                continue

            values = binned_metrics[metric].dropna().to_numpy()
            if len(values) < 2:
                continue

            max_val = values.max()
            threshold_idx = int(np.argmax(values >= 0.9 * max_val))
            summary[f'{metric}_cold_start_threshold_bin'] = str(binned_metrics['bin'].iloc[threshold_idx])
            summary[f'{metric}_max_bin'] = str(binned_metrics['bin'].iloc[int(values.argmax())])
            summary[f'{metric}_improvement_first_to_last'] = float(values[-1] - values[0])

        return summary

    def _compute_summary_for_ranking_rows(
        self,
        ranking_rows_df: pd.DataFrame,
        binned_metrics: pd.DataFrame,
        metric_cols: List[str],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            'total_ranking_rows_analyzed': len(ranking_rows_df),
            'total_users_analyzed': int(ranking_rows_df['did'].nunique()),
            'num_bins': len(binned_metrics),
            'embedding_likes_stats': {
                'mean': float(ranking_rows_df['num_embedding_likes'].mean()),
                'median': float(ranking_rows_df['num_embedding_likes'].median()),
                'std': float(ranking_rows_df['num_embedding_likes'].std()),
                'min': int(ranking_rows_df['num_embedding_likes'].min()),
                'max': int(ranking_rows_df['num_embedding_likes'].max()),
            },
        }
        for metric in metric_cols:
            if metric not in binned_metrics.columns:
                continue
            values = binned_metrics[metric].dropna().to_numpy()
            if len(values) < 2:
                continue
            summary[f'{metric}_max_bin'] = str(binned_metrics['bin'].iloc[int(values.argmax())])
            summary[f'{metric}_improvement_first_to_last'] = float(values[-1] - values[0])
        return summary


# Export the module class
__all__ = ['ColdStartCurvesModule']
