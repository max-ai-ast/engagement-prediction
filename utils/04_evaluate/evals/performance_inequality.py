#!/usr/bin/env python3

"""
Performance Inequality Evaluation Module

This module characterizes between-user inequalities in model performance metrics
by computing Gini coefficients and generating Lorenz curves.

Metrics analyzed:
- Per-user precision (at threshold 0.5)
- Per-user recall (at threshold 0.5)
- Per-user AUC-ROC (where computable)
- Per-user accuracy

Outputs:
- gini_summary.json: Gini coefficients for all metrics
- lorenz_*.png: Lorenz curve plots for each metric
- per_user_metrics.csv: Full per-user metrics table
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
    compute_gini_coefficient,
    compute_lorenz_curve,
    compute_per_user_metrics,
)


class PerformanceInequalityModule(EvalModule):
    """
    Evaluation module for measuring performance inequality across users.
    
    Computes Gini coefficients and Lorenz curves for user-level metrics
    to understand how model benefits are distributed across the user population.
    """
    
    name = "performance_inequality"
    description = "Measures between-user inequalities in precision, recall, and AUC via Gini coefficients and Lorenz curves"
    
    # Metrics to analyze for inequality
    METRICS_TO_ANALYZE = ['precision', 'recall', 'auc_roc', 'accuracy', 'f1']
    
    # Plot styling
    FIGURE_SIZE = (8, 6)
    DPI = 150
    
    def run(self, ctx: EvalContext) -> Dict[str, Any]:
        """
        Run performance inequality analysis.
        
        Args:
            ctx: EvalContext with predictions and output directory.
        
        Returns:
            Dict with Gini coefficients and artifact paths.
        """
        out_dir = self.get_output_dir(ctx)

        if ctx.has_ranking_rows:
            return self._run_ranking_rows(ctx, out_dir)
        
        # Step 1: Compute per-user metrics
        print(f"    Computing per-user metrics for {ctx.num_holdout_users} users...")
        per_user_df = compute_per_user_metrics(ctx.predictions_df)
        
        # Save per-user metrics
        per_user_path = out_dir / "per_user_metrics.csv"
        per_user_df.to_csv(per_user_path, index=False)
        
        # Step 2: Compute Gini coefficients for each metric
        print("    Computing Gini coefficients...")
        gini_results = {}
        for metric in self.METRICS_TO_ANALYZE:
            if metric in per_user_df.columns:
                values = per_user_df[metric].dropna().to_numpy()
                if len(values) > 0:
                    gini = compute_gini_coefficient(values)
                    gini_results[f"gini_{metric}"] = gini
                    gini_results[f"n_users_{metric}"] = len(values)
                    gini_results[f"mean_{metric}"] = float(np.mean(values))
                    gini_results[f"std_{metric}"] = float(np.std(values))
                    gini_results[f"median_{metric}"] = float(np.median(values))
                    gini_results[f"min_{metric}"] = float(np.min(values))
                    gini_results[f"max_{metric}"] = float(np.max(values))
        
        # Add overall stats
        gini_results['total_users'] = len(per_user_df)
        gini_results['total_predictions'] = ctx.num_predictions
        
        # Step 3: Generate Lorenz curve plots
        print("    Generating Lorenz curve plots...")
        plot_paths = {}
        for metric in self.METRICS_TO_ANALYZE:
            if metric in per_user_df.columns:
                values = per_user_df[metric].dropna().to_numpy()
                if len(values) > 1:
                    plot_path = out_dir / f"lorenz_{metric}.png"
                    self._plot_lorenz_curve(
                        values=values,
                        metric_name=metric,
                        gini=gini_results.get(f"gini_{metric}", float('nan')),
                        save_path=plot_path,
                    )
                    plot_paths[f"lorenz_{metric}_path"] = str(plot_path)
        
        # Step 4: Generate combined Lorenz curves plot
        combined_path = out_dir / "lorenz_combined.png"
        self._plot_combined_lorenz(per_user_df, gini_results, combined_path)
        plot_paths['lorenz_combined_path'] = str(combined_path)
        
        # Step 5: Generate distribution plots
        dist_path = out_dir / "metric_distributions.png"
        self._plot_metric_distributions(per_user_df, dist_path)
        plot_paths['distributions_path'] = str(dist_path)
        
        # Save Gini summary
        summary = {
            **gini_results,
            **plot_paths,
            'per_user_metrics_path': str(per_user_path),
        }
        summary_path = out_dir / "gini_summary.json"
        self.save_json(summary, summary_path)
        
        return summary

    def _run_ranking_rows(self, ctx: EvalContext, out_dir: Path) -> Dict[str, Any]:
        """Run inequality analysis on matrix-native ranking metrics."""
        ranking_rows_df = ctx.ranking_rows_df.copy()
        metric_cols = self._ranking_metric_columns(ranking_rows_df)
        if not metric_cols:
            return {
                'status': 'skipped',
                'reason': 'no ranking metric columns available',
            }

        print(f"    Computing per-user ranking metrics for {ctx.num_holdout_users} users...")
        agg_spec: Dict[str, Any] = {
            'num_ranking_rows': ('did', 'size'),
            'num_positive': ('positive_count', 'sum'),
            'mean_candidate_count': ('candidate_count', 'mean'),
        }
        if 'num_embedding_likes' in ranking_rows_df.columns:
            agg_spec['num_embedding_likes'] = ('num_embedding_likes', 'max')
        if 'num_total_likes' in ranking_rows_df.columns:
            agg_spec['num_total_likes'] = ('num_total_likes', 'max')
        for metric in metric_cols:
            agg_spec[metric] = (metric, 'mean')
        per_user_df = ranking_rows_df.groupby('did', as_index=False).agg(**agg_spec)

        per_user_path = out_dir / "per_user_metrics.csv"
        per_user_df.to_csv(per_user_path, index=False)

        print("    Computing Gini coefficients...")
        gini_results = self._compute_gini_summary(per_user_df, metric_cols)
        gini_results['total_users'] = len(per_user_df)
        gini_results['total_predictions'] = ctx.num_predictions
        gini_results['total_ranking_rows'] = ctx.num_ranking_rows

        print("    Generating Lorenz curve plots...")
        plot_paths = {}
        for metric in metric_cols:
            values = per_user_df[metric].dropna().to_numpy()
            if len(values) > 1:
                plot_path = out_dir / f"lorenz_{self._metric_filename(metric)}.png"
                self._plot_lorenz_curve(
                    values=values,
                    metric_name=metric,
                    gini=gini_results.get(f"gini_{metric}", float('nan')),
                    save_path=plot_path,
                )
                plot_paths[f"lorenz_{metric}_path"] = str(plot_path)

        combined_path = out_dir / "lorenz_combined.png"
        self._plot_combined_lorenz_for_metrics(per_user_df, metric_cols, gini_results, combined_path)
        plot_paths['lorenz_combined_path'] = str(combined_path)

        dist_path = out_dir / "metric_distributions.png"
        self._plot_metric_distributions_for_metrics(per_user_df, metric_cols, dist_path)
        plot_paths['distributions_path'] = str(dist_path)

        summary = {
            **gini_results,
            **plot_paths,
            'per_user_metrics_path': str(per_user_path),
        }
        summary_path = out_dir / "gini_summary.json"
        self.save_json(summary, summary_path)

        return summary

    def _ranking_metric_columns(self, ranking_rows_df: pd.DataFrame) -> List[str]:
        return [
            col for col in ranking_rows_df.columns
            if col.startswith('ndcg@') or col.startswith('recall@') or col in ('average_precision', 'auc_roc')
        ]

    def _metric_filename(self, metric: str) -> str:
        return metric.replace('@', '_at_').replace('/', '_')

    def _compute_gini_summary(self, per_user_df: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
        gini_results = {}
        for metric in metrics:
            if metric in per_user_df.columns:
                values = per_user_df[metric].dropna().to_numpy()
                if len(values) > 0:
                    gini = compute_gini_coefficient(values)
                    gini_results[f"gini_{metric}"] = gini
                    gini_results[f"n_users_{metric}"] = len(values)
                    gini_results[f"mean_{metric}"] = float(np.mean(values))
                    gini_results[f"std_{metric}"] = float(np.std(values))
                    gini_results[f"median_{metric}"] = float(np.median(values))
                    gini_results[f"min_{metric}"] = float(np.min(values))
                    gini_results[f"max_{metric}"] = float(np.max(values))
        return gini_results
    
    def _plot_lorenz_curve(
        self,
        values: np.ndarray,
        metric_name: str,
        gini: float,
        save_path: Path,
    ) -> None:
        """Generate a single Lorenz curve plot."""
        fig, ax = plt.subplots(figsize=self.FIGURE_SIZE)
        
        # Compute Lorenz curve
        cum_pop, cum_val = compute_lorenz_curve(values)
        
        # Plot equality line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Equality', linewidth=1.5)
        
        # Plot Lorenz curve
        ax.plot(cum_pop, cum_val, 'b-', linewidth=2, 
                label=f'{metric_name.replace("_", " ").title()} (Gini={gini:.3f})')
        
        # Fill area between curves
        ax.fill_between(cum_pop, cum_val, cum_pop, alpha=0.2, color='blue')
        
        # Styling
        ax.set_xlabel('Cumulative Share of Users (sorted by metric)', fontsize=11)
        ax.set_ylabel(f'Cumulative Share of {metric_name.replace("_", " ").title()}', fontsize=11)
        ax.set_title(f'Lorenz Curve: {metric_name.replace("_", " ").title()}\n'
                     f'Gini Coefficient = {gini:.4f}', fontsize=12)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_combined_lorenz(
        self,
        per_user_df: pd.DataFrame,
        gini_results: Dict[str, Any],
        save_path: Path,
    ) -> None:
        """Generate a combined plot with all Lorenz curves."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot equality line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Equality', linewidth=2)
        
        # Colors for different metrics
        colors = {
            'precision': '#1f77b4',  # blue
            'recall': '#ff7f0e',     # orange
            'auc_roc': '#2ca02c',    # green
            'accuracy': '#d62728',   # red
            'f1': '#9467bd',         # purple
        }
        
        for metric in self.METRICS_TO_ANALYZE:
            if metric not in per_user_df.columns:
                continue
            values = per_user_df[metric].dropna().to_numpy()
            if len(values) < 2:
                continue
            
            cum_pop, cum_val = compute_lorenz_curve(values)
            gini = gini_results.get(f"gini_{metric}", float('nan'))
            
            color = colors.get(metric, '#333333')
            label = f'{metric.replace("_", " ").title()} (Gini={gini:.3f})'
            ax.plot(cum_pop, cum_val, color=color, linewidth=2, label=label)
        
        ax.set_xlabel('Cumulative Share of Users (sorted by metric)', fontsize=12)
        ax.set_ylabel('Cumulative Share of Metric Value', fontsize=12)
        ax.set_title('Performance Inequality: Lorenz Curves for All Metrics', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)

    def _plot_combined_lorenz_for_metrics(
        self,
        per_user_df: pd.DataFrame,
        metrics: List[str],
        gini_results: Dict[str, Any],
        save_path: Path,
    ) -> None:
        """Generate a combined plot for a caller-provided metric list."""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, label='Perfect Equality', linewidth=2)

        cmap = plt.get_cmap('tab10')
        for idx, metric in enumerate(metrics):
            if metric not in per_user_df.columns:
                continue
            values = per_user_df[metric].dropna().to_numpy()
            if len(values) < 2:
                continue

            cum_pop, cum_val = compute_lorenz_curve(values)
            gini = gini_results.get(f"gini_{metric}", float('nan'))
            label = f'{metric.replace("_", " ").title()} (Gini={gini:.3f})'
            ax.plot(cum_pop, cum_val, color=cmap(idx % cmap.N), linewidth=2, label=label)

        ax.set_xlabel('Cumulative Share of Users (sorted by metric)', fontsize=12)
        ax.set_ylabel('Cumulative Share of Metric Value', fontsize=12)
        ax.set_title('Performance Inequality: Lorenz Curves for All Metrics', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(save_path, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)
    
    def _plot_metric_distributions(
        self,
        per_user_df: pd.DataFrame,
        save_path: Path,
    ) -> None:
        """Generate distribution histograms for each metric."""
        metrics = [m for m in self.METRICS_TO_ANALYZE if m in per_user_df.columns]
        self._plot_metric_distributions_for_metrics(per_user_df, metrics, save_path)

    def _plot_metric_distributions_for_metrics(
        self,
        per_user_df: pd.DataFrame,
        metrics: List[str],
        save_path: Path,
    ) -> None:
        """Generate distribution histograms for the provided metrics."""
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            return
        
        # Create subplot grid
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            values = per_user_df[metric].dropna().to_numpy()
            
            if len(values) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=12)
                ax.set_title(metric.replace("_", " ").title())
                continue
            
            # Histogram
            ax.hist(values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
            
            # Add vertical lines for mean and median
            mean_val = np.mean(values)
            median_val = np.median(values)
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle=':', linewidth=2, label=f'Median: {median_val:.3f}')
            
            ax.set_xlabel(metric.replace("_", " ").title(), fontsize=10)
            ax.set_ylabel('Number of Users', fontsize=10)
            ax.set_title(f'Distribution of {metric.replace("_", " ").title()}\n(n={len(values)} users)', fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)


# Export the module class
__all__ = ['PerformanceInequalityModule']
