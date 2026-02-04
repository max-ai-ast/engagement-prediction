#!/usr/bin/env python3
"""
Compare GCS Bucket Data Coverage: Stage vs Prod
================================================

This script provides a systematic comparison of data availability across
the stage and prod GCS buckets, including:

1. Hourly record counts for bsky_posts and bsky_likes
2. Side-by-side coverage visualization
3. Schema comparison to identify differences (like the embeddings column)

Usage:
    python ops/compare_bucket_coverage.py
    python ops/compare_bucket_coverage.py --start 2025-12-01 --end 2026-02-01
    python ops/compare_bucket_coverage.py --output-dir ./reports
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json

import polars as pl
from google.cloud import storage
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


# Bucket configuration
BUCKETS = {
    'stage': 'greenearth-471522-ingex-extract-stage',
    'prod': 'greenearth-471522-ingex-extract-prod',
}

DATA_TYPES = ['bsky_posts', 'bsky_likes']


def list_parquet_files(bucket_name: str, prefix: str) -> List[str]:
    """List all parquet files in a GCS bucket matching a prefix."""
    client = storage.Client()
    blobs = client.list_blobs(bucket_name, prefix=prefix)
    paths = []
    for blob in blobs:
        if blob.name.endswith('.parquet'):
            paths.append(f"gs://{bucket_name}/{blob.name}")
    return sorted(paths)


def get_hourly_counts(paths: List[str], timestamp_col: str = 'record_created_at') -> pl.DataFrame:
    """
    Efficiently scan parquet files and aggregate record counts by hour.
    
    Uses Polars streaming to handle large datasets without loading all into memory.
    """
    if not paths:
        return pl.DataFrame({
            'hour': pl.Series([], dtype=pl.Datetime('us', 'UTC')),
            'count': pl.Series([], dtype=pl.UInt32),
        })
    
    try:
        # Use lazy scanning for memory efficiency
        lf = pl.scan_parquet(paths)
        
        # Check if timestamp column exists
        schema = lf.collect_schema()
        if timestamp_col not in schema:
            print(f"  Warning: Column '{timestamp_col}' not found. Available: {list(schema.keys())[:10]}...")
            return pl.DataFrame({
                'hour': pl.Series([], dtype=pl.Datetime('us', 'UTC')),
                'count': pl.Series([], dtype=pl.UInt32),
            })
        
        # Aggregate to hourly counts
        result = (
            lf
            .with_columns(
                pl.col(timestamp_col)
                .str.to_datetime(time_zone="UTC")
                .dt.truncate("1h")
                .alias("hour")
            )
            .group_by("hour")
            .agg(pl.len().alias("count"))
            .sort("hour")
            .collect(engine="streaming")
        )
        
        return result
    
    except Exception as e:
        print(f"  Error scanning parquet files: {e}")
        return pl.DataFrame({
            'hour': pl.Series([], dtype=pl.Datetime('us', 'UTC')),
            'count': pl.Series([], dtype=pl.UInt32),
        })


def get_schema_info(paths: List[str], sample_size: int = 1) -> Dict[str, Any]:
    """Get schema information and sample data from parquet files."""
    if not paths:
        return {'columns': [], 'dtypes': {}, 'sample': None, 'row_count': 0}
    
    try:
        lf = pl.scan_parquet(paths[:sample_size])  # Only scan first few files for schema
        schema = lf.collect_schema()
        
        # Get a small sample
        sample_df = lf.head(5).collect()
        
        # Get total row count (from first file for speed)
        row_count = pl.scan_parquet(paths[0]).select(pl.len()).collect().item()
        
        return {
            'columns': list(schema.keys()),
            'dtypes': {k: str(v) for k, v in schema.items()},
            'sample': sample_df.to_pandas().to_dict('records') if len(sample_df) > 0 else None,
            'row_count': row_count,
            'n_files': len(paths),
        }
    except Exception as e:
        return {'error': str(e), 'columns': [], 'dtypes': {}}


def check_embeddings_column(paths: List[str]) -> Dict[str, Any]:
    """Specifically investigate the embeddings column."""
    if not paths:
        return {'has_embeddings': False}
    
    try:
        lf = pl.scan_parquet(paths[:3])  # Check first 3 files
        schema = lf.collect_schema()
        
        if 'embeddings' not in schema:
            return {'has_embeddings': False}
        
        # Get sample of embeddings column
        sample = (
            lf
            .select('embeddings')
            .filter(pl.col('embeddings').is_not_null())
            .head(3)
            .collect()
        )
        
        # Analyze embeddings structure
        embeddings_info = {
            'has_embeddings': True,
            'dtype': str(schema['embeddings']),
            'null_rate': None,
            'sample_structure': None,
        }
        
        if len(sample) > 0:
            # Check structure of first non-null embedding
            first_emb = sample['embeddings'][0]
            if first_emb is not None:
                embeddings_info['sample_structure'] = type(first_emb).__name__
                if isinstance(first_emb, list) and len(first_emb) > 0:
                    embeddings_info['list_length'] = len(first_emb)
                    embeddings_info['element_type'] = type(first_emb[0]).__name__
                    if isinstance(first_emb[0], dict):
                        embeddings_info['dict_keys'] = list(first_emb[0].keys())
        
        # Calculate null rate from a sample
        null_check = (
            lf
            .select([
                pl.count().alias('total'),
                pl.col('embeddings').is_null().sum().alias('nulls'),
            ])
            .head(10000)  # Sample first 10k rows
            .collect()
        )
        if null_check['total'][0] > 0:
            embeddings_info['null_rate'] = float(null_check['nulls'][0]) / float(null_check['total'][0])
        
        return embeddings_info
    
    except Exception as e:
        return {'has_embeddings': 'error', 'error': str(e)}


def plot_coverage_comparison(
    coverage_data: Dict[str, Dict[str, pl.DataFrame]],
    output_path: Optional[Path] = None,
    title: str = "GCS Bucket Data Coverage Comparison"
):
    """Create side-by-side coverage plots for stage vs prod."""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    colors = {'stage': '#2ecc71', 'prod': '#3498db'}
    
    for idx, data_type in enumerate(DATA_TYPES):
        ax = axes[idx]
        ax.set_title(f'{data_type}', fontsize=12)
        
        for bucket_name, bucket_id in [('Stage', 'stage'), ('Prod', 'prod')]:
            df = coverage_data[bucket_id].get(data_type)
            if df is None or len(df) == 0:
                continue
            
            # Convert to pandas for matplotlib
            pdf = df.to_pandas()
            pdf['hour'] = pd.to_datetime(pdf['hour'])
            
            ax.bar(
                pdf['hour'],
                pdf['count'],
                width=1/24,  # 1 hour width
                alpha=0.6,
                label=f'{bucket_name}',
                color=colors[bucket_id],
            )
        
        ax.set_ylabel('Records per Hour')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    
    axes[-1].set_xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    
    plt.show()


def plot_daily_comparison(
    coverage_data: Dict[str, Dict[str, pl.DataFrame]],
    output_path: Optional[Path] = None,
):
    """Create a cleaner daily comparison plot."""
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Daily Record Counts: Stage vs Prod", fontsize=14, fontweight='bold')
    
    colors = {'stage': '#2ecc71', 'prod': '#3498db'}
    bar_width = 0.35
    
    for idx, data_type in enumerate(DATA_TYPES):
        ax = axes[idx]
        ax.set_title(f'{data_type}', fontsize=12)
        
        daily_data = {}
        
        for bucket_id in ['stage', 'prod']:
            df = coverage_data[bucket_id].get(data_type)
            if df is None or len(df) == 0:
                daily_data[bucket_id] = pl.DataFrame({
                    'date': pl.Series([], dtype=pl.Date),
                    'daily_count': pl.Series([], dtype=pl.UInt64),
                })
                continue
            
            # Aggregate to daily
            daily = (
                df
                .with_columns(pl.col('hour').dt.date().alias('date'))
                .group_by('date')
                .agg(pl.col('count').sum().alias('daily_count'))
                .sort('date')
            )
            daily_data[bucket_id] = daily
        
        # Get union of all dates
        all_dates = set()
        for bucket_id in ['stage', 'prod']:
            if len(daily_data[bucket_id]) > 0:
                all_dates.update(daily_data[bucket_id]['date'].to_list())
        
        if not all_dates:
            continue
        
        all_dates = sorted(all_dates)
        x = np.arange(len(all_dates))
        
        for i, (bucket_name, bucket_id) in enumerate([('Stage', 'stage'), ('Prod', 'prod')]):
            df = daily_data[bucket_id]
            if len(df) == 0:
                continue
            
            # Create lookup dict
            date_counts = dict(zip(df['date'].to_list(), df['daily_count'].to_list()))
            counts = [date_counts.get(d, 0) for d in all_dates]
            
            offset = -bar_width/2 if bucket_id == 'stage' else bar_width/2
            ax.bar(
                x + offset,
                counts,
                bar_width,
                label=bucket_name,
                color=colors[bucket_id],
                alpha=0.8,
            )
        
        ax.set_ylabel('Records per Day')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Set x-ticks
        tick_step = max(1, len(all_dates) // 20)  # Show ~20 ticks max
        ax.set_xticks(x[::tick_step])
        ax.set_xticklabels([str(d) for d in all_dates[::tick_step]], rotation=45, ha='right')
    
    axes[-1].set_xlabel('Date')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved daily comparison plot to {output_path}")
    
    plt.show()


def generate_summary_report(
    coverage_data: Dict[str, Dict[str, pl.DataFrame]],
    schema_info: Dict[str, Dict[str, Dict[str, Any]]],
    embeddings_info: Dict[str, Dict[str, Any]],
) -> str:
    """Generate a text summary of the comparison."""
    
    lines = [
        "=" * 70,
        "GCS BUCKET DATA COVERAGE REPORT",
        "=" * 70,
        f"Generated: {datetime.now().isoformat()}",
        "",
    ]
    
    # Coverage summary
    lines.extend([
        "-" * 70,
        "DATA COVERAGE SUMMARY",
        "-" * 70,
    ])
    
    for data_type in DATA_TYPES:
        lines.append(f"\n{data_type}:")
        for bucket_id, bucket_name in BUCKETS.items():
            df = coverage_data[bucket_id].get(data_type)
            if df is None or len(df) == 0:
                lines.append(f"  {bucket_id}: No data")
                continue
            
            total_records = df['count'].sum()
            min_hour = df['hour'].min()
            max_hour = df['hour'].max()
            n_hours = len(df)
            avg_per_hour = total_records / n_hours if n_hours > 0 else 0
            
            lines.extend([
                f"  {bucket_id}:",
                f"    Total records: {total_records:,}",
                f"    Date range: {min_hour} to {max_hour}",
                f"    Hours with data: {n_hours:,}",
                f"    Avg records/hour: {avg_per_hour:,.0f}",
            ])
    
    # Schema comparison
    lines.extend([
        "",
        "-" * 70,
        "SCHEMA COMPARISON (bsky_posts)",
        "-" * 70,
    ])
    
    stage_cols = set(schema_info['stage'].get('bsky_posts', {}).get('columns', []))
    prod_cols = set(schema_info['prod'].get('bsky_posts', {}).get('columns', []))
    
    only_in_stage = stage_cols - prod_cols
    only_in_prod = prod_cols - stage_cols
    in_both = stage_cols & prod_cols
    
    lines.extend([
        f"Columns in both: {len(in_both)}",
        f"Only in stage: {sorted(only_in_stage) if only_in_stage else 'None'}",
        f"Only in prod: {sorted(only_in_prod) if only_in_prod else 'None'}",
    ])
    
    # Embeddings investigation
    lines.extend([
        "",
        "-" * 70,
        "EMBEDDINGS COLUMN INVESTIGATION",
        "-" * 70,
    ])
    
    for bucket_id in ['stage', 'prod']:
        emb_info = embeddings_info.get(bucket_id, {})
        lines.append(f"\n{bucket_id}:")
        
        if emb_info.get('has_embeddings'):
            lines.extend([
                f"  Has embeddings: Yes",
                f"  Dtype: {emb_info.get('dtype', 'unknown')}",
                f"  Null rate: {emb_info.get('null_rate', 'unknown'):.2%}" if emb_info.get('null_rate') is not None else "  Null rate: unknown",
                f"  Structure: {emb_info.get('sample_structure', 'unknown')}",
            ])
            if 'list_length' in emb_info:
                lines.append(f"  List length: {emb_info['list_length']}")
            if 'dict_keys' in emb_info:
                lines.append(f"  Dict keys: {emb_info['dict_keys']}")
        else:
            lines.append(f"  Has embeddings: No")
    
    lines.extend([
        "",
        "=" * 70,
    ])
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare GCS bucket data coverage")
    parser.add_argument("--output-dir", type=Path, default=Path("./ops/reports"),
                        help="Directory to save reports and plots")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip generating plots (useful for headless environments)")
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GCS Bucket Data Coverage Comparison")
    print("=" * 70)
    
    coverage_data: Dict[str, Dict[str, pl.DataFrame]] = {}
    schema_info: Dict[str, Dict[str, Dict[str, Any]]] = {}
    embeddings_info: Dict[str, Dict[str, Any]] = {}
    
    for bucket_id, bucket_name in BUCKETS.items():
        print(f"\n[{bucket_id.upper()}] Analyzing {bucket_name}...")
        coverage_data[bucket_id] = {}
        schema_info[bucket_id] = {}
        
        for data_type in DATA_TYPES:
            print(f"  Listing {data_type} files...")
            paths = list_parquet_files(bucket_name, data_type)
            print(f"    Found {len(paths)} parquet files")
            
            if paths:
                print(f"  Getting schema info...")
                schema_info[bucket_id][data_type] = get_schema_info(paths)
                
                print(f"  Aggregating hourly counts (this may take a moment)...")
                coverage_data[bucket_id][data_type] = get_hourly_counts(paths)
                n_hours = len(coverage_data[bucket_id][data_type])
                total = coverage_data[bucket_id][data_type]['count'].sum() if n_hours > 0 else 0
                print(f"    Got {n_hours} hours of data, {total:,} total records")
            else:
                coverage_data[bucket_id][data_type] = pl.DataFrame({
                    'hour': pl.Series([], dtype=pl.Datetime('us', 'UTC')),
                    'count': pl.Series([], dtype=pl.UInt32),
                })
                schema_info[bucket_id][data_type] = {'columns': [], 'dtypes': {}}
        
        # Check embeddings specifically for posts
        print(f"  Investigating embeddings column...")
        posts_paths = list_parquet_files(bucket_name, 'bsky_posts')
        embeddings_info[bucket_id] = check_embeddings_column(posts_paths)
        print(f"    Has embeddings: {embeddings_info[bucket_id].get('has_embeddings', False)}")
    
    # Generate report
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)
    
    report = generate_summary_report(coverage_data, schema_info, embeddings_info)
    print(report)
    
    # Save report
    report_path = args.output_dir / "bucket_coverage_report.txt"
    report_path.write_text(report)
    print(f"\nSaved report to {report_path}")
    
    # Save raw data as JSON
    json_data = {
        'generated': datetime.now().isoformat(),
        'buckets': BUCKETS,
        'schema_info': schema_info,
        'embeddings_info': embeddings_info,
        'coverage_summary': {},
    }
    
    for bucket_id in ['stage', 'prod']:
        json_data['coverage_summary'][bucket_id] = {}
        for data_type in DATA_TYPES:
            df = coverage_data[bucket_id].get(data_type)
            if df is not None and len(df) > 0:
                json_data['coverage_summary'][bucket_id][data_type] = {
                    'total_records': int(df['count'].sum()),
                    'n_hours': len(df),
                    'min_hour': str(df['hour'].min()),
                    'max_hour': str(df['hour'].max()),
                }
    
    json_path = args.output_dir / "bucket_coverage_data.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"Saved data to {json_path}")
    
    # Generate plots
    if not args.no_plot:
        print("\nGenerating plots...")
        try:
            plot_daily_comparison(
                coverage_data,
                output_path=args.output_dir / "daily_coverage_comparison.png"
            )
        except Exception as e:
            print(f"Warning: Could not generate plot: {e}")
            print("Try running with --no-plot in a headless environment")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
