#!/usr/bin/env python3
"""
Inspect the schema of manually-supplied parquet files and compare to pipeline expectations.
"""

import pandas as pd
from pathlib import Path
import json

MANUAL_DATA_DIR = Path(__file__).parent
OUTPUT_FILE = MANUAL_DATA_DIR / "schema_report.json"

def inspect_parquet(path: Path) -> dict:
    """Inspect a parquet file and return schema details."""
    df = pd.read_parquet(path)
    
    info = {
        "file": path.name,
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": {},
        "sample_values": {},
    }
    
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        info["columns"][col] = {
            "dtype": dtype_str,
            "null_count": int(df[col].isnull().sum()),
            "null_pct": round(100 * df[col].isnull().sum() / len(df), 2) if len(df) > 0 else 0,
        }
        
        # Sample non-null values (convert to string for JSON serialization)
        sample = df[col].dropna().head(3).tolist()
        # Handle complex types
        try:
            info["sample_values"][col] = [str(v)[:100] for v in sample]
        except Exception:
            info["sample_values"][col] = ["<complex>"]
        
        # Check if column might contain embeddings (list/array of floats)
        if dtype_str == "object":
            first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(first_val, (list, tuple)):
                info["columns"][col]["is_array"] = True
                info["columns"][col]["array_len"] = len(first_val) if first_val else 0
            
    return info


def main():
    print("=" * 60)
    print("MANUAL DATA SCHEMA INSPECTION")
    print("=" * 60)
    
    # Find all parquet files
    parquet_files = list(MANUAL_DATA_DIR.glob("*.parquet"))
    
    if not parquet_files:
        print("No parquet files found in manual_data/")
        return
    
    report = {
        "files": {},
        "pipeline_expectations": {
            "posts_expected_columns": [
                "did (author DID)",
                "commit_cid (post identifier for join)",
                "text or similar (for embeddings)",
                "optional: image_url",
            ],
            "likes_expected_columns": [
                "did (liker's DID)",
                "subject_cid (liked post's commit_cid)",
            ],
            "join_key": "likes.subject_cid <-> posts.commit_cid"
        },
        "schema_compatibility": {},
    }
    
    for pf in sorted(parquet_files):
        print(f"\n--- {pf.name} ---")
        info = inspect_parquet(pf)
        report["files"][pf.name] = info
        
        print(f"Rows: {info['num_rows']}, Columns: {info['num_columns']}")
        print("Columns:")
        for col, meta in info["columns"].items():
            extra = ""
            if meta.get("is_array"):
                extra = f" [ARRAY len={meta.get('array_len')}]"
            print(f"  - {col}: {meta['dtype']}{extra} (nulls: {meta['null_pct']}%)")
        
        # Sample values
        print("Sample values:")
        for col, samples in info["sample_values"].items():
            print(f"  {col}: {samples}")
    
    # Analyze compatibility
    print("\n" + "=" * 60)
    print("COMPATIBILITY ANALYSIS")
    print("=" * 60)
    
    # Check for posts file
    posts_files = [f for f in parquet_files if "posts" in f.name.lower()]
    likes_files = [f for f in parquet_files if "likes" in f.name.lower()]
    
    compat = {}
    
    for pf in posts_files:
        df = pd.read_parquet(pf)
        cols = set(df.columns)
        
        compat[pf.name] = {
            "has_did": "did" in cols,
            "has_commit_cid": "commit_cid" in cols,
            "text_columns": [c for c in cols if "text" in c.lower()],
            "possible_join_keys": [c for c in cols if "cid" in c.lower() or "id" in c.lower()],
            "possible_embedding_columns": [c for c in cols if any(x in c.lower() for x in ["emb", "embed", "vector", "feature"])],
        }
        
        print(f"\nPOSTS FILE: {pf.name}")
        print(f"  has 'did' (author): {compat[pf.name]['has_did']}")
        print(f"  has 'commit_cid': {compat[pf.name]['has_commit_cid']}")
        print(f"  text columns: {compat[pf.name]['text_columns']}")
        print(f"  possible join keys: {compat[pf.name]['possible_join_keys']}")
        print(f"  possible embedding columns: {compat[pf.name]['possible_embedding_columns']}")
        
        # Check for pre-computed embeddings
        for col in df.columns:
            first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(first_val, (list, tuple)) and len(first_val) > 100:
                print(f"  *** DETECTED EMBEDDING COLUMN: {col} (dim={len(first_val)})")
                compat[pf.name]["detected_embedding"] = {"column": col, "dim": len(first_val)}
    
    for lf in likes_files:
        df = pd.read_parquet(lf)
        cols = set(df.columns)
        
        compat[lf.name] = {
            "has_did": "did" in cols,
            "has_subject_cid": "subject_cid" in cols,
            "possible_join_keys": [c for c in cols if "cid" in c.lower() or "id" in c.lower()],
        }
        
        print(f"\nLIKES FILE: {lf.name}")
        print(f"  has 'did' (liker): {compat[lf.name]['has_did']}")
        print(f"  has 'subject_cid': {compat[lf.name]['has_subject_cid']}")
        print(f"  possible join keys: {compat[lf.name]['possible_join_keys']}")
    
    # Check join compatibility
    print("\n" + "-" * 40)
    print("JOIN COMPATIBILITY CHECK")
    print("-" * 40)
    
    if posts_files and likes_files:
        posts_df = pd.read_parquet(posts_files[0])
        likes_df = pd.read_parquet(likes_files[0])
        
        posts_cols = set(posts_df.columns)
        likes_cols = set(likes_df.columns)
        
        # Standard join check
        if "commit_cid" in posts_cols and "subject_cid" in likes_cols:
            posts_cids = set(posts_df["commit_cid"].dropna().astype(str))
            likes_cids = set(likes_df["subject_cid"].dropna().astype(str))
            overlap = posts_cids & likes_cids
            print(f"Standard join (commit_cid <-> subject_cid):")
            print(f"  Posts commit_cid count: {len(posts_cids)}")
            print(f"  Likes subject_cid count: {len(likes_cids)}")
            print(f"  Overlap: {len(overlap)}")
            compat["join_analysis"] = {
                "join_type": "standard",
                "posts_key": "commit_cid",
                "likes_key": "subject_cid",
                "overlap_count": len(overlap),
            }
        else:
            # Try to find alternative join
            common_cols = posts_cols & likes_cols
            print(f"Standard join NOT available (missing commit_cid or subject_cid)")
            print(f"Common columns: {common_cols}")
            
            for col in common_cols:
                if col != "did":  # did is user, not post
                    posts_vals = set(posts_df[col].dropna().astype(str))
                    likes_vals = set(likes_df[col].dropna().astype(str))
                    overlap = posts_vals & likes_vals
                    if overlap:
                        print(f"  Potential join on '{col}': overlap={len(overlap)}")
                        
            compat["join_analysis"] = {
                "join_type": "non-standard",
                "common_columns": list(common_cols),
                "notes": "Manual mapping required",
            }
    
    report["schema_compatibility"] = compat
    
    # Save report
    with open(OUTPUT_FILE, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n\nFull report saved to: {OUTPUT_FILE}")
    
    # Summary of issues
    print("\n" + "=" * 60)
    print("SUMMARY: REQUIRED CHANGES FOR PIPELINE COMPATIBILITY")
    print("=" * 60)
    
    issues = []
    recommendations = []
    
    for pf in posts_files:
        info = report["files"].get(pf.name, {})
        c = compat.get(pf.name, {})
        
        if not c.get("has_did"):
            issues.append(f"Posts file missing 'did' column (author identifier)")
        if not c.get("has_commit_cid"):
            issues.append(f"Posts file missing 'commit_cid' column (post identifier)")
            if c.get("possible_join_keys"):
                recommendations.append(f"Consider mapping one of these to 'commit_cid': {c['possible_join_keys']}")
        if not c.get("text_columns"):
            issues.append(f"Posts file has no obvious text column")
        if c.get("detected_embedding"):
            emb = c["detected_embedding"]
            recommendations.append(f"Pre-computed embeddings detected in '{emb['column']}' (dim={emb['dim']}). "
                                   f"Consider skipping compute_post_embeddings() and using these directly.")
    
    for lf in likes_files:
        c = compat.get(lf.name, {})
        
        if not c.get("has_did"):
            issues.append(f"Likes file missing 'did' column (liker identifier)")
        if not c.get("has_subject_cid"):
            issues.append(f"Likes file missing 'subject_cid' column (liked post identifier)")
            if c.get("possible_join_keys"):
                recommendations.append(f"Consider mapping one of these to 'subject_cid': {c['possible_join_keys']}")
    
    if issues:
        print("\nISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\nNo critical issues found!")
    
    if recommendations:
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")


if __name__ == "__main__":
    main()

