# Manual Data Schema Compatibility Report

**Generated:** 2025-12-19  
**Files analyzed:**
- `bsky_posts_20251215_170001.parquet` (3,941 rows)
- `bsky_likes_20251215_152342.parquet` (14 rows)

---

## Executive Summary

**Status: ⚠️ NOT USABLE AS-IS**

The manual parquet files have significant schema differences from the standard S3 pipeline data and **critical data quality issues** that prevent immediate use:

1. **Posts have NO text content** (all `record_text` values are empty strings)
2. **Posts are missing the unique identifier** (`commit_cid` / `rkey`)  
3. **Embeddings column exists but is 100% null**
4. **Zero overlap** between posts and likes (completely unrelated datasets)
5. Multiple column naming and format differences

---

## Schema Comparison

### POSTS DataFrame

| Column | Standard (S3) | Manual File | Issue |
|--------|--------------|-------------|-------|
| `id` | int32 | ❌ Missing | Not critical |
| `did` | string | ✅ Present | OK |
| `time_us` | int64 | ❌ Missing | Not critical |
| `commit_cid` | string | ❌ **MISSING** | 🔴 **CRITICAL: Cannot join to likes** |
| `rkey` | string | ❌ **MISSING** | 🔴 **CRITICAL: No post identifier** |
| `record_created_at` | string | ✅ Present | OK |
| `record_text` | string (has content) | ⚠️ ALL EMPTY | 🔴 **CRITICAL: No text for embeddings** |
| `record_langs` | string | ❌ Missing | Not critical |
| `embed_quote_uri` | string | ✅ Present | OK |
| `embed_external_uri` | string | ❌ Missing | Minor |
| `embed_image_uris` | string | ❌ Missing | Minor |
| `reply_parent_uri` | string | ✅ Present | OK |
| `reply_root_uri` | string | ✅ Present | OK |
| `inserted_at` | datetime | ✅ Present (string) | Format difference |
| `embeddings` | N/A | ⚠️ Present but NULL | Pre-computed embeddings not available |

### LIKES DataFrame

| Column | Standard (S3) | Manual File | Issue |
|--------|--------------|-------------|-------|
| `did` | string (lowercase) | `DID` (uppercase) | 🟡 Rename needed |
| `subject_cid` | string (CID hash) | ❌ **MISSING** | 🔴 Use `SubjectURI` to extract |
| N/A | N/A | `SubjectURI` | Contains AT URI (extractable) |
| N/A | N/A | `InsertedAt` | Extra timestamp |
| N/A | N/A | `RecordCreatedAt` | Extra timestamp |

---

## Critical Issues Detail

### Issue 1: Missing Post Identifier (commit_cid / rkey)

The posts table has **no way to uniquely identify each post**. In AT Protocol:
- `rkey` is the record key (e.g., `3m7ycm6jvo22n`)
- `commit_cid` is the content hash (e.g., `bafyreif...`)
- Together with `did`, these form the AT URI: `at://{did}/app.bsky.feed.post/{rkey}`

**Without these, we cannot:**
- Join posts to likes
- Deduplicate posts
- Track specific posts through the pipeline

### Issue 2: Empty Text Content

```
Total posts: 3,941
Empty record_text: 3,941 (100%)
Non-empty record_text: 0
```

**Without text, we cannot:**
- Compute text embeddings (the core of our model)
- Train any content-based model

### Issue 3: No Data Overlap

```
Unique authors in liked posts: 7
Unique authors in posts table: 2,646
Overlap: 0
```

The likes reference posts by authors who **do not appear** in the posts table. These are completely unrelated datasets that cannot be joined.

### Issue 4: Likes Use URI Format (not CID)

Standard pipeline expects `subject_cid` like:
```
bafyreih3rupmwjrcch6jd6hsif4br3fqd3rqfps3rxwlsi7oqzz2nleydi
```

Manual data has `SubjectURI` like:
```
at://did:plc:3xwohwqklgfpwt4ffgpjrafx/app.bsky.feed.post/3m7ycm6jvo22n
```

This can be parsed to extract `author_did` and `rkey`, but we'd need the corresponding CID or a way to join on URI.

---

## Pre-computed Embeddings Status

The manual posts file has an `embeddings` column, but:
- **100% of values are NULL**
- No pre-computed embeddings are available
- We would need to compute embeddings from text (which is also empty)

---

## Recommendations

### For Your Colleague (Data Provider)

Please request updated data exports with:

1. **Add `rkey` and/or `commit_cid` to posts** - Essential for joining
2. **Include actual text content** - `record_text` should not be empty
3. **Use consistent naming** - lowercase `did` in likes, or document the mapping
4. **Ensure data overlap** - Posts should include posts that appear in likes
5. **If pre-computing embeddings** - Ensure they're actually populated

### Option A: Fix at Source (Recommended)

Have your colleague re-export with the correct columns:

```sql
-- Example: What the posts export should include
SELECT 
    did,
    commit_cid,      -- ADD THIS
    rkey,            -- ADD THIS  
    record_text,     -- ENSURE NOT EMPTY
    record_created_at,
    ...
FROM posts
WHERE record_text IS NOT NULL AND record_text != ''
```

### Option B: Pipeline Adapter (If data can be fixed)

If the data issues are resolved, we can add a `ManualDataAdapter` to the pipeline:

```python
# Proposed: utils/adapters/manual_data_adapter.py

def adapt_manual_data(posts_path: Path, likes_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform manual parquet files to match pipeline schema.
    
    Transformations:
    - Rename likes 'DID' -> 'did'
    - Extract subject_cid from SubjectURI (if CID available)
    - Or: switch pipeline to join on URI instead of CID
    """
    posts_df = pd.read_parquet(posts_path)
    likes_df = pd.read_parquet(likes_path)
    
    # Rename columns
    likes_df = likes_df.rename(columns={'DID': 'did'})
    
    # Parse SubjectURI to extract rkey (for URI-based joining)
    likes_df['subject_rkey'] = likes_df['SubjectURI'].str.extract(r'/app\.bsky\.feed\.post/([^/]+)$')
    likes_df['subject_author_did'] = likes_df['SubjectURI'].str.extract(r'at://([^/]+)/')
    
    # Would need corresponding rkey in posts to join
    # posts_df['post_uri'] = 'at://' + posts_df['did'] + '/app.bsky.feed.post/' + posts_df['rkey']
    
    return posts_df, likes_df
```

### Option C: URI-Based Join (Requires Pipeline Changes)

Modify `helpers.py:find_join_key()` to support AT URI joining:

```python
def find_join_key(posts_df, likes_df):
    # Standard CID-based join
    if "subject_cid" in likes_df.columns and "commit_cid" in posts_df.columns:
        return "subject_cid", "commit_cid"
    
    # NEW: URI-based join
    if "SubjectURI" in likes_df.columns and "rkey" in posts_df.columns:
        # Construct post URIs and join on that
        return "subject_uri", "post_uri"  # After constructing these
    
    # ... rest of fallback logic
```

---

## Files Generated

- `inspect_schema.py` - Schema inspection script
- `schema_report.json` - Machine-readable schema details
- `SCHEMA_COMPATIBILITY_REPORT.md` - This report

---

## Next Steps

1. **Share this report with your colleague**
2. **Request corrected data** with:
   - Non-empty `record_text`
   - `commit_cid` and/or `rkey` in posts
   - Consistent column naming
   - Related posts and likes (same time window, overlapping data)
3. Once fixed data is available, we can implement the adapter

---

## Appendix: Sample Data

### Likes Sample
```
DID: did:plc:yorwq3vyuizskuktz2na2mfe
SubjectURI: at://did:plc:3xwohwqklgfpwt4ffgpjrafx/app.bsky.feed.post/3m7ycm6jvo22n
```

### Posts Sample  
```
did: did:plc:5ey3apqbyasywfzyynyavtys
record_text: '' (empty)
record_created_at: 2025-12-15T13:03:56.456Z
embeddings: None
```

### Standard Pipeline Posts Sample (for comparison)
```
did: did:plc:gozgxesrqqci3i2uaafgiqp3
commit_cid: bafyreiferjzpe33wigpz2ja25e5n3dslhyazwnakhtoaviunkr6tivp234
rkey: 3m3oy7ovmps2q
record_text: 'substack.com/@rebahenders...' (actual content)
```
