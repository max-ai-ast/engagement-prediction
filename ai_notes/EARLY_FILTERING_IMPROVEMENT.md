# Early User Filtering Improvement

## Problem

The data pipeline was sampling 100,000 liking users but only ending up with ~67,000 users after filtering (33% drop). This happened because:

1. User sampling occurred in Pass 1 based on all users
2. The `min_likes_per_user` filter was applied much later (after per-user caps)
3. Many sampled users had only 1 like and were filtered out at the end

Example from logs:
```
[2026-01-23 07:24:38.571] Sampled 100,000 liking users (4.0% of total)
...
[2026-01-23 07:26:11.550] Final: 67,431 users with 2,804,977 likes
```

This meant we wasted sampling budget on users who would be excluded anyway.

## Solution

Modified `load_likes_core_polars()` in `utils/helpers.py` to pre-filter users before sampling:

### New Pipeline Flow

1. **Pass 1: Count likes per user** (instead of just collecting unique users)
   - Build `user_counts: Dict[str, int]` during the scan
   - Track how many likes each user has

2. **Pre-filter by min_likes_per_user**
   - Before sampling, filter to `eligible_users` who meet the minimum threshold
   - Log how many users are excluded at this stage

3. **Sample from eligible users only**
   - If `max_liking_users` is set, sample from the eligible pool
   - Ensures sampled users will pass final filters

4. **Pass 2, per-user caps, and verification** proceed as before
   - Final min-likes check now only catches edge cases (e.g., per-user cap reducing someone below threshold)

### Code Changes

**Key modifications to `utils/helpers.py`:**

1. Changed Pass 1 from collecting a set to building a count dictionary:
```python
# Before: all_users: Set[str] = set()
# After:  user_counts: Dict[str, int] = {}

for user in batch_users['did'].to_list():
    user_counts[user] = user_counts.get(user, 0) + 1
```

2. Added pre-filtering step before sampling:
```python
if min_likes_per_user > 0:
    eligible_users = {user for user, count in user_counts.items() 
                      if count >= min_likes_per_user}
    _log(f"Pre-filtering: {n_users_eligible:,} users meet min-likes threshold...")
```

3. Sample from eligible users instead of all users:
```python
if max_liking_users > 0 and len(eligible_users) > max_liking_users:
    # Sample from eligible_users pool
    sampled_user_set = {user_list[i] for i in sampled_indices}
```

4. Updated final min-likes filter to be a verification step:
```python
# Now mainly catches edge cases from per-user caps
if n_removed > 0:
    _log(f"Min-likes verification removed {n_removed:,} likes...")
```

### Benefits

1. **Efficient sampling**: All sampled users will meet minimum requirements
2. **Better resource utilization**: Don't waste sampling budget on users who will be filtered out
3. **More predictable results**: When you request 100k users, you'll actually get close to 100k users in the output
4. **Minimal overhead**: Only requires counting during Pass 1 (already scanning the data)
5. **Transparent logging**: New log messages show how many users are excluded in pre-filtering

### Memory Impact

Negligible increase in Pass 1:
- Before: storing `Set[str]` of all unique users
- After: storing `Dict[str, int]` with counts per user
- Both scale linearly with number of unique users (~2-3 million)
- Dictionary overhead is minimal (~10 bytes per entry vs ~4 bytes for set)

### Memory Estimation Fix

The memory estimation in `estimate_filtered_data_memory()` was also updated to account for:

1. **Pre-filtering logic**: The estimation now calculates eligible users (those meeting 
   `min_likes_per_user`) before computing sampling ratios, since sampled users have 
   more likes on average than the general population.

2. **Corrected bytes-per-row estimates**: Calibrated from multiple observed runs:
   - Likes during chunk accumulation: 12 KB/row (scales with dataset size)
   - Expanded posts: 40 KB/row (text, author, embeddings as float32s)

3. **Fixed phase_2_posts calculation**: Now properly accounts for accumulated liked posts
   during loading, not just the negative reservoir and current batch.

4. **Baseline overhead**: 2 GB fixed overhead + 15% buffer for variance

### Statistics Tracking

New fields added to `stats` dictionary:
- `n_users_eligible_for_sampling`: Users who meet min-likes threshold
- `n_users_excluded_min_likes`: Users filtered out before sampling

These are logged to experiment tracking and summary files.

## Testing

To verify the improvement works:

1. Run the pipeline with `--max-liking-users 100000 --min-likes-per-user 2`
2. Check logs for pre-filtering message showing eligible users
3. Verify final user count is close to 100,000 (not ~67,000)
4. Confirm final min-likes verification removes few or no users

## Related Files

- `utils/helpers.py`: Main implementation (`load_likes_core_polars()`)
- `utils/01_get_data/stage_get_data.py`: Calls this function
- `ai_notes/DATA_PIPELINE_FIXES.md`: Background on pipeline design
