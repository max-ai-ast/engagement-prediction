#!/usr/bin/env python3

"""
Stage 3: Relevel users using Gini-optimized topic discovery with silhouette-optimized KMeans.

Inputs:
- embedding_bundle_*.pkl from Stage 2 (featurize)

Outputs under <run_dir>/relevel/<timestamp>/:
- topic_model.pkl (KMeans with optimal k)
- topic_pca.pkl (optional PCA)
- user_topic_mixtures.parquet
- retained_users.json (if Gini-based selection applied)
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from utils.pipeline.core import new_stage_timestamp_dir, select_prior_output


class TopicArtifacts:
    """
    Container class for topic discovery artifacts produced during clustering.
    
    This class encapsulates all artifacts generated from the topic discovery process,
    including the fitted clustering model, optional dimensionality reduction model,
    and the optimal number of clusters discovered. It provides a clean interface
    for passing these artifacts between pipeline stages.
    
    Attributes:
        topic_model (Optional[Any]): Fitted KMeans clustering model that assigns
            post embeddings to topic clusters. None if topic discovery failed.
        pca_model (Optional[Any]): Fitted PCA model used for dimensionality reduction
            prior to clustering. None if PCA was not applied or not needed.
        global_topic_k (Optional[int]): The optimal number of clusters (topics)
            discovered through silhouette optimization. None if discovery failed.
    """
    def __init__(self, topic_model: Optional[Any], pca_model: Optional[Any], global_topic_k: Optional[int]):
        self.topic_model = topic_model
        self.pca_model = pca_model
        self.global_topic_k = global_topic_k


def optimize_clusters_with_silhouette(
    embeddings_matrix: np.ndarray,
    k_range: tuple = (20, 30),
    sample_size: int = 10000,
    random_state: int = 42
) -> tuple:
    """
    Determine the optimal number of clusters using silhouette score optimization.
    
    This function performs an exhaustive search over a specified range of cluster
    counts (k values) to identify the optimal number of clusters that maximizes
    the silhouette score. The silhouette score measures how well-separated clusters
    are and how similar points within a cluster are to each other, providing a
    quantitative metric for cluster quality.
    
    For large datasets, the function uses random sampling to reduce computational
    cost while maintaining statistical validity. The optimization process fits
    KMeans models for each k value in the range, computes silhouette scores, and
    returns the k value with the highest score along with detailed optimization
    metrics for analysis.
    
    Args:
        embeddings_matrix (np.ndarray): 2D array of shape (n_samples, n_features)
            containing the embedding vectors to cluster. Each row represents a
            single data point (e.g., a post embedding).
        k_range (tuple, optional): Tuple of (min_k, max_k) specifying the range
            of cluster counts to test. Both values are inclusive. Defaults to (20, 30).
        sample_size (int, optional): Maximum number of samples to use for optimization
            when the dataset exceeds this size. For datasets smaller than sample_size,
            the full dataset is used. Defaults to 10000.
        random_state (int, optional): Random seed for reproducibility in sampling
            and KMeans initialization. Defaults to 42.
    
    Returns:
        tuple: A 2-element tuple containing:
            - optimal_k (int): The number of clusters that achieved the highest
              silhouette score within the specified range.
            - optimization_results (dict): Dictionary containing detailed metrics:
                - 'k_values': List of all k values tested
                - 'silhouette_scores': List of silhouette scores for each k
                - 'inertias': List of KMeans inertia values (within-cluster sum
                  of squares) for each k
                - 'computation_times': List of computation times in seconds for each k
                - 'optimal_k': The optimal k value (same as first return value)
                - 'best_silhouette_score': The highest silhouette score achieved
    
    Note:
        The silhouette score ranges from -1 to 1, where:
        - Values close to 1 indicate well-separated, cohesive clusters
        - Values close to 0 indicate overlapping clusters
        - Negative values indicate incorrect clustering assignments
    """
    print(f"🔍 Optimizing cluster count using silhouette score (k_range={k_range})...")
    
    if len(embeddings_matrix) > sample_size:
        print(f"📊 Using {sample_size:,} samples for optimization")
        np.random.seed(random_state)
        sample_indices = np.random.choice(len(embeddings_matrix), sample_size, replace=False)
        sample_embeddings = embeddings_matrix[sample_indices]
    else:
        sample_embeddings = embeddings_matrix
    
    optimization_results = {
        'k_values': [],
        'silhouette_scores': [],
        'inertias': [],
        'computation_times': []
    }
    
    for k in range(k_range[0], k_range[1] + 1):
        start_time = time.time()
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(sample_embeddings)
        sil_score = silhouette_score(sample_embeddings, cluster_labels)
        
        optimization_results['k_values'].append(k)
        optimization_results['silhouette_scores'].append(sil_score)
        optimization_results['inertias'].append(kmeans.inertia_)
        optimization_results['computation_times'].append(time.time() - start_time)
        print(f"   k={k:2d}: Silhouette Score = {sil_score:.3f}")
    
    best_idx = np.argmax(optimization_results['silhouette_scores'])
    optimal_k = optimization_results['k_values'][best_idx]
    best_score = optimization_results['silhouette_scores'][best_idx]
    
    print(f"✅ Optimal k: {optimal_k} (silhouette: {best_score:.3f})")
    optimization_results['optimal_k'] = optimal_k
    optimization_results['best_silhouette_score'] = best_score
    
    return optimal_k, optimization_results


def discover_topics_gini(
    posts_emb_df: pd.DataFrame,
    likes_df_joinable: pd.DataFrame,
    join_like: str,
    join_post: str,
    *,
    global_topic_k: int = 20,
    k_range: tuple = (20, 30),
    random_seed: int = 42,
    use_pca: bool = True,
    pca_components: int = 50,
) -> TopicArtifacts:
    """
    Discover topics from post embeddings using silhouette-optimized KMeans clustering.
    
    This function performs the core topic discovery process by:
    1. Extracting post embeddings from liked posts (user engagement data)
    2. Optionally applying PCA for dimensionality reduction
    3. Optimizing the number of clusters using silhouette score
    4. Fitting a final KMeans model with the optimal number of clusters
    
    The function uses only posts that users have liked (from likes_df_joinable) as
    the training data, which focuses the topic discovery on content that users
    actually engage with. This approach ensures that discovered topics are relevant
    to user behavior patterns rather than arbitrary content clusters.
    
    The silhouette optimization process tests multiple k values and selects the one
    that produces the most well-separated and cohesive clusters, providing a
    data-driven approach to determining the optimal number of topics.
    
    Args:
        posts_emb_df (pd.DataFrame): DataFrame containing post embeddings. Must
            include columns starting with 'post_emb_' or 'image_emb_' for feature
            extraction, and a column specified by join_post for joining with likes.
        likes_df_joinable (pd.DataFrame): DataFrame containing user likes/interactions.
            Must include a column specified by join_like for joining with posts,
            and a 'did' column for user identification.
        join_like (str): Column name in likes_df_joinable that contains post identifiers
            to join with posts_emb_df.
        join_post (str): Column name in posts_emb_df that contains post identifiers
            for joining with likes_df_joinable.
        global_topic_k (int, optional): Initial/expected number of topics. Used as a
            hint for k_range if not explicitly provided. Defaults to 20.
        k_range (tuple, optional): Range of cluster counts (min_k, max_k) to test
            during silhouette optimization. Both values are inclusive. Defaults to (20, 30).
        random_seed (int, optional): Random seed for reproducibility in PCA, KMeans,
            and sampling operations. Defaults to 42.
        use_pca (bool, optional): Whether to apply PCA dimensionality reduction before
            clustering. Recommended for high-dimensional embeddings (>50 dimensions).
            Defaults to True.
        pca_components (int, optional): Number of principal components to retain if
            PCA is enabled. Only applied if the original feature count exceeds this
            value. Defaults to 50.
    
    Returns:
        TopicArtifacts: Object containing:
            - topic_model: Fitted KMeans model with optimal k clusters
            - pca_model: Fitted PCA model (None if PCA was not applied)
            - global_topic_k: The optimal number of clusters discovered
    
    Raises:
        Returns TopicArtifacts(None, None, None) if:
            - No embedding columns found in posts_emb_df
            - No joinable posts found between posts_emb_df and likes_df_joinable
            - Empty join result
    
    Note:
        The function performs an inner join between posts and likes, meaning only
        posts that have been liked by users are used for topic discovery. This
        ensures topics are grounded in actual user engagement patterns.
    """
    feat_cols = [c for c in posts_emb_df.columns if c.startswith('post_emb_') or c.startswith('image_emb_')]
    if not feat_cols:
        return TopicArtifacts(None, None, None)
    
    available_posts = set(posts_emb_df[join_post].astype(str).unique())
    df = likes_df_joinable.copy()
    df[join_like] = df[join_like].astype(str)
    df = df[df[join_like].isin(available_posts)]
    joined = df.merge(posts_emb_df[[join_post] + feat_cols], left_on=join_like, right_on=join_post, how='inner')
    
    if len(joined) == 0:
        return TopicArtifacts(None, None, None)
    
    X = joined[feat_cols].values.astype(np.float32, copy=False)
    pca = None
    
    # Apply PCA if needed
    if use_pca and X.shape[1] > pca_components:
        print(f"📊 Applying PCA: {X.shape[1]} → {pca_components} dimensions")
        pca = PCA(n_components=pca_components, random_state=random_seed)
        X = pca.fit_transform(X)
    
    # Optimize k using silhouette score
    optimal_k, _ = optimize_clusters_with_silhouette(X, k_range=k_range, random_state=random_seed)
    
    # Fit final KMeans with optimal k
    print(f"🎯 Fitting KMeans with optimal k={optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=random_seed, n_init=10)
    kmeans.fit(X)
    
    return TopicArtifacts(kmeans, pca, optimal_k)


def compute_user_topic_mixtures(
    artifacts: TopicArtifacts,
    posts_emb_df: pd.DataFrame,
    likes_df_joinable: pd.DataFrame,
    join_like: str,
    join_post: str
) -> Optional[pd.DataFrame]:
    """
    Compute per-user topic probability distributions from their liked posts.
    
    This function calculates how each user's engagement is distributed across the
    discovered topics. For each user, it:
    1. Identifies all posts they have liked
    2. Assigns each liked post to a topic cluster using the fitted topic model
    3. Counts the number of posts per topic for that user
    4. Normalizes these counts to create a probability distribution (mixture)
    
    The resulting DataFrame represents each user's "topic preference profile",
    showing the proportion of their engagement that goes to each topic. This
    profile is used downstream for user selection, feature engineering, and
    understanding user behavior patterns.
    
    The mixture probabilities sum to 1.0 for each user, representing a proper
    probability distribution over topics. Users with no liked posts or no
    joinable posts are excluded from the result.
    
    Args:
        artifacts (TopicArtifacts): Topic discovery artifacts containing the
            fitted topic_model, optional pca_model, and global_topic_k. The
            topic_model must be non-None for this function to succeed.
        posts_emb_df (pd.DataFrame): DataFrame containing post embeddings with
            columns starting with 'post_emb_' or 'image_emb_', and a column
            specified by join_post for joining.
        likes_df_joinable (pd.DataFrame): DataFrame containing user likes with
            a column specified by join_like for joining, and a 'did' column
            for user identification.
        join_like (str): Column name in likes_df_joinable containing post
            identifiers to join with posts_emb_df.
        join_post (str): Column name in posts_emb_df containing post identifiers
            for joining with likes_df_joinable.
    
    Returns:
        Optional[pd.DataFrame]: DataFrame with:
            - Index: 'did' (user identifiers)
            - Columns: Topic indices (0, 1, 2, ..., global_topic_k-1)
            - Values: Probability of user engagement with each topic (0.0 to 1.0)
            - Each row sums to 1.0 (normalized probability distribution)
        
        Returns None if:
            - artifacts.topic_model is None
            - No joinable posts found between posts_emb_df and likes_df_joinable
            - Empty join result
    
    Note:
        The function ensures all topic columns exist (0 through global_topic_k-1),
        filling missing topics with 0.0 probability. This guarantees consistent
        dimensionality across all users regardless of their actual engagement patterns.
    """
    if artifacts.topic_model is None:
        return None
    
    feat_cols = [c for c in posts_emb_df.columns if c.startswith('post_emb_') or c.startswith('image_emb_')]
    available_posts = set(posts_emb_df[join_post].astype(str).unique())
    df = likes_df_joinable.copy()
    df[join_like] = df[join_like].astype(str)
    df = df[df[join_like].isin(available_posts)]
    joined = df.merge(posts_emb_df[[join_post] + feat_cols], left_on=join_like, right_on=join_post, how='inner')
    
    if len(joined) == 0:
        return None
    
    X = joined[feat_cols].values.astype(np.float32, copy=False)
    if artifacts.pca_model is not None:
        try:
            X = artifacts.pca_model.transform(X)
        except Exception:
            pass
    
    labels = artifacts.topic_model.predict(X)
    joined['_topic'] = labels
    counts = joined.groupby(['did', '_topic']).size().unstack(fill_value=0)
    
    # Normalize to probabilities
    mixtures = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)
    mixtures.index.name = 'did'
    
    # Ensure all topic columns exist
    if artifacts.global_topic_k:
        for t in range(int(artifacts.global_topic_k)):
            if t not in mixtures.columns:
                mixtures[t] = 0.0
        mixtures = mixtures[sorted(mixtures.columns)]
    
    return mixtures


def calculate_gini_coefficient(
    cluster_user_averages: np.ndarray,
    cluster_populations: np.ndarray = None
) -> float:
    """
    Calculate the Gini coefficient to measure inequality in cluster distribution.
    
    The Gini coefficient is a statistical measure of inequality that quantifies
    how unevenly values are distributed across clusters. In this context, it
    measures how balanced user engagement is distributed across different topics.
    
    The coefficient ranges from 0.0 to 1.0:
    - 0.0: Perfect equality (all clusters have equal engagement)
    - 1.0: Perfect inequality (all engagement concentrated in one cluster)
    
    The function automatically selects between weighted and unweighted Gini
    calculations based on whether population sizes are provided. Weighted Gini
    accounts for different cluster sizes (e.g., number of posts or users per
    cluster), providing a more accurate measure when clusters have varying
    populations.
    
    Args:
        cluster_user_averages (np.ndarray): 1D array of shape (n_clusters,)
            containing engagement values per cluster. These represent the
            "wealth" or total engagement for each topic/cluster. Values should
            be non-negative.
        cluster_populations (np.ndarray, optional): 1D array of shape (n_clusters,)
            containing population sizes per cluster (e.g., number of posts,
            number of users). If provided and length matches cluster_user_averages,
            a weighted Gini coefficient is calculated. If None or length mismatch,
            unweighted Gini is used. Defaults to None.
    
    Returns:
        float: Gini coefficient value between 0.0 and 1.0, where:
            - Lower values indicate more balanced/equal distribution
            - Higher values indicate more concentrated/unequal distribution
            - Returns 0.0 for edge cases (empty array, all zeros)
    
    Note:
        The weighted Gini calculation uses the Lorenz curve approach with
        trapezoidal integration. It sorts clusters by engagement value and
        calculates the area between the line of perfect equality and the
        actual distribution curve.
    """
    if len(cluster_user_averages) == 0 or np.sum(cluster_user_averages) == 0:
        return 0.0
    
    if cluster_populations is not None and len(cluster_populations) == len(cluster_user_averages):
        return _calculate_weighted_gini(cluster_user_averages, cluster_populations)
    else:
        return _calculate_unweighted_gini(cluster_user_averages)


def _calculate_unweighted_gini(values: np.ndarray) -> float:
    """
    Calculate unweighted Gini coefficient using the Lorenz curve method.
    
    This is an internal helper function that computes the Gini coefficient
    without considering population sizes. It treats all clusters equally,
    measuring inequality purely based on the distribution of engagement values.
    
    The calculation uses the standard Lorenz curve approach:
    1. Sort values in ascending order
    2. Compute cumulative proportions of both rank and value
    3. Calculate area under the Lorenz curve using trapezoidal integration
    4. Gini = 1 - 2 * (area under Lorenz curve)
    
    Args:
        values (np.ndarray): 1D array of non-negative values representing
            engagement or "wealth" per cluster. Must be non-empty.
    
    Returns:
        float: Unweighted Gini coefficient between 0.0 and 1.0. Returns 0.0
            if all values are zero or the array is empty.
    
    Note:
        This function assumes all clusters have equal "weight" or importance.
        For cases where clusters have different population sizes, use the
        weighted version (_calculate_weighted_gini) instead.
    """
    sorted_values = np.sort(values)
    n = len(sorted_values)
    cumulative_sum = np.cumsum(sorted_values)
    total_sum = cumulative_sum[-1]
    
    if total_sum == 0:
        return 0.0
    
    p = np.arange(1, n + 1) / n
    L = cumulative_sum / total_sum
    gini = 1 - 2 * np.trapz(L, p)
    
    return np.clip(gini, 0.0, 1.0)


def _calculate_weighted_gini(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate weighted Gini coefficient accounting for population sizes.
    
    This is an internal helper function that computes the Gini coefficient
    while considering different population sizes across clusters. This is
    more accurate than unweighted Gini when clusters have varying numbers
    of posts, users, or other population metrics.
    
    The weighted calculation:
    1. Pairs each engagement value with its corresponding population weight
    2. Sorts pairs by engagement value
    3. Computes cumulative proportions of both population weight and
       weighted engagement value
    4. Calculates area under the weighted Lorenz curve
    5. Gini = 1 - 2 * (area under weighted Lorenz curve)
    
    This approach prevents small clusters from being treated equally to large
    clusters when measuring inequality, providing a more accurate representation
    of distribution imbalance in real-world scenarios.
    
    Args:
        values (np.ndarray): 1D array of shape (n_clusters,) containing
            engagement values per cluster (the "wealth" distribution).
        weights (np.ndarray): 1D array of shape (n_clusters,) containing
            population sizes per cluster (e.g., post counts, user counts).
            Must have the same length as values and be non-negative.
    
    Returns:
        float: Weighted Gini coefficient between 0.0 and 1.0. Returns 0.0
            if total weight is zero or all weighted values are zero.
    
    Note:
        The function automatically handles sorting and ensures the result
        is properly clipped to the [0.0, 1.0] range. The weighted approach
        is preferred when cluster populations vary significantly.
    """
    pairs = list(zip(values, weights))
    pairs.sort(key=lambda x: x[0])
    
    sorted_values = np.array([p[0] for p in pairs])
    sorted_weights = np.array([p[1] for p in pairs])
    
    total_weight = np.sum(sorted_weights)
    if total_weight == 0:
        return 0.0
    
    cumulative_weights = np.cumsum(sorted_weights)
    cumulative_weighted_values = np.cumsum(sorted_values * sorted_weights)
    total_weighted_value = cumulative_weighted_values[-1]
    
    if total_weighted_value == 0:
        return 0.0
    
    p = cumulative_weights / total_weight
    L = cumulative_weighted_values / total_weighted_value
    gini = 1 - 2 * np.trapz(L, p)
    
    return np.clip(gini, 0.0, 1.0)


def calculate_user_engagement_gini(
    user_cluster_proportions: pd.DataFrame,
    cluster_populations: np.ndarray = None
) -> float:
    """
    Calculate Gini coefficient for aggregate user engagement across topic clusters.
    
    This function computes the Gini coefficient to measure how evenly user engagement
    is distributed across different topics. It aggregates individual user topic
    preferences into cluster-level engagement totals, then measures the inequality
    of this distribution.
    
    The calculation process:
    1. Sums user engagement probabilities across all users for each topic
       (creating cluster engagement totals)
    2. Uses these totals as the "wealth" distribution across clusters
    3. Optionally weights by cluster populations (e.g., number of posts per topic)
    4. Computes the Gini coefficient of this distribution
    
    A low Gini coefficient indicates balanced engagement across topics (users
    engage with diverse content), while a high coefficient indicates concentrated
    engagement (users focus on a few topics). This metric is useful for evaluating
    topic diversity and identifying potential echo chambers or content imbalances.
    
    Args:
        user_cluster_proportions (pd.DataFrame): DataFrame with:
            - Rows: Users (indexed by 'did' or similar user identifier)
            - Columns: Topic indices (0, 1, 2, ..., n_topics-1)
            - Values: Each user's engagement probability with each topic (0.0 to 1.0)
            Each row should sum to 1.0 (normalized probability distribution).
        cluster_populations (np.ndarray, optional): 1D array of shape (n_topics,)
            containing population sizes per topic (e.g., number of posts per topic).
            If provided, enables weighted Gini calculation. If None, unweighted
            Gini is used. Defaults to None.
    
    Returns:
        float: Gini coefficient between 0.0 and 1.0:
            - 0.0: Perfect balance (all topics have equal total engagement)
            - 1.0: Perfect inequality (all engagement in one topic)
            - Returns 0.0 if user_cluster_proportions is empty
    
    Note:
        This function is a convenience wrapper that aggregates user-level data
        to cluster-level totals before computing Gini. It's equivalent to summing
        the user_cluster_proportions DataFrame along the user axis and then
        computing the Gini coefficient of the resulting cluster totals.
    """
    if user_cluster_proportions.empty:
        return 0.0
    
    cluster_engagement_totals = user_cluster_proportions.sum(axis=0)
    engagement_values = cluster_engagement_totals.values
    
    return calculate_gini_coefficient(engagement_values, cluster_populations)


def gini_based_user_selection(
    users: List[str],
    user_topic_probs: pd.DataFrame,
    global_topic_k: int,
    target_gini: float = 0.1,
    min_users_per_topic: int = 0,
    random_seed: int = 42,
) -> List[str]:
    """
    Select a subset of users to minimize Gini coefficient of engagement distribution.
    
    This function implements a greedy optimization algorithm to select users such
    that the resulting user set has a balanced engagement distribution across
    topics, as measured by the Gini coefficient. The goal is to create a diverse
    user cohort that engages with a wide variety of topics rather than being
    concentrated in a few areas.
    
    Algorithm:
        The function uses an iterative greedy approach:
        1. Initialize with empty user set and zero engagement
        2. For each iteration:
           a. Sample candidate users (for efficiency on large datasets)
           b. For each candidate, simulate adding them to the current set
           c. Calculate the resulting Gini coefficient
           d. Select the user that minimizes Gini (maximizes diversity)
        3. Update cumulative engagement and user counts
        4. Stop when target Gini is reached or all users are selected
        5. Optionally enforce minimum users per topic constraint
    
    Each user contributes to all topics proportionally to their engagement
    probabilities. The weighted engagement for each topic is calculated as:
    weighted_engagement[topic] = sum(user_prob[topic] * cluster_weight[topic])
    where cluster_weight represents the total "wealth" of each topic.
    
    Args:
        users (List[str]): List of user identifiers (DIDs) to select from.
            Only users present in user_topic_probs.index will be considered.
        user_topic_probs (pd.DataFrame): DataFrame with:
            - Index: User identifiers (DIDs)
            - Columns: Topic indices (0, 1, 2, ..., global_topic_k-1)
            - Values: Each user's engagement probability with each topic
            Must contain all users from the users parameter.
        global_topic_k (int): Number of topics/clusters. Must match the number
            of topic columns in user_topic_probs.
        target_gini (float, optional): Target Gini coefficient to achieve.
            The algorithm stops early if this value is reached. Lower values
            indicate more balanced distributions. Defaults to 0.1.
        min_users_per_topic (int, optional): Minimum number of users to select
            per topic. If > 0, the final selection is filtered to ensure each
            topic has at least this many users (based on their primary topic).
            Defaults to 0 (no minimum constraint).
        random_seed (int, optional): Random seed for reproducibility in user
            shuffling and candidate sampling. Defaults to 42.
    
    Returns:
        List[str]: List of selected user identifiers (DIDs) that minimize
            the Gini coefficient. The list is ordered by selection sequence.
            Returns empty list if no eligible users found.
    
    Note:
        The function uses efficient sampling (testing up to 500 candidates per
        iteration) to handle large user sets. Progress is logged every 100
        iterations. The algorithm may not reach the exact target_gini if the
        user pool doesn't contain sufficient diversity, but it will select
        the most balanced subset possible.
    """
    print(f"🎯 Gini-based user selection (target_gini={target_gini:.3f})...")
    
    np.random.seed(random_seed)
    
    # Filter users that exist in mixtures
    eligible_users = [u for u in users if u in user_topic_probs.index]
    if not eligible_users:
        return []
    
    # Calculate cluster engagement weights (x_i values) - sum of user engagement per topic
    # This represents the "wealth" of each topic (EXACTLY like iterative_utils.py)
    cluster_engagement_weights = user_topic_probs.sum(axis=0).values.copy()
    print(f"   Cluster engagement weights (x_i): {cluster_engagement_weights}")
    print(f"   Cluster engagement weights shape: {cluster_engagement_weights.shape}")
    
    # Initialize tracking: weighted engagement per topic (x_i * h_i where h_i = user count per topic)
    # Following EXACT same pattern as iterative_utils.py gini_based_post_sampling
    selected_users = []
    cumulative_weighted_engagement = np.zeros(global_topic_k, dtype=float)
    cumulative_user_counts = np.zeros(global_topic_k, dtype=int)
    
    # Shuffle for randomness
    candidate_users = eligible_users.copy()
    np.random.shuffle(candidate_users)
    
    # Set max iterations (similar to iterative_utils.py)
    max_iterations = min(len(candidate_users) * 2, 1000000)  # Safety limit like iterative_utils
    
    for iteration in range(max_iterations):
        if len(selected_users) >= len(eligible_users):
            break
        
        best_user = None
        best_gini = float('inf')
        
        # Sample candidates for efficiency (EXACT same approach as iterative_utils.py)
        remaining = [u for u in candidate_users if u not in selected_users]
        if not remaining:
            break
        
        # Greedy sampling: test fewer users for speed (like iterative_utils.py)
        if len(remaining) > 500:
            sample_size = min(500, len(remaining) // 20)  # 5% or 500, whichever is smaller
            candidates = np.random.choice(remaining, sample_size, replace=False)
            if iteration == 0:
                print(f"   Strategy: Testing {sample_size:,} users per iteration (greedy sampling)")
        else:
            candidates = remaining
            if iteration == 0:
                print(f"   Strategy: Testing all {len(remaining):,} users per iteration")
        
        # Test candidate users to find the one that minimizes Gini (EXACT same logic as iterative_utils.py)
        for user in candidates:
            if user not in user_topic_probs.index:
                continue
            
            # Get user's primary topic (the topic with highest engagement probability)
            # This is how we assign users to topics (similar to how posts belong to one cluster)
            user_probs = user_topic_probs.loc[user].values
            primary_topic = int(np.argmax(user_probs))
            
            # Validate topic ID bounds
            if primary_topic < 0 or primary_topic >= global_topic_k:
                continue
            
            # Store original values to restore after testing (EXACT same pattern as iterative_utils.py)
            original_weighted = cumulative_weighted_engagement[primary_topic]
            original_count = cumulative_user_counts[primary_topic]
            
            # Temporarily update cumulative variables for this test
            # EXACT same calculation as iterative_utils.py: x_i * h_i
            # For users: add the cluster engagement weight for the user's primary topic
            cumulative_weighted_engagement[primary_topic] += cluster_engagement_weights[primary_topic]
            cumulative_user_counts[primary_topic] += 1
            
            # Use the updated cumulative variables directly for Gini calculation
            test_weighted_engagement = cumulative_weighted_engagement
            test_user_counts = cumulative_user_counts
            
            # Use weighted Gini calculation with x_i * h_i values (EXACT same as iterative_utils.py)
            test_gini = calculate_gini_coefficient(test_weighted_engagement, test_user_counts)
            
            # Restore original cumulative values after testing this user
            cumulative_weighted_engagement[primary_topic] = original_weighted
            cumulative_user_counts[primary_topic] = original_count
            
            # Choose the user that gives the LOWEST Gini (highest diversity)
            if test_gini < best_gini:
                best_user = user
                best_gini = test_gini
                
                # Early stopping thresholds (matching iterative_utils.py pattern)
                if test_gini < 0.05:  # Very good diversity threshold
                    if iteration == 0 or iteration % 100 == 0:
                        print(f"   🎯 Excellent diversity found! Gini: {test_gini:.4f}")
                    break
                elif test_gini < 0.15 and iteration > 50:  # Acceptable diversity after some iterations
                    break
                elif test_gini < 0.25 and iteration > 100:  # Good diversity after many iterations
                    break
        
        if best_user is None:
            # Fallback: select first remaining user
            if remaining:
                best_user = remaining[0]
                # Calculate Gini for fallback user
                user_probs = user_topic_probs.loc[best_user].values
                primary_topic = int(np.argmax(user_probs))
                cumulative_weighted_engagement[primary_topic] += cluster_engagement_weights[primary_topic]
                cumulative_user_counts[primary_topic] += 1
                best_gini = calculate_gini_coefficient(cumulative_weighted_engagement, cumulative_user_counts)
            else:
                break
        
        # Add best user
        selected_users.append(best_user)
        user_probs = user_topic_probs.loc[best_user].values
        primary_topic = int(np.argmax(user_probs))
        
        # Update cumulative variables for next iteration (EXACT same pattern as iterative_utils.py)
        cumulative_weighted_engagement[primary_topic] += cluster_engagement_weights[primary_topic]
        cumulative_user_counts[primary_topic] += 1
        
        # Check if we've reached target Gini
        # Need at least 2 users for Gini to be meaningful (with 1 user, Gini is always 0)
        current_gini = calculate_gini_coefficient(cumulative_weighted_engagement, cumulative_user_counts)
        min_users_for_gini = max(2, min_users_per_topic * global_topic_k) if min_users_per_topic > 0 else 2
        if len(selected_users) >= min_users_for_gini and current_gini <= target_gini:
            print(f"✅ Target Gini reached: {current_gini:.4f} (selected {len(selected_users)} users)")
            break
        
        # Progress indicators (matching iterative_utils.py pattern)
        if iteration % 100 == 0:
            print(f"   Iter {iteration:4d}: {len(selected_users):4d} users, Gini: {current_gini:.4f}")
            print(f"      Current user counts per topic: {cumulative_user_counts}")
        elif iteration % 25 == 0 and len(eligible_users) > 1000:
            print(f"   Iter {iteration:4d}: {len(selected_users):4d} users, Gini: {current_gini:.4f}")
    
    # Calculate final weighted engagement for Gini calculation (EXACT same pattern as iterative_utils.py)
    # This ensures consistency between selection optimization and final measurement
    final_user_counts = cumulative_user_counts.copy()
    
    # Calculate final weighted engagement: x_i * h_i for each topic (EXACT same as iterative_utils.py)
    final_weighted_engagement = np.zeros(global_topic_k, dtype=float)
    for topic_id in range(global_topic_k):
        final_weighted_engagement[topic_id] = cluster_engagement_weights[topic_id] * final_user_counts[topic_id]
    
    # Validate final calculation inputs
    if len(final_weighted_engagement) != len(final_user_counts):
        print(f"⚠️  WARNING: Final Gini calculation mismatch!")
        print(f"   Weighted engagement length: {len(final_weighted_engagement)}")
        print(f"   User counts length: {len(final_user_counts)}")
    
    # Calculate final Gini using weighted calculation (EXACT same as iterative_utils.py)
    try:
        final_gini = calculate_gini_coefficient(final_weighted_engagement, final_user_counts)
    except Exception as e:
        print(f"❌ ERROR: Failed to calculate final Gini coefficient: {e}")
        print(f"   Falling back to simple calculation")
        final_gini = calculate_gini_coefficient(final_weighted_engagement)
    
    print(f"   Final user counts per topic (h_i): {final_user_counts}")
    print(f"   Final cluster engagement weights (x_i): {cluster_engagement_weights}")
    print(f"   Final weighted engagement (x_i * h_i): {final_weighted_engagement}")
    
    # Ensure minimum users per topic if specified
    if min_users_per_topic > 0:
        per_topic_counts = {t: 0 for t in range(global_topic_k)}
        final_users = []
        for user in selected_users:
            if user not in user_topic_probs.index:
                continue
            user_probs = user_topic_probs.loc[user].values
            top_topic = int(np.argmax(user_probs))
            if per_topic_counts[top_topic] < min_users_per_topic:
                per_topic_counts[top_topic] += 1
                final_users.append(user)
        
        if len(final_users) >= min_users_per_topic * global_topic_k:
            return final_users
    
    print(f"\n🎉 USER SELECTION COMPLETE!")
    print(f"   Users selected: {len(selected_users):,}")
    print(f"   Final Gini: {final_gini:.4f}")
    print(f"   Iterations: {iteration + 1}")
    
    return selected_users


def run(context, args) -> Dict[str, Any]:
    """
    Main pipeline execution function for Gini-optimized topic discovery and user releveling.
    
    This function orchestrates the complete Stage 3 releveling process using
    Gini-optimized topic discovery. It follows the standard pipeline interface
    pattern, taking a context object (containing run directory and configuration)
    and an args object (containing hyperparameters and options), and returning
    a dictionary of output artifacts and metadata.
    
    Pipeline Flow:
        1. Locate and load embedding bundle from Stage 2 (featurize)
        2. Extract posts and likes data, identify joinable interactions
        3. Discover topics using silhouette-optimized KMeans clustering
        4. Compute user topic mixture distributions
        5. Optionally apply Gini-based user selection for balanced cohorts
        6. Save all artifacts (topic model, PCA model, mixtures, retained users)
        7. Generate stage info metadata
    
    The function produces outputs compatible with downstream pipeline stages,
    ensuring seamless integration with the existing workflow. All outputs are
    saved to a timestamped directory under the run directory.
    
    Args:
        context: Pipeline context object containing:
            - run_dir (Path): Base directory for this pipeline run
            - use_latest (bool): Whether to use latest prior outputs
            - prior_outputs (Dict): Dictionary mapping stage names to output paths
            - artifacts (Dict): Dictionary for storing stage artifacts
        args: Arguments object containing hyperparameters (accessed via getattr):
            - global_topic_k (int): Initial number of topics (default: 20)
            - k_range (tuple): Range for silhouette optimization (default: (20, 30))
            - random_seed (int): Random seed for reproducibility (default: 42)
            - use_pca (bool): Whether to apply PCA (default: True)
            - pca_components (int): PCA dimensions if enabled (default: 50)
            - relevel_strategy (str): Selection strategy, 'gini_based' to enable
              Gini selection (default: 'gini_based')
            - target_gini (float): Target Gini coefficient (default: 0.1)
            - min_likes_per_user (int): Minimum likes required for user eligibility (default: 4)
            - relevel_min_users_per_topic (int): Minimum users per topic (default: 0)
    
    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'output_dir' (str): Path to the output directory for this stage
            - 'artifacts' (Dict): Dictionary of artifact paths:
                - 'mixtures_path' (str): Path to user_topic_mixtures.parquet
                - 'retained_users_path' (str, optional): Path to retained_users.json
                  (only if Gini selection was applied)
                - 'topic_model_path' (str): Path to topic_model.pkl
                - 'topic_pca_path' (str, optional): Path to topic_pca.pkl
                  (only if PCA was applied)
                - 'embedding_bundle_path' (str): Path to input embedding bundle
    
    Raises:
        FileNotFoundError: If Stage 2 (featurize) output not found or embedding
            bundle not found in the featurize output directory.
        KeyError: If likes_df is missing the required join_like column.
        RuntimeError: If topic discovery fails (scikit-learn missing, no joinable
            likes, or empty results) or if user topic mixture computation fails.
    
    Output Files:
        All files are saved to <run_dir>/relevel/<timestamp>/:
        - topic_model.pkl: Fitted KMeans model with optimal k
        - topic_pca.pkl: Fitted PCA model (if PCA was applied)
        - user_topic_mixtures.parquet: User topic probability distributions
        - retained_users.json: Selected user list (if Gini selection applied)
        - stage_info.txt: Metadata and statistics about the run
    
    Note:
        This function is designed to be called by the pipeline orchestrator.
        It follows the same interface pattern as stage_relevel_uniform.py,
        allowing it to be used as a drop-in replacement or alternative
        releveling strategy via CLI flags.
    """
    run_dir = Path(context.run_dir).resolve()
    out_dir = new_stage_timestamp_dir(run_dir, '03_relevel')
    
    # Locate embedding bundle from prior featurize stage
    prior_featurize = select_prior_output(
        run_dir, '02_featurize',
        use_latest=context.use_latest,
        prior_path=context.prior_outputs.get('02_featurize')
    )
    if prior_featurize is None:
        raise FileNotFoundError("Featurize output not found. Run Stage 2 first or provide --prior-output-featurize.")
    
    bundle_candidates = sorted(
        prior_featurize.glob('embedding_bundle_*.pkl'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not bundle_candidates:
        raise FileNotFoundError(f"No embedding_bundle_*.pkl found under {prior_featurize}")
    bundle_path = bundle_candidates[0]
    
    # Load bundle
    with open(bundle_path, 'rb') as f:
        bundle = pickle.load(f)
    posts_emb_df: pd.DataFrame = bundle['posts_emb_df']
    likes_df: pd.DataFrame = bundle['likes_df']
    join_like: str = str(bundle['join_like'])
    join_post: str = str(bundle['join_post'])
    
    # Eligibility (for mixtures; selection can be applied later)
    available_posts = set(posts_emb_df[join_post].astype(str).unique())
    likes_df_local = likes_df.copy()
    if join_like not in likes_df_local.columns:
        raise KeyError(f"likes_df missing join_like column: {join_like}")
    likes_df_local[join_like] = likes_df_local[join_like].astype(str)
    likes_joinable = likes_df_local[likes_df_local[join_like].isin(available_posts)]
    
    # Topic discovery with Gini optimization
    global_topic_k = int(args.global_topic_k)
    k_range = tuple(getattr(args, 'k_range', (20, 30)))
    if isinstance(k_range, (list, tuple)) and len(k_range) == 2:
        k_range = (int(k_range[0]), int(k_range[1]))
    else:
        k_range = (max(global_topic_k - 5, 10), global_topic_k + 5)
    
    random_seed = int(args.random_seed)
    use_pca = bool(getattr(args, 'use_pca', True))
    pca_components = int(getattr(args, 'pca_components', 50))
    
    t0 = time.time()
    artifacts = discover_topics_gini(
        posts_emb_df,
        likes_joinable,
        join_like,
        join_post,
        global_topic_k=global_topic_k,
        k_range=k_range,
        random_seed=random_seed,
        use_pca=use_pca,
        pca_components=pca_components,
    )
    
    if artifacts.topic_model is None:
        raise RuntimeError("Topic discovery unavailable (scikit-learn missing or no joinable likes)")
    
    # Update global_topic_k with actual optimal k
    global_topic_k = artifacts.global_topic_k
    
    # Compute mixtures
    mixtures = compute_user_topic_mixtures(artifacts, posts_emb_df, likes_joinable, join_like, join_post)
    if mixtures is None or mixtures.empty:
        raise RuntimeError("Failed to compute user topic mixtures")
    
    # Save mixtures
    mixtures_path = out_dir / 'user_topic_mixtures.parquet'
    mixtures.to_parquet(mixtures_path, index=True)
    
    # Optional Gini-based selection
    relevel_strategy = str(getattr(args, 'relevel_strategy', 'gini_based'))
    target_gini = float(getattr(args, 'target_gini', 0.1))
    relevel_min_users_per_topic = int(args.relevel_min_users_per_topic)
    
    retained_users_path = None
    if relevel_strategy == 'gini_based' and artifacts.global_topic_k:
        # Eligible users based on min likes per user
        min_likes_per_user = int(args.min_likes_per_user)
        counts = likes_joinable.groupby('did', observed=True)[join_like].nunique().astype(int)
        eligible_users = counts[counts >= min_likes_per_user].index.astype(str).tolist()
        
        kept_users = gini_based_user_selection(
            users=eligible_users,
            user_topic_probs=mixtures,
            global_topic_k=int(artifacts.global_topic_k),
            target_gini=target_gini,
            min_users_per_topic=int(relevel_min_users_per_topic),
            random_seed=random_seed,
        )
        
        retained_users_path = out_dir / 'retained_users.json'
        with open(retained_users_path, 'w') as f:
            json.dump({'retained_users': kept_users}, f, indent=2)
    
    # Save topic artifacts
    topic_model_path = out_dir / 'topic_model.pkl'
    with open(topic_model_path, 'wb') as f:
        pickle.dump(artifacts.topic_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    pca_model_path = None
    if artifacts.pca_model is not None:
        pca_model_path = out_dir / 'topic_pca.pkl'
        with open(pca_model_path, 'wb') as f:
            pickle.dump(artifacts.pca_model, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Stage info
    info_lines = [
        f"stage: relevel",
        f"method: gini_optimized",
        f"runtime_seconds: {time.time()-t0:.2f}",
        f"settings: global_topic_k={global_topic_k}, k_range={k_range}, relevel_strategy={relevel_strategy}, target_gini={target_gini}, relevel_min_users_per_topic={relevel_min_users_per_topic}",
        f"inputs: embedding_bundle",
        f"N_posts_emb: {len(posts_emb_df)}",
        f"N_likes_joinable: {len(likes_joinable)}",
        f"N_users_mixtures: {len(mixtures)}",
        f"N_retained_users: {len(kept_users) if 'kept_users' in locals() else 0}",
    ]
    (out_dir / 'stage_info.txt').write_text('\n'.join(info_lines) + '\n')
    
    return {
        'output_dir': out_dir,
        'artifacts': {
            'mixtures_path': str(mixtures_path),
            **({'retained_users_path': str(retained_users_path)} if retained_users_path else {}),
            'topic_model_path': str(topic_model_path),
            **({'topic_pca_path': str(pca_model_path)} if pca_model_path else {}),
            'embedding_bundle_path': str(bundle_path.resolve()),
        }
    }
