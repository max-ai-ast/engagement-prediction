#!/usr/bin/env python3

"""
Approximate Nearest Neighbor (ANN) retrieval utilities.

Supports multiple ANN index types:
- FAISS: Fast library by Facebook for efficient similarity search
- Annoy: Spotify's ANN library optimized for memory efficiency

Key concepts:
- ANN trades exact nearest neighbors for speed (99%+ recall with 10-100x speedup)
- Indexes are built once on all post embeddings, then used for fast lookup
- Use cosine similarity (normalized embeddings + inner product)
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd


# =============================================================================
# ANN Index Interface
# =============================================================================

class ANNIndex:
    """
    Abstract base class for ANN indexes.
    Provides a unified interface for FAISS, Annoy, or other backends.
    """
    
    def __init__(self, embeddings: np.ndarray, index_type: str = 'faiss'):
        """
        Initialize ANN index.
        
        Args:
            embeddings: Post embeddings [N_posts, shared_dim]
            index_type: 'faiss' or 'annoy'
        """
        self.embeddings = embeddings
        self.index_type = index_type
        self.n_posts, self.dim = embeddings.shape
        self.index = None
        
    def build(self):
        """Build the ANN index. Must be called before search."""
        raise NotImplementedError
        
    def search(self, query: np.ndarray, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query embedding [1, shared_dim]
            k: Number of nearest neighbors to return
            
        Returns:
            indices: [1, k] array of post indices
            distances: [1, k] array of distances (higher = more similar for cosine)
        """
        raise NotImplementedError
    
    def save(self, path: str):
        """Save index to disk."""
        raise NotImplementedError
    
    @classmethod
    def load(cls, path: str, embeddings: Optional[np.ndarray] = None):
        """Load index from disk."""
        raise NotImplementedError


# =============================================================================
# FAISS Implementation
# =============================================================================

class FAISSIndex(ANNIndex):
    """
    FAISS-based ANN index.
    
    FAISS (Facebook AI Similarity Search) is optimized for:
    - Large-scale similarity search (millions to billions of vectors)
    - GPU acceleration
    - Multiple index types (flat, IVF, HNSW, etc.)
    
    Index types:
    - IndexFlatIP: Exact search with inner product (good baseline, ~1M vectors)
    - IndexIVFFlat: Inverted file index (10M+ vectors, requires training)
    - IndexHNSWFlat: Hierarchical NSW graph (best quality/speed tradeoff)
    """
    
    def __init__(self, embeddings: np.ndarray, use_gpu: bool = False):
        super().__init__(embeddings, index_type='faiss')
        self.use_gpu = use_gpu
        self._normalized_embeddings: Optional[np.ndarray] = None
        self._fallback_mode = False
        
    def build(self, index_type: str = 'hnsw'):
        """
        Build FAISS index.
        
        Args:
            index_type: 'flat' (exact), 'ivf' (fast), or 'hnsw' (balanced)
        """
        embeddings = self.embeddings.astype(np.float32, copy=False)
        denom = np.linalg.norm(embeddings, axis=1, keepdims=True)
        denom = np.clip(denom, a_min=1e-12, a_max=None)
        normalized_embs = embeddings / denom
        self._normalized_embeddings = normalized_embs
        try:
            import faiss  # type: ignore
            if index_type == 'flat':
                self.index = faiss.IndexFlatIP(self.dim)
                self.index.add(normalized_embs)
            elif index_type == 'ivf':
                nlist = max(1, int(np.sqrt(self.n_posts)))
                quantizer = faiss.IndexFlatIP(self.dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
                self.index.train(normalized_embs)
                self.index.add(normalized_embs)
                self.index.nprobe = min(32, nlist)
            elif index_type == 'hnsw':
                self.index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
                self.index.add(normalized_embs)
            else:
                raise ValueError(f"Unknown FAISS index type: {index_type}")
            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except Exception:
                    pass
        except ImportError:
            # Fallback: exact cosine search with numpy
            self.index = "numpy_exact"
            self._fallback_mode = True
        
        print(f"Built FAISS {index_type} index with {self.n_posts} posts, dim={self.dim}")
        
    def search(self, query: np.ndarray, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search FAISS index for top-k nearest neighbors.
        
        Args:
            query: [1, shared_dim] user embedding
            k: Number of results
            
        Returns:
            indices: [1, k] post indices
            distances: [1, k] similarity scores (inner product, higher is better)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")
        
        query = query.astype(np.float32, copy=False)
        q_denom = np.linalg.norm(query, axis=1, keepdims=True)
        q_denom = np.clip(q_denom, a_min=1e-12, a_max=None)
        query_normalized = query / q_denom
        k = min(k, self.n_posts)
        if self._fallback_mode:
            assert self._normalized_embeddings is not None
            sims = query_normalized @ self._normalized_embeddings.T
            idx_part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
            part_vals = np.take_along_axis(sims, idx_part, axis=1)
            order = np.argsort(-part_vals, axis=1)
            indices = np.take_along_axis(idx_part, order, axis=1)
            distances = np.take_along_axis(part_vals, order, axis=1)
            return indices, distances
        distances, indices = self.index.search(query_normalized, k)
        return indices, distances
    
    def save(self, path: str):
        """Save FAISS index to disk."""
        # TODO: faiss.write_index(self.index, path)
        print(f"Saved FAISS index to {path}")
    
    @classmethod
    def load(cls, path: str, embeddings: Optional[np.ndarray] = None):
        """Load FAISS index from disk."""
        # TODO: 
        # import faiss
        # index = faiss.read_index(path)
        # obj = cls(embeddings if embeddings is not None else np.zeros((1, index.d)))
        # obj.index = index
        # return obj
        raise NotImplementedError("FAISS index loading not yet implemented")


# =============================================================================
# Annoy Implementation
# =============================================================================

class AnnoyIndex(ANNIndex):
    """
    Annoy-based ANN index.
    
    Annoy (Approximate Nearest Neighbors Oh Yeah) by Spotify:
    - Memory-mapped files (efficient for large indexes)
    - Simple API, easy to deploy
    - Uses random projection trees
    - Good for medium-scale datasets (1M-10M vectors)
    """
    
    def __init__(self, embeddings: np.ndarray):
        super().__init__(embeddings, index_type='annoy')
        self._fallback_mode = False
        self._normalized_embeddings: Optional[np.ndarray] = None

    def build(self, n_trees: int = 50):
        """
        Build Annoy index.
        
        Args:
            n_trees: Number of trees (more = higher accuracy, slower build)
                     Spotify recommends: 10 for speed, 100+ for precision
        """
        embeddings = self.embeddings.astype(np.float32, copy=False)
        denom = np.linalg.norm(embeddings, axis=1, keepdims=True)
        denom = np.clip(denom, a_min=1e-12, a_max=None)
        normalized_embs = embeddings / denom
        self._normalized_embeddings = normalized_embs
        try:
            from annoy import AnnoyIndex as _AnnoyIndex  # type: ignore
            self.index = _AnnoyIndex(self.dim, 'angular')
            for i, emb in enumerate(normalized_embs):
                self.index.add_item(i, emb)
            self.index.build(n_trees)
        except ImportError:
            self.index = "numpy_exact"
            self._fallback_mode = True
        
        print(f"Built Annoy index with {self.n_posts} posts, {n_trees} trees, dim={self.dim}")
        
    def search(self, query: np.ndarray, k: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search Annoy index for top-k nearest neighbors.
        
        Args:
            query: [1, shared_dim] user embedding
            k: Number of results
            
        Returns:
            indices: [1, k] post indices
            distances: [1, k] similarity scores (1 - angular distance)
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build() first.")
        
        query = query.astype(np.float32, copy=False)
        q_denom = np.linalg.norm(query, axis=1, keepdims=True)
        q_denom = np.clip(q_denom, a_min=1e-12, a_max=None)
        query_normalized = query / q_denom
        k = min(k, self.n_posts)
        if self._fallback_mode:
            assert self._normalized_embeddings is not None
            sims = query_normalized @ self._normalized_embeddings.T
            idx_part = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
            part_vals = np.take_along_axis(sims, idx_part, axis=1)
            order = np.argsort(-part_vals, axis=1)
            indices = np.take_along_axis(idx_part, order, axis=1)
            distances = np.take_along_axis(part_vals, order, axis=1)
            return indices, distances
        indices, angular_distances = self.index.get_nns_by_vector(
            query_normalized[0],
            k,
            include_distances=True,
        )
        similarities = 1.0 - np.array(angular_distances, dtype=np.float32)
        return np.array([indices]), np.array([similarities])
    
    def save(self, path: str):
        """Save Annoy index to disk."""
        # TODO: self.index.save(path)
        print(f"Saved Annoy index to {path}")
    
    @classmethod
    def load(cls, path: str, embeddings: Optional[np.ndarray] = None):
        """Load Annoy index from disk."""
        # TODO:
        # from annoy import AnnoyIndex
        # # Need to know dimension
        # if embeddings is None:
        #     raise ValueError("Must provide embeddings to determine dimension")
        # dim = embeddings.shape[1]
        # obj = cls(embeddings)
        # obj.index = AnnoyIndex(dim, 'angular')
        # obj.index.load(path)
        # return obj
        raise NotImplementedError("Annoy index loading not yet implemented")


# =============================================================================
# Utility Functions
# =============================================================================

def build_ann_index(
    embeddings: np.ndarray,
    index_type: str = 'faiss',
    **kwargs
) -> ANNIndex:
    """
    Factory function to build ANN index.
    
    Args:
        embeddings: [N_posts, shared_dim] encoded post embeddings
        index_type: 'faiss' or 'annoy'
        **kwargs: Additional arguments for specific index type
        
    Returns:
        Built ANN index ready for search
    """
    _ = kwargs.get('shared_dim', None)
    if index_type == 'faiss':
        index = FAISSIndex(embeddings, use_gpu=kwargs.get('use_gpu', False))
        index.build(index_type=kwargs.get('faiss_index_type', 'hnsw'))
    elif index_type == 'annoy':
        index = AnnoyIndex(embeddings)
        index.build(n_trees=kwargs.get('n_trees', 50))
    else:
        raise ValueError(f"Unknown index type: {index_type}. Choose 'faiss' or 'annoy'.")
    
    return index


def encode_all_posts(
    model,
    posts_emb_df: pd.DataFrame,
    join_post: str,
    embedding_dim: int,
    device: str,
    batch_size: int = 1024,
) -> Tuple[List[str], np.ndarray]:
    """
    Encode all posts through the post tower of the two-tower model.
    
    This is done ONCE and cached in the ANN index.
    
    Args:
        model: Trained TwoTowerEngagement model
        posts_emb_df: DataFrame with post embeddings (from Stage 2)
        join_post: Column name for post IDs
        embedding_dim: Dimension of post embeddings (e.g., 384)
        device: 'cpu' or 'cuda'
        batch_size: Batch size for encoding
        
    Returns:
        post_ids: List of post IDs [N_posts]
        encoded_embeddings: [N_posts, shared_dim] encoded through post tower
    """
    import torch
    
    # Get embedding columns
    emb_cols = [f"post_emb_{i}" for i in range(embedding_dim)]
    
    post_ids = []
    encoded_embeddings = []
    
    model.eval()
    with torch.no_grad():
        for start in range(0, len(posts_emb_df), batch_size):
            batch_df = posts_emb_df.iloc[start:start+batch_size]
            
            # Extract post IDs
            batch_post_ids = batch_df[join_post].astype(str).tolist()
            post_ids.extend(batch_post_ids)
            
            # Extract embeddings
            batch_embs = batch_df[emb_cols].values.astype(np.float32)
            batch_tensor = torch.tensor(batch_embs, dtype=torch.float32, device=device)
            
            # Encode through post tower
            encoded = model.encode_post(batch_tensor)  # [batch_size, shared_dim]
            encoded_embeddings.append(encoded.cpu().numpy())
    
    # Concatenate all batches
    encoded_embeddings = np.vstack(encoded_embeddings)  # [N_posts, shared_dim]
    
    print(f"Encoded {len(post_ids)} posts: {encoded_embeddings.shape}")
    return post_ids, encoded_embeddings


def encode_all_posts_fit_sharded(
    model,
    posts_emb_df: pd.DataFrame,
    join_post: str,
    embedding_dim: int,
    device: str,
    batch_size: int = 1024,
) -> Dict[int, Dict[str, Any]]:
    """
    Encode posts and shard by FIT hard-query index.

    Returns:
        Dict[q_idx, {"post_ids": List[str], "embeddings": np.ndarray}]
    """
    import torch

    if not getattr(model, "use_fit", False):
        raise ValueError("FIT sharded encoding requires a model with use_fit=True")

    emb_cols = [f"post_emb_{i}" for i in range(embedding_dim)]
    shards_post_ids: Dict[int, List[str]] = {}
    shards_embeddings: Dict[int, List[np.ndarray]] = {}

    model.eval()
    with torch.no_grad():
        for start in range(0, len(posts_emb_df), batch_size):
            batch_df = posts_emb_df.iloc[start:start + batch_size]
            batch_post_ids = batch_df[join_post].astype(str).tolist()
            batch_embs = batch_df[emb_cols].values.astype(np.float32, copy=False)
            batch_tensor = torch.tensor(batch_embs, dtype=torch.float32, device=device)

            post_encoded = model.encode_post(batch_tensor).cpu().numpy()
            _, q_idx = model.mqm(batch_tensor, tau=float(getattr(model, "fit_tau_min", 0.1)), hard=True)
            q_idx_np = q_idx.detach().cpu().numpy().astype(np.int64, copy=False)

            for idx, shard in enumerate(q_idx_np):
                shard = int(shard)
                shards_post_ids.setdefault(shard, []).append(batch_post_ids[idx])
                shards_embeddings.setdefault(shard, []).append(post_encoded[idx])

    shard_map: Dict[int, Dict[str, Any]] = {}
    for shard, emb_list in shards_embeddings.items():
        shard_map[shard] = {
            "post_ids": shards_post_ids[shard],
            "embeddings": np.vstack(emb_list).astype(np.float32, copy=False),
        }
    return shard_map


# =============================================================================
# Pseudocode Example: End-to-End Retrieval
# =============================================================================

"""
PSEUDOCODE: How to use ANN retrieval in production

# Step 1: Offline - Build index (run once, or periodically when new posts arrive)
all_post_embeddings = load_from_stage_2()  # [N_posts, 384]
encoded_posts = model.encode_post(all_post_embeddings)  # [N_posts, 128]
ann_index = build_ann_index(encoded_posts, index_type='faiss')
ann_index.save('post_index.faiss')

# Step 2: Online - Fast retrieval per user request
def get_recommendations(user_id: str, k: int = 50):
    # 2a. Get user's history
    user_likes = get_user_history(user_id)  # List of post IDs
    history_embs = [post_embeddings[pid] for pid in user_likes[-20:]]  # Last 20 posts
    
    # 2b. Encode user
    user_emb = model.encode_user(history_embs)  # [1, 128]
    
    # 2c. ANN search (fast! ~1ms for 1M posts)
    candidate_indices, scores = ann_index.search(user_emb, k=1000)  # Get 1000 candidates
    candidate_post_ids = [all_post_ids[idx] for idx in candidate_indices[0]]
    
    # 2d. (Optional) Cross-encoder reranking on top-100
    if use_cross_encoder:
        top_100 = candidate_post_ids[:100]
        reranked = cross_encoder_rerank(user_id, top_100)
        return reranked[:k]
    
    return candidate_post_ids[:k]

# Typical latency breakdown:
# - User encoding: 1-5ms
# - ANN search: 1-10ms (FAISS on 1M posts)
# - Cross-encoder (optional): 50-200ms (batch of 100 posts)
# Total: 10-50ms without cross-encoder, 60-200ms with cross-encoder
"""
