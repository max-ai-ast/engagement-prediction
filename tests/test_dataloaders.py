"""Comprehensive tests for dataloader classes and functions."""
import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader

from utils.dataloaders import (
    SummarizedEngagementDataset,
    SequenceEngagementDataset,
    create_data_loaders,
    MeanSummarizer,
)


# =============================================================================
# Fixtures for test data
# =============================================================================

@pytest.fixture
def mock_embeddings_mmap():
    """Create a mock embeddings memmap with 100 posts, each with 64-dim embeddings."""
    np.random.seed(42)
    return np.random.randn(100, 64).astype(np.float32)


@pytest.fixture
def mock_target_posts_df():
    """Create a mock target_posts DataFrame with train/val splits."""
    return pl.DataFrame({
        "split": ["train"] * 8 + ["val"] * 4,
        "target_did": [f"user{i}" for i in range(12)],
        "like_uri": [f"post{i}" for i in range(12)],
        "neg_uri": [f"neg{i}" for i in range(12)],
        "like_emb_idx": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "neg_emb_idx": [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31],
    })


@pytest.fixture
def mock_history_df():
    """Create a mock history DataFrame with variable-length user histories."""
    # Create some users with different history lengths
    return pl.DataFrame({
        "target_did": [f"user{i}" for i in range(12)],
        "like_uri": [f"post{i}" for i in range(12)],
        "prior_emb_indices": [
            [40, 41, 42],  # user0: 3 items
            [43, 44],      # user1: 2 items
            [],            # user2: empty history
            [45, 46, 47, 48, 49],  # user3: 5 items
            [50],          # user4: 1 item
            [51, 52, 53, 54],  # user5: 4 items
            [55, 56],      # user6: 2 items
            [57],          # user7: 1 item
            [58, 59, 60],  # user8: 3 items (val split starts here)
            [],            # user9: empty history
            [61, 62, 63, 64, 65, 66],  # user10: 6 items
            [67, 68],      # user11: 2 items
        ],
    })


# =============================================================================
# SummarizedEngagementDataset Tests
# =============================================================================

def test_summarized_dataset_initialization(mock_embeddings_mmap, mock_target_posts_df, mock_history_df):
    """Test SummarizedEngagementDataset initializes correctly."""
    summarizer = MeanSummarizer()
    dataset = SummarizedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="train",
        summarizer=summarizer,
        embed_dim=64,
    )
    
    # Should have 2 samples per target post (positive + negative)
    assert len(dataset) == 8 * 2  # 8 training posts
    assert dataset.embed_dim == 64


def test_summarized_dataset_item_shape(mock_embeddings_mmap, mock_target_posts_df, mock_history_df):
    """Test SummarizedEngagementDataset returns correctly shaped items."""
    summarizer = MeanSummarizer()
    dataset = SummarizedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="train",
        summarizer=summarizer,
        embed_dim=64,
    )
    
    # Test positive sample (even index)
    pos_sample = dataset[0]
    assert "features" in pos_sample
    assert "label" in pos_sample
    assert "user_id" in pos_sample
    assert "post_id" in pos_sample
    
    # Features should be [user_summary || post_embedding] = 2 * embed_dim
    assert pos_sample["features"].shape == (128,)
    assert pos_sample["features"].dtype == torch.float32
    assert pos_sample["label"].item() == 1.0
    assert pos_sample["user_id"] == "user0"
    assert pos_sample["post_id"] == "post0"
    
    # Test negative sample (odd index)
    neg_sample = dataset[1]
    assert neg_sample["features"].shape == (128,)
    assert neg_sample["label"].item() == 0.0
    assert neg_sample["user_id"] == "user0"
    assert neg_sample["post_id"] == "neg0"


def test_summarized_dataset_indexing_pattern(mock_embeddings_mmap, mock_target_posts_df, mock_history_df):
    """Test SummarizedEngagementDataset indexing follows positive/negative pattern."""
    summarizer = MeanSummarizer()
    dataset = SummarizedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="train",
        summarizer=summarizer,
        embed_dim=64,
    )
    
    # Check first few samples follow the pattern
    for i in range(0, min(8, len(dataset)), 2):
        pos = dataset[i]
        neg = dataset[i + 1]
        
        # Positive and negative samples for same user should have same user_id
        assert pos["user_id"] == neg["user_id"]
        
        # Labels should be correct
        assert pos["label"].item() == 1.0
        assert neg["label"].item() == 0.0
        
        # Post IDs should differ
        assert pos["post_id"] != neg["post_id"]


def test_summarized_dataset_empty_history(mock_embeddings_mmap, mock_target_posts_df, mock_history_df):
    """Test SummarizedEngagementDataset handles users with empty history."""
    summarizer = MeanSummarizer()
    dataset = SummarizedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="train",
        summarizer=summarizer,
        embed_dim=64,
    )
    
    # Find a sample for user2 (which has empty history)
    # user2 is at index 2, so samples are at indices 4 and 5
    sample = dataset[4]
    
    # User summary should be zero vector (first 64 dims)
    user_summary = sample["features"][:64]
    assert torch.allclose(user_summary, torch.zeros(64)), "Empty history should produce zero user summary"


def test_summarized_dataset_split_filtering(mock_embeddings_mmap, mock_target_posts_df, mock_history_df):
    """Test SummarizedEngagementDataset correctly filters by split."""
    summarizer = MeanSummarizer()
    
    train_dataset = SummarizedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="train",
        summarizer=summarizer,
        embed_dim=64,
    )
    
    val_dataset = SummarizedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="val",
        summarizer=summarizer,
        embed_dim=64,
    )
    
    # Train has 8 posts, val has 4 posts
    assert len(train_dataset) == 16  # 8 * 2
    assert len(val_dataset) == 8     # 4 * 2


# =============================================================================
# SequenceEngagementDataset Tests
# =============================================================================

def test_sequence_dataset_initialization(mock_embeddings_mmap, mock_target_posts_df, mock_history_df):
    """Test SequenceEngagementDataset initializes correctly."""
    dataset = SequenceEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="train",
        max_history_len=10,
        embed_dim=64,
    )
    
    assert len(dataset) == 8 * 2  # 8 training posts, 2 samples each
    assert dataset.max_history_len == 10
    assert dataset.embed_dim == 64


def test_sequence_dataset_item_shape(mock_embeddings_mmap, mock_target_posts_df, mock_history_df):
    """Test SequenceEngagementDataset returns correctly shaped items."""
    max_history_len = 10
    dataset = SequenceEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="train",
        max_history_len=max_history_len,
        embed_dim=64,
    )
    
    sample = dataset[0]
    
    assert "history_embeddings" in sample
    assert "history_mask" in sample
    assert "target_post_embedding" in sample
    assert "label" in sample
    assert "user_id" in sample
    assert "post_id" in sample
    
    # Check shapes
    assert sample["history_embeddings"].shape == (max_history_len, 64)
    assert sample["history_mask"].shape == (max_history_len,)
    assert sample["target_post_embedding"].shape == (64,)
    assert sample["label"].shape == ()  # scalar
    
    # Check types
    assert sample["history_embeddings"].dtype == torch.float32
    assert sample["history_mask"].dtype == torch.bool
    assert sample["target_post_embedding"].dtype == torch.float32
    assert sample["label"].dtype == torch.float32


def test_sequence_dataset_padding_and_mask(mock_embeddings_mmap, mock_target_posts_df, mock_history_df):
    """Test SequenceEngagementDataset correctly pads and masks sequences."""
    max_history_len = 10
    dataset = SequenceEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="train",
        max_history_len=max_history_len,
        embed_dim=64,
    )
    
    # user0 has 3 history items
    sample = dataset[0]
    mask = sample["history_mask"]
    
    # First 3 positions should be True, rest False
    assert mask[:3].all(), "First 3 positions should be valid"
    assert not mask[3:].any(), "Remaining positions should be padding"
    
    # Padded positions should be zero
    padded_embeddings = sample["history_embeddings"][3:]
    assert torch.allclose(padded_embeddings, torch.zeros_like(padded_embeddings))


def test_sequence_dataset_empty_history(mock_embeddings_mmap, mock_target_posts_df, mock_history_df):
    """Test SequenceEngagementDataset handles empty history correctly."""
    max_history_len = 10
    dataset = SequenceEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="train",
        max_history_len=max_history_len,
        embed_dim=64,
    )
    
    # user2 has empty history (index 4 and 5)
    sample = dataset[4]
    
    # All mask should be False
    assert not sample["history_mask"].any(), "Empty history should have all-False mask"
    
    # All embeddings should be zero
    assert torch.allclose(sample["history_embeddings"], torch.zeros_like(sample["history_embeddings"]))


def test_sequence_dataset_truncation(mock_embeddings_mmap, mock_target_posts_df, mock_history_df):
    """Test SequenceEngagementDataset truncates long sequences."""
    max_history_len = 3  # Shorter than some histories
    dataset = SequenceEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="train",
        max_history_len=max_history_len,
        embed_dim=64,
    )
    
    # user3 has 5 history items, but should be truncated to 3
    sample = dataset[6]  # user3 is at index 3, samples at 6 and 7
    mask = sample["history_mask"]
    
    # All positions should be valid (truncated)
    assert mask.all(), "Truncated sequence should have all valid positions"


def test_sequence_dataset_positive_negative_labeling(mock_embeddings_mmap, mock_target_posts_df, mock_history_df):
    """Test SequenceEngagementDataset correctly labels positive and negative samples."""
    dataset = SequenceEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="train",
        max_history_len=10,
        embed_dim=64,
    )
    
    # Check pattern for several users
    for i in range(0, 8, 2):
        pos = dataset[i]
        neg = dataset[i + 1]
        
        assert pos["label"].item() == 1.0, f"Even index {i} should be positive"
        assert neg["label"].item() == 0.0, f"Odd index {i+1} should be negative"
        
        # Same user, same history
        assert pos["user_id"] == neg["user_id"]
        assert torch.equal(pos["history_embeddings"], neg["history_embeddings"])
        assert torch.equal(pos["history_mask"], neg["history_mask"])
        
        # Different target posts
        assert not torch.equal(pos["target_post_embedding"], neg["target_post_embedding"])


# =============================================================================
# create_data_loaders Tests
# =============================================================================

def test_create_data_loaders_returns_correct_loaders(mock_embeddings_mmap, mock_target_posts_df, mock_history_df):
    """Test create_data_loaders creates loaders with correct configuration."""
    summarizer = MeanSummarizer()
    
    train_dataset = SummarizedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="train",
        summarizer=summarizer,
        embed_dim=64,
    )
    
    val_dataset = SummarizedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="val",
        summarizer=summarizer,
        embed_dim=64,
    )
    
    train_loader, val_loader, holdout_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=4,
        num_workers=0,  # No multiprocessing for tests
        pin_memory=False,
    )
    
    # Check loaders are DataLoader instances
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert holdout_loader is None  # No holdout dataset provided
    
    # Check batch sizes
    assert train_loader.batch_size == 4
    assert val_loader.batch_size == 4


def test_create_data_loaders_with_collate_fn(mock_embeddings_mmap, mock_target_posts_df, mock_history_df):
    """Test create_data_loaders works with custom collate function."""
    train_dataset = SequenceEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="train",
        max_history_len=10,
        embed_dim=64,
    )
    
    val_dataset = SequenceEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="val",
        max_history_len=10,
        embed_dim=64,
    )
    
    train_loader, val_loader, _ = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )
    
    # Get a batch and verify it's properly collated
    batch = next(iter(train_loader))
    
    assert "history_embeddings" in batch
    assert "history_mask" in batch
    assert batch["history_embeddings"].ndim == 3  # [batch, seq, dim]
    assert batch["history_mask"].ndim == 2  # [batch, seq]


def test_create_data_loaders_with_holdout(mock_embeddings_mmap, mock_target_posts_df, mock_history_df):
    """Test create_data_loaders creates holdout loader when provided."""
    summarizer = MeanSummarizer()
    
    train_dataset = SummarizedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="train",
        summarizer=summarizer,
        embed_dim=64,
    )
    
    val_dataset = SummarizedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="val",
        summarizer=summarizer,
        embed_dim=64,
    )
    
    holdout_dataset = val_dataset  # Reuse val for testing
    
    train_loader, val_loader, holdout_loader = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        holdout_dataset=holdout_dataset,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )
    
    assert holdout_loader is not None
    assert isinstance(holdout_loader, DataLoader)
    assert holdout_loader.batch_size == 4


def test_create_data_loaders_iteration(mock_embeddings_mmap, mock_target_posts_df, mock_history_df):
    """Test that created data loaders can be iterated."""
    summarizer = MeanSummarizer()
    
    train_dataset = SummarizedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="train",
        summarizer=summarizer,
        embed_dim=64,
    )
    
    val_dataset = SummarizedEngagementDataset(
        embeddings_mmap=mock_embeddings_mmap,
        target_posts_df=mock_target_posts_df,
        history_df=mock_history_df,
        split="val",
        summarizer=summarizer,
        embed_dim=64,
    )
    
    train_loader, val_loader, _ = create_data_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=4,
        num_workers=0,
        pin_memory=False,
    )
    
    # Should be able to iterate through loaders
    train_batches = list(train_loader)
    assert len(train_batches) > 0
    
    val_batches = list(val_loader)
    assert len(val_batches) > 0
    
    # Check batch contents
    batch = train_batches[0]
    assert "features" in batch
    assert "label" in batch
