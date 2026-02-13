"""Tests for UserSummarizer classes in dataloaders."""
import numpy as np
import pytest

from utils.dataloaders import MeanSummarizer, EMASummarizer, LinearRecencySummarizer


def test_mean_summarizer_empty_input():
    """MeanSummarizer should return zero vector for empty input."""
    summarizer = MeanSummarizer()
    empty_embeddings = np.zeros((0, 128), dtype=np.float32)
    result = summarizer.summarize(empty_embeddings)
    
    assert result.shape == (128,)
    assert result.dtype == np.float32
    assert np.allclose(result, 0.0)


def test_mean_summarizer_non_empty_input():
    """MeanSummarizer should compute mean for non-empty input."""
    summarizer = MeanSummarizer()
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    result = summarizer.summarize(embeddings)
    
    expected = np.array([3.0, 4.0], dtype=np.float32)
    assert result.shape == (2,)
    assert np.allclose(result, expected)


def test_ema_summarizer_empty_input():
    """EMASummarizer should return zero vector for empty input."""
    summarizer = EMASummarizer(alpha=0.1)
    empty_embeddings = np.zeros((0, 64), dtype=np.float32)
    result = summarizer.summarize(empty_embeddings)
    
    assert result.shape == (64,)
    assert result.dtype == np.float32
    assert np.allclose(result, 0.0)


def test_ema_summarizer_non_empty_input():
    """EMASummarizer should compute weighted average for non-empty input."""
    summarizer = EMASummarizer(alpha=0.5)
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    result = summarizer.summarize(embeddings)
    
    # Verify more weight on first (most recent) embedding
    assert result.shape == (2,)
    assert result[0] > result[1]


def test_linear_recency_summarizer_empty_input():
    """LinearRecencySummarizer should return zero vector for empty input."""
    summarizer = LinearRecencySummarizer()
    empty_embeddings = np.zeros((0, 256), dtype=np.float32)
    result = summarizer.summarize(empty_embeddings)
    
    assert result.shape == (256,)
    assert result.dtype == np.float32
    assert np.allclose(result, 0.0)


def test_linear_recency_summarizer_non_empty_input():
    """LinearRecencySummarizer should compute linearly weighted average."""
    summarizer = LinearRecencySummarizer()
    embeddings = np.array([[3.0, 0.0], [2.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    result = summarizer.summarize(embeddings)
    
    # Weights: [3, 2, 1] normalized = [0.5, 0.333, 0.167]
    # Applied to embeddings: [3.0, 0.0]*0.5 + [2.0, 0.0]*0.333 + [1.0, 0.0]*0.167
    # Result first dim: 3.0*0.5 + 2.0*0.333 + 1.0*0.167 ≈ 2.333
    expected = np.array([2.333, 0.0], dtype=np.float32)
    assert result.shape == (2,)
    assert np.allclose(result, expected, rtol=1e-3)


def test_all_summarizers_consistent_empty_behavior():
    """All summarizers should return zero vectors of correct shape for empty input."""
    summarizers = [
        MeanSummarizer(),
        EMASummarizer(alpha=0.1),
        LinearRecencySummarizer(),
    ]
    
    for embed_dim in [32, 64, 128, 256]:
        empty_embeddings = np.zeros((0, embed_dim), dtype=np.float32)
        for summarizer in summarizers:
            result = summarizer.summarize(empty_embeddings)
            assert result.shape == (embed_dim,), f"{type(summarizer).__name__} failed for dim {embed_dim}"
            assert result.dtype == np.float32
            assert np.allclose(result, 0.0)
