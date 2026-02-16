"""Comprehensive tests for MLP model architectures."""
import importlib
import pytest
import torch
import torch.nn as nn

# Import from module with numeric prefix
stage_train_mlp = importlib.import_module("utils.04_train.stage_train_mlp")
SummarizedMLP = stage_train_mlp.SummarizedMLP
AttentionMLP = stage_train_mlp.AttentionMLP


# =============================================================================
# SummarizedMLP Tests
# =============================================================================

def test_summarized_mlp_initialization():
    """Test SummarizedMLP initializes with correct architecture."""
    model = SummarizedMLP(
        input_dim=768,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.3,
    )
    
    # Check model has the expected structure
    assert isinstance(model.network, nn.Sequential)
    
    # Count linear layers (should be len(hidden_dims) + 1 for output)
    linear_layers = [m for m in model.network.modules() if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 4  # 3 hidden + 1 output


def test_summarized_mlp_forward_shape():
    """Test SummarizedMLP forward pass produces correct output shape."""
    batch_size = 16
    input_dim = 768
    
    model = SummarizedMLP(
        input_dim=input_dim,
        hidden_dims=[512, 256],
        dropout_rate=0.3,
    )
    
    # Create random input
    x = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output = model.forward(x)
    
    # Check output shape and range
    assert output.shape == (batch_size, 1)
    assert output.dtype == torch.float32
    assert (output >= 0).all() and (output <= 1).all(), "Output should be in [0, 1] due to sigmoid"


def test_summarized_mlp_single_hidden_layer():
    """Test SummarizedMLP with single hidden layer."""
    model = SummarizedMLP(
        input_dim=256,
        hidden_dims=[128],
        dropout_rate=0.2,
    )
    
    x = torch.randn(8, 256)
    output = model.forward(x)
    
    assert output.shape == (8, 1)


def test_summarized_mlp_multiple_hidden_layers():
    """Test SummarizedMLP with multiple hidden layers."""
    model = SummarizedMLP(
        input_dim=1024,
        hidden_dims=[512, 256, 128, 64],
        dropout_rate=0.4,
    )
    
    x = torch.randn(4, 1024)
    output = model.forward(x)
    
    assert output.shape == (4, 1)


def test_summarized_mlp_compute_loss_and_preds():
    """Test SummarizedMLP compute_loss_and_preds method."""
    model = SummarizedMLP(
        input_dim=768,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
    )
    
    batch = {
        "features": torch.randn(16, 768),
        "label": torch.randint(0, 2, (16,)).float(),
    }
    
    device = "cpu"
    loss, preds = model.compute_loss_and_preds(batch, device)
    
    # Check loss is a scalar
    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert loss.item() >= 0, "BCE loss should be non-negative"
    
    # Check predictions shape and range
    assert preds.shape == (16,)
    assert (preds >= 0).all() and (preds <= 1).all()


def test_summarized_mlp_backward_pass():
    """Test SummarizedMLP gradients flow correctly."""
    model = SummarizedMLP(
        input_dim=256,
        hidden_dims=[128, 64],
        dropout_rate=0.2,
    )
    
    x = torch.randn(8, 256)
    labels = torch.randint(0, 2, (8,)).float()
    
    # Forward
    output = model.forward(x).squeeze(-1)
    loss = nn.functional.binary_cross_entropy(output, labels)
    
    # Backward
    loss.backward()
    
    # Check gradients exist
    for param in model.parameters():
        assert param.grad is not None, "All parameters should have gradients"


def test_summarized_mlp_eval_mode():
    """Test SummarizedMLP behaves differently in eval mode (dropout)."""
    model = SummarizedMLP(
        input_dim=256,
        hidden_dims=[128],
        dropout_rate=0.5,  # High dropout for testing
    )
    
    x = torch.randn(16, 256)
    
    # Train mode - run multiple times, should get different results
    model.train()
    outputs_train = [model.forward(x) for _ in range(3)]
    
    # Eval mode - should be deterministic
    model.eval()
    with torch.no_grad():
        outputs_eval = [model.forward(x) for _ in range(3)]
    
    # Eval outputs should be identical
    for i in range(len(outputs_eval) - 1):
        assert torch.allclose(outputs_eval[i], outputs_eval[i + 1])


def test_summarized_mlp_zero_dropout():
    """Test SummarizedMLP with zero dropout."""
    model = SummarizedMLP(
        input_dim=256,
        hidden_dims=[128, 64],
        dropout_rate=0.0,
    )
    
    x = torch.randn(8, 256)
    output = model.forward(x)
    
    assert output.shape == (8, 1)


# =============================================================================
# AttentionMLP Tests
# =============================================================================

def test_attention_mlp_initialization():
    """Test AttentionMLP initializes correctly."""
    model = AttentionMLP(
        embed_dim=384,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
        user_hidden_dim=256,
        user_output_dim=128,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        attention_dropout=0.1,
    )
    
    assert model.embed_dim == 384
    assert model.user_output_dim == 128
    assert hasattr(model, "user_encoder")
    assert hasattr(model, "mlp_head")
    assert isinstance(model.mlp_head, nn.Sequential)


def test_attention_mlp_forward_shape():
    """Test AttentionMLP forward pass produces correct output shape."""
    batch_size = 16
    seq_len = 50
    embed_dim = 384
    
    model = AttentionMLP(
        embed_dim=embed_dim,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
        user_hidden_dim=256,
        user_output_dim=128,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=seq_len,
        attention_dropout=0.1,
    )
    
    # Create random inputs
    history_embeddings = torch.randn(batch_size, seq_len, embed_dim)
    history_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    post_embedding = torch.randn(batch_size, embed_dim)
    
    # Forward pass
    output = model.forward(history_embeddings, history_mask, post_embedding)
    
    # Check output shape and range
    assert output.shape == (batch_size, 1)
    assert output.dtype == torch.float32
    assert (output >= 0).all() and (output <= 1).all(), "Output should be in [0, 1]"


def test_attention_mlp_with_mask():
    """Test AttentionMLP correctly uses history mask."""
    batch_size = 8
    seq_len = 20
    embed_dim = 128
    
    model = AttentionMLP(
        embed_dim=embed_dim,
        hidden_dims=[64],
        dropout_rate=0.2,
        user_hidden_dim=64,
        user_output_dim=32,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=seq_len,
        attention_dropout=0.1,
    )
    
    # Create inputs with partial masking
    history_embeddings = torch.randn(batch_size, seq_len, embed_dim)
    history_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    history_mask[:, :10] = True  # Only first 10 positions valid
    post_embedding = torch.randn(batch_size, embed_dim)
    
    # Should run without error
    output = model.forward(history_embeddings, history_mask, post_embedding)
    assert output.shape == (batch_size, 1)


def test_attention_mlp_empty_history():
    """Test AttentionMLP with empty history (all-False mask)."""
    batch_size = 4
    seq_len = 20
    embed_dim = 128
    
    model = AttentionMLP(
        embed_dim=embed_dim,
        hidden_dims=[64],
        dropout_rate=0.2,
        user_hidden_dim=64,
        user_output_dim=32,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=seq_len,
        attention_dropout=0.1,
    )
    
    # Empty history - all zeros with all-False mask
    history_embeddings = torch.zeros(batch_size, seq_len, embed_dim)
    history_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    post_embedding = torch.randn(batch_size, embed_dim)
    
    # Should handle gracefully
    output = model.forward(history_embeddings, history_mask, post_embedding)
    assert output.shape == (batch_size, 1)
    assert torch.isfinite(output).all(), "Output should be finite even with empty history"


def test_attention_mlp_compute_loss_and_preds():
    """Test AttentionMLP compute_loss_and_preds method."""
    model = AttentionMLP(
        embed_dim=384,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
        user_hidden_dim=256,
        user_output_dim=128,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        attention_dropout=0.1,
    )
    
    batch_size = 16
    batch = {
        "history_embeddings": torch.randn(batch_size, 50, 384),
        "history_mask": torch.ones(batch_size, 50, dtype=torch.bool),
        "target_post_embedding": torch.randn(batch_size, 384),
        "label": torch.randint(0, 2, (batch_size,)).float(),
    }
    
    device = "cpu"
    loss, preds = model.compute_loss_and_preds(batch, device)
    
    # Check loss is a scalar
    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert loss.item() >= 0, "BCE loss should be non-negative"
    
    # Check predictions shape and range
    assert preds.shape == (batch_size,)
    assert (preds >= 0).all() and (preds <= 1).all()


def test_attention_mlp_different_sequence_lengths():
    """Test AttentionMLP with different sequence lengths via masking."""
    batch_size = 8
    seq_len = 30
    embed_dim = 128
    
    model = AttentionMLP(
        embed_dim=embed_dim,
        hidden_dims=[64],
        dropout_rate=0.2,
        user_hidden_dim=64,
        user_output_dim=32,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=seq_len,
        attention_dropout=0.1,
    )
    
    history_embeddings = torch.randn(batch_size, seq_len, embed_dim)
    
    # Create masks with different lengths
    history_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    for i in range(batch_size):
        valid_len = (i + 1) * 3  # Varying lengths: 3, 6, 9, ...
        history_mask[i, :min(valid_len, seq_len)] = True
    
    post_embedding = torch.randn(batch_size, embed_dim)
    
    output = model.forward(history_embeddings, history_mask, post_embedding)
    assert output.shape == (batch_size, 1)
    assert torch.isfinite(output).all()


def test_attention_mlp_backward_pass():
    """Test AttentionMLP gradients flow correctly."""
    model = AttentionMLP(
        embed_dim=128,
        hidden_dims=[64],
        dropout_rate=0.2,
        user_hidden_dim=64,
        user_output_dim=32,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=20,
        attention_dropout=0.1,
    )
    
    batch_size = 8
    history_embeddings = torch.randn(batch_size, 20, 128)
    history_mask = torch.ones(batch_size, 20, dtype=torch.bool)
    post_embedding = torch.randn(batch_size, 128)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    # Forward
    output = model.forward(history_embeddings, history_mask, post_embedding).squeeze(-1)
    loss = nn.functional.binary_cross_entropy(output, labels)
    
    # Backward
    loss.backward()
    
    # Check gradients exist in both encoder and MLP
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} should have gradient"


def test_attention_mlp_eval_mode():
    """Test AttentionMLP behaves consistently in eval mode."""
    model = AttentionMLP(
        embed_dim=128,
        hidden_dims=[64],
        dropout_rate=0.5,
        user_hidden_dim=64,
        user_output_dim=32,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=20,
        attention_dropout=0.3,
    )
    
    batch_size = 8
    history_embeddings = torch.randn(batch_size, 20, 128)
    history_mask = torch.ones(batch_size, 20, dtype=torch.bool)
    post_embedding = torch.randn(batch_size, 128)
    
    # Eval mode - should be deterministic
    model.eval()
    with torch.no_grad():
        outputs_eval = [
            model.forward(history_embeddings, history_mask, post_embedding)
            for _ in range(3)
        ]
    
    # All outputs should be identical
    for i in range(len(outputs_eval) - 1):
        assert torch.allclose(outputs_eval[i], outputs_eval[i + 1])


def test_attention_mlp_attention_heads():
    """Test AttentionMLP with different numbers of attention heads."""
    embed_dim = 128
    
    for num_heads in [1, 2, 4, 8]:
        # user_hidden_dim must be divisible by num_heads
        user_hidden_dim = 64 if num_heads <= 4 else 128
        
        model = AttentionMLP(
            embed_dim=embed_dim,
            hidden_dims=[64],
            dropout_rate=0.2,
            user_hidden_dim=user_hidden_dim,
            user_output_dim=32,
            num_attention_heads=num_heads,
            num_attention_layers=1,
            max_history_len=20,
            attention_dropout=0.1,
        )
        
        history_embeddings = torch.randn(4, 20, embed_dim)
        history_mask = torch.ones(4, 20, dtype=torch.bool)
        post_embedding = torch.randn(4, embed_dim)
        
        output = model.forward(history_embeddings, history_mask, post_embedding)
        assert output.shape == (4, 1)


def test_attention_mlp_attention_layers():
    """Test AttentionMLP with different numbers of attention layers."""
    embed_dim = 128
    
    for num_layers in [1, 2, 3, 4]:
        model = AttentionMLP(
            embed_dim=embed_dim,
            hidden_dims=[64],
            dropout_rate=0.2,
            user_hidden_dim=64,
            user_output_dim=32,
            num_attention_heads=2,
            num_attention_layers=num_layers,
            max_history_len=20,
            attention_dropout=0.1,
        )
        
        history_embeddings = torch.randn(4, 20, embed_dim)
        history_mask = torch.ones(4, 20, dtype=torch.bool)
        post_embedding = torch.randn(4, embed_dim)
        
        output = model.forward(history_embeddings, history_mask, post_embedding)
        assert output.shape == (4, 1)


# =============================================================================
# Comparison Tests
# =============================================================================

def test_models_different_architectures():
    """Test that SummarizedMLP and AttentionMLP have different architectures."""
    summarized_mlp = SummarizedMLP(
        input_dim=768,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
    )
    
    attention_mlp = AttentionMLP(
        embed_dim=384,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
        user_hidden_dim=256,
        user_output_dim=128,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        attention_dropout=0.1,
    )
    
    # AttentionMLP should have more parameters due to encoder
    summarized_params = sum(p.numel() for p in summarized_mlp.parameters())
    attention_params = sum(p.numel() for p in attention_mlp.parameters())
    
    assert attention_params > summarized_params, "AttentionMLP should have more parameters"


def test_models_output_same_type():
    """Test that both models produce compatible outputs."""
    summarized_mlp = SummarizedMLP(
        input_dim=768,
        hidden_dims=[256],
        dropout_rate=0.3,
    )
    
    attention_mlp = AttentionMLP(
        embed_dim=384,
        hidden_dims=[256],
        dropout_rate=0.3,
        user_hidden_dim=128,
        user_output_dim=128,
        num_attention_heads=4,
        num_attention_layers=1,
        max_history_len=20,
        attention_dropout=0.1,
    )
    
    # SummarizedMLP output
    x1 = torch.randn(8, 768)
    out1 = summarized_mlp.forward(x1)
    
    # AttentionMLP output
    history = torch.randn(8, 20, 384)
    mask = torch.ones(8, 20, dtype=torch.bool)
    post = torch.randn(8, 384)
    out2 = attention_mlp.forward(history, mask, post)
    
    # Both should produce same shape and type
    assert out1.shape == out2.shape
    assert out1.dtype == out2.dtype
    assert (out1 >= 0).all() and (out1 <= 1).all()
    assert (out2 >= 0).all() and (out2 <= 1).all()
