"""Comprehensive tests for MLP model architectures."""
import importlib
import pytest
import torch
import torch.nn as nn

# Import from module with numeric prefix
stage_train_mlp = importlib.import_module("utils.04_train.stage_train_mlp")
MLPModel = stage_train_mlp.MLPModel
CrossAttentionPoolingEncoder = stage_train_mlp.CrossAttentionPoolingEncoder


# =============================================================================
# MLPModel (summarized) Tests
# =============================================================================

def test_summarized_mlp_initialization():
    """Test summarized MLPModel initializes with correct architecture."""
    model = MLPModel(
        post_embedding_dim=384,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.3,
        user_hidden_dim=256,
        user_output_dim=384,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        attention_dropout=0.1,
        user_encoder_type="summarized",
    )
    
    # Check model has the expected structure
    assert isinstance(model.mlp_head, nn.Sequential)
    
    # Count linear layers (should be len(hidden_dims) + 1 for output)
    linear_layers = [m for m in model.mlp_head.modules() if isinstance(m, nn.Linear)]
    assert len(linear_layers) == 4  # 3 hidden + 1 output


def test_summarized_mlp_forward_shape():
    """Test summarized MLPModel forward pass produces correct output shape."""
    batch_size = 16
    embed_dim = 384
    
    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[512, 256],
        dropout_rate=0.3,
        user_hidden_dim=256,
        user_output_dim=embed_dim,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        attention_dropout=0.1,
        user_encoder_type="summarized",
    )
    
    history_embeddings = torch.randn(batch_size, 1, embed_dim)
    history_mask = torch.ones(batch_size, 1, dtype=torch.bool)
    post_embedding = torch.randn(batch_size, embed_dim)
    
    # Forward pass
    output = model.forward(history_embeddings, history_mask, post_embedding)
    
    # Check output shape and range
    assert output.shape == (batch_size, 1)
    assert output.dtype == torch.float32
    assert (output >= 0).all() and (output <= 1).all(), "Output should be in [0, 1] due to sigmoid"


def test_summarized_mlp_single_hidden_layer():
    """Test summarized MLPModel with single hidden layer."""
    embed_dim = 128
    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[128],
        dropout_rate=0.2,
        user_hidden_dim=64,
        user_output_dim=embed_dim,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=20,
        attention_dropout=0.1,
        user_encoder_type="summarized",
    )
    
    history_embeddings = torch.randn(8, 1, embed_dim)
    history_mask = torch.ones(8, 1, dtype=torch.bool)
    post_embedding = torch.randn(8, embed_dim)
    output = model.forward(history_embeddings, history_mask, post_embedding)
    
    assert output.shape == (8, 1)


def test_summarized_mlp_multiple_hidden_layers():
    """Test summarized MLPModel with multiple hidden layers."""
    embed_dim = 512
    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[512, 256, 128, 64],
        dropout_rate=0.4,
        user_hidden_dim=256,
        user_output_dim=embed_dim,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        attention_dropout=0.1,
        user_encoder_type="summarized",
    )
    
    history_embeddings = torch.randn(4, 1, embed_dim)
    history_mask = torch.ones(4, 1, dtype=torch.bool)
    post_embedding = torch.randn(4, embed_dim)
    output = model.forward(history_embeddings, history_mask, post_embedding)
    
    assert output.shape == (4, 1)


def test_summarized_mlp_compute_loss_and_preds():
    """Test summarized MLPModel compute_loss_and_preds method (features path)."""
    embed_dim = 384
    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
        user_hidden_dim=256,
        user_output_dim=embed_dim,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        attention_dropout=0.1,
        user_encoder_type="summarized",
    )
    
    batch = {
        "features": torch.randn(16, 2 * embed_dim),
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


def test_summarized_mlp_compute_loss_and_preds_empty_history_uses_empty_embedding():
    """Empty-history summarized batches should route through the cold-start embedding."""
    torch.manual_seed(0)
    embed_dim = 16
    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[],
        dropout_rate=0.0,
        user_hidden_dim=32,
        user_output_dim=embed_dim,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=10,
        attention_dropout=0.0,
        user_encoder_type="summarized",
    )
    model.eval()

    batch_size = 8
    user_summary = torch.zeros(batch_size, embed_dim)
    post_embedding = torch.randn(batch_size, embed_dim)
    features = torch.cat([user_summary, post_embedding], dim=1)
    batch = {"features": features, "label": torch.randint(0, 2, (batch_size,)).float()}

    with torch.no_grad():
        model.user_encoder.empty_user_embedding.fill_(0.1)
    _, preds1 = model.compute_loss_and_preds(batch, device="cpu")

    with torch.no_grad():
        model.user_encoder.empty_user_embedding.fill_(0.2)
    _, preds2 = model.compute_loss_and_preds(batch, device="cpu")

    assert not torch.allclose(preds1, preds2), "Predictions should depend on the cold-start embedding for empty histories"


def test_summarized_mlp_compute_loss_and_preds_non_empty_history_ignores_empty_embedding():
    """Non-empty summarized batches should ignore the cold-start embedding."""
    torch.manual_seed(0)
    embed_dim = 16
    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[],
        dropout_rate=0.0,
        user_hidden_dim=32,
        user_output_dim=embed_dim,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=10,
        attention_dropout=0.0,
        user_encoder_type="summarized",
    )
    model.eval()

    batch_size = 8
    user_summary = torch.ones(batch_size, embed_dim)
    post_embedding = torch.randn(batch_size, embed_dim)
    features = torch.cat([user_summary, post_embedding], dim=1)
    batch = {"features": features, "label": torch.randint(0, 2, (batch_size,)).float()}

    with torch.no_grad():
        model.user_encoder.empty_user_embedding.fill_(0.1)
    _, preds1 = model.compute_loss_and_preds(batch, device="cpu")

    with torch.no_grad():
        model.user_encoder.empty_user_embedding.fill_(0.2)
    _, preds2 = model.compute_loss_and_preds(batch, device="cpu")

    assert torch.allclose(preds1, preds2), "Predictions should not depend on the cold-start embedding when history exists"


def test_summarized_mlp_backward_pass():
    """Test summarized MLPModel gradients flow correctly."""
    embed_dim = 128
    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[128, 64],
        dropout_rate=0.2,
        user_hidden_dim=64,
        user_output_dim=embed_dim,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=20,
        attention_dropout=0.1,
        user_encoder_type="summarized",
    )
    
    history_embeddings = torch.randn(8, 1, embed_dim)
    history_mask = torch.ones(8, 1, dtype=torch.bool)
    post_embedding = torch.randn(8, embed_dim)
    labels = torch.randint(0, 2, (8,)).float()
    
    # Forward
    output = model.forward(history_embeddings, history_mask, post_embedding).squeeze(-1)
    loss = nn.functional.binary_cross_entropy(output, labels)
    
    # Backward
    loss.backward()
    
    # Check gradients exist
    for param in model.parameters():
        assert param.grad is not None, "All parameters should have gradients"


def test_summarized_mlp_eval_mode():
    """Test summarized MLPModel behaves differently in eval mode (dropout)."""
    embed_dim = 128
    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[128],
        dropout_rate=0.5,  # High dropout for testing
        user_hidden_dim=64,
        user_output_dim=embed_dim,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=20,
        attention_dropout=0.1,
        user_encoder_type="summarized",
    )
    
    history_embeddings = torch.randn(16, 1, embed_dim)
    history_mask = torch.ones(16, 1, dtype=torch.bool)
    post_embedding = torch.randn(16, embed_dim)
    
    # Train mode - run multiple times, should get different results
    model.train()
    outputs_train = [model.forward(history_embeddings, history_mask, post_embedding) for _ in range(3)]
    
    # Eval mode - should be deterministic
    model.eval()
    with torch.no_grad():
        outputs_eval = [model.forward(history_embeddings, history_mask, post_embedding) for _ in range(3)]
    
    # Eval outputs should be identical
    for i in range(len(outputs_eval) - 1):
        assert torch.allclose(outputs_eval[i], outputs_eval[i + 1])


def test_summarized_mlp_zero_dropout():
    """Test summarized MLPModel with zero dropout."""
    embed_dim = 128
    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[128, 64],
        dropout_rate=0.0,
        user_hidden_dim=64,
        user_output_dim=embed_dim,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=20,
        attention_dropout=0.1,
        user_encoder_type="summarized",
    )
    
    history_embeddings = torch.randn(8, 1, embed_dim)
    history_mask = torch.ones(8, 1, dtype=torch.bool)
    post_embedding = torch.randn(8, embed_dim)
    output = model.forward(history_embeddings, history_mask, post_embedding)
    
    assert output.shape == (8, 1)

def test_summarized_mlp_torchscript():
    """Test summarized MLPModel can be TorchScript scripted (serving artifact)."""
    embed_dim = 128
    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[64],
        dropout_rate=0.1,
        user_hidden_dim=64,
        user_output_dim=embed_dim,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=20,
        attention_dropout=0.1,
        user_encoder_type="summarized",
    )

    scripted = torch.jit.script(model)
    history_embeddings = torch.randn(4, 1, embed_dim)
    history_mask = torch.ones(4, 1, dtype=torch.bool)
    post_embedding = torch.randn(4, embed_dim)
    out = scripted(history_embeddings, history_mask, post_embedding)
    assert out.shape == (4, 1)

def test_full_transformer_mlp_torchscript_no_sdpa():
    """TorchScript for full_transformer should avoid aten::scaled_dot_product_attention (Triton compatibility)."""
    embed_dim = 128
    seq_len = 10
    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[64],
        dropout_rate=0.1,
        user_hidden_dim=64,
        user_output_dim=32,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=seq_len,
        attention_dropout=0.1,
        user_encoder_type="full_transformer",
    )

    scripted = torch.jit.script(model)
    assert "scaled_dot_product_attention" not in scripted.code

    history_embeddings = torch.randn(2, seq_len, embed_dim)
    history_mask = torch.ones(2, seq_len, dtype=torch.bool)
    post_embedding = torch.randn(2, embed_dim)
    out = scripted(history_embeddings, history_mask, post_embedding)
    assert out.shape == (2, 1)


def test_cross_attention_mlp_initialization():
    """Test cross_attention MLPModel initializes with the efficient sequence encoder."""
    model = MLPModel(
        post_embedding_dim=384,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
        user_hidden_dim=256,
        user_output_dim=128,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        attention_dropout=0.1,
        user_encoder_type="cross_attention",
    )

    assert model.post_embedding_dim == 384
    assert model.user_output_dim == 128
    assert isinstance(model.user_encoder, CrossAttentionPoolingEncoder)
    assert isinstance(model.mlp_head, nn.Sequential)


def test_cross_attention_mlp_forward_shape():
    """Test cross_attention MLPModel forward pass produces correct output shape."""
    batch_size = 16
    seq_len = 50
    embed_dim = 384

    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
        user_hidden_dim=256,
        user_output_dim=128,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=seq_len,
        attention_dropout=0.1,
        user_encoder_type="cross_attention",
    )

    history_embeddings = torch.randn(batch_size, seq_len, embed_dim)
    history_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    post_embedding = torch.randn(batch_size, embed_dim)

    output = model.forward(history_embeddings, history_mask, post_embedding)

    assert output.shape == (batch_size, 1)
    assert output.dtype == torch.float32
    assert (output >= 0).all() and (output <= 1).all(), "Output should be in [0, 1]"


def test_cross_attention_mlp_compute_loss_and_preds():
    """Test cross_attention MLPModel compute_loss_and_preds uses sequence batches."""
    model = MLPModel(
        post_embedding_dim=384,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
        user_hidden_dim=256,
        user_output_dim=128,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        attention_dropout=0.1,
        user_encoder_type="cross_attention",
    )

    batch_size = 16
    batch = {
        "history_embeddings": torch.randn(batch_size, 50, 384),
        "history_mask": torch.ones(batch_size, 50, dtype=torch.bool),
        "target_post_embedding": torch.randn(batch_size, 384),
        "label": torch.randint(0, 2, (batch_size,)).float(),
    }

    loss, preds = model.compute_loss_and_preds(batch, "cpu")

    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert loss.item() >= 0
    assert preds.shape == (batch_size,)
    assert (preds >= 0).all() and (preds <= 1).all()


def test_cross_attention_mlp_torchscript():
    """Test cross_attention MLPModel can be TorchScript scripted for serving."""
    embed_dim = 128
    seq_len = 10
    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[64],
        dropout_rate=0.1,
        user_hidden_dim=64,
        user_output_dim=32,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=seq_len,
        attention_dropout=0.1,
        user_encoder_type="cross_attention",
    )

    scripted = torch.jit.script(model)
    assert "scaled_dot_product_attention" not in scripted.code

    history_embeddings = torch.randn(2, seq_len, embed_dim)
    history_mask = torch.ones(2, seq_len, dtype=torch.bool)
    post_embedding = torch.randn(2, embed_dim)
    out = scripted(history_embeddings, history_mask, post_embedding)
    assert out.shape == (2, 1)

# =============================================================================
# MLPModel (full_transformer) Tests
# =============================================================================

def test_attention_mlp_initialization():
    """Test full_transformer MLPModel initializes correctly."""
    model = MLPModel(
        post_embedding_dim=384,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
        user_hidden_dim=256,
        user_output_dim=128,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        attention_dropout=0.1,
        user_encoder_type="full_transformer",
    )
    
    assert model.post_embedding_dim == 384
    assert model.user_output_dim == 128
    assert hasattr(model, "user_encoder")
    assert hasattr(model, "mlp_head")
    assert isinstance(model.mlp_head, nn.Sequential)


def test_attention_mlp_forward_shape():
    """Test full_transformer MLPModel forward pass produces correct output shape."""
    batch_size = 16
    seq_len = 50
    embed_dim = 384
    
    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
        user_hidden_dim=256,
        user_output_dim=128,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=seq_len,
        attention_dropout=0.1,
        user_encoder_type="full_transformer",
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
    """Test full_transformer MLPModel correctly uses history mask."""
    batch_size = 8
    seq_len = 20
    embed_dim = 128
    
    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[64],
        dropout_rate=0.2,
        user_hidden_dim=64,
        user_output_dim=32,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=seq_len,
        attention_dropout=0.1,
        user_encoder_type="full_transformer",
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
    """Test full_transformer MLPModel with empty history (all-False mask)."""
    batch_size = 4
    seq_len = 20
    embed_dim = 128
    
    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[64],
        dropout_rate=0.2,
        user_hidden_dim=64,
        user_output_dim=32,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=seq_len,
        attention_dropout=0.1,
        user_encoder_type="full_transformer",
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
    """Test full_transformer MLPModel compute_loss_and_preds method."""
    model = MLPModel(
        post_embedding_dim=384,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
        user_hidden_dim=256,
        user_output_dim=128,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        attention_dropout=0.1,
        user_encoder_type="full_transformer",
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
    """Test full_transformer MLPModel with different sequence lengths via masking."""
    batch_size = 8
    seq_len = 30
    embed_dim = 128
    
    model = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[64],
        dropout_rate=0.2,
        user_hidden_dim=64,
        user_output_dim=32,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=seq_len,
        attention_dropout=0.1,
        user_encoder_type="full_transformer",
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
    """Test full_transformer MLPModel gradients flow correctly."""
    model = MLPModel(
        post_embedding_dim=128,
        hidden_dims=[64],
        dropout_rate=0.2,
        user_hidden_dim=64,
        user_output_dim=32,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=20,
        attention_dropout=0.1,
        user_encoder_type="full_transformer",
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
        # `empty_history_embedding` is only used when an example has an empty history
        # (all-masked). For fully non-empty batches, it's expected to have no grad.
        if "empty_history_embedding" in name:
            continue
        assert param.grad is not None, f"Parameter {name} should have gradient"


def test_attention_mlp_eval_mode():
    """Test full_transformer MLPModel behaves consistently in eval mode."""
    model = MLPModel(
        post_embedding_dim=128,
        hidden_dims=[64],
        dropout_rate=0.5,
        user_hidden_dim=64,
        user_output_dim=32,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=20,
        attention_dropout=0.3,
        user_encoder_type="full_transformer",
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
    """Test full_transformer MLPModel with different numbers of attention heads."""
    embed_dim = 128
    
    for num_heads in [1, 2, 4, 8]:
        # user_hidden_dim must be divisible by num_heads
        user_hidden_dim = 64 if num_heads <= 4 else 128
        
        model = MLPModel(
            post_embedding_dim=embed_dim,
            hidden_dims=[64],
            dropout_rate=0.2,
            user_hidden_dim=user_hidden_dim,
            user_output_dim=32,
            num_attention_heads=num_heads,
            num_attention_layers=1,
            max_history_len=20,
            attention_dropout=0.1,
            user_encoder_type="full_transformer",
        )
        
        history_embeddings = torch.randn(4, 20, embed_dim)
        history_mask = torch.ones(4, 20, dtype=torch.bool)
        post_embedding = torch.randn(4, embed_dim)
        
        output = model.forward(history_embeddings, history_mask, post_embedding)
        assert output.shape == (4, 1)


def test_attention_mlp_attention_layers():
    """Test full_transformer MLPModel with different numbers of attention layers."""
    embed_dim = 128
    
    for num_layers in [1, 2, 3, 4]:
        model = MLPModel(
            post_embedding_dim=embed_dim,
            hidden_dims=[64],
            dropout_rate=0.2,
            user_hidden_dim=64,
            user_output_dim=32,
            num_attention_heads=2,
            num_attention_layers=num_layers,
            max_history_len=20,
            attention_dropout=0.1,
            user_encoder_type="full_transformer",
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
    """Test that summarized and full_transformer variants have different parameter counts."""
    summarized_mlp = MLPModel(
        post_embedding_dim=384,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
        user_hidden_dim=256,
        user_output_dim=384,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        attention_dropout=0.1,
        user_encoder_type="summarized",
    )
    
    attention_mlp = MLPModel(
        post_embedding_dim=384,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
        user_hidden_dim=256,
        user_output_dim=128,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        attention_dropout=0.1,
        user_encoder_type="full_transformer",
    )
    
    # full_transformer variant should have more parameters due to encoder
    summarized_params = sum(p.numel() for p in summarized_mlp.parameters())
    attention_params = sum(p.numel() for p in attention_mlp.parameters())
    
    assert attention_params > summarized_params, "full_transformer variant should have more parameters"


def test_cross_attention_mlp_fewer_params_than_full_transformer():
    """Test cross_attention MLP has fewer params than full_transformer."""
    full_transformer_mlp = MLPModel(
        post_embedding_dim=384,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
        user_hidden_dim=256,
        user_output_dim=128,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        attention_dropout=0.1,
        user_encoder_type="full_transformer",
    )

    cross_attention_mlp = MLPModel(
        post_embedding_dim=384,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
        user_hidden_dim=256,
        user_output_dim=128,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        attention_dropout=0.1,
        user_encoder_type="cross_attention",
    )

    full_transformer_params = sum(p.numel() for p in full_transformer_mlp.parameters())
    cross_attention_params = sum(p.numel() for p in cross_attention_mlp.parameters())

    assert cross_attention_params < full_transformer_params


def test_models_output_same_type():
    """Test that both models produce compatible outputs."""
    embed_dim = 384
    summarized_mlp = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[256],
        dropout_rate=0.3,
        user_hidden_dim=128,
        user_output_dim=embed_dim,
        num_attention_heads=4,
        num_attention_layers=1,
        max_history_len=20,
        attention_dropout=0.1,
        user_encoder_type="summarized",
    )
    
    attention_mlp = MLPModel(
        post_embedding_dim=embed_dim,
        hidden_dims=[256],
        dropout_rate=0.3,
        user_hidden_dim=128,
        user_output_dim=128,
        num_attention_heads=4,
        num_attention_layers=1,
        max_history_len=20,
        attention_dropout=0.1,
        user_encoder_type="full_transformer",
    )
    
    # Summarized output (summary token at position 0)
    history1 = torch.randn(8, 1, embed_dim)
    mask1 = torch.ones(8, 1, dtype=torch.bool)
    post1 = torch.randn(8, embed_dim)
    out1 = summarized_mlp.forward(history1, mask1, post1)
    
    # Full transformer output
    history = torch.randn(8, 20, embed_dim)
    mask = torch.ones(8, 20, dtype=torch.bool)
    post = torch.randn(8, embed_dim)
    out2 = attention_mlp.forward(history, mask, post)
    
    # Both should produce same shape and type
    assert out1.shape == out2.shape
    assert out1.dtype == out2.dtype
    assert (out1 >= 0).all() and (out1 <= 1).all()
    assert (out2 >= 0).all() and (out2 <= 1).all()
