"""Comprehensive tests for Two Tower model architecture."""
import importlib
import pytest
import torch
import torch.nn as nn

# Import from module with numeric prefix
stage_train_two_tower = importlib.import_module("utils.04_train.stage_train_two_tower")
PostTower = stage_train_two_tower.PostTower
SharedPostFeatureEncoder = stage_train_two_tower.SharedPostFeatureEncoder
TwoTowerModel = stage_train_two_tower.TwoTowerModel


# =============================================================================
# PostTower Tests
# =============================================================================

def test_post_tower_initialization():
    """Test PostTower initializes correctly."""
    tower = PostTower(
        input_dim=384,
        hidden_dim=256,
        output_dim=128,
        dropout_rate=0.3,
    )
    
    assert isinstance(tower.network, nn.Sequential)
    
    # Check it has the expected layers
    modules = list(tower.network.modules())
    assert any(isinstance(m, nn.Linear) for m in modules)
    assert any(isinstance(m, nn.LayerNorm) for m in modules)
    assert any(isinstance(m, nn.GELU) for m in modules)
    assert any(isinstance(m, nn.Dropout) for m in modules)


def test_post_tower_forward_shape():
    """Test PostTower forward pass produces correct output shape."""
    batch_size = 16
    input_dim = 384
    output_dim = 128
    
    tower = PostTower(
        input_dim=input_dim,
        hidden_dim=256,
        output_dim=output_dim,
        dropout_rate=0.3,
    )
    
    # Create random input
    post_embeddings = torch.randn(batch_size, input_dim)
    
    # Forward pass
    output = tower.forward(post_embeddings)
    
    # Check output shape
    assert output.shape == (batch_size, output_dim)
    assert output.dtype == torch.float32


def test_post_tower_different_dimensions():
    """Test PostTower with various dimension configurations."""
    configs = [
        (256, 128, 64),
        (512, 256, 128),
        (384, 384, 256),
        (128, 64, 32),
    ]
    
    for input_dim, hidden_dim, output_dim in configs:
        tower = PostTower(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout_rate=0.2,
        )
        
        x = torch.randn(8, input_dim)
        output = tower.forward(x)
        
        assert output.shape == (8, output_dim)


def test_post_tower_batch_sizes():
    """Test PostTower handles different batch sizes."""
    tower = PostTower(
        input_dim=384,
        hidden_dim=256,
        output_dim=128,
        dropout_rate=0.3,
    )
    
    for batch_size in [1, 4, 16, 32, 64]:
        x = torch.randn(batch_size, 384)
        output = tower.forward(x)
        assert output.shape == (batch_size, 128)


def test_post_tower_backward_pass():
    """Test PostTower gradients flow correctly."""
    tower = PostTower(
        input_dim=384,
        hidden_dim=256,
        output_dim=128,
        dropout_rate=0.2,
    )
    
    x = torch.randn(8, 384, requires_grad=True)
    output = tower.forward(x)
    
    # Compute a dummy loss
    loss = output.sum()
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None
    for param in tower.parameters():
        assert param.grad is not None


def test_post_tower_eval_mode():
    """Test PostTower behaves consistently in eval mode."""
    tower = PostTower(
        input_dim=384,
        hidden_dim=256,
        output_dim=128,
        dropout_rate=0.5,  # High dropout for testing
    )
    
    x = torch.randn(16, 384)
    
    # Eval mode - should be deterministic
    tower.eval()
    with torch.no_grad():
        outputs = [tower.forward(x) for _ in range(3)]
    
    # All outputs should be identical
    for i in range(len(outputs) - 1):
        assert torch.allclose(outputs[i], outputs[i + 1])


def test_shared_post_feature_encoder_zeroes_padding_row():
    encoder = SharedPostFeatureEncoder(
        post_embedding_dim=16,
        num_embedding_table_rows=32,
        n_hashes_author_emb_table=3,
    )

    assert torch.all(encoder.collab_table.weight[0] == 0)


# =============================================================================
# TwoTowerModel Tests - Initialization
# =============================================================================

def test_two_tower_model_full_transformer_encoder():
    """Test TwoTowerModel with full-transformer encoder."""
    model = TwoTowerModel(
        post_embedding_dim=384,
        shared_dim=128,
        user_hidden_dim=256,
        post_hidden_dim=256,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        dropout_rate=0.3,
        similarity_temperature=0.2,
        user_encoder_type="full_transformer",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    
    assert model.shared_dim == 128
    assert model.post_embedding_dim == 384
    assert model.user_encoder_type == "full_transformer"
    assert hasattr(model, "user_tower")
    assert hasattr(model, "post_tower")


def test_two_tower_model_cross_attention_encoder():
    """Test TwoTowerModel with cross-attention encoder."""
    model = TwoTowerModel(
        post_embedding_dim=384,
        shared_dim=128,
        user_hidden_dim=256,
        post_hidden_dim=256,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        dropout_rate=0.3,
        similarity_temperature=0.2,
        user_encoder_type="cross_attention",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    
    assert model.user_encoder_type == "cross_attention"
    assert hasattr(model, "user_tower")
    assert hasattr(model, "post_tower")


def test_two_tower_model_invalid_encoder_type():
    """Test TwoTowerModel raises error for invalid encoder type."""
    with pytest.raises(ValueError, match="Unknown user_encoder_type"):
        TwoTowerModel(
            post_embedding_dim=384,
            shared_dim=128,
            user_hidden_dim=256,
            post_hidden_dim=256,
            num_attention_heads=4,
            num_attention_layers=2,
            max_history_len=50,
            dropout_rate=0.3,
            similarity_temperature=0.2,
            user_encoder_type="invalid_type",
            use_post_encoder=True,
        l2_normalize_embeddings=True,
        )


# =============================================================================
# TwoTowerModel Tests - encode_user
# =============================================================================

def test_two_tower_encode_user_shape():
    """Test TwoTowerModel encode_user produces correct output shape."""
    batch_size = 16
    seq_len = 50
    input_dim = 384
    shared_dim = 128
    
    model = TwoTowerModel(
        post_embedding_dim=input_dim,
        shared_dim=shared_dim,
        user_hidden_dim=256,
        post_hidden_dim=256,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=seq_len,
        dropout_rate=0.3,
        similarity_temperature=0.2,
        user_encoder_type="full_transformer",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    
    history_embeddings = torch.randn(batch_size, seq_len, input_dim)
    history_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    user_emb = model.encode_user(history_embeddings, history_mask)
    
    assert user_emb.shape == (batch_size, shared_dim)
    assert user_emb.dtype == torch.float32
    assert torch.allclose(user_emb.norm(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_two_tower_encode_user_with_mask():
    """Test TwoTowerModel encode_user respects history mask."""
    batch_size = 8
    seq_len = 30
    
    model = TwoTowerModel(
        post_embedding_dim=384,
        shared_dim=128,
        user_hidden_dim=256,
        post_hidden_dim=256,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=seq_len,
        dropout_rate=0.3,
        similarity_temperature=0.2,
        user_encoder_type="full_transformer",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    
    history_embeddings = torch.randn(batch_size, seq_len, 384)
    history_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    history_mask[:, :15] = True  # Only first half valid
    
    user_emb = model.encode_user(history_embeddings, history_mask)
    
    assert user_emb.shape == (batch_size, 128)
    assert torch.isfinite(user_emb).all()
    assert torch.allclose(user_emb.norm(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_two_tower_encode_user_empty_history():
    """Test TwoTowerModel encode_user with empty history."""
    batch_size = 4
    seq_len = 20
    
    model = TwoTowerModel(
        post_embedding_dim=384,
        shared_dim=128,
        user_hidden_dim=256,
        post_hidden_dim=256,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=seq_len,
        dropout_rate=0.3,
        similarity_temperature=0.2,
        user_encoder_type="full_transformer",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    
    # Empty history
    history_embeddings = torch.zeros(batch_size, seq_len, 384)
    history_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    
    user_emb = model.encode_user(history_embeddings, history_mask)
    
    assert user_emb.shape == (batch_size, 128)
    assert torch.isfinite(user_emb).all()
    assert torch.allclose(user_emb.norm(dim=-1), torch.ones(batch_size), atol=1e-5)

def test_two_tower_empty_history_scores_vary_with_post_full_transformer():
    """Cold-start users should not get identical Two-Tower scores for all posts."""
    torch.manual_seed(0)
    batch_size = 2
    seq_len = 10
    embed_dim = 32

    model = TwoTowerModel(
        post_embedding_dim=embed_dim,
        shared_dim=16,
        user_hidden_dim=32,
        post_hidden_dim=32,
        num_attention_heads=4,
        num_attention_layers=1,
        max_history_len=seq_len,
        dropout_rate=0.0,
        similarity_temperature=0.2,
        user_encoder_type="full_transformer",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    model.eval()

    history_embeddings = torch.zeros(batch_size, seq_len, embed_dim)
    history_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    post_embeddings = torch.stack([torch.ones(embed_dim), -torch.ones(embed_dim)], dim=0)

    with torch.no_grad():
        scores = model.forward(history_embeddings, history_mask, post_embeddings)

    assert scores.shape == (batch_size,)
    assert torch.isfinite(scores).all()
    assert not torch.allclose(scores[0], scores[1]), "Scores should depend on the post even for empty histories"

def test_two_tower_empty_history_scores_vary_with_post_summarized():
    """Cold-start users should not get identical Two-Tower scores in summarized mode."""
    torch.manual_seed(0)
    batch_size = 2
    embed_dim = 32

    model = TwoTowerModel(
        post_embedding_dim=embed_dim,
        shared_dim=embed_dim,
        user_hidden_dim=32,
        post_hidden_dim=32,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=10,
        dropout_rate=0.0,
        similarity_temperature=0.2,
        user_encoder_type="summarized",
        use_post_encoder=False,
        l2_normalize_embeddings=True,
    )
    model.eval()

    history_embeddings = torch.zeros(batch_size, 1, embed_dim)
    history_mask = torch.zeros(batch_size, 1, dtype=torch.bool)  # indicates "no history"
    post_embeddings = torch.stack([torch.ones(embed_dim), -torch.ones(embed_dim)], dim=0)

    with torch.no_grad():
        scores = model.forward(history_embeddings, history_mask, post_embeddings)

    assert scores.shape == (batch_size,)
    assert torch.isfinite(scores).all()
    assert not torch.allclose(scores[0], scores[1]), "Scores should depend on the post even for empty summarized histories"


# =============================================================================
# TwoTowerModel Tests - encode_post
# =============================================================================

def test_two_tower_encode_post_shape():
    """Test TwoTowerModel encode_post produces correct output shape."""
    batch_size = 16
    input_dim = 384
    shared_dim = 128
    
    model = TwoTowerModel(
        post_embedding_dim=input_dim,
        shared_dim=shared_dim,
        user_hidden_dim=256,
        post_hidden_dim=256,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        dropout_rate=0.3,
        similarity_temperature=0.2,
        user_encoder_type="full_transformer",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    
    post_embeddings = torch.randn(batch_size, input_dim)
    
    post_emb = model.encode_post(post_embeddings)
    
    assert post_emb.shape == (batch_size, shared_dim)
    assert post_emb.dtype == torch.float32
    assert torch.allclose(post_emb.norm(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_two_tower_encode_post_single():
    """Test TwoTowerModel encode_post with single post."""
    model = TwoTowerModel(
        post_embedding_dim=384,
        shared_dim=128,
        user_hidden_dim=256,
        post_hidden_dim=256,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        dropout_rate=0.3,
        similarity_temperature=0.2,
        user_encoder_type="full_transformer",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    
    post_embedding = torch.randn(1, 384)
    post_emb = model.encode_post(post_embedding)
    
    assert post_emb.shape == (1, 128)
    assert torch.allclose(post_emb.norm(dim=-1), torch.ones(1), atol=1e-5)


def test_two_tower_encode_post_can_skip_l2_normalization():
    """Test TwoTowerModel encode_post can return unnormalized embeddings."""
    model = TwoTowerModel(
        post_embedding_dim=384,
        shared_dim=128,
        user_hidden_dim=256,
        post_hidden_dim=256,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        dropout_rate=0.3,
        similarity_temperature=0.2,
        user_encoder_type="full_transformer",
        use_post_encoder=True,
        l2_normalize_embeddings=False,
    )

    post_embeddings = torch.randn(16, 384)
    post_emb = model.encode_post(post_embeddings)

    assert post_emb.shape == (16, 128)
    assert post_emb.dtype == torch.float32
    assert not torch.allclose(post_emb.norm(dim=-1), torch.ones(16), atol=1e-5)


# =============================================================================
# TwoTowerModel Tests - forward (dot product scoring)
# =============================================================================

def test_two_tower_forward_shape():
    """Test TwoTowerModel forward pass produces correct output shape."""
    batch_size = 16
    seq_len = 50
    input_dim = 384
    
    model = TwoTowerModel(
        post_embedding_dim=input_dim,
        shared_dim=128,
        user_hidden_dim=256,
        post_hidden_dim=256,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=seq_len,
        dropout_rate=0.3,
        similarity_temperature=0.2,
        user_encoder_type="full_transformer",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    
    history_embeddings = torch.randn(batch_size, seq_len, input_dim)
    history_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    post_embeddings = torch.randn(batch_size, input_dim)
    
    scores = model.forward(history_embeddings, history_mask, post_embeddings)
    
    # Scores should be raw logits (before sigmoid)
    assert scores.shape == (batch_size,)
    assert scores.dtype == torch.float32


def test_two_tower_forward_dot_product():
    """Test TwoTowerModel forward computes cosine similarity from normalized towers."""
    model = TwoTowerModel(
        post_embedding_dim=384,
        shared_dim=128,
        user_hidden_dim=256,
        post_hidden_dim=256,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        dropout_rate=0.0,  # No dropout for deterministic test
        similarity_temperature=0.2,
        user_encoder_type="full_transformer",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    
    model.eval()
    with torch.no_grad():
        batch_size = 4
        history_embeddings = torch.randn(batch_size, 50, 384)
        history_mask = torch.ones(batch_size, 50, dtype=torch.bool)
        post_embeddings = torch.randn(batch_size, 384)
        
        # Get scores via forward
        scores = model.forward(history_embeddings, history_mask, post_embeddings)
        
        # Compute manually
        user_emb = model.encode_user(history_embeddings, history_mask)
        post_emb = model.encode_post(post_embeddings)
        manual_scores = (user_emb * post_emb).sum(dim=-1) / model.similarity_temperature
        
        # Should match
        assert torch.allclose(scores, manual_scores, rtol=1e-5)


def test_two_tower_forward_without_l2_normalization_uses_raw_dot_product():
    """Disabling L2 normalization should switch scoring from cosine to raw dot product."""
    model = TwoTowerModel(
        post_embedding_dim=2,
        shared_dim=2,
        user_hidden_dim=4,
        post_hidden_dim=4,
        num_attention_heads=1,
        num_attention_layers=1,
        max_history_len=10,
        dropout_rate=0.0,
        similarity_temperature=1.0,
        user_encoder_type="summarized",
        use_post_encoder=False,
        l2_normalize_embeddings=False,
    )

    history_embeddings = torch.tensor([[[3.0, 4.0]]])
    history_mask = torch.ones(1, 1, dtype=torch.bool)
    post_embeddings = torch.tensor([[0.0, 5.0]])

    score = model.forward(history_embeddings, history_mask, post_embeddings)

    assert torch.allclose(score, torch.tensor([20.0]))


def test_two_tower_forward_with_l2_normalization_uses_cosine_similarity():
    """Enabling L2 normalization should preserve cosine-style scoring."""
    model = TwoTowerModel(
        post_embedding_dim=2,
        shared_dim=2,
        user_hidden_dim=4,
        post_hidden_dim=4,
        num_attention_heads=1,
        num_attention_layers=1,
        max_history_len=10,
        dropout_rate=0.0,
        similarity_temperature=1.0,
        user_encoder_type="summarized",
        use_post_encoder=False,
        l2_normalize_embeddings=True,
    )

    history_embeddings = torch.tensor([[[3.0, 4.0]]])
    history_mask = torch.ones(1, 1, dtype=torch.bool)
    post_embeddings = torch.tensor([[0.0, 5.0]])

    score = model.forward(history_embeddings, history_mask, post_embeddings)

    assert torch.allclose(score, torch.tensor([0.8]))


def test_two_tower_forward_both_encoder_types():
    """Test TwoTowerModel forward works with both encoder types."""
    batch_size = 8
    history_embeddings = torch.randn(batch_size, 30, 384)
    history_mask = torch.ones(batch_size, 30, dtype=torch.bool)
    post_embeddings = torch.randn(batch_size, 384)
    
    for encoder_type in ["full_transformer", "cross_attention"]:
        model = TwoTowerModel(
            post_embedding_dim=384,
            shared_dim=128,
            user_hidden_dim=256,
            post_hidden_dim=256,
            num_attention_heads=4,
            num_attention_layers=2,
            max_history_len=30,
            dropout_rate=0.3,
            similarity_temperature=0.2,
            user_encoder_type=encoder_type,
            use_post_encoder=True,
        l2_normalize_embeddings=True,
        )
        
        scores = model.forward(history_embeddings, history_mask, post_embeddings)
        assert scores.shape == (batch_size,)

def test_two_tower_summarized_user_tower_torchscript():
    """Test summarized user_tower can be TorchScript scripted (serving artifact)."""
    embed_dim = 128
    model = TwoTowerModel(
        post_embedding_dim=embed_dim,
        shared_dim=embed_dim,
        user_hidden_dim=64,
        post_hidden_dim=64,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=20,
        dropout_rate=0.1,
        similarity_temperature=0.2,
        user_encoder_type="summarized",
        use_post_encoder=False,
        l2_normalize_embeddings=True,
    )

    scripted = torch.jit.script(model.user_tower)
    history_embeddings = torch.randn(4, 1, embed_dim)
    history_mask = torch.ones(4, 1, dtype=torch.bool)
    out = scripted(history_embeddings, history_mask)
    assert out.shape == (4, embed_dim)


# =============================================================================
# TwoTowerModel Tests - compute_loss_and_preds
# =============================================================================

def test_two_tower_compute_loss_and_preds():
    """Test TwoTowerModel compute_loss_and_preds method."""
    model = TwoTowerModel(
        post_embedding_dim=384,
        shared_dim=128,
        user_hidden_dim=256,
        post_hidden_dim=256,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        dropout_rate=0.3,
        similarity_temperature=0.2,
        user_encoder_type="full_transformer",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    
    batch_size = 16
    history_embeddings = torch.randn(batch_size, 50, 384)
    history_mask = torch.ones(batch_size, 50, dtype=torch.bool)
    post_embeddings = torch.randn(batch_size, 384)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    batch = {
        "history_embeddings": history_embeddings,
        "history_mask": history_mask,
        "target_post_embedding": post_embeddings,
        "label": labels,
    }
    loss, scores = model.compute_loss_and_preds(batch, device="cpu", embed_dim=384)
    
    # Check loss
    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert loss.item() >= 0, "BCE loss should be non-negative"
    
    # Check scores (raw logits)
    assert scores.shape == (batch_size,)
    assert scores.dtype == torch.float32
    
    # Verify sigmoid(scores) is in [0, 1]
    probs = torch.sigmoid(scores)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_two_tower_compute_loss_and_preds_summarized_empty_history_uses_empty_embedding():
    """Summarized compute_loss_and_preds should use cold-start embedding for empty histories."""
    torch.manual_seed(0)
    embed_dim = 16
    model = TwoTowerModel(
        post_embedding_dim=embed_dim,
        shared_dim=embed_dim,
        user_hidden_dim=32,
        post_hidden_dim=32,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=10,
        dropout_rate=0.0,
        similarity_temperature=0.2,
        user_encoder_type="summarized",
        use_post_encoder=False,
        l2_normalize_embeddings=True,
    )
    model.eval()

    batch_size = 8
    user_summary = torch.zeros(batch_size, embed_dim)
    post_embeddings = torch.arange(embed_dim, dtype=torch.float32).unsqueeze(0).expand(batch_size, -1)
    features = torch.cat([user_summary, post_embeddings], dim=1)
    batch = {"features": features, "label": torch.randint(0, 2, (batch_size,)).float()}

    with torch.no_grad():
        model.user_tower.tower.empty_user_embedding.zero_()
        model.user_tower.tower.empty_user_embedding[0] = 1.0
    _, scores1 = model.compute_loss_and_preds(batch, device="cpu", embed_dim=embed_dim)

    with torch.no_grad():
        model.user_tower.tower.empty_user_embedding.zero_()
        model.user_tower.tower.empty_user_embedding[1] = 1.0
    _, scores2 = model.compute_loss_and_preds(batch, device="cpu", embed_dim=embed_dim)

    assert not torch.allclose(scores1, scores2), "Scores should depend on the cold-start embedding for empty histories"


def test_two_tower_compute_loss_all_positive():
    """Test TwoTowerModel compute_loss with all positive labels."""
    model = TwoTowerModel(
        post_embedding_dim=128,
        shared_dim=64,
        user_hidden_dim=128,
        post_hidden_dim=128,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=20,
        dropout_rate=0.2,
        similarity_temperature=0.2,
        user_encoder_type="full_transformer",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    
    batch_size = 8
    history_embeddings = torch.randn(batch_size, 20, 128)
    history_mask = torch.ones(batch_size, 20, dtype=torch.bool)
    post_embeddings = torch.randn(batch_size, 128)
    labels = torch.ones(batch_size)  # All positive
    
    batch = {
        "history_embeddings": history_embeddings,
        "history_mask": history_mask,
        "target_post_embedding": post_embeddings,
        "label": labels,
    }
    loss, scores = model.compute_loss_and_preds(batch, device="cpu", embed_dim=128)
    
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_two_tower_compute_loss_all_negative():
    """Test TwoTowerModel compute_loss with all negative labels."""
    model = TwoTowerModel(
        post_embedding_dim=128,
        shared_dim=64,
        user_hidden_dim=128,
        post_hidden_dim=128,
        num_attention_heads=2,
        num_attention_layers=1,
        max_history_len=20,
        dropout_rate=0.2,
        similarity_temperature=0.2,
        user_encoder_type="full_transformer",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    
    batch_size = 8
    history_embeddings = torch.randn(batch_size, 20, 128)
    history_mask = torch.ones(batch_size, 20, dtype=torch.bool)
    post_embeddings = torch.randn(batch_size, 128)
    labels = torch.zeros(batch_size)  # All negative
    
    batch = {
        "history_embeddings": history_embeddings,
        "history_mask": history_mask,
        "target_post_embedding": post_embeddings,
        "label": labels,
    }
    loss, scores = model.compute_loss_and_preds(batch, device="cpu", embed_dim=128)
    
    assert torch.isfinite(loss)
    assert loss.item() >= 0


# =============================================================================
# TwoTowerModel Tests - Gradient Flow
# =============================================================================

def test_two_tower_backward_pass():
    """Test TwoTowerModel gradients flow through both towers."""
    model = TwoTowerModel(
        post_embedding_dim=384,
        shared_dim=128,
        user_hidden_dim=256,
        post_hidden_dim=256,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        dropout_rate=0.3,
        similarity_temperature=0.2,
        user_encoder_type="full_transformer",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    
    batch_size = 8
    history_embeddings = torch.randn(batch_size, 50, 384)
    history_mask = torch.ones(batch_size, 50, dtype=torch.bool)
    post_embeddings = torch.randn(batch_size, 384)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    batch = {
        "history_embeddings": history_embeddings,
        "history_mask": history_mask,
        "target_post_embedding": post_embeddings,
        "label": labels,
    }
    loss, _ = model.compute_loss_and_preds(batch, device="cpu", embed_dim=384)
    
    loss.backward()
    
    # Check gradients exist in both towers
    for name, param in model.named_parameters():
        # `empty_history_embedding` is only used when an example has an empty history
        # (all-masked). For fully non-empty batches, it's expected to have no grad.
        if "empty_history_embedding" in name:
            assert (param.grad is None) or torch.isfinite(param.grad).all()
            continue
        assert param.grad is not None, f"Parameter {name} should have gradient"
        assert torch.isfinite(param.grad).all(), f"Gradient for {name} should be finite"


# =============================================================================
# TwoTowerModel Tests - Eval Mode
# =============================================================================

def test_two_tower_eval_mode():
    """Test TwoTowerModel behaves consistently in eval mode."""
    model = TwoTowerModel(
        post_embedding_dim=384,
        shared_dim=128,
        user_hidden_dim=256,
        post_hidden_dim=256,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        dropout_rate=0.5,  # High dropout
        similarity_temperature=0.2,
        user_encoder_type="full_transformer",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    
    batch_size = 8
    history_embeddings = torch.randn(batch_size, 50, 384)
    history_mask = torch.ones(batch_size, 50, dtype=torch.bool)
    post_embeddings = torch.randn(batch_size, 384)
    
    # Eval mode
    model.eval()
    with torch.no_grad():
        outputs = [
            model.forward(history_embeddings, history_mask, post_embeddings)
            for _ in range(3)
        ]
    
    # All outputs should be identical
    for i in range(len(outputs) - 1):
        assert torch.allclose(outputs[i], outputs[i + 1])


# =============================================================================
# TwoTowerModel Tests - Parameter Counts
# =============================================================================

def test_two_tower_parameter_count():
    """Test TwoTowerModel has reasonable number of parameters."""
    model = TwoTowerModel(
        post_embedding_dim=384,
        shared_dim=128,
        user_hidden_dim=256,
        post_hidden_dim=256,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        dropout_rate=0.3,
        similarity_temperature=0.2,
        user_encoder_type="full_transformer",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    user_tower_params = sum(p.numel() for p in model.user_tower.parameters())
    post_tower_params = sum(p.numel() for p in model.post_tower.parameters())
    
    # User tower should have more parameters than post tower (has attention)
    assert user_tower_params > post_tower_params
    
    # Total should be sum of both
    assert total_params == user_tower_params + post_tower_params
    
    # Should have a reasonable number of parameters (not too small or huge)
    assert total_params > 1000  # At least 1K params
    assert total_params < 100_000_000  # Less than 100M params


def test_two_tower_cross_attention_fewer_params():
    """Test cross_attention encoder has fewer params than full_transformer."""
    full_transformer_model = TwoTowerModel(
        post_embedding_dim=384,
        shared_dim=128,
        user_hidden_dim=256,
        post_hidden_dim=256,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        dropout_rate=0.3,
        similarity_temperature=0.2,
        user_encoder_type="full_transformer",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    
    cross_attention_model = TwoTowerModel(
        post_embedding_dim=384,
        shared_dim=128,
        user_hidden_dim=256,
        post_hidden_dim=256,
        num_attention_heads=4,
        num_attention_layers=2,
        max_history_len=50,
        dropout_rate=0.3,
        similarity_temperature=0.2,
        user_encoder_type="cross_attention",
        use_post_encoder=True,
        l2_normalize_embeddings=True,
    )
    
    attention_params = sum(p.numel() for p in full_transformer_model.parameters())
    cross_attention_params = sum(p.numel() for p in cross_attention_model.parameters())
    
    # Cross-attention should have fewer parameters
    assert cross_attention_params < attention_params


# =============================================================================
# TwoTowerModel Tests - Different Configurations
# =============================================================================

def test_two_tower_different_shared_dims():
    """Test TwoTowerModel with different shared dimensions."""
    for shared_dim in [32, 64, 128, 256]:
        model = TwoTowerModel(
            post_embedding_dim=384,
            shared_dim=shared_dim,
            user_hidden_dim=256,
            post_hidden_dim=256,
            num_attention_heads=4,
            num_attention_layers=2,
            max_history_len=50,
            dropout_rate=0.3,
            similarity_temperature=0.2,
            user_encoder_type="full_transformer",
            use_post_encoder=True,
        l2_normalize_embeddings=True,
        )
        
        batch_size = 4
        history_embeddings = torch.randn(batch_size, 50, 384)
        history_mask = torch.ones(batch_size, 50, dtype=torch.bool)
        post_embeddings = torch.randn(batch_size, 384)
        
        scores = model.forward(history_embeddings, history_mask, post_embeddings)
        assert scores.shape == (batch_size,)


def test_two_tower_different_num_heads():
    """Test TwoTowerModel with different numbers of attention heads."""
    for num_heads in [1, 2, 4, 8]:
        # user_hidden_dim must be divisible by num_heads
        user_hidden_dim = 256 if num_heads <= 4 else 512
        
        model = TwoTowerModel(
            post_embedding_dim=384,
            shared_dim=128,
            user_hidden_dim=user_hidden_dim,
            post_hidden_dim=256,
            num_attention_heads=num_heads,
            num_attention_layers=2,
            max_history_len=50,
            dropout_rate=0.3,
            similarity_temperature=0.2,
            user_encoder_type="full_transformer",
            use_post_encoder=True,
        l2_normalize_embeddings=True,
        )
        
        batch_size = 4
        history_embeddings = torch.randn(batch_size, 50, 384)
        history_mask = torch.ones(batch_size, 50, dtype=torch.bool)
        post_embeddings = torch.randn(batch_size, 384)
        
        scores = model.forward(history_embeddings, history_mask, post_embeddings)
        assert scores.shape == (batch_size,)
