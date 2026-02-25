import importlib

import torch
import torch.nn as nn


def _copy_ts_block_to_torch_layer(ts_block: nn.Module, layer: nn.TransformerEncoderLayer) -> None:
    # Attention projections
    layer.self_attn.in_proj_weight.data.copy_(ts_block.qkv_proj.weight.data)
    layer.self_attn.in_proj_bias.data.copy_(ts_block.qkv_proj.bias.data)
    layer.self_attn.out_proj.weight.data.copy_(ts_block.out_proj.weight.data)
    layer.self_attn.out_proj.bias.data.copy_(ts_block.out_proj.bias.data)

    # Layer norms
    layer.norm1.weight.data.copy_(ts_block.norm1.weight.data)
    layer.norm1.bias.data.copy_(ts_block.norm1.bias.data)
    layer.norm2.weight.data.copy_(ts_block.norm2.weight.data)
    layer.norm2.bias.data.copy_(ts_block.norm2.bias.data)

    # Feed-forward
    layer.linear1.weight.data.copy_(ts_block.ff1.weight.data)
    layer.linear1.bias.data.copy_(ts_block.ff1.bias.data)
    layer.linear2.weight.data.copy_(ts_block.ff2.weight.data)
    layer.linear2.bias.data.copy_(ts_block.ff2.bias.data)


def test_ts_transformer_block_matches_torch_transformer_encoder_layer():
    dataloaders = importlib.import_module("utils.dataloaders")
    TSBlock = getattr(dataloaders, "_TS_TransformerBlock")

    torch.manual_seed(0)
    B, T, D = 3, 7, 32
    H = 4

    ts_block = TSBlock(hidden_dim=D, num_attention_heads=H, dropout_rate=0.0).eval()
    torch_layer = nn.TransformerEncoderLayer(
        d_model=D,
        nhead=H,
        dim_feedforward=4 * D,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=False,
    ).eval()
    _copy_ts_block_to_torch_layer(ts_block, torch_layer)

    x = torch.randn(B, T, D)
    # True = ignore/padding (PyTorch convention)
    key_padding_mask = torch.zeros(B, T, dtype=torch.bool)
    key_padding_mask[0, -1] = True
    key_padding_mask[1, -2:] = True
    key_padding_mask[2, -3:] = True

    out_ts = ts_block(x, key_padding_mask)
    out_ref = torch_layer(x, src_key_padding_mask=key_padding_mask)

    assert torch.allclose(out_ts, out_ref, rtol=1e-4, atol=1e-5)


def test_ts_transformer_stack_matches_torch_stack():
    dataloaders = importlib.import_module("utils.dataloaders")
    TSBlock = getattr(dataloaders, "_TS_TransformerBlock")

    torch.manual_seed(0)
    B, T, D = 2, 9, 64
    H = 8
    L = 3

    ts_blocks = nn.ModuleList([TSBlock(hidden_dim=D, num_attention_heads=H, dropout_rate=0.0).eval() for _ in range(L)])
    torch_layers = nn.ModuleList(
        [
            nn.TransformerEncoderLayer(
                d_model=D,
                nhead=H,
                dim_feedforward=4 * D,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            ).eval()
            for _ in range(L)
        ]
    )
    for b, l in zip(ts_blocks, torch_layers, strict=True):
        _copy_ts_block_to_torch_layer(b, l)

    x = torch.randn(B, T, D)
    key_padding_mask = torch.zeros(B, T, dtype=torch.bool)
    key_padding_mask[0, -2:] = True
    key_padding_mask[1, -4:] = True

    out_ts = x
    for b in ts_blocks:
        out_ts = b(out_ts, key_padding_mask)

    out_ref = x
    for l in torch_layers:
        out_ref = l(out_ref, src_key_padding_mask=key_padding_mask)

    assert torch.allclose(out_ts, out_ref, rtol=1e-4, atol=1e-5)

