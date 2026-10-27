"""Explicit BF16 operations matching the Qwen-TTS PyTorch implementation."""
import types

import torch
from torch import nn
from torch.nn import functional as F


def rms_forward(self, x, residual=None):
    if residual is not None:
        x = x + residual
    value = x.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.variance_epsilon)
    result = self.weight * value.to(x.dtype)
    return result if residual is None else (result, x)


class BFloatRope(nn.Module):
    def __init__(self, dim, theta):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)), persistent=False)
        self.dim = dim

    def forward(self, positions, q, k):
        angles = positions.float().reshape(-1, 1) * self.inv_freq.float().reshape(1, -1)
        angles = torch.cat((angles, angles), -1)
        cos, sin = angles.cos().to(q.dtype)[:, None, :], angles.sin().to(q.dtype)[:, None, :]
        def apply(x):
            shape = x.shape
            x = x.view(-1, shape[-1] // self.dim, self.dim)
            rotate = torch.cat((-x[..., self.dim // 2:], x[..., :self.dim // 2]), -1)
            return (x * cos + rotate * sin).reshape(shape)
        return apply(q), apply(k)


def split_linear(self, x):
    return torch.cat([F.linear(x, weight) for weight in self.weight.split(self.parity_sizes)], -1), None


def mlp_forward(self, x):
    gate_up, _ = self.gate_up_proj(x)
    gate, up = gate_up.chunk(2, -1)
    out, _ = self.down_proj(F.silu(gate) * up)
    return out


def apply_bf16(backbone, config):
    """Preserve reference rounding boundaries while retaining native attention."""
    for module in backbone.modules():
        if hasattr(module, "variance_epsilon") and hasattr(module, "weight"):
            module.forward = types.MethodType(rms_forward, module)
    theta = getattr(config, "rope_theta", None)
    if theta is None:
        theta = config.rope_parameters["rope_theta"]
    for layer in backbone.layers:
        attn = layer.self_attn
        attn.rotary_emb = BFloatRope(config.head_dim, theta)
        attn.qkv_proj.parity_sizes = [attn.q_size, attn.kv_size, attn.kv_size]
        attn.qkv_proj.forward = types.MethodType(split_linear, attn.qkv_proj)
        layer.mlp.gate_up_proj.parity_sizes = [config.intermediate_size] * 2
        layer.mlp.gate_up_proj.forward = types.MethodType(split_linear, layer.mlp.gate_up_proj)
        layer.mlp.forward = types.MethodType(mlp_forward, layer.mlp)
