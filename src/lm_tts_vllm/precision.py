"""Explicit BF16 operations matching the Qwen-TTS PyTorch implementation."""
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
    # Slice instead of Tensor.split: torch.compile cannot trace split on vLLM's parameter subclass.
    starts = [sum(self.parity_sizes[:i]) for i in range(len(self.parity_sizes))]
    return torch.cat([F.linear(x, self.weight[start:start + size])
                      for start, size in zip(starts, self.parity_sizes)], -1), None


def mlp_forward(self, x):
    gate_up, _ = self.gate_up_proj(x)
    gate, up = gate_up.chunk(2, -1)
    out, _ = self.down_proj(F.silu(gate) * up)
    return out


_PATCHED = {}


def _override_forward(module, forward):
    # Swap in a subclass rather than binding a bound method on the instance:
    # torch.compile cannot trace ``function.__get__`` from instance patches.
    key = (type(module), forward)
    if key not in _PATCHED:
        _PATCHED[key] = type(f"BF16{type(module).__name__}", (type(module),), {"forward": forward})
    module.__class__ = _PATCHED[key]


def apply_bf16(backbone, config):
    """Preserve reference rounding boundaries while retaining native attention."""
    for module in backbone.modules():
        if hasattr(module, "variance_epsilon") and hasattr(module, "weight"):
            _override_forward(module, rms_forward)
    theta = getattr(config, "rope_theta", None)
    if theta is None:
        theta = config.rope_parameters["rope_theta"]
    for layer in backbone.layers:
        attn = layer.self_attn
        attn.rotary_emb = BFloatRope(config.head_dim, theta)
        attn.qkv_proj.parity_sizes = [attn.q_size, attn.kv_size, attn.kv_size]
        _override_forward(attn.qkv_proj, split_linear)
        layer.mlp.gate_up_proj.parity_sizes = [config.intermediate_size] * 2
        _override_forward(layer.mlp.gate_up_proj, split_linear)
        _override_forward(layer.mlp, mlp_forward)
