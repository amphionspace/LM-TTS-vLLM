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
    return torch.cat([F.linear(x, weight) for weight in self.weight.split(self.parity_sizes)], -1), None


def mlp_forward(self, x):
    gate_up, _ = self.gate_up_proj(x)
    gate, up = gate_up.chunk(2, -1)
    out, _ = self.down_proj(F.silu(gate) * up)
    return out


