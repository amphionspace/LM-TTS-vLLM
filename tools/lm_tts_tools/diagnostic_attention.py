"""SDPA substitution for diagnosis only; excluded from the plugin wheel."""
import types
from torch.nn import functional as F


def sdpa_attention(self, query, key, value):
    """Single-request, full-prefix diagnostic oracle; no paging or chunking."""
    q = query.view(1, -1, self.num_heads, self.head_size).transpose(1, 2)
    k = key.view(1, -1, self.num_kv_heads, self.head_size).transpose(1, 2)
    v = value.view(1, -1, self.num_kv_heads, self.head_size).transpose(1, 2)
    out = F.scaled_dot_product_attention(
        q, k, v, is_causal=q.shape[2] > 1, scale=self.impl.scale, enable_gqa=True
    )
    return out.transpose(1, 2).reshape_as(query)


def patch_sdpa(model):
    for layer in model.model.layers:
        attention = layer.self_attn.attn
        attention.forward = types.MethodType(sdpa_attention, attention)
