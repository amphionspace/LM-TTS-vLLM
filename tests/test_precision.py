import types
import unittest

import torch
from torch import nn

from lm_tts_vllm.precision import BFloatRope, rms_forward, split_linear
from lm_tts_tools.diagnostic_attention import sdpa_attention


class PrecisionTests(unittest.TestCase):
    def test_rmsnorm_matches_qwen_with_bf16_weight_and_residual(self):
        from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSRMSNorm
        torch.manual_seed(3)
        reference = Qwen3TTSRMSNorm(128).to(torch.bfloat16)
        reference.weight.data.normal_()
        x = torch.randn(2, 37, 128).bfloat16()
        residual = torch.randn_like(x)
        actual, added = rms_forward(reference, x, residual)
        self.assertTrue(torch.equal(added, x + residual))
        self.assertTrue(torch.equal(actual, reference(x + residual)))

    def test_rope_matches_equal_axis_interleaved_mrope(self):
        from qwen_tts.core.models.modeling_qwen3_tts import apply_multimodal_rotary_pos_emb
        torch.manual_seed(4)
        n, dim = 37, 128
        positions = torch.arange(113, 113 + n)
        q, k = torch.randn(n, 16 * dim).bfloat16(), torch.randn(n, 8 * dim).bfloat16()
        rope = BFloatRope(dim, 1_000_000)
        angles = (rope.inv_freq[None, :, None] @ positions[None, None, :].float()).transpose(1, 2)
        angles = torch.cat((angles, angles), -1)
        cos, sin = angles.cos().bfloat16().expand(3, -1, -1, -1), angles.sin().bfloat16().expand(3, -1, -1, -1)
        rq, rk = apply_multimodal_rotary_pos_emb(
            q.reshape(1, n, 16, dim).transpose(1, 2), k.reshape(1, n, 8, dim).transpose(1, 2),
            cos, sin, [24, 20, 20], True)
        aq, ak = rope(positions, q, k)
        self.assertTrue(torch.equal(aq, rq.transpose(1, 2).reshape_as(aq)))
        self.assertTrue(torch.equal(ak, rk.transpose(1, 2).reshape_as(ak)))

    def test_split_qkv_preserves_separate_projection_outputs(self):
        torch.manual_seed(5)
        projections = [nn.Linear(64, n, bias=False).bfloat16() for n in [128, 64, 64]]
        combined = nn.Linear(64, 256, bias=False).bfloat16()
        combined.weight.data.copy_(torch.cat([p.weight for p in projections]))
        combined.parity_sizes = [128, 64, 64]
        combined.forward = types.MethodType(split_linear, combined)
        x = torch.randn(7, 64).bfloat16()
        actual, _ = combined(x)
        self.assertTrue(torch.equal(actual, torch.cat([p(x) for p in projections], -1)))

    def test_sdpa_diagnostic_matches_transformers_grouped_attention(self):
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        torch.manual_seed(6)
        heads, kv_heads, dim, length = 4, 2, 16, 9
        module = nn.Module()
        module.num_key_value_groups = heads // kv_heads
        module.is_causal = True
        module.eval()
        q = torch.randn(1, heads, length, dim).bfloat16()
        k = torch.randn(1, kv_heads, length, dim).bfloat16()
        v = torch.randn_like(k)
        expected, _ = sdpa_attention_forward(module, q, k, v, attention_mask=None, scaling=dim**-0.5)
        native = types.SimpleNamespace(num_heads=heads, num_kv_heads=kv_heads, head_size=dim, impl=types.SimpleNamespace(scale=dim**-0.5))
        flatten = lambda x: x.transpose(1, 2).reshape(length, -1)
        actual = sdpa_attention(native, flatten(q), flatten(k), flatten(v))
        self.assertTrue(torch.equal(actual, expected.reshape_as(actual)))


if __name__ == "__main__":
    unittest.main()
