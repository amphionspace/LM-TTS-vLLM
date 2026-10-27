"""Causal Talker returning unnormalized last-token hidden states via vLLM pooling.

The BF16 profile preserves Qwen-TTS's intermediate rounding boundaries. Attention
still uses vLLM's backend; parity is measured, never assumed to be bitwise exact.
"""
import os
import types

import torch
from torch import nn

from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.models.qwen3 import Qwen3Model


from .precision import BFloatRope, mlp_forward, rms_forward, sdpa_attention, split_linear


class LMTTSTalkerModel(nn.Module):
    is_pooling_model = True
    default_seq_pooling_type = "LAST"
    default_tok_pooling_type = "ALL"

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        if vllm_config.parallel_config.tensor_parallel_size != 1 or vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("The parity plugin currently requires TP=PP=1")
        self.model = Qwen3Model(vllm_config=vllm_config, prefix=f"{prefix}.model".strip("."))
        self.pooler = DispatchPooler.for_embedding(vllm_config.model_config.pooler_config)
        self.profile = os.environ.get("LM_TTS_PRECISION", "bf16")
        if self.profile not in {"bf16", "native", "sdpa"}:
            raise ValueError("LM_TTS_PRECISION must be bf16, native or sdpa")
        if self.profile in {"bf16", "sdpa"}:
            cfg = vllm_config.model_config.hf_config
            for module in self.model.modules():
                if hasattr(module, "variance_epsilon") and hasattr(module, "weight"):
                    module.forward = types.MethodType(rms_forward, module)
            for layer in self.model.layers:
                attn = layer.self_attn
                attn.rotary_emb = BFloatRope(cfg.head_dim, cfg.rope_parameters["rope_theta"])
                attn.qkv_proj.parity_sizes = [attn.q_size, attn.kv_size, attn.kv_size]
                attn.qkv_proj.forward = types.MethodType(split_linear, attn.qkv_proj)
                layer.mlp.gate_up_proj.parity_sizes = [cfg.intermediate_size] * 2
                layer.mlp.gate_up_proj.forward = types.MethodType(split_linear, layer.mlp.gate_up_proj)
                layer.mlp.forward = types.MethodType(mlp_forward, layer.mlp)
                if self.profile == "sdpa":
                    if vllm_config.cache_config.enable_prefix_caching:
                        raise ValueError("The SDPA diagnostic requires full-prefix reference mode")
                    attn.attn.forward = types.MethodType(sdpa_attention, attn.attn)

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def load_weights(self, weights):
        loaded = self.model.load_weights((name.removeprefix("model."), value) for name, value in weights)
        return {"model." + name for name in loaded}
