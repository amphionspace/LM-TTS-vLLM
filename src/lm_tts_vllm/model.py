"""Causal Talker returning unnormalized last-token hidden states via vLLM pooling.

The BF16 profile preserves Qwen-TTS's intermediate rounding boundaries. Attention
still uses vLLM's backend; parity is measured, never assumed to be bitwise exact.
"""
import os

from torch import nn

from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.models.qwen3 import Qwen3Model

from .precision import apply_bf16


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
        if self.profile not in {"bf16", "native"}:
            raise ValueError("LM_TTS_PRECISION must be bf16 or native")
        if self.profile == "bf16":
            apply_bf16(self.model, vllm_config.model_config.hf_config)

    def embed_input_ids(self, input_ids):
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def load_weights(self, weights):
        loaded = self.model.load_weights((name.removeprefix("model."), value) for name, value in weights)
        return {"model." + name for name in loaded}
