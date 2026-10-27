"""Single-frame BF16 code predictor using vLLM FlashAttention and local KV cache.

Inference-only, single request, greedy or sampled decoding. Greedy residual
codebooks can be captured in one CUDA graph; sampling currently uses eager FA2.
No SDPA or Transformers runtime is used.
"""
import json
from pathlib import Path

import torch
from torch.nn import functional as F


class CodePredictor:
    def __init__(self, source):
        from safetensors import safe_open
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        self.attention = flash_attn_varlen_func
        source = Path(source)
        talker = json.loads((source / "config.json").read_text())["talker_config"]
        self.config = c = talker["code_predictor_config"]
        if c["hidden_size"] != talker["hidden_size"] or c["attention_bias"]:
            raise ValueError("Code predictor requires matching hidden dimensions and bias-free attention")
        if c["hidden_act"] != "silu" or any(t != "full_attention" for t in c["layer_types"]):
            raise ValueError("Code predictor currently requires SiLU and full attention")
        self.weights = {}
        prefix = "talker.code_predictor."
        for shard in sorted(source.glob("*.safetensors")):
            with safe_open(shard, framework="pt", device="cpu") as reader:
                for name in reader.keys():
                    if name.startswith(prefix):
                        self.weights[name.removeprefix(prefix)] = reader.get_tensor(name).to(device="cuda", dtype=torch.bfloat16)
        expected = {"model.norm.weight"}
        for index in range(c["num_hidden_layers"]):
            for name in ["input_layernorm", "post_attention_layernorm", "self_attn.q_proj",
                         "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                         "self_attn.q_norm", "self_attn.k_norm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
                expected.add(f"model.layers.{index}.{name}.weight")
        for index in range(c["num_code_groups"] - 1):
            expected.update([f"model.codec_embedding.{index}.weight", f"lm_head.{index}.weight"])
        if self.weights.keys() != expected:
            raise ValueError(f"Code predictor weights mismatch: missing={expected - self.weights.keys()}, "
                             f"unexpected={self.weights.keys() - expected}")
        self.groups = c["num_code_groups"]
        self.head_dim = d = c["head_dim"]
        self.heads = c["num_attention_heads"]
        self.kv_heads = c["num_key_value_heads"]
        self.eps = c["rms_norm_eps"]
        # Same FP32 frequency initialization and BF16 rotation boundaries as HF.
        inv = 1.0 / (c["rope_theta"] ** (torch.arange(0, d, 2, dtype=torch.float32, device="cpu") / d))
        angles = torch.arange(self.groups, dtype=torch.float32, device="cpu")[:, None] * inv[None, :]
        angles = torch.cat([angles, angles], -1).cuda()
        self.cos, self.sin = angles.cos().bfloat16(), angles.sin().bfloat16()
        self.cache = [(torch.empty(self.groups, self.kv_heads, d, device="cuda", dtype=torch.bfloat16),
                       torch.empty(self.groups, self.kv_heads, d, device="cuda", dtype=torch.bfloat16))
                      for _ in range(c["num_hidden_layers"])]
        self.lengths = {n: torch.tensor([0, n], device="cuda", dtype=torch.int32)
                        for n in range(1, self.groups + 1)}
        self.graph = None

    def norm(self, x, name):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return self.weights[name + ".weight"] * value.to(x.dtype)

    def linear(self, x, name):
        return F.linear(x, self.weights[name + ".weight"])

    def rope(self, x, start):
        cos = self.cos[start:start + len(x), None, :]
        sin = self.sin[start:start + len(x), None, :]
        half = self.head_dim // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], -1)
        return x * cos + rotated * sin

    def step(self, embeddings, start):
        x = embeddings
        end = start + len(x)
        for index, (keys, values) in enumerate(self.cache):
            p = f"model.layers.{index}."
            residual = x
            normalized = self.norm(x, p + "input_layernorm")
            q = self.linear(normalized, p + "self_attn.q_proj").view(-1, self.heads, self.head_dim)
            k = self.linear(normalized, p + "self_attn.k_proj").view(-1, self.kv_heads, self.head_dim)
            v = self.linear(normalized, p + "self_attn.v_proj").view(-1, self.kv_heads, self.head_dim)
            q = self.rope(self.norm(q, p + "self_attn.q_norm"), start)
            k = self.rope(self.norm(k, p + "self_attn.k_norm"), start)
            keys[start:end].copy_(k)
            values[start:end].copy_(v)
            attention = self.attention(
                q, keys[:end], values[:end], cu_seqlens_q=self.lengths[len(x)],
                cu_seqlens_k=self.lengths[end], max_seqlen_q=len(x), max_seqlen_k=end,
                softmax_scale=self.head_dim ** -0.5, causal=True, fa_version=2)
            x = residual + self.linear(attention.reshape(len(x), -1), p + "self_attn.o_proj")
            residual = x
            normalized = self.norm(x, p + "post_attention_layernorm")
            activated = F.silu(self.linear(normalized, p + "mlp.gate_proj"))
            activated = activated * self.linear(normalized, p + "mlp.up_proj")
            x = residual + self.linear(activated, p + "mlp.down_proj")
        return self.norm(x, "model.norm")[-1:]

    @staticmethod
    def sample(scores, temperature=0.9, top_k=50, top_p=1.0, generator=None):
        if temperature <= 0 or not 0 < top_p <= 1 or top_k < 0:
            raise ValueError("Invalid code predictor sampling parameters")
        scores = scores / temperature
        if top_k:
            threshold = scores.topk(min(top_k, scores.shape[-1]), dim=-1).values[..., -1:]
            scores = scores.masked_fill(scores < threshold, -float("inf"))
        if top_p < 1:
            sorted_scores, indices = scores.sort(dim=-1)
            remove = sorted_scores.softmax(-1).cumsum(-1) <= (1 - top_p)
            remove[..., -1] = False
            scores = scores.masked_fill(torch.zeros_like(remove).scatter(-1, indices, remove), -float("inf"))
        return torch.multinomial(scores.softmax(-1), 1, generator=generator).squeeze(-1)

    @torch.inference_mode()
    def eager(self, inputs, forced_codes=None, sampling=None):
        """Forced prefixes are only for teacher-forced numeric diagnostics."""
        x = inputs.reshape(2, self.config["hidden_size"])
        codes, logits = [], []
        for step in range(self.groups - 1):
            hidden = self.step(x, 0 if step == 0 else step + 1)
            scores = self.linear(hidden, f"lm_head.{step}").float()
            token = scores.argmax(-1) if sampling is None else self.sample(scores, **sampling)
            codes.append(token)
            logits.append(scores)
            if step < self.groups - 2:
                selected = token if forced_codes is None else forced_codes[step:step + 1]
                x = F.embedding(selected, self.weights[f"model.codec_embedding.{step}.weight"])
        return torch.stack(codes, -1), torch.stack(logits, 1)

    @torch.inference_mode()
    def capture(self):
        if self.graph is not None:
            return
        self.static_inputs = torch.zeros(2, self.config["hidden_size"], device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.eager(self.static_inputs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.static_outputs = self.eager(self.static_inputs)
        self.graph = graph

    @torch.inference_mode()
    def run(self, inputs, use_graph=True):
        if inputs.shape not in [(2, self.config["hidden_size"]), (1, 2, self.config["hidden_size"])]:
            raise ValueError("Code predictor supports one hidden state plus one first-codebook embedding")
        if not use_graph:
            return self.eager(inputs)
        self.capture()
        self.static_inputs.copy_(inputs.reshape_as(self.static_inputs))
        self.graph.replay()
        return tuple(value.clone() for value in self.static_outputs)
