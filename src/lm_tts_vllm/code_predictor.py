"""Batched BF16 code predictor using vLLM FlashAttention and a frame-local KV cache.

Inference-only. All rows of a scheduler batch decode the residual codebooks
together; greedy and sampled rows share one CUDA graph per batch bucket.
Sampling is Gumbel-max over request-local uniforms drawn outside the graph,
so every row keeps its own temperature, top-k, top-p and generator.
No SDPA or Transformers runtime is used.
"""
import json
from pathlib import Path

import torch
from torch.nn import functional as F


class CodePredictor:
    def __init__(self, source, max_batch=1):
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
        self.max_batch = max_batch
        self.cache = [(torch.empty(max_batch, self.groups, self.kv_heads, d, device="cuda", dtype=torch.bfloat16),
                       torch.empty(max_batch, self.groups, self.kv_heads, d, device="cuda", dtype=torch.bfloat16))
                      for _ in range(c["num_hidden_layers"])]
        self.vocab = self.weights["lm_head.0.weight"].shape[0]
        self.static = {
            "inputs": torch.zeros(max_batch, 2, c["hidden_size"], device="cuda", dtype=torch.bfloat16),
            "uniforms": torch.full((max_batch, self.groups - 1, self.vocab), 0.5, device="cuda"),
            # Per-row [inverse temperature, top-k, top-p, noise]; greedy rows use top-k 1 and no noise,
            # so BF16 logit ties resolve to the first index like argmax.
            "options": torch.ones(max_batch, 4, device="cuda"),
        }
        self.graphs = {}
        self.pool = None

    def norm(self, x, name):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + self.eps)
        return self.weights[name + ".weight"] * value.to(x.dtype)

    def linear(self, x, name):
        return F.linear(x, self.weights[name + ".weight"])

    def rope(self, x, start):
        # x: [batch, tokens, heads, head_dim]
        cos = self.cos[start:start + x.shape[1], None, :]
        sin = self.sin[start:start + x.shape[1], None, :]
        half = self.head_dim // 2
        rotated = torch.cat([-x[..., half:], x[..., :half]], -1)
        return x * cos + rotated * sin

    def step(self, x, start):
        """x: [batch, tokens, hidden] continuing every row at position ``start``."""
        batch, tokens, _ = x.shape
        end = start + tokens
        lengths_q = torch.arange(batch + 1, device=x.device, dtype=torch.int32) * tokens
        lengths_k = torch.arange(batch + 1, device=x.device, dtype=torch.int32) * end
        for index, (keys, values) in enumerate(self.cache):
            p = f"model.layers.{index}."
            residual = x
            normalized = self.norm(x, p + "input_layernorm")
            q = self.linear(normalized, p + "self_attn.q_proj").view(batch, tokens, self.heads, self.head_dim)
            k = self.linear(normalized, p + "self_attn.k_proj").view(batch, tokens, self.kv_heads, self.head_dim)
            v = self.linear(normalized, p + "self_attn.v_proj").view(batch, tokens, self.kv_heads, self.head_dim)
            q = self.rope(self.norm(q, p + "self_attn.q_norm"), start)
            k = self.rope(self.norm(k, p + "self_attn.k_norm"), start)
            keys[:batch, start:end].copy_(k)
            values[:batch, start:end].copy_(v)
            attention = self.attention(
                q.reshape(batch * tokens, self.heads, self.head_dim),
                keys[:batch, :end].reshape(batch * end, self.kv_heads, self.head_dim),
                values[:batch, :end].reshape(batch * end, self.kv_heads, self.head_dim),
                cu_seqlens_q=lengths_q, cu_seqlens_k=lengths_k, max_seqlen_q=tokens, max_seqlen_k=end,
                softmax_scale=self.head_dim ** -0.5, causal=True, fa_version=2)
            x = residual + self.linear(attention.reshape(batch, tokens, -1), p + "self_attn.o_proj")
            residual = x
            normalized = self.norm(x, p + "post_attention_layernorm")
            activated = F.silu(self.linear(normalized, p + "mlp.gate_proj"))
            activated = activated * self.linear(normalized, p + "mlp.up_proj")
            x = residual + self.linear(activated, p + "mlp.down_proj")
        return self.norm(x[:, -1], "model.norm")

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

    @staticmethod
    def pick(scores, uniforms, options):
        """Graph-safe per-row sampling: the same filter as ``sample``, then Gumbel-max."""
        inv_temperature, top_k, top_p, noise = options[:, :1], options[:, 1:2].long(), options[:, 2:3], options[:, 3:]
        scores = scores * inv_temperature
        ordered = scores.sort(-1, descending=True).values
        kth = ordered.gather(-1, top_k - 1)
        ordered = ordered.masked_fill(ordered < kth, -float("inf"))
        probs = ordered.softmax(-1)
        # Keep a token while the mass ranked strictly above it is below top_p.
        cutoff = ordered.masked_fill(probs.cumsum(-1) - probs >= top_p, float("inf")).amin(-1, keepdim=True)
        scores = scores.masked_fill(scores < torch.maximum(cutoff, kth), -float("inf"))
        return (scores - noise * torch.log(-torch.log(uniforms))).argmax(-1)

    @torch.inference_mode()
    def decode(self, inputs, uniforms=None, options=None, forced_codes=None):
        """inputs: [batch, 2, hidden] (Talker hidden, first-codebook embedding).

        Without options every row is greedy. Forced prefixes are only for
        teacher-forced numeric diagnostics.
        """
        x, codes, logits = inputs, [], []
        for step in range(self.groups - 1):
            hidden = self.step(x, 0 if step == 0 else step + 1)
            scores = self.linear(hidden, f"lm_head.{step}").float()
            token = scores.argmax(-1) if options is None else self.pick(scores, uniforms[:, step], options)
            codes.append(token)
            logits.append(scores)
            if step < self.groups - 2:
                selected = token if forced_codes is None else forced_codes[:, step]
                x = F.embedding(selected, self.weights[f"model.codec_embedding.{step}.weight"])[:, None]
        return torch.stack(codes, -1), torch.stack(logits, 1)

    def bucket(self, batch):
        size = 1
        while size < batch:
            size *= 2
        return min(size, self.max_batch)

    @torch.inference_mode()
    def capture(self):
        sizes = sorted({self.bucket(n) for n in range(1, self.max_batch + 1)}, reverse=True)
        s = self.static
        for size in sizes:
            if size in self.graphs:
                continue
            args = (s["inputs"][:size], s["uniforms"][:size], s["options"][:size])
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.decode(*args)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self.pool):
                codes, _ = self.decode(*args)
            self.pool = graph.pool()
            self.graphs[size] = (graph, codes)

    @torch.inference_mode()
    def run_batch(self, inputs, sampling=None, use_graph=True):
        """Residual codes [batch, groups - 1] for every row; ``sampling`` holds one dict or None (greedy) per row."""
        if inputs.ndim != 3 or inputs.shape[1:] != (2, self.config["hidden_size"]):
            raise ValueError("Expected code predictor inputs [batch, 2, hidden_size]")
        batch = len(inputs)
        if not 0 < batch <= self.max_batch:
            raise ValueError(f"Code predictor batch must be in 1..{self.max_batch}")
        options = [None] * batch if sampling is None else sampling
        if len(options) != batch:
            raise ValueError("Each request must have its own sampling configuration")
        rows = []
        for option in options:
            if option is None:
                rows.append((1.0, 1, 2.0, 0.0))
                continue
            temperature, top_k, top_p = option["temperature"], option.get("top_k", 0), option.get("top_p", 1.0)
            if temperature <= 0 or not 0 < top_p <= 1 or top_k < 0:
                raise ValueError("Invalid code predictor sampling parameters")
            rows.append((1.0 / temperature, min(top_k, self.vocab) or self.vocab, 2.0 if top_p >= 1 else top_p, 1.0))
        size = self.bucket(batch) if use_graph else batch
        s = self.static
        s["inputs"][:batch].copy_(inputs)
        s["options"][:batch].copy_(torch.tensor(rows, dtype=torch.float32), non_blocking=True)
        generators = [None if o is None else o.get("generator") for o in options]
        if any(g is not None for g in generators):
            # One request-local draw per frame keeps rows independent of batch order.
            for row, generator in enumerate(generators):
                s["uniforms"][row].uniform_(1e-20, 1 - 1e-7, generator=generator)
            s["uniforms"][batch:size].uniform_(1e-20, 1 - 1e-7)
        else:
            s["uniforms"][:size].uniform_(1e-20, 1 - 1e-7)
        if not use_graph:
            codes, _ = self.decode(s["inputs"][:batch], s["uniforms"][:batch], s["options"][:batch])
            return codes, None
        if size not in self.graphs:
            self.capture()
        graph, codes = self.graphs[size]
        graph.replay()
        return codes[:batch].clone(), None
