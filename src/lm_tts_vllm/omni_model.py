"""LM-TTS adaptations for the native vLLM-Omni TTS model lifecycle.

The optional runtime is pinned separately. It owns requests, KV allocation,
sampling, scheduling and waveform delivery; no development tools are imported.
"""
import os

import torch
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    Qwen3TTSTalkerForConditionalGeneration,
)

from .precision import apply_bf16
from .code_predictor import CodePredictor


class LMTTSForConditionalGeneration(Qwen3TTSTalkerForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if vllm_config.parallel_config.tensor_parallel_size != 1 or vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("LM-TTS native validation currently requires TP=PP=1")
        # Original Qwen-TTS permits codec zero. The upstream default masks it.
        self._codec_disallowed_mask[0] = False
        profile = os.environ.get("LM_TTS_PRECISION", "bf16")
        if profile == "bf16":
            apply_bf16(self.model, self.talker_config)
        elif profile != "native":
            raise ValueError("Native LM-TTS supports bf16 or native precision profiles")
        self.mtp_sample_uniforms = False
        self.mtp_accepts_req_infos = True
        self._native_cp = None

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        self._native_cp = CodePredictor(self.model_path)
        self._native_cp.capture()
        return loaded

    @staticmethod
    def _validate_prompt(info):
        mode = info.get("non_streaming_mode")
        mode = mode[0] if isinstance(mode, list) and mode else mode
        if mode is not True:
            raise ValueError("LM-TTS native validation requires non_streaming_mode=True")

    def preprocess_batch_mrv2(self, *, req_infos, device):
        for info in req_infos:
            self._validate_prompt(info)
        return super().preprocess_batch_mrv2(req_infos=req_infos, device=device)

    def preprocess_batch(self, *, req_ids, model_intermediate_buffer, device):
        for req_id in req_ids:
            self._validate_prompt(model_intermediate_buffer[req_id])
        return super().preprocess_batch(req_ids=req_ids, model_intermediate_buffer=model_intermediate_buffer, device=device)

    @torch.inference_mode()
    def talker_mtp(self, input_ids, input_embeds, last_talker_hidden, text_step, **kwargs):
        do_sample = kwargs.get("do_sample")
        if do_sample is None:
            do_sample = self._subtalker_sampling_params.get("do_sample", True)
        infos = kwargs.get("req_infos") or []
        info = infos[0] if infos else {}
        def param(name, default):
            value = info.get(name, default)
            return value[0] if isinstance(value, list) and value else value
        do_sample = param("subtalker_dosample", do_sample)
        batch = input_ids.shape[0]
        if batch != 1:
            raise ValueError("LM-TTS native code predictor currently supports batch size one")
        inputs = torch.cat([last_talker_hidden.reshape(batch, 1, -1),
                            input_embeds.reshape(batch, 1, -1)], dim=1).bfloat16()
        if do_sample:
            defaults = self._subtalker_sampling_params
            sampling = {"temperature": float(param("subtalker_temperature", kwargs.get("temperature") or defaults.get("temperature", 0.9))),
                        "top_k": int(param("subtalker_top_k", kwargs.get("top_k") if kwargs.get("top_k") is not None else defaults.get("top_k", 50))),
                        "top_p": float(param("subtalker_top_p", kwargs.get("top_p") or defaults.get("top_p", 1.0))),
                        "generator": kwargs.get("generator")}
            residual_codes, _ = self._native_cp.eager(inputs, sampling=sampling)
        else:
            residual_codes, _ = self._native_cp.run(inputs, use_graph=not torch.cuda.is_current_stream_capturing())
        codes = torch.cat([input_ids.reshape(batch, 1).long(), residual_codes], dim=1)
        invalid = (codes[:, :1] < 0) | (codes[:, :1] >= self._codebook_vocab_size)
        codes = torch.where(invalid, torch.zeros_like(codes), codes)
        rows = torch.arange(codes.shape[1] - 1, device=codes.device)[None, :]
        residual = self._stacked_codec_embed[rows, codes[:, 1:]]
        # Reference sums all 16 BF16 embeddings together. Adding CB0 after
        # reducing the other 15 introduces an extra BF16 rounding boundary.
        embeddings = torch.cat([input_embeds.reshape(batch, 1, -1), residual], dim=1)
        combined = embeddings.sum(dim=1) + text_step.reshape(batch, -1)
        return combined, codes

    # Upstream MRv2 resolves this alias directly, not through talker_mtp.
    mtp = talker_mtp
