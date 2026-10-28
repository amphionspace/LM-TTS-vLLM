"""LM-TTS adaptations for the native vLLM-Omni TTS model lifecycle.

The optional runtime is pinned separately. It owns requests, KV allocation,
sampling, scheduling and waveform delivery; no development tools are imported.
"""
import os
from contextlib import contextmanager

import torch
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import (
    Qwen3TTSTalkerForConditionalGeneration,
)

from .precision import apply_bf16
from .code_predictor import CodePredictor
from .native_payload import normalize_voice_prompt
from .audio import canonical_audio_string
from .interleaved import TrainingPromptBuilder, uses_training_layout


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
        self._training_layout = uses_training_layout(self.talker_config)
        self._training_plans = {}
        self._training_sample_counts = None
        if self._training_layout:
            self._prompt_builder = TrainingPromptBuilder(self._prompt_builder, self)
            self.logitsprocs_need_output_token_ids = True
            self.mtp_disable_graph = True  # CP retains its own per-frame graph.

    @contextmanager
    def mrv2_sampling_context(self, *, req_ids, num_output_tokens):
        with super().mrv2_sampling_context(req_ids=req_ids, num_output_tokens=num_output_tokens):
            self._training_sample_counts = num_output_tokens.cpu().tolist() if self._training_layout else None
            try:
                yield
            finally:
                self._training_sample_counts = None

    def mrv2_custom_sampler(self, sampler):
        if self._training_layout:
            return None
        return super().mrv2_custom_sampler(sampler)

    def compute_logits(self, hidden_states, sampling_metadata=None):
        logits = super().compute_logits(hidden_states, sampling_metadata)
        if logits is not None and self._training_sample_counts is not None:
            for row, (req_id, count) in enumerate(zip(self._batch_req_ids, self._training_sample_counts)):
                plan = self._training_plans.get(req_id)
                if plan is None or count >= len(plan.slots):
                    continue
                if plan.slots[count] >= 0:
                    logits[row].fill_(-float('inf'))
                    logits[row, self.talker_config.codec_pad_id] = 0
                elif not plan.text_done[count]:
                    logits[row, self._codec_eos_token_id] = -float('inf')
        return logits

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        self._native_cp = CodePredictor(self.model_path)
        self._native_cp.capture()
        return loaded

    @torch.inference_mode()
    def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
        """Export one reusable CPU payload through vLLM's standard apply_model RPC.

        The existing native speaker/codec encoders own preprocessing and BF16
        execution. No second Transformers model or custom worker is needed.
        """
        if not isinstance(x_vector_only_mode, bool):
            raise ValueError("x_vector_only_mode must be a boolean")
        if not x_vector_only_mode and (not isinstance(ref_text, str) or not ref_text.strip()):
            raise ValueError("ICL voice cloning requires reference text")
        wav, sr = self._prompt_builder.normalize_ref_audio(canonical_audio_string(ref_audio))
        speaker = self._prompt_builder.extract_speaker_embedding(wav, sr)
        codes = None
        if not x_vector_only_mode:
            codes = self._encode_ref_audio_batch([wav], sr, device=speaker.device)[0].cpu().tolist()
        return {"ref_code": codes, "ref_spk_embedding": speaker.float().cpu().tolist(),
                "ref_text": ref_text, "x_vector_only_mode": x_vector_only_mode,
                "icl_mode": not x_vector_only_mode}

    @staticmethod
    def _validate_prompt(info):
        info.update(normalize_voice_prompt(info))
        if isinstance(info.get("ref_audio"), list):
            info["ref_audio"] = [canonical_audio_string(value) for value in info["ref_audio"]]
        mode = info.get("non_streaming_mode")
        mode = mode[0] if isinstance(mode, list) and mode else mode
        if mode is not None and not isinstance(mode, bool):
            raise ValueError("non_streaming_mode must be a boolean")

    def preprocess(self, input_ids, input_embeds, **info):
        self._validate_prompt(info)
        return super().preprocess(input_ids, input_embeds, **info)

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
        if self._training_layout:
            return self._training_mtp(input_ids, input_embeds, last_talker_hidden, text_step, **kwargs)
        return self._codec_mtp(input_ids, input_embeds, last_talker_hidden, text_step, **kwargs)

    def _training_mtp(self, input_ids, input_embeds, last_talker_hidden, text_step, **kwargs):
        # Text slots must not run CP or consume its sampling generator.
        ids = input_ids.reshape(-1)
        real = (ids >= 0) & (ids < self._codebook_vocab_size)
        indices = real.nonzero().flatten()
        combined = input_embeds.reshape(len(ids), -1).clone()
        codes = torch.full((len(ids), 16), -1, dtype=torch.long, device=ids.device)
        if indices.numel():
            options = dict(kwargs)
            selected = indices.cpu().tolist()
            for key in ('req_infos', 'generators'):
                if options.get(key) is not None:
                    options[key] = [options[key][i] for i in selected]
            frame_embeds, frames = self._codec_mtp(
                ids[indices], input_embeds.reshape(len(ids), -1)[indices],
                last_talker_hidden.reshape(len(ids), -1)[indices],
                text_step.reshape(len(ids), -1)[indices], **options)
            combined[indices] = frame_embeds
            codes[indices] = frames
        return combined, codes

    def _codec_mtp(self, input_ids, input_embeds, last_talker_hidden, text_step, **kwargs):
        do_sample = kwargs.get("do_sample")
        if do_sample is None:
            do_sample = self._subtalker_sampling_params.get("do_sample", True)
        infos = kwargs.get("req_infos") or []
        batch = input_ids.shape[0]
        inputs = torch.cat([last_talker_hidden.reshape(batch, 1, -1),
                            input_embeds.reshape(batch, 1, -1)], dim=1).bfloat16()
        defaults = self._subtalker_sampling_params
        generators = kwargs.get("generators")
        options = []
        for index in range(batch):
            info = infos[index] if index < len(infos) else {}
            def param(name, default):
                value = info.get(name, default)
                return value[0] if isinstance(value, list) and value else value
            if not param("subtalker_dosample", do_sample):
                options.append(None)
                continue
            options.append({
                "temperature": float(param("subtalker_temperature", kwargs.get("temperature") or defaults.get("temperature", 0.9))),
                "top_k": int(param("subtalker_top_k", kwargs.get("top_k") if kwargs.get("top_k") is not None else defaults.get("top_k", 50))),
                "top_p": float(param("subtalker_top_p", kwargs.get("top_p") or defaults.get("top_p", 1.0))),
                "generator": generators[index] if generators is not None else kwargs.get("generator"),
            })
        residual_codes, _ = self._native_cp.run_batch(
            inputs, sampling=options, use_graph=not torch.cuda.is_current_stream_capturing())
        codes = torch.cat([input_ids.reshape(batch, 1).long(), residual_codes], dim=1)
        invalid = (codes[:, :1] < 0) | (codes[:, :1] >= self._codebook_vocab_size)
        codes = torch.where(invalid, torch.zeros_like(codes), codes)
        rows = torch.arange(codes.shape[1] - 1, device=codes.device)[None, :]
        residual = self._stacked_codec_embed[rows, codes[:, 1:]]
        # Reference sums all 16 BF16 embeddings together. Adding CB0 after
        # reducing the other 15 introduces an extra BF16 rounding boundary.
        embeddings = torch.cat([input_embeds.reshape(batch, 1, -1), residual], dim=1)
        if self._training_layout:
            combined = embeddings[:, 0]
            for group in range(1, embeddings.shape[1]):
                combined = combined + embeddings[:, group]
        else:
            combined = embeddings.sum(dim=1)
        combined = combined + text_step.reshape(batch, -1)
        return combined, codes

    # Upstream MRv2 resolves this alias directly, not through talker_mtp.
    mtp = talker_mtp
