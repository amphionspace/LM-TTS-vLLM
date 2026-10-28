"""LM-TTS request options on the standard Omni speech endpoint."""
import math

from vllm_omni.entrypoints.openai.tts_adapters.qwen3_tts import Qwen3TTSAdapter
from .audio import canonical_audio_string

CP_KEYS = ('subtalker_dosample', 'subtalker_temperature', 'subtalker_top_k', 'subtalker_top_p')


class LMTTSAdapter(Qwen3TTSAdapter):
    name = 'lm_tts'
    stage_keys = frozenset()
    model_archs = frozenset({'LMTTSForConditionalGeneration'})
    detect_priority = 50
    max_new_tokens_min = 2

    def normalize(self, request):
        if isinstance(request.ref_audio, list):
            request.ref_audio = [canonical_audio_string(value) for value in request.ref_audio]
        else:
            request.ref_audio = canonical_audio_string(request.ref_audio)
        return super().normalize(request)

    def validate(self, request):
        error = super().validate(request)
        if error:
            return error
        extra = request.extra_params or {}
        for key in ('do_sample', 'subtalker_dosample'):
            if key in extra and not isinstance(extra[key], bool):
                return f'{key} must be a boolean'
        for key in ('temperature', 'subtalker_temperature', 'top_p', 'subtalker_top_p', 'repetition_penalty'):
            if key not in extra:
                continue
            value = extra[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return f'{key} must be a finite number'
            lower = value >= 0 if key == 'temperature' else value > 0
            if not lower or (key.endswith('top_p') and value > 1):
                return f'{key} is outside its supported range'
        for key in ('top_k', 'subtalker_top_k'):
            if key in extra and (type(extra[key]) is not int or extra[key] < (-1 if key == 'top_k' else 0)):
                return f'{key} must be an integer with no limit represented by 0'
        return None

    def _build_tts_params(self, request):
        params = super()._build_tts_params(request)
        extra = request.extra_params or {}
        for key in CP_KEYS:
            if key in extra:
                params[key] = [extra[key]]
        if request.max_new_tokens is not None:
            params['lm_tts_explicit_cap'] = [True]
        return params

    def apply_sampling_overrides(self, sampling_params_list, request, prompt=None, request_id=None):
        params = super().apply_sampling_overrides(sampling_params_list, request, prompt, request_id)
        extra = request.extra_params or {}
        if extra.get('do_sample') is False:
            params[0].temperature = 0.0
        if extra.get('top_k') == 0:
            params[0].top_k = -1
        if 'repetition_penalty' in extra:
            params[0].repetition_penalty = float(extra['repetition_penalty'])
        # Preserve stock Qwen's full-payload behavior under our model-type name.
        if request.is_streaming() and not getattr(self.ctx.server.model_config, 'async_chunk', True):
            from vllm.sampling_params import RequestOutputKind
            for param in params:
                param.output_kind = RequestOutputKind.FINAL_ONLY
        return params

    def validate_generation(self, tts_params, *, stage0_finish_reason, output_tokens):
        # Original Qwen's generate returns the available frames at an explicit
        # caller limit. Keep automatic budget exhaustion as an error.
        if stage0_finish_reason == 'length' and tts_params.get('lm_tts_explicit_cap') == [True]:
            return
        return super().validate_generation(tts_params, stage0_finish_reason=stage0_finish_reason,
                                           output_tokens=output_tokens)
