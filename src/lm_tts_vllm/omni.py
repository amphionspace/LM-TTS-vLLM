"""Optional native TTS pipeline registration, loaded only by vLLM-Omni."""
from dataclasses import replace
from pathlib import Path

from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig


class LMTTSConfig(Qwen3TTSConfig):
    model_type = "lm_tts"

    def __init__(self, **kwargs):
        # Transformers 5 generates a dataclass initializer for subclasses
        # without their own __init__, bypassing the parent's nested parsing.
        speaker = kwargs.get("speaker_encoder_config")
        if isinstance(speaker, dict):
            speaker = dict(speaker)
            # Upstream's speaker config has a closed constructor signature,
            # while HF serialization adds these metadata keys.
            speaker.pop("model_type", None)
            speaker.pop("transformers_version", None)
            kwargs["speaker_encoder_config"] = speaker
        super().__init__(**kwargs)
        # Supported non-streaming TTS uses equal positions on every MRoPE
        # axis, which is ordinary 1-D RoPE (as in the pooling export).
        rope = dict(self.talker_config.rope_parameters or self.talker_config.rope_scaling or {})
        rope.setdefault("rope_type", rope.pop("type", "default"))
        rope.setdefault("rope_theta", self.talker_config.rope_theta)
        rope.pop("mrope_section", None)
        rope.pop("interleaved", None)
        self.talker_config.rope_parameters = rope
        self.talker_config.rope_scaling = dict(rope)

    def update(self, config_dict):
        # vLLM applies per-stage overrides after HF construction. Keep nested
        # configs typed when Code2Wav adjusts max_position_embeddings.
        remaining = dict(config_dict)
        for name in ("talker_config", "speaker_encoder_config"):
            value = remaining.get(name)
            if isinstance(value, dict):
                getattr(self, name).update(value)
                remaining.pop(name)
        super().update(remaining)


def register():
    from transformers import AutoConfig
    from vllm import ModelRegistry
    from vllm.transformers_utils.config import _CONFIG_REGISTRY
    from vllm_omni.config.pipeline_registry import register_pipeline
    from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_PIPELINE
    from vllm_omni.model_executor.models.registry import OmniModelRegistry
    AutoConfig.register("lm_tts", LMTTSConfig, exist_ok=True)
    _CONFIG_REGISTRY["lm_tts"] = LMTTSConfig
    target = "lm_tts_vllm.omni_model:LMTTSForConditionalGeneration"
    for registry in [ModelRegistry, OmniModelRegistry]:
        registry.register_model("LMTTSForConditionalGeneration", target)
    register_pipeline(replace(
        QWEN3_TTS_PIPELINE, model_type="lm_tts", model_arch="LMTTSForConditionalGeneration",
        stages=(replace(QWEN3_TTS_PIPELINE.stages[0], custom_process_next_stage_input_func=
                        "lm_tts_vllm.native_payload.talker2code2wav_full_payload"),
                QWEN3_TTS_PIPELINE.stages[1]),
        default_deploy_config_name=str(Path(__file__).with_name("native_validation.yaml"))))
