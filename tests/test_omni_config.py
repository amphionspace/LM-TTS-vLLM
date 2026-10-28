"""Run in the optional Omni environment; no weights or GPU required."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("vllm_omni"), "optional Omni runtime not installed")
class OmniConfigTests(unittest.TestCase):
    def test_standard_vllm_hook_registers_native_pipeline_and_only_our_adapter(self):
        from lm_tts_vllm import register
        from lm_tts_vllm.omni import register as register_omni
        from transformers import AutoConfig
        from vllm_omni.config.pipeline_registry import resolve_pipeline_config
        from vllm_omni.entrypoints.openai.tts_adapters import detect_tts_model_type
        register()
        register_omni()  # Both public discovery hooks may run in one process.
        self.assertEqual(AutoConfig.for_model('lm_tts').model_type, 'lm_tts')
        self.assertEqual(resolve_pipeline_config('lm_tts').model_arch, 'LMTTSForConditionalGeneration')
        self.assertEqual(detect_tts_model_type('qwen3_tts', 'LMTTSForConditionalGeneration'), 'lm_tts')
        self.assertEqual(detect_tts_model_type('qwen3_tts', 'Qwen3TTSTalkerForConditionalGeneration'), 'qwen3_tts')

    def test_nested_config_survives_transformers_roundtrip_and_stage_override(self):
        from lm_tts_vllm.omni import LMTTSConfig
        raw = {"model_type": "lm_tts", "talker_config": {
            "hidden_size": 1024, "num_attention_heads": 16, "num_hidden_layers": 28,
            "codec_eos_token_id": 2150,
            "rope_scaling": {"rope_type": "default", "mrope_section": [24, 20, 20],
                             "interleaved": True}, "rope_theta": 1000000}}
        config = LMTTSConfig.from_dict(raw)
        self.assertEqual(config.get_text_config().num_attention_heads, 16)
        self.assertEqual(config.talker_config.codec_eos_token_id, 2150)
        self.assertNotIn("mrope_section", config.talker_config.rope_parameters)
        copy = LMTTSConfig.from_dict(config.to_dict())
        copy.update({"architectures": ["Qwen3TTSCode2Wav"],
                     "talker_config": {"max_position_embeddings": 8192}})
        self.assertEqual(copy.talker_config.hidden_size, 1024)
        self.assertEqual(copy.talker_config.num_hidden_layers, 28)
        self.assertEqual(copy.talker_config.max_position_embeddings, 8192)
        self.assertEqual(copy.get_text_config().num_attention_heads, 16)


if __name__ == "__main__":
    unittest.main()
