def register():
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "LMTTSTalkerModel", "lm_tts_vllm.model:LMTTSTalkerModel"
    )
