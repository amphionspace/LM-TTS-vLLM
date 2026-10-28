def register():
    from importlib.util import find_spec
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "LMTTSTalkerModel", "lm_tts_vllm.model:LMTTSTalkerModel"
    )
    # Omni resolves HF config and the pipeline before its own general-plugin
    # hook. The standard vLLM hook runs early enough for automatic CLI startup.
    # Keep Omni optional for installations using only the diagnostic backbone.
    if find_spec("vllm_omni") is not None:
        from .omni import register as register_omni
        register_omni()
