def register():
    # Omni resolves HF config and the pipeline before its own general-plugin
    # hook, so the standard vLLM hook registers early enough for CLI startup.
    from .omni import register as register_omni
    register_omni()
