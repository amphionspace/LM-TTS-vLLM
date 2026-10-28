from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODELS = {
    "56384": {"source": Path('/workspace/workspace/yanglin/UltraEval-Audio/init_model/emilia-v4-emilia2-cbinit-langid-step-00056384'),
              "native": ROOT / 'artifacts/native-56384'},
    "10000": {"source": Path('/workspace/workspace/yanglin/UltraEval-Audio/init_model/emilia-v4-emilia2-cbinit-langid-interleaved5x10-step-00010000'),
              "native": ROOT / 'artifacts/native-10000'},
}
REFERENCE = '/workspace/workspace/yanglin/UltraEval-Audio/raw_data/voice_clone_manifests/seed_tts_eval_zh/000000.wav'
