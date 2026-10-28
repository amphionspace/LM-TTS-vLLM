from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODELS = {
    "56384": {"source": Path('/workspace/workspace/yanglin/UltraEval-Audio/init_model/emilia-v4-emilia2-cbinit-langid-step-00056384'),
              "native": ROOT / 'artifacts/native-56384'},
    "10000": {"source": Path('/workspace/workspace/yanglin/UltraEval-Audio/init_model/emilia-v4-emilia2-cbinit-langid-interleaved5x10-step-00010000'),
              "native": ROOT / 'artifacts/native-10000'},
}
REFERENCE = str(ROOT / 'assets/reference/vo_anecdote_part1_4_zhongli_04.wav')
REFERENCE_TEXT = '早听说枫丹的土壤水质都与璃月大不相同，不知道这枫丹种出来的茶叶，究竟是什么味道。'
