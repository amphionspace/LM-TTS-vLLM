"""Generate long-text samples with the PyTorch LM-TTS baseline (UltraEval CodecSynthesizer, BF16 SDPA)."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--ref-audio', type=Path, required=True)
    p.add_argument('--ref-text', required=True)
    p.add_argument('--texts', type=Path, required=True, help='JSON list of target texts')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3])
    p.add_argument('--max-frames', type=int, default=400)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import soundfile as sf
    import torch
    sys.path.insert(0, str(ROOT.parent / 'LM-TTS-Training'))
    from lm_tts.inference.codec import CodecSynthesizer
    engine = CodecSynthesizer(args.model, 'cuda:0', torch.bfloat16, 'sdpa')
    args.output.mkdir(parents=True, exist_ok=True)
    for t, text in enumerate(json.loads(args.texts.read_text())):
        for seed in args.seeds:
            torch.manual_seed(seed)
            wav, rate = engine.synthesize(text, reference_audio=str(args.ref_audio), reference_text=args.ref_text,
                                          speaker_audio=str(args.ref_audio), streaming=True, language_codec_id=-1,
                                          penalize_reference=False, use_cache=True, max_new_tokens=args.max_frames - 1,
                                          min_new_tokens=2, do_sample=True, temperature=0.9, top_k=50, top_p=1.0,
                                          repetition_penalty=1.05, subtalker_dosample=True, subtalker_temperature=0.9,
                                          subtalker_top_k=50, subtalker_top_p=1.0)
            sf.write(args.output / f'pytorch-t{t}-s{seed}.wav', wav, rate)
            print(f't{t} s{seed} {len(wav) / rate:.2f}s', flush=True)


if __name__ == '__main__':
    main()
