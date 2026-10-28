"""Export the UltraEval LM-TTS BF16 prefill and shared voice prompt for parity checks."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--ref-audio', type=Path, required=True)
    parser.add_argument('--ref-text', default='在此奉劝大家别乱打美白针。')
    parser.add_argument('--text', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'LM-TTS-Training'))
    import torch
    from lm_tts.inference.codec import CodecSynthesizer, extract_speaker_embedding
    synth = CodecSynthesizer(args.model, 'cuda:0', torch.bfloat16, 'sdpa')
    with torch.inference_mode():
        codes = synth.codec.encode(str(args.ref_audio)).audio_codes[0].to('cuda')
        speaker = extract_speaker_embedding(synth.model.speaker_encoder, args.ref_audio)
        ids = [synth.raw['tts_bos_token_id'],
               *synth.tokenizer.encode(args.ref_text, add_special_tokens=False),
               *synth.tokenizer.encode(args.text, add_special_tokens=False), synth.raw['tts_eos_token_id']]
        ids = torch.tensor(ids, device='cuda')
        batch = {'text_ids': ids, 'text_lengths': ids.new_tensor([len(ids)]),
                 'codes': codes, 'frame_lengths': ids.new_tensor([len(codes)]),
                 'streaming': torch.tensor([True], device='cuda'),
                 'language_codec_ids': ids.new_tensor([synth.raw['talker_config']['codec_language_id']['chinese']]),
                 'speaker_embeddings': speaker.unsqueeze(0)}
        embeds, _ = synth.model.input_embeddings(batch)
        hidden = synth.model.talker.model(**embeds, use_cache=False).last_hidden_state[:, -1]
        logits = synth.model.talker.codec_head(hidden).float()
        args.output.mkdir(parents=True, exist_ok=False)
        torch.save({'prefill': embeds['inputs_embeds'][0].cpu(), 'hidden': hidden.cpu(), 'logits': logits.cpu()},
                   args.output / 'reference.pt')
        info = {'text': [args.text], 'ref_text': [args.ref_text], 'language': ['Chinese'],
                'task_type': ['Base'], 'non_streaming_mode': [False], 'max_new_tokens': [80],
                'subtalker_dosample': [False], 'voice_clone_prompt': [{
                    'ref_code': codes.cpu().tolist(), 'ref_spk_embedding': speaker.float().cpu().tolist(),
                    'ref_text': args.ref_text, 'icl_mode': True, 'x_vector_only_mode': False}]}
        (args.output / 'prompt.json').write_text(json.dumps(info, ensure_ascii=False))
        print(json.dumps({'prefill_shape': list(embeds['inputs_embeds'].shape),
                          'reference_frames': len(codes), 'argmax': int(logits.argmax())}), flush=True)


if __name__ == '__main__':
    main()
