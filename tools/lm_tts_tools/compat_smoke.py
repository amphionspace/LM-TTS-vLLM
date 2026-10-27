"""Exercise native Base voice-cloning modes and scheduler batching."""
import argparse
import base64
import copy
import json
import math
from operator import methodcaller
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--ref-audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cached-prompt', action='store_true',
                        help='Export native voice prompts over standard apply_model RPC and reuse them')
    parser.add_argument('--audio-formats', action='store_true',
                        help='Exercise path, file URI, base64, and waveform inputs across the four cases')
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoTokenizer
    from vllm_omni import Omni
    from vllm_omni.plugins import load_omni_general_plugins
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder
    load_omni_general_plugins()
    model = args.model.resolve()
    config = json.loads((model / 'config.json').read_text())['talker_config']
    codec = json.loads((model / 'speech_tokenizer/config.json').read_text())
    tok = AutoTokenizer.from_pretrained(model)
    ref = sf.info(args.ref_audio)
    ref_len = math.ceil(ref.frames * codec['input_sample_rate'] / ref.samplerate / codec['encode_downsample_rate'])
    requests, cases = [], []
    for xvec in [False, True]:
        for nonstream in [True, False]:
            info = {'task_type': ['Base'], 'ref_audio': [str(args.ref_audio.resolve())],
                    'text': ['你好，这是语音合成测试。'], 'language': ['Chinese'],
                    'x_vector_only_mode': [xvec], 'non_streaming_mode': [nonstream],
                    'max_new_tokens': [8], 'full_utterance_decode': [True],
                    'subtalker_dosample': [not nonstream], 'subtalker_temperature': [0.9],
                    'subtalker_top_k': [50], 'subtalker_top_p': [0.9]}
            if not xvec:
                info.update(ref_text=['在此奉劝大家别乱打美白针。'], ref_code_length=[ref_len])
            length = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
                info, task_type='Base', tokenize_prompt=lambda t: tok(t)['input_ids'],
                codec_language_id=config['codec_language_id'], spk_is_dialect=config.get('spk_is_dialect'))
            requests.append({'prompt_token_ids': [0] * length, 'additional_information': info})
            cases.append({'x_vector_only': xvec, 'non_streaming_mode': nonstream, 'prompt_length': length,
                          'cp_sampling': not nonstream})
    if args.audio_formats:
        if args.cached_prompt:
            parser.error('--audio-formats and --cached-prompt exercise different input paths')
        wav, sr = sf.read(args.ref_audio, dtype='float32')
        inputs = [str(args.ref_audio.resolve()), args.ref_audio.resolve().as_uri(),
                  'data:audio/wav;base64,' + base64.b64encode(args.ref_audio.read_bytes()).decode(),
                  (wav, sr)]
        for request, case, audio, label in zip(requests, cases, inputs, ['path', 'file_uri', 'base64', 'waveform']):
            request['additional_information']['ref_audio'] = [audio]
            case['audio_format'] = label
    args.output.mkdir(parents=True, exist_ok=False)
    engine = Omni(model=str(model), stage_overrides={
        '0': {'max_num_seqs': 2, 'kv_cache_memory_bytes': 536870912}, '1': {'max_num_seqs': 2}})
    try:
        if args.cached_prompt:
            for xvec in [False, True]:
                result = engine.engine.collective_rpc('apply_model', timeout=120,
                    args=(methodcaller('create_voice_clone_prompt', str(args.ref_audio.resolve()),
                                       None if xvec else '在此奉劝大家别乱打美白针。', xvec),), stage_ids=[0])
                # Omni aggregates stages, replicas, and worker results.
                while isinstance(result, list) and len(result) == 1:
                    result = result[0]
                if not isinstance(result, dict) or 'ref_spk_embedding' not in result:
                    raise RuntimeError(f'Unexpected voice prompt RPC result: {type(result)}')
                (args.output / f'voice-prompt-{xvec}.json').write_text(json.dumps(result))
                for request, case in zip(requests, cases):
                    if case['x_vector_only'] == xvec:
                        info = request['additional_information']
                        info.pop('ref_audio')
                        info['voice_clone_prompt'] = [result]
                        case['cached_prompt'] = True
        params = copy.deepcopy(engine.default_sampling_params_list)
        params[0].max_tokens = 8
        params[0].seed = 42
        params[0].extra_args = {'tts_local_seed': 42}
        rows = []
        for output in engine.generate(requests, sampling_params_list=params, use_tqdm=False):
            index = int(output.request_id.split('_', 1)[0])
            mm = output.outputs[0].multimodal_output
            audio, codes, sr = mm['audio'], mm['codec_frames'], mm['sr']
            if isinstance(audio, list): audio = torch.cat(audio, dim=-1)
            if isinstance(codes, list): codes = torch.cat(codes, dim=0)
            if isinstance(sr, list): sr = sr[-1]
            assert torch.isfinite(audio).all() and audio.numel() > 0
            assert codes.ndim == 2 and codes.shape[1] == 16 and codes.numel() > 0
            assert (codes >= 0).all() and (codes < 2048).all()
            sf.write(args.output / f'{index}.wav', audio.float().cpu().numpy().flatten(), int(sr))
            np.save(args.output / f'{index}-codes.npy', codes.cpu().numpy())
            rows.append(dict(cases[index], index=index, codec_shape=list(codes.shape), audio_seconds=audio.numel()/int(sr)))
        assert len(rows) == len(cases), (len(rows), len(cases))
        (args.output / 'report.json').write_text(json.dumps(rows, indent=2)+'\n')
        print(json.dumps(rows), flush=True)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
