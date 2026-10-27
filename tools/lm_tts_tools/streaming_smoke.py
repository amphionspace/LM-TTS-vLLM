"""Check real incremental audio delivery through the standard AsyncOmni API."""
import argparse
import asyncio
import copy
import json
import math
from pathlib import Path
import time


async def run(args):
    import numpy as np
    import soundfile as sf
    import torch
    from vllm.sampling_params import RequestOutputKind
    from transformers import AutoTokenizer
    from vllm_omni import AsyncOmni
    from vllm_omni.plugins import load_omni_general_plugins
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder
    import lm_tts_vllm

    load_omni_general_plugins()
    model = args.model.resolve()
    config = json.loads((model / 'config.json').read_text())['talker_config']
    codec = json.loads((model / 'speech_tokenizer/config.json').read_text())
    tok = AutoTokenizer.from_pretrained(model)
    ref = sf.info(args.ref_audio)
    info = {'task_type': ['Base'], 'ref_audio': [str(args.ref_audio.resolve())],
            'ref_text': ['在此奉劝大家别乱打美白针。'],
            'text': ['你好，这是语音合成测试。我们正在测试流式音频输出，检查每一段声音是否正常。'],
            'language': ['Chinese'], 'x_vector_only_mode': [False],
            'non_streaming_mode': [True], 'full_utterance_decode': [False],
            'max_new_tokens': [80], 'subtalker_dosample': [False],
            'ref_code_length': [math.ceil(ref.frames * codec['input_sample_rate'] /
                                          ref.samplerate / codec['encode_downsample_rate'])]}
    length = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        info, task_type='Base', tokenize_prompt=lambda t: tok(t)['input_ids'],
        codec_language_id=config['codec_language_id'], spk_is_dialect=config.get('spk_is_dialect'))
    args.output.mkdir(parents=True, exist_ok=False)
    engine = AsyncOmni(model=str(model), deploy_config=str(
        Path(lm_tts_vllm.__file__).parent / 'native_streaming.yaml'),
        stage_overrides={'0': {'kv_cache_memory_bytes': 536870912}})
    try:
        assert engine.async_chunk, 'Streaming deploy config was not applied'
        params = copy.deepcopy(engine.default_sampling_params_list)
        for param in params:
            param.output_kind = RequestOutputKind.DELTA
        params[0].max_tokens = 80
        params[0].min_tokens = 80  # Force the capped-generation boundary.
        params[0].seed = 42
        params[0].extra_args = {'tts_local_seed': 42}
        start, chunks, final = time.perf_counter(), [], None
        async for output in engine.generate(
                {'prompt_token_ids': [0] * length, 'additional_information': info},
                request_id='stream-cap', sampling_params_list=params):
            mm = output.outputs[0].multimodal_output
            audio = mm.get('audio')
            if audio is None:
                continue
            audio = torch.cat(audio, dim=-1) if isinstance(audio, list) else audio
            assert audio.numel() > 0 and torch.isfinite(audio).all()
            sr = mm['sr'][-1] if isinstance(mm['sr'], list) else mm['sr']
            row = {'finished': output.finished, 'samples': audio.numel(),
                   'elapsed_seconds': time.perf_counter() - start}
            print(json.dumps(row), flush=True)
            if output.finished:
                final = row
                sf.write(args.output / 'final.wav', audio.float().cpu().numpy().flatten(), int(sr))
                codes = mm.get('codec_frames')
                if codes is not None:
                    codes = torch.cat(codes, dim=0) if isinstance(codes, list) else codes
                    np.save(args.output / 'codes.npy', codes.cpu().numpy())
                    final['codec_shape'] = list(codes.shape)
            else:
                sf.write(args.output / f'chunk-{len(chunks)}.wav',
                         audio.float().cpu().numpy().flatten(), int(sr))
                chunks.append(row)
        report = {'chunks': chunks, 'final': final}
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        assert final is not None, 'Missing final output'
        assert len(chunks) > 1, 'No real incremental audio delivery'
        assert final['samples'] == 79 * 1920, 'Capped generation must retain 79 codec frames'
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--ref-audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == '__main__':
    main()
