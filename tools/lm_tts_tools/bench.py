"""Measure generation speed (RTF) of full-text and streamed-text requests for one deploy variant."""
import argparse
import asyncio
import copy
import json
import time
from pathlib import Path

TEXT = ('今天天气这么好，阳光照在身上暖洋洋的，我们一起去公园散散步，顺便买杯咖啡吧。'
        '回来的路上还可以去书店看看，听说最近新到了不少好书。')


async def run(args):
    import torch
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.protocol import StreamingInput
    from vllm.sampling_params import RequestOutputKind
    from vllm_omni import AsyncOmni
    from vllm_omni.plugins import load_omni_general_plugins
    import lm_tts_vllm
    from lm_tts_vllm.input_stream import TextStreamBuffer
    from lm_tts_vllm.interleaved import request_layout

    load_omni_general_plugins()
    base = json.loads(args.prompt.read_text())
    tok = AutoTokenizer.from_pretrained(args.model)
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    raw = json.loads((args.model / 'config.json').read_text())
    talker = raw['talker_config']
    refs = len(base['voice_clone_prompt'][0]['ref_code'])
    overrides = {'0': {'kv_cache_memory_bytes': 536870912}}
    for stage, values in json.loads(args.overrides).items():
        overrides.setdefault(stage, {}).update(values)
    started = time.perf_counter()
    engine = AsyncOmni(model=str(args.model), deploy_config=str(
        Path(lm_tts_vllm.__file__).parent / 'native_input_streaming.yaml'), stage_overrides=overrides)
    load_seconds = time.perf_counter() - started

    def params(max_tokens, **extra):
        return SamplingParams(temperature=0.9, top_k=50, repetition_penalty=1.05, seed=42, max_tokens=max_tokens,
                              detokenize=False, output_kind=RequestOutputKind.DELTA,
                              extra_args={'tts_local_seed': 42}, **extra)

    async def timed(prompt, request_id, stage0):
        stage_params = copy.deepcopy(engine.default_sampling_params_list)
        if stage0 is not None:
            stage_params[0] = stage0
        for p in stage_params:
            p.output_kind = RequestOutputKind.DELTA
        start, first, samples, arrivals, sizes = time.perf_counter(), None, 0, [], []
        async for output in engine.generate(prompt, request_id=request_id, sampling_params_list=stage_params):
            audio = (output.outputs[0].multimodal_output or {}).get('audio')
            if audio is None:
                continue
            audio = torch.cat(audio, dim=-1) if isinstance(audio, list) else audio
            if audio.numel():
                now = time.perf_counter() - start
                first = first if first is not None else now
                samples += audio.numel()
                arrivals.append(now)
                sizes.append(audio.numel() / 24000)
        wall = time.perf_counter() - start
        seconds = samples / 24000
        gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
        # Steady RTF: generation time per second of audio after the first chunk; < 1 plays without stalls.
        steady = (arrivals[-1] - arrivals[0]) / max(seconds - sizes[0], 1e-6) if len(arrivals) > 1 else None
        return {'first': round(first, 3), 'wall': round(wall, 3), 'audio': round(seconds, 2),
                'rtf': round(wall / seconds, 3), 'steady_rtf': round(steady, 3) if steady else None,
                'chunks': len(arrivals), 'mean_gap': round(sum(gaps) / len(gaps), 3) if gaps else None}

    def full_prompt():
        info = dict(base, text=[TEXT], max_new_tokens=[args.frames], full_utterance_decode=[False])
        length, plan, _, _ = request_layout(info, talker, encode)
        return ({'prompt_token_ids': [0] * length, 'additional_information': info},
                params(len(plan.slots), stop_token_ids=[talker['codec_eos_token_id']]))

    def stream_inputs():
        ref_ids, target_ids = encode(base['ref_text'][0]), encode(TEXT)
        buffer = TextStreamBuffer(ref_ids, refs, raw['tts_eos_token_id'], max_frames=args.frames - 1)
        buffer.append(target_ids, final=True)

        async def gen():
            for chunk in buffer.drain():
                info = dict(base, lm_tts_input_stream=[True], lm_tts_chunk_index=[chunk.index],
                            lm_tts_chunk_text_ids=list(chunk.token_ids), lm_tts_chunk_final=[chunk.final],
                            lm_tts_chunk_frames=[chunk.audio_frames], full_utterance_decode=[False], text=[TEXT])
                n = (9 + refs + len(chunk.token_ids)) if chunk.index == 0 else len(chunk.token_ids)
                p = params(chunk.audio_frames + 1, min_tokens=max(0, 2 - chunk.frames_before),
                           stop_token_ids=[talker['codec_pad_id'], talker['codec_eos_token_id']], ignore_eos=True)
                yield StreamingInput(prompt={'prompt_token_ids': [0] * n, 'additional_information': info,
                                             'model_intermediate_buffer': info}, sampling_params=p)
        return gen()

    results = {'variant': args.name, 'overrides': overrides, 'load_seconds': round(load_seconds, 1)}
    try:
        prompt, p = full_prompt()
        await timed(prompt, 'warmup', p)
        for i in range(args.repeats):
            prompt, p = full_prompt()
            results.setdefault('full', []).append(await timed(prompt, f'full-{i}', p))
            results.setdefault('stream', []).append(await timed(stream_inputs(), f'stream-{i}', None))
    finally:
        engine.close()
    print('RESULT ' + json.dumps(results, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--prompt', type=Path, required=True)
    p.add_argument('--name', default='baseline')
    p.add_argument('--overrides', default='{}', help='JSON stage overrides, e.g. {"1": {"enforce_eager": false}}')
    p.add_argument('--frames', type=int, default=400)
    p.add_argument('--repeats', type=int, default=2)
    asyncio.run(run(p.parse_args()))


if __name__ == '__main__':
    main()
