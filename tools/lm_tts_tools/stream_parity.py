"""Greedy streamed-text input must reproduce the full-text interleaved request.

Both requests run on one engine with the same deploy config, so any audio
difference comes from how streamed text segments are fed, not from kernels.
"""
import argparse
import asyncio
import copy
import json
import time
from pathlib import Path


async def run(args):
    import numpy as np
    import soundfile as sf
    import torch
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from vllm.engine.protocol import StreamingInput
    from vllm.sampling_params import RequestOutputKind
    from vllm_omni import AsyncOmni
    from vllm_omni.plugins import load_omni_general_plugins
    import lm_tts_vllm
    from lm_tts_vllm.input_stream import TextStreamBuffer, segment_prompt_length
    from lm_tts_vllm.interleaved import request_layout

    load_omni_general_plugins()
    base = json.loads(args.prompt.read_text())
    tok = AutoTokenizer.from_pretrained(args.model)
    encode = lambda s: tok.encode(s, add_special_tokens=False)
    raw = json.loads((args.model / 'config.json').read_text())
    talker = raw['talker_config']
    text = args.text or tok.decode(encode(base['text'][0])[:args.text_tokens])
    ref_ids, target_ids = encode(base['ref_text'][0]), encode(text)
    refs = len(base['voice_clone_prompt'][0]['ref_code'])
    args.output.mkdir(parents=True, exist_ok=False)
    engine = AsyncOmni(model=str(args.model), deploy_config=str(
        Path(lm_tts_vllm.__file__).parent / 'native_input_streaming.yaml'),
        stage_overrides={'0': {'kv_cache_memory_bytes': 536870912}})

    def params(max_tokens, **extra):
        return SamplingParams(temperature=0, repetition_penalty=1.0, seed=42, max_tokens=max_tokens,
                              detokenize=False, output_kind=RequestOutputKind.DELTA,
                              extra_args={'tts_local_seed': 42}, **extra)

    async def collect(prompt, request_id, on_audio=None, stage0=None):
        stage_params = copy.deepcopy(engine.default_sampling_params_list)
        if stage0 is not None:
            stage_params[0] = stage0
        for p in stage_params:
            p.output_kind = RequestOutputKind.DELTA
        parts = []
        async for output in engine.generate(prompt, request_id=request_id, sampling_params_list=stage_params):
            audio = (output.outputs[0].multimodal_output or {}).get('audio')
            if audio is None:
                continue
            audio = torch.cat(audio, dim=-1) if isinstance(audio, list) else audio
            if audio.numel():
                parts.append(audio.float().cpu().flatten())
                if on_audio:
                    on_audio()
        return torch.cat(parts).numpy()

    try:
        # Full text in one request: the reference for streamed input.
        info = dict(base, text=[text], max_new_tokens=[args.frames + 1], full_utterance_decode=[False])
        length, plan, _, _ = request_layout(info, talker, encode)
        full = await collect({'prompt_token_ids': [0] * length, 'additional_information': info}, 'full',
                             stage0=params(len(plan.slots), stop_token_ids=[talker['codec_eos_token_id']]))

        # Streamed text: the remainder is withheld until audio has arrived.
        buffer = TextStreamBuffer(ref_ids, refs, raw['tts_eos_token_id'], max_frames=args.frames)
        first_need = (refs // 10 + 1) * 5 - len(ref_ids)
        ready = asyncio.Event()
        timeline, start = [], time.perf_counter()

        def stream_prompt(chunk):
            chunk_info = dict(base, lm_tts_input_stream=[True], lm_tts_chunk_index=[chunk.index],
                              lm_tts_chunk_text_ids=list(chunk.token_ids), lm_tts_chunk_final=[chunk.final],
                              lm_tts_chunk_frames=[chunk.audio_frames], full_utterance_decode=[False],
                              text=[tok.decode(target_ids[:max(1, chunk.submitted_target_tokens)])])
            n = segment_prompt_length(chunk, base['language'][0], refs)
            p = params(chunk.audio_frames + 1, min_tokens=chunk.audio_frames,
                       stop_token_ids=[talker['codec_pad_id'], talker['codec_eos_token_id']], ignore_eos=True)
            return StreamingInput(prompt={'prompt_token_ids': [0] * n, 'additional_information': chunk_info,
                                          'model_intermediate_buffer': chunk_info}, sampling_params=p)

        async def inputs():
            buffer.append(target_ids[:first_need])
            for chunk in buffer.drain():
                timeline.append(('send', chunk.index, time.perf_counter() - start))
                yield stream_prompt(chunk)
            await asyncio.wait_for(ready.wait(), timeout=120)
            await asyncio.sleep(args.pause)
            for i in range(first_need, len(target_ids), args.step):
                if buffer.exhausted:  # --frames reached before the text ran out
                    return
                buffer.append(target_ids[i:i + args.step])
                for chunk in buffer.drain():
                    timeline.append(('send', chunk.index, time.perf_counter() - start))
                    yield stream_prompt(chunk)
                await asyncio.sleep(args.pause)
            if buffer.exhausted:
                return
            buffer.append([], final=True)
            for chunk in buffer.drain():
                timeline.append(('send', chunk.index, time.perf_counter() - start))
                yield stream_prompt(chunk)

        def on_audio():
            timeline.append(('audio', None, time.perf_counter() - start))
            ready.set()
        streamed = await collect(inputs(), 'streamed', on_audio)
    finally:
        engine.close()

    sf.write(args.output / 'full.wav', full, 24000)
    sf.write(args.output / 'streamed.wav', streamed, 24000)
    n = min(len(full), len(streamed))
    diff = np.flatnonzero(full[:n] != streamed[:n])
    report = {'text': text, 'full_samples': len(full), 'streamed_samples': len(streamed),
              'identical': len(full) == len(streamed) and not len(diff),
              'first_diff_frame': int(diff[0] // 1920) if len(diff) else None,
              'max_abs_diff': float(abs(full[:n] - streamed[:n]).max()) if n else None,
              'timeline': timeline}
    (args.output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != 'timeline'}, ensure_ascii=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--prompt', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--text')
    p.add_argument('--text-tokens', type=int, default=40)
    p.add_argument('--frames', type=int, default=79)
    p.add_argument('--step', type=int, default=3, help='text tokens per client delta after the first')
    p.add_argument('--pause', type=float, default=0.5)
    asyncio.run(run(p.parse_args()))


if __name__ == '__main__':
    main()
