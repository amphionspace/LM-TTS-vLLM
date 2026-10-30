"""Loopback-only demo backend, with incremental native audio delivery."""
import argparse
import asyncio
import copy
import json
import math
import sys
import time
import uuid

from .common import MODELS, REFERENCE, REFERENCE_TEXT, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pytorch', 'vllm'], required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--warm-model', default='20000')
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field

    class Request(BaseModel):
        model: str = '20000'
        ref_audio: str
        ref_text: str = ''
        text: str
        x_vector_only_mode: bool = False
        non_streaming_mode: bool = False
        language: str = 'Auto'
        temperature: float = Field(default=0.9, ge=0, le=2)
        top_k: int = Field(default=50, ge=0, le=2048)
        top_p: float = Field(default=1, gt=0, le=1)
        penalty: float = Field(default=1.05, ge=1, le=3)
        greedy_cp: bool = False
        cp_temperature: float = Field(default=0.9, gt=0, le=2)
        cp_top_k: int = Field(default=50, ge=0, le=2048)
        cp_top_p: float = Field(default=1, gt=0, le=1)
        seed: int = Field(default=42, ge=0, le=2147483647)
        max_tokens: int = Field(default=512, ge=3, le=2048)

    app = FastAPI()
    state = {'model': None, 'engine': None, 'tokenizer': None, 'training': False, 'warm': False}
    lock = asyncio.Lock()
    outdir = ROOT / 'results/demo'
    outdir.mkdir(parents=True, exist_ok=True)

    def load(name):
        if name not in MODELS:
            raise ValueError('Unknown checkpoint')
        if state['model'] == name:
            return state['engine']
        if state['engine'] is not None:
            if args.mode == 'vllm':
                state['engine'].close()
            state.update(engine=None, model=None, tokenizer=None)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        raw = json.loads((MODELS[name]['source'] / 'config.json').read_text())
        state['native_path'] = MODELS[name]['native']
        training = raw['talker_config'].get('lm_tts_streaming_layout') == 'interleaved'
        if args.mode == 'pytorch':
            if training:
                # Same implementation used by UltraEval scripts/local_qwen.py.
                sys.path.insert(0, str(ROOT.parent / 'LM-TTS-Training'))
                from lm_tts.inference.codec import CodecSynthesizer
                engine = CodecSynthesizer(MODELS[name]['source'], 'cuda:0', torch.bfloat16, 'sdpa')
            else:
                from qwen_tts import Qwen3TTSModel
                engine = Qwen3TTSModel.from_pretrained(str(MODELS[name]['source']), device_map='cuda:0',
                    dtype=torch.bfloat16, attn_implementation='sdpa')
        else:
            import lm_tts_vllm
            from pathlib import Path
            from vllm_omni import AsyncOmni
            from vllm_omni.plugins import load_omni_general_plugins
            from transformers import AutoTokenizer
            load_omni_general_plugins()
            engine = AsyncOmni(model=str(MODELS[name]['native']), deploy_config=str(
                Path(lm_tts_vllm.__file__).parent / 'native_input_streaming.yaml'), stage_init_timeout=1200, init_timeout=1800, stage_overrides={
                '1': {'max_model_len': 65536, 'max_num_batched_tokens': 65536,
                      'default_sampling_params': {'max_tokens': 65536}}})
            state['tokenizer'] = AutoTokenizer.from_pretrained(str(MODELS[name]['native']))
        state.update(model=name, engine=engine, training=training)
        return engine

    def pytorch_audio(engine, req):
        settings = dict(max_new_tokens=req.max_tokens, min_new_tokens=2,
            do_sample=req.temperature > 0, temperature=max(req.temperature, 1e-5), top_k=req.top_k,
            top_p=req.top_p, repetition_penalty=req.penalty, subtalker_dosample=not req.greedy_cp,
            subtalker_temperature=req.cp_temperature, subtalker_top_k=req.cp_top_k, subtalker_top_p=req.cp_top_p)
        if state['training']:
            settings['max_new_tokens'] -= 1  # Match UltraEval's cap convention.
            language = -1 if req.language.lower() == 'auto' else engine.raw['talker_config']['codec_language_id'][req.language.lower()]
            return engine.synthesize(req.text.strip(),
                reference_audio=None if req.x_vector_only_mode else req.ref_audio,
                reference_text=None if req.x_vector_only_mode else req.ref_text.strip(),
                speaker_audio=req.ref_audio, streaming=not req.non_streaming_mode,
                language_codec_id=language, penalize_reference=False, use_cache=True, **settings)
        prompt = engine.create_voice_clone_prompt(ref_audio=req.ref_audio, ref_text=req.ref_text.strip(),
                                                  x_vector_only_mode=req.x_vector_only_mode)
        wavs, rate = engine.generate_voice_clone(text=[req.text.strip()], language=[req.language],
            voice_clone_prompt=prompt, non_streaming_mode=req.non_streaming_mode,
            x_vector_only_mode=req.x_vector_only_mode, **settings)
        return wavs[0], rate

    async def native_audio(engine, req, audio_info):
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind
        from lm_tts_vllm.interleaved import request_layout
        from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder
        raw = json.loads((MODELS[req.model]['native'] / 'config.json').read_text())['talker_config']
        codec = json.loads((MODELS[req.model]['native'] / 'speech_tokenizer/config.json').read_text())
        frames = math.ceil(audio_info.frames * codec['input_sample_rate'] / audio_info.samplerate / codec['encode_downsample_rate'])
        info = {'task_type': ['Base'], 'ref_audio': [req.ref_audio], 'ref_text': [req.ref_text.strip()],
            'text': [req.text.strip()], 'language': [req.language], 'x_vector_only_mode': [req.x_vector_only_mode],
            'non_streaming_mode': [req.non_streaming_mode], 'max_new_tokens': [req.max_tokens],
            'full_utterance_decode': [False], 'ref_code_length': [frames],
            'subtalker_dosample': [not req.greedy_cp], 'subtalker_temperature': [req.cp_temperature],
            'subtalker_top_k': [req.cp_top_k], 'subtalker_top_p': [req.cp_top_p]}
        maximum = req.max_tokens
        if state['training']:
            length, plan, _, _ = request_layout(info, raw,
                lambda text: state['tokenizer'].encode(text, add_special_tokens=False))
            maximum = len(plan.slots)
        else:
            length = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
                info, task_type='Base', tokenize_prompt=lambda text: state['tokenizer'](text)['input_ids'],
                codec_language_id=raw['codec_language_id'], spk_is_dialect=raw.get('spk_is_dialect'))
        if length + maximum > 4096:
            raise ValueError('输入与生成长度合计超过当前 4096 token 上限')
        params = copy.deepcopy(engine.default_sampling_params_list)
        params[0] = SamplingParams(temperature=req.temperature, top_k=req.top_k or -1, top_p=req.top_p,
            repetition_penalty=req.penalty, seed=req.seed, max_tokens=maximum, min_tokens=2,
            stop_token_ids=[2150], detokenize=False, extra_args={'tts_local_seed': req.seed})
        for param in params:
            param.output_kind = RequestOutputKind.DELTA
        async for output in engine.generate({'prompt_token_ids': [0] * length, 'additional_information': info},
                request_id=uuid.uuid4().hex, sampling_params_list=params):
            mm = output.outputs[0].multimodal_output
            wav = mm.get('audio')
            if wav is None:
                continue
            if isinstance(wav, list):
                wav = torch.cat(wav, dim=-1)
            if not wav.numel():
                continue
            rate = mm['sr'][-1] if isinstance(mm['sr'], list) else mm['sr']
            yield wav.float().cpu().numpy().flatten(), int(rate)

    async def generate(req):
        if not req.text.strip():
            raise ValueError('目标文本不能为空')
        if not req.x_vector_only_mode and not req.ref_text.strip():
            raise ValueError('ICL 克隆需要参考文本')
        audio_info = sf.info(req.ref_audio)
        if not 0 < audio_info.duration <= 30:
            raise ValueError('参考音频须在 0–30 秒之间')
        async with lock:
            started = time.perf_counter()
            engine = load(req.model)
            load_seconds = time.perf_counter() - started
            torch.manual_seed(req.seed)
            started = time.perf_counter()
            path = outdir / f'{args.mode}-{req.model}-{uuid.uuid4().hex}.wav'
            parts, rate, first_audio_seconds = [], None, None
            async def audio_iter():
                if args.mode == 'pytorch':
                    yield await asyncio.to_thread(pytorch_audio, engine, req)
                else:
                    async for chunk in native_audio(engine, req, audio_info):
                        yield chunk
            async for wav, chunk_rate in audio_iter():
                wav = np.asarray(wav, dtype=np.float32).reshape(-1)
                if not len(wav) or not np.isfinite(wav).all():
                    raise ValueError('模型返回了空音频或非有限值')
                if rate is not None and chunk_rate != rate:
                    raise ValueError('音频块采样率变化')
                rate = chunk_rate
                if first_audio_seconds is None:
                    first_audio_seconds = time.perf_counter() - started
                chunk_path = path.with_name(path.stem + f'-chunk-{len(parts)}.wav')
                sf.write(chunk_path, wav, rate)
                parts.append(wav)
                yield {'type': 'chunk', 'audio': str(chunk_path), 'index': len(parts)-1,
                       'elapsed_seconds': time.perf_counter()-started}
            if not parts:
                raise ValueError('模型未生成音频')
            wav = np.concatenate(parts)
            sf.write(path, wav, rate)
            result = {'type': 'result', 'audio': str(path), 'seconds': time.perf_counter()-started,
                'first_audio_seconds': first_audio_seconds, 'chunks': len(parts), 'load_seconds': load_seconds,
                'audio_seconds': len(wav)/rate, 'mode': args.mode, 'model': req.model,
                'code_predictor_sampling': not req.greedy_cp}
            path.with_suffix('.json').write_text(json.dumps({'request': req.model_dump(), 'result': result},
                                                            ensure_ascii=False, indent=2))
            yield result

    @app.get('/health')
    def health():
        return {'mode': args.mode, 'model': state['model'], 'ready': state['engine'] is not None and state['warm']}

    @app.post('/synthesize')
    async def synthesize(req: Request):
        try:
            result = None
            async for event in generate(req):
                if event['type'] == 'result':
                    result = event
            return result
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post('/synthesize_stream')
    async def synthesize_stream(req: Request):
        async def events():
            try:
                async for event in generate(req):
                    yield json.dumps(event, ensure_ascii=False) + '\n'
            except Exception as exc:
                import traceback
                traceback.print_exc()
                yield json.dumps({'type': 'error', 'detail': str(exc)}, ensure_ascii=False) + '\n'
        return StreamingResponse(events(), media_type='application/x-ndjson')

    from .input_sessions import install_input_sessions
    warmup = install_input_sessions(app, Request, mode=args.mode, load=load, state=state, lock=lock, outdir=outdir)
    load(args.warm_model)

    @app.on_event('startup')
    async def warm():
        async def run():
            started = time.perf_counter()
            try:
                if state['training']:
                    await warmup(Request(model=args.warm_model, ref_audio=REFERENCE, ref_text=REFERENCE_TEXT, text=''),
                                 '你好，这是一段预热用的流式语音合成。文本分段送入，让首次请求也和后面的请求一样快。')
                print(f'warmup finished in {time.perf_counter() - started:.1f}s', flush=True)
            except Exception:
                import traceback
                traceback.print_exc()
            state['warm'] = True
        asyncio.create_task(run())
    try:
        uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='info')
    finally:
        if args.mode == 'vllm' and state['engine'] is not None:
            state['engine'].close()


if __name__ == '__main__':
    main()
