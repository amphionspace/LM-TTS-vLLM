"""Loopback-only demo backend, launched in the selected runtime environment."""
import argparse
import copy
import json
import math
import threading
import time
import uuid
from pathlib import Path

from .common import MODELS, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['pytorch', 'vllm'], required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--warm-model', default='56384')
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field

    class Request(BaseModel):
        model: str = '56384'
        ref_audio: str
        ref_text: str = ''
        text: str
        x_vector_only_mode: bool = False
        non_streaming_mode: bool = True
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
        max_tokens: int = Field(default=512, ge=2, le=1024)

    app = FastAPI()
    state = {'model': None, 'engine': None, 'tokenizer': None}
    lock = threading.Lock()
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
            state['engine'] = None
            state['model'] = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        if args.mode == 'pytorch':
            from qwen_tts import Qwen3TTSModel
            engine = Qwen3TTSModel.from_pretrained(str(MODELS[name]['source']), device_map='cuda:0',
                                                  dtype=torch.bfloat16, attn_implementation='sdpa')
        else:
            from vllm_omni import Omni
            from vllm_omni.plugins import load_omni_general_plugins
            from transformers import AutoTokenizer
            load_omni_general_plugins()
            engine = Omni(model=str(MODELS[name]['native']), stage_overrides={
                '0': {'kv_cache_memory_bytes': 536870912},
                '1': {'max_model_len': 32768, 'max_num_batched_tokens': 32768,
                      'default_sampling_params': {'max_tokens': 32768}}})
            state['tokenizer'] = AutoTokenizer.from_pretrained(str(MODELS[name]['native']))
        state.update(model=name, engine=engine)
        return engine

    @app.get('/health')
    def health():
        return {'mode': args.mode, 'model': state['model'], 'ready': state['engine'] is not None}

    @app.post('/synthesize')
    def synthesize(req: Request):
        try:
            if not req.text.strip():
                raise ValueError('目标文本不能为空')
            if not req.x_vector_only_mode and not req.ref_text.strip():
                raise ValueError('ICL 克隆需要参考文本')
            audio_info = sf.info(req.ref_audio)
            if not 0 < audio_info.duration <= 30:
                raise ValueError('参考音频须在 0–30 秒之间')
            with lock:
                started = time.perf_counter()
                engine = load(req.model)
                load_seconds = time.perf_counter() - started
                torch.manual_seed(req.seed)
                started = time.perf_counter()
                codes = None
                if args.mode == 'pytorch':
                    prompt = engine.create_voice_clone_prompt(ref_audio=req.ref_audio, ref_text=req.ref_text.strip(), x_vector_only_mode=req.x_vector_only_mode)
                    wavs, rate = engine.generate_voice_clone(
                        text=[req.text.strip()], language=[req.language], voice_clone_prompt=prompt,
                        non_streaming_mode=req.non_streaming_mode, x_vector_only_mode=req.x_vector_only_mode, max_new_tokens=req.max_tokens,
                        do_sample=req.temperature > 0, temperature=max(req.temperature, 1e-5), top_k=req.top_k,
                        top_p=req.top_p, repetition_penalty=req.penalty,
                        subtalker_dosample=not req.greedy_cp, subtalker_temperature=req.cp_temperature,
                        subtalker_top_k=req.cp_top_k, subtalker_top_p=req.cp_top_p)
                    wav = wavs[0]
                else:
                    from vllm import SamplingParams
                    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder
                    raw = json.loads((MODELS[req.model]['native'] / 'config.json').read_text())['talker_config']
                    codec = json.loads((MODELS[req.model]['native'] / 'speech_tokenizer/config.json').read_text())
                    frames = math.ceil(audio_info.frames * codec['input_sample_rate'] / audio_info.samplerate / codec['encode_downsample_rate'])
                    info = {'task_type': ['Base'], 'ref_audio': [req.ref_audio], 'ref_text': [req.ref_text.strip()],
                            'text': [req.text.strip()], 'language': [req.language], 'x_vector_only_mode': [req.x_vector_only_mode],
                            'non_streaming_mode': [req.non_streaming_mode], 'max_new_tokens': [req.max_tokens], 'full_utterance_decode': [True],
                            'ref_code_length': [frames], 'subtalker_dosample': [not req.greedy_cp],
                            'subtalker_temperature': [req.cp_temperature], 'subtalker_top_k': [req.cp_top_k],
                            'subtalker_top_p': [req.cp_top_p]}
                    length = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
                        info, task_type='Base', tokenize_prompt=lambda t: state['tokenizer'](t)['input_ids'],
                        codec_language_id=raw['codec_language_id'], spk_is_dialect=raw.get('spk_is_dialect'))
                    if length + req.max_tokens > 4096:
                        raise ValueError('输入与生成长度合计超过当前 4096 token 上限')
                    params = copy.deepcopy(engine.default_sampling_params_list)
                    params[0] = SamplingParams(temperature=req.temperature, top_k=req.top_k or -1, top_p=req.top_p,
                        repetition_penalty=req.penalty, seed=req.seed, max_tokens=req.max_tokens, min_tokens=2,
                        stop_token_ids=[2150], detokenize=False, extra_args={'tts_local_seed': req.seed})
                    outputs = engine.generate([{'prompt_token_ids': [0] * length, 'additional_information': info}],
                                              sampling_params_list=params, use_tqdm=False)
                    output = list(outputs)[-1].outputs[0].multimodal_output
                    wav = output['audio']
                    if isinstance(wav, list): wav = torch.cat(wav, dim=-1)
                    wav = wav.float().cpu().numpy().flatten()
                    rate = output['sr']
                    if isinstance(rate, list): rate = rate[-1]
                    rate = int(rate)
                    codes = output.get('codec_frames')
                    if isinstance(codes, list): codes = torch.cat(codes, dim=0)
                elapsed = time.perf_counter() - started
                path = outdir / f'{args.mode}-{req.model}-{uuid.uuid4().hex}.wav'
                sf.write(path, wav, rate)
                if codes is not None:
                    np.save(path.with_suffix('.npy'), codes.cpu().numpy())
                result = {'audio': str(path), 'seconds': elapsed, 'load_seconds': load_seconds,
                          'audio_seconds': len(wav) / rate, 'mode': args.mode, 'model': req.model,
                          'code_predictor_sampling': not req.greedy_cp}
                path.with_suffix('.json').write_text(json.dumps({'request': req.model_dump(), 'result': result}, ensure_ascii=False, indent=2))
                return result
        except Exception as exc:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    load(args.warm_model)
    try:
        uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='info')
    finally:
        if args.mode == 'vllm' and state['engine'] is not None:
            state['engine'].close()


if __name__ == '__main__':
    main()
