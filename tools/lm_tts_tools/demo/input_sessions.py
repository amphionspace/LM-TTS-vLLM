"""Demo transport for real incremental text input; the model stays in the plugin."""
import asyncio
import copy
import json
import time
import uuid
import threading


def install_input_sessions(app, Request, *, mode, load, state, lock, outdir):
    import numpy as np
    import soundfile as sf
    import torch
    from fastapi import HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    from lm_tts_vllm.input_stream import TextStreamBuffer, segment_prompt_length

    sessions = {}

    class Delta(BaseModel):
        text: str = ''
        final: bool = False

    def progress(session, status):
        buffer = session.get('buffer')
        return {'type': 'input_progress', 'received_text': session['text'],
                'submitted_chars': session['submitted_chars'], 'segments': session['segments'], 'status': status,
                'need_tokens': buffer.needed() if buffer is not None else None}

    async def mark_submitted(session, chunk, tokenizer, target_ids):
        decoded = tokenizer.decode(target_ids[:chunk.submitted_target_tokens], skip_special_tokens=False)
        chars = 0
        for a, b in zip(decoded, session['text']):
            if a != b:
                break
            chars += 1
        session['submitted_chars'] = max(session['submitted_chars'], chars)
        session['segments'] = chunk.index + 1
        await session['output'].put(progress(session, '输入已结束' if chunk.final else '生成中'))

    async def chunks(session, tokenizer, buffer):
        target_ids = []
        while not buffer.exhausted:
            await session['output'].put(progress(session, '等待下一段输入'))
            delta = await asyncio.wait_for(session['input'].get(), timeout=180)
            ids = tokenizer.encode(delta.text, add_special_tokens=False)
            target_ids.extend(ids)
            buffer.append(ids, final=delta.final)
            for chunk in buffer.drain():
                yield chunk, target_ids

    async def native(session, req, engine, audio_info):
        from vllm import SamplingParams
        from vllm.engine.protocol import StreamingInput
        from vllm.sampling_params import RequestOutputKind
        tokenizer = state['tokenizer']
        raw = engine.model_config.hf_config
        # The native speech encoder uses 1920 samples per 24-kHz frame.
        codec = json.loads((state['native_path'] / 'speech_tokenizer/config.json').read_text())
        import math
        refs = math.ceil(audio_info.frames * codec['input_sample_rate'] / audio_info.samplerate / codec['encode_downsample_rate'])
        buffer = TextStreamBuffer(tokenizer.encode(req.ref_text, add_special_tokens=False), refs,
                                  raw.tts_eos_token_id, req.max_tokens-1)
        session['buffer'] = buffer
        base = {'task_type': ['Base'], 'ref_audio': [req.ref_audio], 'ref_text': [req.ref_text],
                'language': [req.language], 'x_vector_only_mode': [False], 'non_streaming_mode': [False],
                'full_utterance_decode': [False], 'ref_code_length': [refs], 'max_new_tokens': [req.max_tokens],
                'subtalker_dosample': [not req.greedy_cp], 'subtalker_temperature': [req.cp_temperature],
                'subtalker_top_k': [req.cp_top_k], 'subtalker_top_p': [req.cp_top_p]}

        async def prompts():
            async for chunk, target_ids in chunks(session, tokenizer, buffer):
                info = dict(base, text=[session['text']], lm_tts_input_stream=[True],
                    lm_tts_chunk_index=[chunk.index], lm_tts_chunk_text_ids=list(chunk.token_ids),
                    lm_tts_chunk_final=[chunk.final], lm_tts_chunk_frames=[chunk.audio_frames])
                length = segment_prompt_length(chunk, req.language, refs)
                params = SamplingParams(temperature=req.temperature, top_k=req.top_k or -1, top_p=req.top_p,
                    repetition_penalty=req.penalty, seed=req.seed, max_tokens=chunk.audio_frames,
                    min_tokens=max(0, 2-chunk.frames_before), stop_token_ids=[2148,2150], ignore_eos=True,
                    detokenize=False, output_kind=RequestOutputKind.DELTA, extra_args={'tts_local_seed': req.seed})
                yield StreamingInput(prompt={'prompt_token_ids': [0]*length,
                    'additional_information': info, 'model_intermediate_buffer': info}, sampling_params=params)
                await mark_submitted(session, chunk, tokenizer, target_ids)
        params = copy.deepcopy(engine.default_sampling_params_list)
        for param in params:
            param.output_kind = RequestOutputKind.DELTA
        async for output in engine.generate(prompts(), request_id=session['id'], sampling_params_list=params):
            mm = output.outputs[0].multimodal_output or {}
            audio = mm.get('audio')
            if audio is None:
                continue
            audio = torch.cat(audio, dim=-1) if isinstance(audio, list) else audio
            if audio.numel():
                rate = mm['sr'][-1] if isinstance(mm['sr'], list) else mm['sr']
                yield audio.float().cpu().numpy().flatten(), int(rate)

    async def pytorch(session, req, engine):
        from transformers import DynamicCache
        from lm_tts.inference.codec import extract_speaker_embedding
        from lm_tts.inference.frame import next_frame
        async def in_thread(fn):
            task = asyncio.create_task(asyncio.to_thread(fn))
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                session['cancel'].set()
                await task
                raise
        tokenizer = engine.tokenizer
        def prepare():
            with torch.inference_mode():
                codes = engine.codec.encode(req.ref_audio).audio_codes[0].to('cuda')
                speaker = extract_speaker_embedding(engine.model.speaker_encoder, req.ref_audio)
                return codes, speaker
        codes, speaker = await in_thread(prepare)
        refs = len(codes)
        buffer = TextStreamBuffer(tokenizer.encode(req.ref_text, add_special_tokens=False), refs,
                                  engine.raw['tts_eos_token_id'], req.max_tokens-1)
        session['buffer'] = buffer
        cache = DynamicCache()
        text_ids = [engine.raw['tts_bos_token_id']]
        language = -1 if req.language.lower() == 'auto' else engine.raw['talker_config']['codec_language_id'][req.language.lower()]
        settings = dict(do_sample=req.temperature>0, temperature=max(req.temperature,1e-5), top_k=req.top_k,
            top_p=req.top_p, repetition_penalty=req.penalty, subtalker_dosample=not req.greedy_cp,
            subtalker_temperature=req.cp_temperature, subtalker_top_k=req.cp_top_k, subtalker_top_p=req.cp_top_p)
        async for chunk, target_ids in chunks(session, tokenizer, buffer):
            text_ids.extend(chunk.token_ids)
            if len(text_ids)+len(codes)+chunk.audio_frames+9 > 4096:
                raise ValueError('输入与生成长度超过当前会话上限')
            await mark_submitted(session, chunk, tokenizer, target_ids)
            def generate_chunk():
                nonlocal codes
                stop = False
                before = len(codes)
                with torch.inference_mode():
                    text = codes.new_tensor(text_ids)
                    batch = {'text_ids': text, 'text_lengths': text.new_tensor([len(text)]),
                             'streaming': torch.tensor([True], device=codes.device),
                             'language_codec_ids': text.new_tensor([language]), 'speaker_embeddings': speaker[None]}
                    for _ in range(chunk.audio_frames):
                        if session['cancel'].is_set():
                            break
                        batch.update(codes=codes, frame_lengths=text.new_tensor([len(codes)]), history=codes[refs:,0])
                        frame, stopped = next_frame(engine.model, batch, cache=cache, generation=settings,
                            suppress_eos=not chunk.final or len(codes)-refs<2)
                        if bool(stopped.item()):
                            stop = True
                            break
                        codes = torch.cat([codes, frame])
                    count = len(codes)-before
                    if not count:
                        return None, None, stop
                    window = codes[max(0,before-72):]
                    wavs, rate = engine.codec.decode({'audio_codes':[window]})
                    trim = int(len(wavs[0])*(len(window)-count)/len(window))
                    return wavs[0][trim:], rate, stop
            wav, rate, stop = await in_thread(generate_chunk)
            if wav is not None:
                yield wav, rate
            if stop:
                break

    async def run(session, req):
        try:
            async with lock:
                await session['output'].put(progress(session, '正在加载模型'))
                engine = load(req.model)
                if not state['training']:
                    raise ValueError('当前流式文本输入支持 step-20000')
                torch.manual_seed(req.seed)
                started = time.perf_counter()
                parts, rate, first_audio = [], None, None
                source = native(session, req, engine, sf.info(req.ref_audio)) if mode == 'vllm' else pytorch(session, req, engine)
                async for wav, rate in source:
                    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
                    if not np.isfinite(wav).all():
                        raise ValueError('模型返回非有限音频值')
                    if first_audio is None:
                        first_audio = time.perf_counter()-started
                    path = outdir/f'{mode}-input-{session["id"]}-chunk-{len(parts)}.wav'
                    sf.write(path,wav,rate)
                    parts.append(wav)
                    await session['output'].put({'type':'chunk','audio':str(path),'index':len(parts)-1,
                                                'elapsed_seconds':time.perf_counter()-started})
                if not parts:
                    raise ValueError('没有生成音频；请发送文本后结束输入')
                path = outdir/f'{mode}-input-{session["id"]}.wav'
                audio = np.concatenate(parts)
                sf.write(path,audio,rate)
                result = {'type':'result','audio':str(path),'mode':mode,'model':req.model,'chunks':len(parts),
                          'first_audio_seconds':first_audio,'seconds':time.perf_counter()-started,
                          'audio_seconds':len(audio)/rate,'received_text':session['text'],
                          'submitted_chars':session['submitted_chars']}
                path.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
                await session['output'].put(result)
        except asyncio.CancelledError:
            await session['output'].put({'type':'error','detail':'会话已取消'})
            raise
        except Exception as exc:
            import traceback
            traceback.print_exc()
            await session['output'].put({'type':'error','detail':str(exc)})
        finally:
            session['closed'] = True
            await session['output'].put(None)

    @app.post('/input_sessions')
    async def create(req: Request):
        if req.model != '20000':
            raise HTTPException(400, '当前流式文本输入支持 step-20000')
        if not req.ref_text.strip():
            raise HTTPException(400, '需要参考文本')
        if not 0 < sf.info(req.ref_audio).duration <= 30:
            raise HTTPException(400, '参考音频须在 0–30 秒之间')
        for old_id in [key for key, value in sessions.items() if value['closed']]:
            sessions.pop(old_id, None)
        if len(sessions) >= 8:
            raise HTTPException(429, '请先结束已有会话')
        return {'session_id':open_session(req)['id']}

    def open_session(req):
        sid = uuid.uuid4().hex
        session = {'id':sid,'input':asyncio.Queue(),'output':asyncio.Queue(),'text':'',
                   'submitted_chars':0,'segments':0,'closed':False,'input_closed':False,'cancel':threading.Event()}
        sessions[sid]=session
        session['task']=asyncio.create_task(run(session,req))
        return session

    @app.post('/input_sessions/{sid}/text')
    async def send(sid: str, delta: Delta):
        session=sessions.get(sid)
        if session is None or session['closed'] or session['input_closed']:
            raise HTTPException(409,'会话已结束')
        if len(session['text'])+len(delta.text)>8000:
            raise HTTPException(400,'输入过长')
        session['text']+=delta.text
        session['input_closed']=delta.final
        await session['input'].put(delta)
        event=progress(session,'输入已结束' if delta.final else '已接收文本')
        await session['output'].put(event)
        return event

    @app.get('/input_sessions/{sid}/events')
    async def events(sid: str):
        session=sessions.get(sid)
        if session is None:
            raise HTTPException(404,'会话不存在')
        async def stream():
            try:
                while True:
                    event=await session['output'].get()
                    if event is None:
                        break
                    yield json.dumps(event,ensure_ascii=False)+'\n'
            finally:
                if not session['task'].done():
                    session['cancel'].set()
                    session['task'].cancel()
                sessions.pop(sid,None)
        return StreamingResponse(stream(),media_type='application/x-ndjson')

    @app.delete('/input_sessions/{sid}')
    async def cancel(sid: str):
        session=sessions.pop(sid,None)
        if session and not session['task'].done():
            session['cancel'].set()
            session['task'].cancel()
        return {'cancelled':True}

    async def warmup(req, text):
        """Run one streamed session end to end; the first request otherwise pays seconds of lazy setup."""
        session = open_session(req)
        await session['input'].put(Delta(text=text, final=True))
        await session['task']
        sessions.pop(session['id'], None)
        while not session['output'].empty():
            event = session['output'].get_nowait()
            if event and event['type'] == 'error':
                raise RuntimeError(event['detail'])

    return warmup
