"""Reference cloning with full-text or real incremental text input."""
import argparse
import html
import json
import threading
import time
import urllib.error
import urllib.request

import gradio as gr
import soundfile as sf

from .common import MODELS, REFERENCE, REFERENCE_TEXT, ROOT

PRESETS = {
    '低温采样（原 demo）': (0.3, 10, 1.0, 2.0, True, 0.9, 50, 1.0),
    '评测默认采样': (0.9, 50, 1.0, 1.05, False, 0.9, 50, 1.0),
    'Greedy 对齐': (0.0, 50, 1.0, 1.05, True, 0.9, 50, 1.0),
}
EXAMPLE = ('你好，这是一段流式语音合成测试。文本会分成几段逐步发送，模型收到前面的内容后就开始生成声音。'
           '在播放第一段声音的时候，后面的文字还没有全部发送完成。你也可以手动发送下一段文本，体验边输入边说话。')


def input_progress(event):
    text = event.get('received_text', '')
    at = max(0,min(len(text),int(event.get('submitted_chars',0))))
    status = html.escape(event.get('status',''))
    return (f'<div><strong>{status}</strong> · 已送入模型 {at} / 已接收 {len(text)} 字'
            f' · {event.get("segments",0)} 个文本块</div>'
            '<div style="white-space:pre-wrap;line-height:1.9;padding:12px;border:1px solid #ddd;border-radius:8px">'
            f'<mark style="background:#d1fae5;color:#065f46">{html.escape(text[:at])}</mark>'
            f'<span style="color:#888">{html.escape(text[at:])}</span></div>')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=7861)
    parser.add_argument('--pytorch-port',type=int,default=7862)
    parser.add_argument('--vllm-port',type=int,default=7863)
    args=parser.parse_args()
    ports={'pytorch':args.pytorch_port,'vllm':args.vllm_port}
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def open_request(mode,path,data=None,method=None):
        req=urllib.request.Request(f'http://127.0.0.1:{ports[mode]}{path}',
            data=json.dumps(data).encode() if data is not None else None,
            headers={'Content-Type':'application/json'},method=method)
        try:
            return opener.open(req,timeout=1200)
        except urllib.error.HTTPError as exc:
            raise gr.Error(json.loads(exc.read()).get('detail',str(exc))) from exc
        except urllib.error.URLError as exc:
            raise gr.Error('所选后端正在加载或暂不可用，请稍后重试') from exc

    def rpc(mode,path,data=None,method=None):
        with open_request(mode,path,data,method) as response:
            return json.load(response)

    def payload(mode,model,ref_audio,ref_text,language,temperature,top_k,top_p,penalty,
                greedy_cp,cp_temperature,cp_top_k,cp_top_p,seed,max_tokens):
        if not ref_audio or not ref_text.strip():
            raise gr.Error('请提供参考音频及对应文本')
        return dict(model=model,ref_audio=ref_audio,ref_text=ref_text,language=language,text='',
            temperature=float(temperature),top_k=int(top_k),top_p=float(top_p),penalty=float(penalty),
            greedy_cp=bool(greedy_cp),cp_temperature=float(cp_temperature),cp_top_k=int(cp_top_k),
            cp_top_p=float(cp_top_p),seed=int(seed),max_tokens=int(max_tokens),x_vector_only_mode=False)

    def output_events(response,mode,model):
        for line in response:
            event=json.loads(line)
            if event['type']=='error':
                raise gr.Error(event['detail'])
            if event['type']=='input_progress':
                yield gr.skip(),gr.skip(),gr.skip(),input_progress(event)
            elif event['type']=='chunk':
                wav,sr=sf.read(event['audio'],dtype='int16')
                yield (sr,wav),None,f"收到第 {event['index']+1} 块音频 · {event['elapsed_seconds']:.2f} 秒",gr.skip()
            elif event['type']=='result':
                info=(f"{mode} · step-{model}｜音频 {event['audio_seconds']:.2f} 秒"
                      f"｜首块 {event['first_audio_seconds']:.2f} 秒｜合成 {event['seconds']:.2f} 秒｜{event['chunks']} 块")
                indicator=input_progress(dict(event,status='生成完成')) if 'received_text' in event else gr.skip()
                yield gr.skip(),event['audio'],info,indicator

    def synthesize(text,*settings):
        if not text.strip():
            raise gr.Error('请输入文本')
        data=payload(*settings)
        data.update(text=text,non_streaming_mode=True)
        yield None,None,'正在生成',input_progress({'received_text':text,'submitted_chars':len(text),'segments':1,'status':'整段输入'})
        with open_request(settings[0],'/synthesize_stream',data) as response:
            yield from output_events(response,settings[0],settings[1])

    def start_session(previous,*settings):
        if previous:
            rpc(previous['mode'],f"/input_sessions/{previous['id']}",method='DELETE')
        data=payload(*settings)
        data['non_streaming_mode']=False
        result=rpc(settings[0],'/input_sessions',data)
        session={'id':result['session_id'],'mode':settings[0],'model':settings[1]}
        return session,'会话已开始，请发送第一段文本',input_progress({'status':'等待文本'})

    def listen(session):
        if not session:
            return
        with open_request(session['mode'],f"/input_sessions/{session['id']}/events") as response:
            yield from output_events(response,session['mode'],session['model'])

    def send_text(session,text,final=False):
        if not session:
            raise gr.Error('请先开始会话')
        if not text.strip() and not final:
            raise gr.Error('请输入本段文本')
        event=rpc(session['mode'],f"/input_sessions/{session['id']}/text",{'text':text,'final':final})
        return '',input_progress(event)

    def cancel(session):
        if session:
            rpc(session['mode'],f"/input_sessions/{session['id']}",method='DELETE')
        return None,'会话已取消'

    def simulate(text,interval,previous,*settings):
        if not text.strip():
            raise gr.Error('请输入用于模拟的完整文本')
        session,_,_=start_session(previous,*settings)
        stop=threading.Event()
        errors=[]
        def feed():
            try:
                for offset in range(0,len(text),8):
                    if stop.is_set():
                        return
                    rpc(session['mode'],f"/input_sessions/{session['id']}/text",{'text':text[offset:offset+8]})
                    if stop.wait(float(interval)):
                        return
                rpc(session['mode'],f"/input_sessions/{session['id']}/text",{'text':'','final':True})
            except Exception as exc:
                errors.append(exc)
                rpc(session['mode'],f"/input_sessions/{session['id']}",method='DELETE')
        worker=threading.Thread(target=feed,daemon=True)
        worker.start()
        yield session,None,None,'正在分段发送',input_progress({'status':'开始模拟'})
        try:
            for values in listen(session):
                yield (session,*values)
            if errors:
                raise errors[0]
        finally:
            stop.set()
            worker.join(timeout=2)

    config=json.loads((MODELS['10000']['source']/'config.json').read_text())
    languages=['Auto']+[s.capitalize() for s in config['talker_config']['codec_language_id'] if 'dialect' not in s]
    with gr.Blocks(title='LM-TTS · PyTorch / vLLM') as demo:
        gr.Markdown('# LM-TTS 声音克隆试听\n选择整段输入，或分段发送文本并边输入边听。绿色表示已送入模型的文本，不是已播放的内容。')
        session=gr.State(None)
        with gr.Row():
            with gr.Column():
                mode=gr.Radio([('PyTorch（BF16 基线）','pytorch'),('vLLM（原生插件）','vllm')],value='vllm',label='推理模式')
                model=gr.Dropdown([('Emilia step-56384','56384'),('Emilia step-10000','10000')],value='10000',label='模型')
                ref_audio=gr.Audio(value=REFERENCE,type='filepath',label='参考音频（最长 30 秒）')
                ref_text=gr.Textbox(value=REFERENCE_TEXT,label='参考音频文本',lines=2)
                input_mode=gr.Radio(['整段输入','流式输入'],value='流式输入',label='文本输入方式')
                with gr.Group(visible=False) as full_group:
                    full_text=gr.Textbox(value=EXAMPLE,label='要合成的完整文本',lines=4)
                    run=gr.Button('合成',variant='primary')
                with gr.Group(visible=True) as stream_group:
                    gr.Markdown('先开始会话，再逐段发送。全部发完后点击“结束输入”。流式文本输入当前支持 step-10000。')
                    start=gr.Button('开始会话')
                    delta=gr.Textbox(label='本次发送的文本',lines=3,placeholder='在上一段播放时继续输入下一段…')
                    with gr.Row():
                        send=gr.Button('发送本段',variant='primary')
                        finish=gr.Button('结束输入')
                        cancel_button=gr.Button('取消会话')
                    with gr.Accordion('一键体验：模拟分段输入',open=True):
                        example=gr.Textbox(value=EXAMPLE,label='模拟输入文本',lines=4)
                        interval=gr.Slider(0.1,2,value=0.6,step=0.1,label='发送间隔（秒，每次 8 字）')
                        simulate_button=gr.Button('模拟流式输入')
                language=gr.Dropdown(languages,value='Auto',label='语言')
                preset=gr.Radio(list(PRESETS),value='评测默认采样',label='采样预设')
                with gr.Accordion('采样参数',open=False):
                    temperature=gr.Slider(0,1.5,value=0.9,step=0.05,label='Talker temperature（0 为 greedy）')
                    top_k=gr.Slider(0,100,value=50,step=1,label='Talker top-k（0 不限制）')
                    top_p=gr.Slider(0.05,1,value=1,step=0.05,label='Talker top-p')
                    penalty=gr.Slider(1,3,value=1.05,step=0.05,label='重复惩罚')
                    greedy_cp=gr.Checkbox(value=False,label='Code predictor greedy（关闭则采样）')
                    cp_temperature=gr.Slider(0.05,1.5,value=0.9,step=0.05,label='Code predictor temperature')
                    cp_top_k=gr.Slider(0,100,value=50,step=1,label='Code predictor top-k（0 不限制）')
                    cp_top_p=gr.Slider(0.05,1,value=1,step=0.05,label='Code predictor top-p')
                    seed=gr.Number(value=42,precision=0,label='随机种子')
                    max_tokens=gr.Slider(3,2048,value=512,step=1,label='最大生成帧数')
            with gr.Column():
                indicator=gr.HTML(input_progress({'status':'等待输入'}),label='文本输入进度')
                audio=gr.Audio(label='实时播放',streaming=True,autoplay=True)
                complete_audio=gr.Audio(label='完整音频')
                info=gr.Markdown('首次加载或切换模型可能需要数分钟。两种推理模式尚未完全数值对齐。')
        settings=[mode,model,ref_audio,ref_text,language,temperature,top_k,top_p,penalty,greedy_cp,
                  cp_temperature,cp_top_k,cp_top_p,seed,max_tokens]
        outputs=[audio,complete_audio,info,indicator]
        input_mode.change(lambda value:(gr.update(visible=value=='整段输入'),gr.update(visible=value=='流式输入')),
                          input_mode,[full_group,stream_group],queue=False)
        preset.change(lambda p:PRESETS[p],preset,[temperature,top_k,top_p,penalty,greedy_cp,cp_temperature,cp_top_k,cp_top_p])
        run.click(synthesize,[full_text,*settings],outputs,api_name='synthesize',concurrency_limit=1)
        start.click(start_session,[session,*settings],[session,info,indicator],queue=False,api_name='start_input').then(
            listen,session,outputs,concurrency_limit=4,api_name='listen_input')
        send.click(send_text,[session,delta],[delta,indicator],queue=False,api_name='send_text')
        finish.click(lambda s,t:send_text(s,t,True),[session,delta],[delta,indicator],queue=False,api_name='finish_input')
        cancel_button.click(cancel,session,[session,info],queue=False,api_name='cancel_input')
        simulate_button.click(simulate,[example,interval,session,*settings],[session,*outputs],
                              concurrency_limit=1,api_name='simulate_input')
    demo.queue().launch(server_name='0.0.0.0',server_port=args.port,
                        allowed_paths=[str(ROOT/'results/demo'),REFERENCE])


if __name__=='__main__':
    main()
