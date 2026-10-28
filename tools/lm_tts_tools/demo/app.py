"""Gradio listening page shared by the original and native vLLM runtimes."""
import argparse
import json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--pytorch-port', type=int, default=7862)
    parser.add_argument('--vllm-port', type=int, default=7863)
    args = parser.parse_args()
    ports = {'pytorch': args.pytorch_port, 'vllm': args.vllm_port}
    # Local backends must not use an inherited HTTP/SOCKS proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def synthesize(mode, model, ref_audio, ref_text, text, language, temperature, top_k, top_p,
                   penalty, greedy_cp, cp_temperature, cp_top_k, cp_top_p, seed, max_tokens,
                   non_streaming_mode,
                   progress=gr.Progress()):
        if not ref_audio or not text.strip():
            raise gr.Error('请提供参考音频和要合成的文本')
        if not ref_text.strip():
            raise gr.Error('ICL 克隆需要参考文本')
        progress(0, desc='正在合成；首次加载或切换模型可能需要数分钟')
        data = dict(mode=mode, model=model, ref_audio=ref_audio, ref_text=ref_text, text=text, language=language,
                    temperature=float(temperature), top_k=int(top_k), top_p=float(top_p), penalty=float(penalty),
                    greedy_cp=bool(greedy_cp), cp_temperature=float(cp_temperature), cp_top_k=int(cp_top_k),
                    cp_top_p=float(cp_top_p), seed=int(seed), max_tokens=int(max_tokens),
                    x_vector_only_mode=False, non_streaming_mode=bool(non_streaming_mode))
        req = urllib.request.Request(f'http://127.0.0.1:{ports[mode]}/synthesize_stream',
                                     data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
        try:
            with opener.open(req, timeout=1200) as response:
                for line in response:
                    event = json.loads(line)
                    if event['type'] == 'error':
                        raise gr.Error(event['detail'])
                    if event['type'] == 'chunk':
                        wav, sr = sf.read(event['audio'], dtype='int16')
                        yield (sr, wav), None, f"已收到第 {event['index']+1} 块音频 · {event['elapsed_seconds']:.2f} 秒"
                    elif event['type'] == 'result':
                        result = event
                        info = (f"{mode} · step-{model}｜音频 {result['audio_seconds']:.2f} 秒"
                                f"｜首块 {result['first_audio_seconds']:.2f} 秒｜合成 {result['seconds']:.2f} 秒"
                                f"｜{result['chunks']} 块｜CP {'greedy' if greedy_cp else 'sampling'}")
                        yield gr.skip(), result['audio'], info
        except urllib.error.HTTPError as exc:
            raise gr.Error(json.loads(exc.read()).get('detail', str(exc))) from exc
        except urllib.error.URLError as exc:
            raise gr.Error('所选推理服务正在启动或暂不可用，请稍后重试') from exc

    config = json.loads((MODELS['56384']['source'] / 'config.json').read_text())
    languages = ['Auto'] + [s.capitalize() for s in config['talker_config']['codec_language_id'] if 'dialect' not in s]
    with gr.Blocks(title='LM-TTS · PyTorch / vLLM') as demo:
        gr.Markdown('# LM-TTS 声音克隆试听\nvLLM 支持边生成边播放。step-10000 的 PyTorch 基线使用 UltraEval 相同的 LM-TTS 推理实现；PyTorch 当前整段返回。两种模式尚未完全数值对齐。')
        with gr.Row():
            with gr.Column():
                mode = gr.Radio([('PyTorch（BF16 基线）', 'pytorch'), ('vLLM（原生插件）', 'vllm')], value='vllm', label='推理模式')
                model = gr.Dropdown([('Emilia step-56384', '56384'), ('Emilia step-10000（5×10 interleaved）', '10000')], value='10000', label='模型')
                ref_audio = gr.Audio(value=REFERENCE, type='filepath', label='参考音频（最长 30 秒）')
                ref_text = gr.Textbox(value=REFERENCE_TEXT, label='参考音频文本', lines=2)
                non_streaming_mode = gr.Checkbox(value=False, label='一次性输入全部文本（关闭：10000 使用 5×10；56384 使用 Qwen overlay）')
                text = gr.Textbox(value='你好，这是语音合成测试。', label='要合成的文本', lines=3)
                language = gr.Dropdown(languages, value='Auto', label='语言')
                preset = gr.Radio(list(PRESETS), value='评测默认采样', label='采样预设')
                with gr.Accordion('采样参数', open=True):
                    temperature = gr.Slider(0, 1.5, value=0.9, step=0.05, label='Talker temperature（0 为 greedy）')
                    top_k = gr.Slider(0, 100, value=50, step=1, label='Talker top-k（0 不限制）')
                    top_p = gr.Slider(0.05, 1, value=1, step=0.05, label='Talker top-p')
                    penalty = gr.Slider(1, 3, value=1.05, step=0.05, label='重复惩罚')
                    greedy_cp = gr.Checkbox(value=False, label='Code predictor greedy（关闭则采样）')
                    cp_temperature = gr.Slider(0.05, 1.5, value=0.9, step=0.05, label='Code predictor temperature')
                    cp_top_k = gr.Slider(0, 100, value=50, step=1, label='Code predictor top-k（0 不限制）')
                    cp_top_p = gr.Slider(0.05, 1, value=1, step=0.05, label='Code predictor top-p')
                    seed = gr.Number(value=42, precision=0, label='随机种子')
                    max_tokens = gr.Slider(3, 2048, value=512, step=1, label='最大生成帧数')
                run = gr.Button('合成', variant='primary')
            with gr.Column():
                audio = gr.Audio(label='实时播放', streaming=True, autoplay=True)
                complete_audio = gr.Audio(label='完整音频')
                info = gr.Markdown('首次加载及切换模型可能需要数分钟。vLLM 音频分块返回与文本输入方式分别控制。')
        preset.change(lambda p: PRESETS[p], preset, [temperature, top_k, top_p, penalty, greedy_cp, cp_temperature, cp_top_k, cp_top_p])
        run.click(synthesize, [mode, model, ref_audio, ref_text, text, language, temperature, top_k, top_p, penalty,
                              greedy_cp, cp_temperature, cp_top_k, cp_top_p, seed, max_tokens,
                              non_streaming_mode], [audio, complete_audio, info],
                  api_name='synthesize', concurrency_limit=1)
    demo.queue().launch(server_name='0.0.0.0', server_port=args.port,
                        allowed_paths=[str(ROOT / 'results/demo'), REFERENCE])


if __name__ == '__main__':
    main()
