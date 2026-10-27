"""Gradio listening page shared by the original and native vLLM runtimes."""
import argparse
import json
import urllib.error
import urllib.request

import gradio as gr

from .common import MODELS, REFERENCE, ROOT

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
                   x_vector_only_mode, non_streaming_mode,
                   progress=gr.Progress()):
        if not ref_audio or not text.strip():
            raise gr.Error('请提供参考音频和要合成的文本')
        if not x_vector_only_mode and not ref_text.strip():
            raise gr.Error('ICL 克隆需要参考文本')
        progress(0, desc='正在合成；首次加载或切换模型可能需要数分钟')
        data = dict(mode=mode, model=model, ref_audio=ref_audio, ref_text=ref_text, text=text, language=language,
                    temperature=float(temperature), top_k=int(top_k), top_p=float(top_p), penalty=float(penalty),
                    greedy_cp=bool(greedy_cp), cp_temperature=float(cp_temperature), cp_top_k=int(cp_top_k),
                    cp_top_p=float(cp_top_p), seed=int(seed), max_tokens=int(max_tokens),
                    x_vector_only_mode=bool(x_vector_only_mode), non_streaming_mode=bool(non_streaming_mode))
        req = urllib.request.Request(f'http://127.0.0.1:{ports[mode]}/synthesize',
                                     data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
        try:
            with opener.open(req, timeout=1200) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise gr.Error(json.loads(exc.read()).get('detail', str(exc))) from exc
        except urllib.error.URLError as exc:
            raise gr.Error('所选推理服务正在启动或暂不可用，请稍后重试') from exc
        info = (f"{mode} · step-{model}｜音频 {result['audio_seconds']:.2f} 秒｜合成 {result['seconds']:.2f} 秒"
                f"｜模型加载 {result['load_seconds']:.2f} 秒｜CP {'greedy' if greedy_cp else 'sampling'}")
        return result['audio'], info

    config = json.loads((MODELS['56384']['source'] / 'config.json').read_text())
    languages = ['Auto'] + [s.capitalize() for s in config['talker_config']['codec_language_id'] if 'dialect' not in s]
    with gr.Blocks(title='LM-TTS · PyTorch / vLLM') as demo:
        gr.Markdown('# LM-TTS 声音克隆试听\n选择 PyTorch 原 demo 或 vLLM 插件，使用相同参考音频比较。两种模式当前尚未完全数值对齐。')
        with gr.Row():
            with gr.Column():
                mode = gr.Radio([('PyTorch（原 demo）', 'pytorch'), ('vLLM（原生插件）', 'vllm')], value='pytorch', label='推理模式')
                model = gr.Dropdown([('Emilia step-56384', '56384'), ('训练 checkpoint step-100', '100')], value='56384', label='模型')
                ref_audio = gr.Audio(value=REFERENCE, type='filepath', label='参考音频（最长 30 秒）')
                ref_text = gr.Textbox(value='在此奉劝大家别乱打美白针。', label='参考音频文本', lines=2)
                x_vector_only_mode = gr.Checkbox(value=False, label='仅克隆说话人音色（无需参考文本）')
                non_streaming_mode = gr.Checkbox(value=True, label='一次性输入文本（关闭使用交错文本条件，不影响音频返回方式）')
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
                    max_tokens = gr.Slider(2, 1024, value=512, step=1, label='最大生成帧数')
                run = gr.Button('合成', variant='primary')
            with gr.Column():
                audio = gr.Audio(label='合成结果')
                info = gr.Markdown('首次加载及切换模型可能需要数分钟。step-100 是训练早期模型，可能很快结束。')
        preset.change(lambda p: PRESETS[p], preset, [temperature, top_k, top_p, penalty, greedy_cp, cp_temperature, cp_top_k, cp_top_p])
        run.click(synthesize, [mode, model, ref_audio, ref_text, text, language, temperature, top_k, top_p, penalty,
                              greedy_cp, cp_temperature, cp_top_k, cp_top_p, seed, max_tokens,
                              x_vector_only_mode, non_streaming_mode], [audio, info],
                  api_name='synthesize', concurrency_limit=1)
    demo.queue().launch(server_name='0.0.0.0', server_port=args.port,
                        allowed_paths=[str(ROOT / 'results/demo'), REFERENCE])


if __name__ == '__main__':
    main()
