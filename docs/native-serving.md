# 标准原生语音服务

安装主插件与可选 Omni 依赖后，直接使用标准命令；不需要启动本仓库的
Gradio 后端、客户端桥接或自定义 worker：

```bash
pip install -e '.[omni]'
CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/native-model --omni \
  --host 127.0.0.1 --port 8000 --kv-cache-memory-bytes 536870912
```

模型路径应是 `prepare_native` 生成的视图。默认配置为 BF16 验证配置；TP/PP=1，
Talker graph、prefix cache、async scheduling 尚未启用。原生音频分块可指定
`--deploy-config /path/to/lm_tts_vllm/native_streaming.yaml`。
当前已验证 SDK 分块及 HTTP 整句生成；HTTP 分块端到端尚未单独验证。

共享测试环境里另有 `smg-servicer` 插件，其 CLI 参数补丁与当前版本冲突；在
这个环境需要加 `VLLM_PLUGINS=lm_tts,vllm_omni_register_models`，保留本插件与
官方 Omni 模型注册。独立安装环境不需要该过滤项。

向 `/v1/audio/speech` 提交 JSON，例如：

```json
{
  "input": "你好，这是语音合成测试。",
  "task_type": "Base",
  "language": "Chinese",
  "ref_audio": "data:audio/wav;base64,<参考音频编码>",
  "ref_text": "在此奉劝大家别乱打美白针。",
  "x_vector_only_mode": false,
  "non_streaming_mode": true,
  "response_format": "wav",
  "max_new_tokens": 512,
  "seed": 42,
  "extra_params": {
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.05,
    "subtalker_dosample": true,
    "subtalker_temperature": 0.9,
    "subtalker_top_k": 50,
    "subtalker_top_p": 1.0
  }
}
```

`x_vector_only_mode=true` 时参考文本可省略。文本条件模式与音频是否流式返回
相互独立。`extra_params.do_sample=false` 使 Talker greedy；CP 用自己的
`subtalker_dosample` 控制。默认验证配置是 greedy，采样请求需显式设置参数。

LM-TTS 的 serving adapter 只匹配 `LMTTSForConditionalGeneration`，普通
Qwen-TTS 仍使用上游 adapter。它将 CP 参数传入请求元数据，并在用户明确指定
长度上限时返回已有音频，与原 Qwen `generate` 一致；默认自动预算耗尽仍保留
上游错误行为。返回的是部分音频时，不代表模型已自然生成 EOS。

2026-10-28 实测：标准 CLI 自动注册和启动，四种 Base 克隆组合均返回 24 kHz
音频；max_new_tokens=8 时为 7 帧 / 0.56 秒。只对 CP 采样时，同 seed 的 WAV
相同，seed 42 改为 43 后 WAV 改变。记录位于 `results/native-http/`；测试入口
是独立工具 `python -m lm_tts_tools.serve_smoke`。这不构成跨引擎精度或性能结论。
