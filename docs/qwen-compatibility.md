# Qwen-TTS 功能兼容进度

目标以原 `qwen_tts.Qwen3TTSModel` 的公开推理接口为准，先补功能，再集中检查
数值差异与性能。当前工作在独立 `feat/qwen-tts-compat` worktree；在线 Gradio
仍运行主分支，避免开发中的改动影响试听。

两份指定 checkpoint 的 `tts_model_type` 均为 `base`。VoiceDesign 的自然语言
音色设计及 CustomVoice 的预置 speaker / instruct 需要对应权重，不能通过
让 Base 模型接受同名字段来宣称兼容。

| 能力 | 当前状态 |
| --- | --- |
| BF16，Talker / CP FA2，KV cache | 已验证 |
| Talker / CP greedy、temperature、top-k、top-p | 已验证；CP sampling 暂为 eager |
| 请求级随机种子 | 在线服务验证了同种子复现、不同种子变化 |
| Base：参考文本 + 参考音频 ICL | 单请求及混合批量已验证 |
| Base：仅说话人向量，不要求参考文本 | 56384 已验证 |
| non_streaming_mode=True / False 文本条件方式 | 四种克隆组合已验证；不是音频传输方式 |
| 批量 / 连续批处理 | max_num_seqs=2 混合四请求已验证；CP 暂逐请求执行，尚未向量化 |
| 缓存 voice_clone_prompt 输入 / 生成 | 上游有消费接口，公开导出接口待接入和验证 |
| 文件、URL、base64、(waveform, sr) 输入 | 上游有加载实现，待逐项端到端验证 |
| 真正音频分块返回 | 已增加独立 streaming 配置与末帧处理，待验证 |
| 任意文本的 prompt 长度 | Gradio 已按原生模板和 codec 下采样率计算；待统一为公共工具 |
| min/max token、EOS、repetition penalty | 主分支已验证基本路径；批量 / streaming 边界待补 |
| CustomVoice / VoiceDesign | 需要对应模型权重，未宣称已通过 |

每一行的“已实现”不等于“已验证”。当前两个 Base 模型的完整 codec 一致性
仍未通过；不会把功能冒烟结果当作精度或性能结论。

开发分支验证记录：`python -m lm_tts_tools.compat_smoke` 在 56384 上运行四种
组合，各返回 7 × 16 个有效 codec 和 0.56 秒音频（max_tokens=8）。
结果见 `results/base-compat/report.json`。单元测试 13 项，12 通过、1 项因
基线环境不包含可选原生运行时而跳过。

复现入口（运行于原生 Omni 环境，设置 `PYTHONPATH=src:tools`）：

```bash
python -m lm_tts_tools.compat_smoke --model artifacts/native-56384 \
  --ref-audio /path/to/reference.wav --output results/base-compat
python -m lm_tts_tools.compat_smoke --model artifacts/native-100 \
  --ref-audio /path/to/reference.wav --output results/audio-formats --audio-formats
python -m lm_tts_tools.streaming_smoke --model artifacts/native-56384 \
  --ref-audio /path/to/reference.wav --output results/streaming
```

这些固定用例的参考文本是“在此奉劝大家别乱打美白针。”，应使用对应音频。
音频格式测试覆盖路径、file URI、data URI/base64 和 `(waveform, sr)`。
流式测试显式使用 `deploy_config` 和 `RequestOutputKind.DELTA`，并断言实际
收到多个中间音频块；只收到完整音频会判失败。

`--cached-prompt` 使用标准 `apply_model` RPC 导出 JSON 可序列化的提示，
再移除请求中的参考音频，验证完全复用提示的生成路径。该上游 RPC 的 callable
需要 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`；只在可信本地离线测试进程中设置，
不写入插件默认配置或在线服务。普通生成和直接传入缓存提示不需要这个开关。
