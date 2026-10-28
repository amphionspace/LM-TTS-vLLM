# Qwen-TTS 功能兼容进度

> 本页含 step-100 的历史验证，不能自动推广到 step-10000。当前模型与整体进度以
> [README](../README.md) 为准。客户端文本续传仍未完成，音频分块不等于输入流式。

目标以原 `qwen_tts.Qwen3TTSModel` 的公开推理接口为准，先补功能，再集中检查
数值差异与性能。功能开发与验证在独立 `feat/qwen-tts-compat` worktree 中进行。

两份指定 checkpoint 的 `tts_model_type` 均为 `base`。VoiceDesign 的自然语言
音色设计及 CustomVoice 的预置 speaker / instruct 需要对应权重，不能通过
让 Base 模型接受同名字段来宣称兼容。

| 能力 | 当前状态 |
| --- | --- |
| BF16，Talker / CP FA2，KV cache | 已验证 |
| Talker / CP greedy、temperature、top-k、top-p | 已验证；CP sampling 暂为 eager |
| 请求级随机种子 | 在线服务验证了同种子复现、不同种子变化 |
| Base：参考文本 + 参考音频 ICL | 单请求及混合批量已验证 |
| Base：仅说话人向量，不要求参考文本 | 两份指定 checkpoint 均已验证 |
| non_streaming_mode=True / False 文本条件方式 | 两份模型的四种克隆组合已验证；不是音频传输方式 |
| 批量 / 连续批处理 | max_num_seqs=2 混合四请求已验证；CP 暂逐请求执行，尚未向量化 |
| 标准 CLI / HTTP | 自动插件加载及四种克隆组合通过；HTTP CP 采样与 seed 已验证 |
| 缓存 voice_clone_prompt 输入 / 生成 | 原生导出及不携带参考音频的四种组合复用已验证 |
| 文件、URL、base64、(waveform, sr) 输入 | 六种格式的解码波形一致；路径、file URI、data URI、waveform 已端到端验证 |
| 真正音频分块返回 | AsyncOmni 分块与长度截断通过：10 + 25 + 25 + 19 帧 |
| 任意文本的 prompt 长度 | Gradio 已按原生模板和 codec 下采样率计算；待统一为公共工具 |
| min/max token、EOS、repetition penalty | 批量 EOS 与截断已验证；流式 max=80 返回 79 帧已验证 |
| CustomVoice / VoiceDesign | 需要对应模型权重，未宣称已通过 |

每一行的“已实现”不等于“已验证”。当前两个 Base 模型的完整 codec 一致性
仍未通过；不会把功能冒烟结果当作精度或性能结论。

开发分支验证记录：`python -m lm_tts_tools.compat_smoke` 在 56384 上运行四种
组合，各返回 7 × 16 个有效 codec 和 0.56 秒音频（max_tokens=8）。
结果见 `results/base-compat/report.json`。基础回归 15 项通过；另在 Omni 环境
验证了配置 roundtrip 和标准插件入口 / serving adapter 注册两项测试。

step-100 的同样四种组合与四种参考音频格式均已通过，见
`results/base-100-formats-v2/report.json`，分别返回 2 / 2 / 5 / 7 帧。
这同时覆盖 EOS 提前结束与 max_tokens 截断，不把早期模型的短输出视为接口失败。

2026-10-28：`results/cached-prompt-resume/report.json` 验证原生导出提示后，不再
传入参考音频的四种组合；各生成 7 × 16 codec 和 0.56 秒音频。与之前的直接
参考音频用例比较，两个 x-vector 用例的 codec 完全相同；两个 ICL 用例存在
差异，尚未定位，见同目录 `raw-comparison.json`。不能据此宣称 ICL 缓存数值对齐。

`lm_tts_tools.input_formats_smoke` 对路径、file URI、本机 HTTP、裸 base64、
data URI、waveform 的解码结果逐样本比较，六项均与原 WAV 完全相同。插件
识别裸 base64 音频头，避免编码中的 `/` 被上游误判成文件路径。

`results/streaming-cap-ack/report.json`：真实 DELTA 音频总计 151680 samples /
24 kHz = 6.32 秒，对应 79 帧。三个中间块为 10、25、25 帧，最后一个块为
19 帧。MRv2 的输出线程会在调度器确认结束之前发布达到上限的 token，因此
插件依据 `output_token_count` 与 `sampling_params.max_tokens` 提前过滤末帧。
流式 `codec_frames` 含解码上下文和参考帧，不可直接当作生成帧数；WAV 由所有
DELTA 块拼接，最终事件并非累计整段音频。

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
通过 SDK 传入 NumPy waveform 时，用工具层 `reference_audio_payload` 转成
显式 inline-array 载荷；固定版本的上游 metadata 序列化会丢失嵌套 NumPy
数组的数据缓冲区。字符串输入不需要这个转换。
流式测试显式使用 `deploy_config` 和 `RequestOutputKind.DELTA`，并断言实际
收到多个中间音频块；只收到完整音频会判失败。

`--cached-prompt` 使用标准 `apply_model` RPC 导出 JSON 可序列化的提示，
再移除请求中的参考音频，验证完全复用提示的生成路径。该上游 RPC 的 callable
需要 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`；只在可信本地离线测试进程中设置，
不写入插件默认配置或在线服务。普通生成和直接传入缓存提示不需要这个开关。
