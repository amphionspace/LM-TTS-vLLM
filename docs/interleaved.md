# LM-TTS 5×10 输入协议

step-10000 的 `talker_config.lm_tts_streaming_layout=interleaved` 自动选择训练协议。
`non_streaming_mode=False` 每 5 个文本 token 插入 10 帧 codec，文本 EOS 所在块之后
可以连续生成音频。参考 codec 同样占用音频帧位置。`True` 使用训练的非流式布局。
两种布局都按 LM-TTS-Training 的顺序逐项累加 BF16 codebook embeddings。

原生插件用 codec-pad 占位 token 为插入的文本分配普通 vLLM KV 位置，强制这些位置
不采样音频、不运行 CP，也不进入音频解码器。文本结束前屏蔽 codec EOS。
用户的 `max_new_tokens` 仍按音频计数；内部调度预算另外计入文本位置。
长度上限延续 UltraEval 的约定：达到 N 的上限时返回 N−1 帧音频。

输入目前一次提交全文，模型内部按 5×10 交错执行；这与客户端逐次补充文本是两个接口。
音频分块由标准 AsyncOmni 和 Code2Wav stage 提供。插件本身没有专用客户端或 worker。
Gradio 的 loopback HTTP 后端只负责隔离 PyTorch / vLLM 两套环境和转发音频块。

PyTorch 基线直接调用 sibling `LM-TTS-Training/lm_tts/inference/codec.py` 的
`CodecSynthesizer`，与 `UltraEval-Audio/scripts/local_qwen.py` 一致，使用 BF16 SDPA。
vLLM Talker / CP 仍使用 FA2。streaming 输入布局与音频分块不意味着数值对齐已经通过。

验证工具：

- `lm_tts_tools.streaming_smoke --interleaved`：跨文本边界、真实分块输出、长度上限。
- `lm_tts_tools.training_reference`：从原始导出模型保存 BF16 prefill、hidden、logits 和
  共享参考 codec / speaker embedding，避免前处理差异影响 Talker 对照。
- `streaming_smoke --reference <目录>`：受信任的本机离线 RPC 验证 prompt 和首个 hidden。
  此诊断模式需要 `VLLM_ALLOW_INSECURE_SERIALIZATION=1`；demo 不启用此设置。

当前验证阶段使用 eager Talker 和标准 vLLM sampler；CP greedy 仍保留自己的 CUDA graph。
这些设置用于先验证协议，不能作为最终速度验收结果。

2026-10-28 首轮功能验证：长文本插入 35 个文本位置、115 个调度步骤，
返回 10 + 25 + 25 + 19 共 79 帧音频（151680 samples，24 kHz）。
23850 组前缀布局与当前 LM-TTS-Training 实现一致。
这验证了布局与分块完整性，不代表跨引擎生成的 codec 已数值一致。
