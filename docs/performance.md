# 精度与速度必须共同验收

关闭优化是用于定位误差的验证条件，不能作为完整插件的默认实现或交付结论。
基准为 `tts-demo` / UltraEval 的 BF16 SDPA，不以 FP32 改变用户实际的推理条件。

## 已实现的验证路径

| 配置 | 数值操作 | 跨帧 KV 复用 | 用途 |
| --- | --- | --- | --- |
| `bf16 + reference` | 保留 Qwen-TTS 的 BF16 舍入边界 | 无 | 数值定位 |
| `native + reference` | vLLM 原生融合算子 | 无 | 算子误差对照 |
| `bf16 + cached` | 保留 BF16 舍入边界 | vLLM prefix blocks | 缓存精度与性能验证 |

`cached` 通过 vLLM 0.32 的 embedding-content hash 区分前缀，不能仅用虚拟 token ID
来共享 cache，否则不同音频/说话人条件会误命中。报告分别记录 submitted/cached tokens。
teacher-forced 与 free-running 之间清空 cache；质量与速度数据不能来自不同配置。

## 完整推理通路的要求

1. Talker、code predictor 的逐帧调用应在 GPU worker 内完成，使用请求独立的状态，
   正式原生路径已消除对齐工具的逐帧 IPC。真正的 autoregressive decode 复用 KV cache，
   不能把每步全前缀重算包装成最终加速版本。
2. code predictor 已接入原生 Omni 调度，支持 greedy 和随机采样；greedy 的独立
   CUDA graph 已有局部验证，sampling 当前使用 eager FA2。批量 CP 仍需优化，
   功能冒烟不能替代端到端吞吐验收。帧 embedding 的求和必须按 checkpoint 协议：
   step-10000 的训练布局采用逐 codebook BF16 连加，旧 Qwen 路径保留原来的求和语义。
3. 逐项验证 CUDA graph、融合 norm/activation、continuous batching、异步输出。
   多请求的 speaker/ref-code/hidden state 必须隔离，并验证取消、结束与重用时的清理。
4. 音频 decoder 的流式块必须是增量输出，参考音频裁剪和边界上下文不能因加速改变。
5. interleaved 5×10 是独立输入协议，需要按照训练和 `local_interleaved.py` 验证。
   当前 non-streaming 跑通不能代替 interleaved streaming 验收。

当前原生 pipeline 使用固定 vLLM-Omni 提交，版本与已知边界见 [README](../README.md)。
step-10000 的全文输入 5×10 和音频分块已验证；真正客户端文本续传尚未打通，
当前原型被 stage 1 runner v2 不支持 resumable input 的检查拒绝。

## 验收数据

- 相同输入、权重、BF16 dtype、语言和条件；teacher-forced hidden/logits 误差、有效
  codec argmax、top-2 margin。区别原始 logits 与施加 repetition penalty 后的决策。
- greedy：全 codebook、首个分歧帧、EOS、生成长度；随机采样：检查 logits 分布和
  多样本质量，不把相同 seed 等同于跨内核 bitwise 保证。
- 当前不运行 CER/WER 或说话人相似度评分；UltraEval 仅用于核对实际推理协议与输入。
  对齐验证覆盖中英文、Auto/显式语言、长短句和参考音频，直接比较数值和 codec 输出。
- 速度：预热后的端到端 RTF、真实首个音频块延迟、p50/p95、音频秒/墙钟秒，
  并发 1/4/8、显存峰值。当前 `first_codec_frame_seconds` 只是 codec 首帧，不能标成 TTFA。
- 同卡、相同请求集、相同输出长度条件下比较；记录 warmup、缓存冷热和 GPU 竞争。
  验证期间共享 A800 的其他任务会动态改变显存和负载，因此单次 wall time 只作冒烟数据，
  不能据此发布加速倍数。

只有数值/codec 一致性回归与性能数据同时满足目标，才把对应优化设为默认。
