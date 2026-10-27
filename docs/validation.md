# BF16 验证记录（2026-10-27）

当前结论：两个 checkpoint 都已通过插件生成音频；数值误差已量化，但完整 codec
一致性尚未通过，当前不能作为精度和速度均验收完成的 serving 插件。

## 环境和输入

- PyTorch 基线：原 Qwen-TTS 0.1.1、Torch 2.8.0+cu126、Transformers 4.57.3，BF16 SDPA。
- vLLM：0.32.0、Torch 2.13.0+cu130、Transformers 5.19.0，BF16；A800 80GB，TP/PP=1。
- 参考音频：Seed-TTS 中文 `000000.wav`，参考文本“在此奉劝大家别乱打美白针。”。
- Auto 组文本：“简单地说，这相当于惠普把消费领域市场拱手相让了。”，seed=42，
  Talker 和 code predictor 均采用 T=0.9、top-k=50，Talker repetition penalty=1.05。
- greedy 组文本：“你好，这是语音合成测试。”，Chinese、两个阶段均 greedy，max_new_tokens=80。
- 所有对照保持同一请求的 frontend、权重、采样和音频 decoder 不变。

## 数值与生成

以下 argmax 是同一 PyTorch 前缀下的 Talker 原始有效 codec logits argmax，
尚未施加 repetition penalty，不能视作全码本或自由生成一致。

| checkpoint / 配置 | teacher-forced 步数 | argmax 一致率 | 最大 logits 绝对误差 | 自由生成首个分歧帧（从 0 起） | PyTorch / vLLM 帧数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| step-100 / bf16 reference Auto | 6 | 100% | 0.109375 | 4 | 5 / 7 |
| step-56384 / bf16 reference Auto | 91 | 100% | 0.25 | 6 | 90 / 95 |
| step-100 / native reference Auto | 6 | 100% | 0.125 | 3 | 5 / 5 |
| step-56384 / bf16 cached greedy | 80 | 100% | 0.1875 | 0 | 79 / 77 |
| step-56384 / sdpa reference greedy | 80 | 100% | 0.25 | 1 | 79 / 79 |

这些运行的完整 codec 序列均不完全相同。greedy cached 组的第一帧在 residual
codebook 5 首次不同；它说明 hidden 的微小偏差会经过 code predictor 放大，
即使 Talker 的同前缀 argmax 一致也不能跳过后续码本验证。
step-100 是训练早期 checkpoint，短输出本身不能单独归因为插件。
greedy 组 PyTorch 达到长度上限；vLLM 在 77 帧后自然 EOS，因此不是等长速度比较。

## 缓存与耗时

`56384-cached-greedy` 自由生成阶段提交 9867 tokens，其中 9120 命中 prefix cache
（约 92.4%）；teacher-forced 后先清空 cache。其端到端墙钟时间为 9.958 秒，
音频 6.16 秒，RTF 1.617，Talker engine 时间累计 2.918 秒。

以上仅证明缓存生效。共享 GPU 的竞争、不同输出长度、未做系统预热和多次重复，
使这些数据不足以支持加速倍数结论。当前逐帧 IPC 与 PyTorch code predictor 仍有开销。
完整性能目标和回归矩阵见 [performance.md](performance.md)。

## 可复查产物

本地 `results/100-auto-compare-v2`、`results/56384-auto-compare`、
`results/100-native-compare-v2`、`results/56384-cached-greedy` 含 report、
codec 数组和音频。大文件不进入 Git；上述表格保留可审阅的测量摘要。

## 原评测入口冒烟测试

`lm_tts_vllm.ultraeval` 已执行原 `voice_clone_infer.py` 的 Seed-TTS 中文首条输入，
step-100、BF16 cached、non-streaming、Auto、batch=1。原 journal 返回 `status=ok`，
24 kHz / 0.56 秒，speaker embedding 与 ICL 调用都通过原入口的检查。
结果在 `results/ultraeval-100-smoke`。这只是推理接口验证，不是质量评分。
按当前任务范围，不运行 CER/WER 或说话人相似度评分。

## 排除 vLLM 后的前缀重算对照

在原 Torch 2.8.0+cu126 / BF16 SDPA 环境，用 step-56384 greedy trace 的同一
输入逐步完整前缀重算，比较原增量 KV 解码保存的 hidden/logits：首步误差为 0，
80 步 Talker argmax 一致率 100%，全程最大 logits 绝对误差 0.125。
结果在 `results/56384-greedy/pytorch-full-prefix-audit.json`。

因此完整前缀重算本身就会引入 BF16 计算形状相关的差异；不能把与增量基线的
所有误差都归因于 vLLM attention 内核。它也不能证明插件误差已消除。

## SDPA 内核诊断

`results/56384-sdpa-greedy-v3` 使用 vLLM 运行时的 PyTorch SDPA 替换 attention，
其余保留 BF16 profile。首步 hidden relative-L2 为 0.010294，logits 最大绝对误差
0.0625；全程 teacher-forced logits 最大绝对误差 0.25。第一个完整 codec 帧一致，
第二帧开始不同；两边均生成 79 帧并到达 token 上限。

`--require-identical-codes` 保存报告后返回失败，符合回归门槛的预期。SDPA 诊断
没有消除差异，且仍跨 Torch/CUDA 版本，不能据此将剩余误差归因于单一算子。
这是定位模式，不作为最终性能路径。
