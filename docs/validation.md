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

`lm_tts_tools.ultraeval` 已执行原 `voice_clone_infer.py` 的 Seed-TTS 中文首条输入，
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

## vLLM 原生 FlashAttention 的逐层定位

`results/56384-layer-audit-qkv` 对相同的首步 prefill 做逐层 hook，实际计算仍使用
vLLM `FLASH_ATTN`。第一层 RMSNorm、Q/K/V 投影、Q/K norm，以及经过 RoPE 后
真正送入 attention 的 Q、K、V，均逐元素完全一致。

首个差异出现在 attention 输出：最大绝对误差 0.0078125，平均绝对误差
0.000001909，相对 L2 约 0.0001916。由此可以把这条路径上的初始差异定位到
attention 计算，而不是前端输入、QKV 投影或 RoPE。后续层和 code predictor
会继续传播、放大差异；这个定位结果不等于完整 codec 一致性已经通过。

## 原生 attention backend 对照

同一 step-56384 greedy 请求，均使用 prefix cache；以下没有 SDPA 替换：

| backend | teacher-forced argmax | 最大 logits 绝对误差 | 首个完整帧分歧 | 输出帧数 |
| --- | ---: | ---: | ---: | ---: |
| FLASH_ATTN | 100% | 0.1875 | 0 | 77 |
| TRITON_ATTN | 100% | 0.125 | 1 | 79 |
| FLASHINFER | 100% | 0.1875 | 0 | 79 |

结果分别在 `56384-cached-greedy`、`56384-triton-cached-greedy` 和
`56384-flashinfer-cached-greedy`。这不支持只凭单句最大误差更低就更改默认 backend。

进一步固定第一帧首码本 1174，仅替换 code predictor 的 Talker hidden 输入，
重现了 residual codebook 5 的分歧：基线 token 810 与 1412 的 logits 都为 8.0625，
替换 hidden 后变为 8.0 与 8.0625。原 hidden 路径精确重现已保存的首帧码本。
产物为 `56384-layer-audit-qkv/code-predictor-margin.json`；它说明该处存在 BF16
平局敏感性，不应把所有 greedy 分歧都归为协议错误，也不能据此跳过数值检查。

同一保存的 attention 输入在原 Torch 2.8 环境重放，通过 profiler 确认 SDPA 调用
`aten::_scaled_dot_product_flash_attention`，且输出精确等于基线保存值。
因此这里的基线 SDPA 不是纯 math attention；它与 vLLM FA2 是不同的执行路径。

## 原生 code predictor 与 CUDA graph

插件内的新 code predictor 使用 vLLM FlashAttention 2、BF16 和帧内增量 KV cache，
支持 greedy 的 15 步 CUDA graph，不依赖 Qwen-TTS 或 Transformers，也不调用 SDPA。
独立测试覆盖 step-56384 的 8 个帧输入、step-100 的 5 个帧输入：

- 13 个输入的原生 eager 与 CUDA graph，码本和 logits 均逐元素一致。
- 交替输入后重用第一个输入，结果一致；帧内 cache 未观察到跨调用残留。
- 与原 PyTorch 相比，step-56384 有 5/8 个输入的完整 residual codes 一致，
  teacher-forced 最大 logits 误差 0.15625；step-100 为 5/5 一致，最大误差 0.03125。
- CP graph 独立微基准的中位耗时分别约 12.33 ms 和 12.37 ms；原 PyTorch
  这些输入的暖调用约 69–81 ms。条件并非完整 serving 基准，不据此发布端到端倍数。

微基准原始数据在 `cp-native-{56384,100}.json`；
`cp-native-{56384,100}-cli.json` 保留可复现工具的复验结果；图与 eager 一致性、
有限值和 cache 重用是强制检查。step-100 还通过了原码本严格一致性门槛。

整链路使用 `--engine-mode cached --code-predictor-engine vllm-graph --greedy`：

| 模型 | PyTorch / vLLM 帧数 | vLLM 音频秒数 | vLLM 耗时 | RTF | 首个码本帧分歧 |
| --- | ---: | ---: | ---: | ---: | ---: |
| step-56384 | 79 / 79 | 6.32 | 4.523 s | 0.716 | 1 |
| step-100 | 2 / 2 | 0.16 | 0.266 s | 1.665 | 1 |

两组都没有通过原 PyTorch 的严格全 codec 一致性；step-56384 两边到达长度上限，
step-100 两边自然 EOS。step-56384 CP 累计约 1.000 s，Talker engine 累计 2.902 s。
这两次测试和其他任务共享 GPU，属于集成冒烟数据，尚非正式速度验收。

## 插件与工具拆分

主 wheel 仅含 `lm_tts_vllm/{__init__,model,precision,code_predictor}.py`，
开发工具位于独立 `lm-tts-vllm-tools` wheel。检查确认主包不导入工具、
Qwen-TTS 或自定义进程启动代码；SDPA 替换诊断也已移到工具包。
这轮推理移除了工具中的显式 `register()` 调用，模型通过安装包 entry point 自动注册。
当前仍是 pooling 对齐通路，最终原生完整 TTS 启动与音频输出接口留到后续整体验证。
