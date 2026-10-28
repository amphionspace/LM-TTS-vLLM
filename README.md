# LM-TTS-vLLM

为 Emilia / LM-TTS checkpoint 提供 **vLLM 外部模型插件**。目标是在遵循训练和
UltraEval 推理协议的前提下，对齐 PyTorch BF16 推理，同时获得原生 vLLM 的速度与服务能力。

**当前状态（2026-10-28）：能通过原生 vLLM 生成和分块输出音频，但项目尚未完成。
客户端持续追加文本的真正流式输入尚未打通，完整 BF16 codec 一致性与性能也尚未验收。**

## 最终要实现什么

1. **纯插件接入**：安装后由 vLLM / vLLM-Omni 的插件入口自动注册；模型计算位于
   `src/lm_tts_vllm/`，不要求用户启动本仓库专用 client 或 worker。
2. **正确的推理协议**：支持 step-56384 和新的 step-10000；后者严格遵循训练的
   5 个文本 token / 10 帧音频交错布局，正确处理参考音频、文本 EOS、音频 EOS 和长度限制。
3. **真正的文本与音频双向流式体验**：同一个请求保留 KV 状态，文本分段到达；
   全部文本到达之前即可产出并播放音频，缺少后续文本时等待，明确结束输入后再收尾。
   正常追加每段文本不重启服务、不重新加载模型、不拆成多个独立 TTS 请求拼接。
4. **与实际 BF16 基线对齐**：使用原 demo / UltraEval 的实际模型、输入、采样和解码协议。
   分别比较同前缀 hidden/logits、greedy codec、EOS 和长度。不能用“能出声音”代替精度验收。
5. **精度和速度都要**：前期允许关闭优化定位问题，最终须验证 KV cache、CUDA graph、
   批处理与采样配置的正确性及端到端收益。正式原生 attention 使用 FlashAttention，不以 SDPA 冒充 vLLM 加速。
6. **可试听的双模式 demo**：PyTorch / vLLM 可切换，支持真实采样；流式输入页面显示
   已接收文本、已提交给模型的文本以及等待/生成/输入结束状态。

当前不要求 CER/WER 或说话人评分。CustomVoice / VoiceDesign 需要对应权重，不能仅靠 Base 权重声明支持。

## 当前做到什么程度

下表描述 `main` 的能力。历史 step-100 的验证不能自动算作 step-10000 已通过。

| 项目 | 当前状态 | 边界 |
| --- | --- | --- |
| 外部模型注册、标准 Omni CLI / HTTP | 已实现并有功能验证 | 完整 TTS pipeline 依赖固定提交的 vLLM-Omni |
| 原生 Talker + code predictor + Code2Wav | 已跑通 | 不再通过对齐工具逐帧 IPC 驱动正式 demo |
| BF16、Talker / CP FlashAttention、KV cache | 已接入 | 不代表与 PyTorch 完整数值一致 |
| Talker / CP greedy 与随机采样 | 已验证功能 | CP greedy 有独立 CUDA graph；CP 采样当前用 eager |
| step-10000 的 5×10 模型内部布局 | 已实现并验证布局与音频输出 | 输入仍然是一次提交全文 |
| vLLM 分块音频输出 | 已验证 | Gradio / 浏览器有转码和播放缓冲 |
| PyTorch demo 基线 | 可用 | 当前上线版整段生成后返回 |
| 同一请求持续追加文本 | **未完成** | 开发原型首轮端到端测试被上游能力检查拒绝 |
| 输入文本进度指示、新的输入方式 UI | **未上线** | 仅在未完成原型中；当前页面仍有旧文本布局复选框 |
| BF16 完整 codec / 长序列一致性 | **未验收** | 当前只有局部数值对照，不能宣称精度已对齐 |
| 高吞吐、并发、端到端加速收益 | **未验收** | 当前 demo 单请求配置，测试 GPU 存在其他负载 |

参考音频＋参考文本的 ICL 克隆已可用。仅说话人向量克隆保留在后端，页面已去掉该选项。
更详细的功能记录见 [兼容矩阵](docs/qwen-compatibility.md)，其中旧模型用例均须按记录的 checkpoint 解读。

## 三种容易混淆的“流式”

| 含义 | 当前上线版是否支持 |
| --- | --- |
| 模型内部按 5×10 交错消费预先给定的全文 | 是，step-10000 |
| 音频每生成一块就传给播放器 | 是，vLLM；不是 PyTorch 基线 |
| 客户端后续才发送新文本，并继续同一个推理会话 | **否** |

页面的 `non_streaming_mode` 旧复选框只改变文本条件布局，不代表客户端可以追加文本。
“实时播放”表示分块音频输出，不保证零缓冲，也不表示已达到低延迟性能目标。
首块音频返回时间、浏览器开始播放时间、整段生成耗时是三个不同指标。

## 真正流式输入：原型与明确阻塞

未完成代码保存在 `feat/qwen-tts-compat` 分支（`32e2864`），**不合并进当前可用的 main**。
原型包含文本分块缓冲、会话接口、输入进度 UI，以及通过标准
`AsyncOmni.generate(StreamingInput)` 续传同一个请求的尝试。

首轮端到端测试未通过，上游返回：

```text
streaming (resumable) input is not supported:
stage 1 runs on model_runner v2, which supports turn-based requests only;
use model_runner: v1 for that stage
```

当前部署两个阶段都使用 runner v2。这说明现有原型配置不可用；不能只隐藏异常，
也不能直接宣称改成 v1 就完成了。必须确认解码阶段的 runner / connector 兼容性，
然后验证跨文本段的 KV、参考音频、codec 历史和终止状态都能正确延续。

当前运行进程没有加载这套原型，也没有追加文本接口。完成实现后需要一次部署重启；
部署成功后的日常分段提交不应再要求重启。现有线上服务保持可用，开发验证应独立进行。

## 已验证的证据与仍然缺失的证据

- step-10000 的 23,850 组前缀布局与 LM-TTS-Training 协议一致。
- 长文本样例有 35 个额外文本位置、115 个调度步骤，输出
  `10 + 25 + 25 + 19 = 79` 帧音频，即 24 kHz 下 151,680 samples / 6.32 秒。
  这验证布局、分块和截断，不验证跨引擎 codec 一致性。
- 共享参考 codec / speaker 条件的一个 BF16 样例，修正 BOS 文本投影批次后，
  prefill embedding 的 `104 × 1024` 元素完全一致。
  首步 hidden 最大绝对误差 `0.09375`，logits 最大绝对误差 `0.09375`，
  两边原始 argmax 均为 `892`。这只是一个前缀的结果，不是整条生成序列通过。
- 在线验证覆盖 interleaved / 非流式布局、ICL / 仅音色、greedy / 采样，
  并验证了固定用例的同 seed 重复输出和逐块 PCM 拼接等于完整 WAV。
- Gradio 上传、分块转码与完整音频下载已有验证；新默认参考音频已通过 vLLM
  采样冒烟，返回两块、总计 1.84 秒音频（24 帧上限用例，不是自然结束或速度基准）。
- **尚缺**：完整序列 teacher-forced 对照、完整 codec 分歧分析、真实文本续传成功样例、
  浏览器双向流式体验、多请求生命周期回归、统一配置的性能报告。

本地验证结果不随 Git 发布。部分早期开发结果归档在
`results/archive/feat-qwen-tts-compat/`；线上记录在 `results/demo-service/`。
数值证据主要来自 `interleaved-10000-shared-reference-v2`，默认参考冒烟记录为
`results/demo-service/default-reference-smoke.json`。旧结果里的参考音频可能不同于当前默认值。

## 模型与默认参考

| 模型 | 当前用途 | 源路径（工作区相对路径） |
| --- | --- | --- |
| step-10000 | 当前默认，5×10 interleaved | `../UltraEval-Audio/init_model/emilia-v4-emilia2-cbinit-langid-interleaved5x10-step-00010000` |
| step-56384 | 保留的旧模型 | `../UltraEval-Audio/init_model/emilia-v4-emilia2-cbinit-langid-step-00056384` |
| step-100 | 仅历史验证 | 已从 demo 模型列表替换掉 |

原始权重保持只读，转换后的原生模型视图在 `artifacts/native-10000` / `artifacts/native-56384`。
默认参考音频已移动至 [`assets/reference/vo_anecdote_part1_4_zhongli_04.wav`](assets/reference/vo_anecdote_part1_4_zhongli_04.wav)。
默认文本：

> 早听说枫丹的土壤水质都与璃月大不相同，不知道这枫丹种出来的茶叶，究竟是什么味道。

step-10000 的 PyTorch 基线使用 `LM-TTS-Training` 的 `CodecSynthesizer`，
与 `UltraEval-Audio/scripts/local_qwen.py` 调用的实现一致，BF16 + SDPA。
step-56384 使用原 Qwen-TTS demo 路径。参考音频编码前处理的差异需单独核查，
共享参考条件的 Talker 对照不能代替整个前处理链路的对齐。

## 运行现有 demo

以下命令面向已经准备好权重、依赖及本地环境的当前工作区，不是一键干净环境安装教程。

```bash
# 在仓库根目录执行；按空闲 GPU 调整
PYTORCH_GPU=5 VLLM_GPU=6 bash scripts/run_demo.sh

# 页面启动不代表模型已就绪，必须检查后端
curl --noproxy '*' http://127.0.0.1:7862/health
curl --noproxy '*' http://127.0.0.1:7863/health
```

页面为 `http://<服务器地址>:7861`，后端仅监听本机 7862 / 7863。
健康接口应返回对应模型和 `ready: true`；冷启动可能需要数分钟。
启动脚本会同时管理三个进程，任一退出会清理其他进程，因此重启页面目前也会影响后端。
日志和 PID 位于 `results/demo-service/`，合成音频与请求参数位于 `results/demo/`。

当前环境：

- 原生环境：`/tmp/lm-tts-omni/bin/python`，vLLM `0.32.0`、Transformers `5.19`。
- vLLM-Omni 固定提交：`ee37ec0620a99b3787dab5e5e3bd1f5cdfd0615f`。
  启动时仍有版本号不匹配警告，当前组合有功能验证，不代表所有上游功能兼容。
- 基线环境：`../envs/lm-tts/bin/python`，Torch `2.8.0+cu126`、Transformers `4.57.3`、Qwen-TTS `0.1.1`。
- 可通过 `TTS_PYTHON` / `OMNI_PYTHON` 覆盖环境路径。
  Gradio 分块转码要求 `ffmpeg` 和 `ffprobe`；启动器支持 `artifacts/ffmpeg/bin/` 或 `FFMPEG_BIN`。

详见 [demo 使用说明](docs/demo.md) 和 [原生服务入口](docs/native-serving.md)。

## 目录与开发状态

| 路径 | 职责 |
| --- | --- |
| `src/lm_tts_vllm/` | 纯插件：注册、模型适配、原生调度接口及部署配置 |
| `tools/lm_tts_tools/` | 模型准备、PyTorch 基线、对齐诊断、demo；不进入插件 wheel |
| `scripts/`、`tests/`、`docs/` | 启动辅助、回归检查与文档 |
| `assets/reference/` | demo 默认参考音频 |
| `artifacts/`、`results/`、`vendor-reference/` | 本地权重视图、结果与上游参考源码，不随 Git 发布 |

`main` 保存可用实现和此状态文档。未完成流式输入原型单独保留在
`feat/qwen-tts-compat`，不是交付功能。独立的 `LM-TTS-vLLM-next` 只是 Git worktree，
不是另一个运行依赖；该工作目录现已删除，验证结果已归档，原型分支已推送保留。
未通过的原型没有合并进 main。

## 接下来按什么顺序做

1. **打通真正文本续传**：解决 stage 1 runner 的兼容限制；验证文本不足时等待、
   新段到达后复用同一 KV 会话、结束输入只发送一次文本 EOS、参考帧只注入一次。
2. **以严格用例验收输入流**：第二段文本故意等第一段音频输出后才发送；
   服务不能预先看到全文。记录每次文本到达、模型接收、首音频和结束时间。
   验证空段、短尾段、取消、超时、长度上限和连续新会话；明确分段 tokenizer 边界语义。
3. **上线 UI**：将旧复选框替换成“整段输入 / 流式输入”，提供发送本段、结束输入、取消，
   显示已提交文本和等待状态。模拟输入与真实手动追加都须通过，不用全量预填伪装流式。
4. **完成 BF16 验收**：先隔离参考前处理差异，再比较同前缀 hidden/logits，
   对完整 greedy codec 记录首个分歧与 EOS；单独检查采样行为，不要求不同引擎同 seed 必然逐 token 一致。
5. **完成速度验收**：在相同正确性配置下逐项启用优化，测预热后首音频延迟、RTF、
   p50/p95、并发与显存。不能用冷启动、短截断用例或有其他 GPU 负载的单次数据宣称加速倍数。
6. **再合并交付**：通过上述相关回归后合并原型、清理临时文件与分支，并更新能力矩阵。
   暂未完成的代码应明确保留为开发状态，不为了清理分支而合进 main。

## 进一步阅读

- [5×10 布局与验证](docs/interleaved.md)
- [原生 pipeline 接入边界](docs/native-pipeline.md)
- [精度和速度验收要求](docs/performance.md)
- [数值对齐与历史复现命令](docs/alignment-tools.md)
- [历史验证记录](docs/validation.md)

历史工具中的 pooling、逐帧 IPC 和 SDPA 替换只用于诊断，不能当作当前原生服务或最终性能方案。
