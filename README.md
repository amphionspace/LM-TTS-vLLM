# LM-TTS-vLLM

针对本地 Emilia Qwen3-TTS checkpoint 的 **vLLM 0.32.0 外部模型插件**，以
`tts-demo/app.py` 和 `UltraEval-Audio/scripts/voice_clone_infer.py` 的
**BF16 + PyTorch SDPA** 推理为数值基准。

本实现将 Talker backbone 放入 vLLM；文本投影、speaker encoder、16-codebook
帧的组合、Talker 采样和 waveform 解码沿用原来的 `qwen-tts`。
code predictor 默认沿用原实现作为对照，也可选择下面的原生 FlashAttention / CUDA graph 路径。
通过进程隔离保留原 demo 的 Transformers 4.57.3，同时使用 vLLM 所需的
Transformers 5.x。插件通过 `vllm.general_plugins` 注册 `LMTTSTalkerModel`，
无需修改 vLLM。

## 包边界与最终启动方式

- `src/lm_tts_vllm/`：纯 vLLM 插件，只包含注册、模型和计算实现。主 wheel 不包含工具包，
  不导入 Qwen-TTS 前端，不创建自定义 client/worker，也不包含 SDPA 替换诊断。
- `tools/lm_tts_tools/`：独立开发工具包，包含转换、基线、对齐、UltraEval 适配以及环境隔离进程。
  `tools/pyproject.toml` 可以单独安装；运行插件不要求安装它。
- `tests/`、`docs/`、`scripts/`：验证、记录与开发辅助，不属于插件的运行依赖。

最终启动约定是安装插件后由 vLLM 自动加载 `vllm.general_plugins` entry point，
用户不需要另行启动本仓库的 client 或 worker。当前已注册的 `LMTTSTalkerModel`
仍是用于对齐的 Talker pooling 模型，**尚不是可用 `vllm serve` 完整生成音频的 TTS 接口**。
后者还需要把完整生成状态与音频输出协议接入原生运行链路，不能把转换后的 backbone
加载成功当成端到端 TTS 服务已完成。

## 当前边界

- 支持两份指定模型的 **non-streaming / speaker+ICL / batch size 1** 推理。
- step-100 的训练是 mixed interleaved 5×10，但也支持 non-streaming；本入口明确
  使用与 demo 相同的 non-streaming 协议，**不将它当成 interleaved streaming 评测**。
- 通过 vLLM causal pooling 获取未归一化的最后一个 hidden state，每步提交完整 embedding
  前缀。`--engine-mode reference` 重算前缀；`--engine-mode cached` 使用 vLLM 基于
  embedding 内容哈希的 prefix cache，复用完整的 16-token KV blocks，只计算未命中的尾部。
  两者都仍有逐帧 IPC，尚不是完整的高吞吐 serving 实现。
- 当前 TP/PP 固定为 1，Talker 关闭 CUDA graph 和量化，code predictor 可独立启用 graph。
  这是验证阶段的设置，完整实现必须同时
  验收精度和速度，见 [性能与精度验收](docs/performance.md)。
- `bf16` profile 保留原 PyTorch 的 RMSNorm、残差、RoPE、SiLU 乘法的舍入边界，并分别
  计算 Q/K/V 和 gate/up 投影；attention 仍使用 vLLM FlashAttention backend。
  `native` profile 保留 vLLM 原生融合算子，可用于误差对照。
- 不承诺 BF16 不同 attention 内核逐 bit 相同。先用相同前缀做 teacher-forced 比较，
  再检查 free-running codec 的首个分歧位置，避免把误差累积误认为权重/协议错误。

## 环境

推荐两个独立环境，不要升级正在使用的 demo 环境。

```bash
# vLLM 环境：使用支持 vLLM 0.32.0 的 Python/CUDA 环境
python3.12 -m venv .venv-vllm
.venv-vllm/bin/pip install -e '.[engine]'

# 前后处理：使用原 demo / evaluation 环境
export TTS_PYTHON=../envs/lm-tts/bin/python
export VLLM_PYTHON="$PWD/.venv-vllm/bin/python"
export PYTHONPATH="$PWD/tools:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

前后处理环境使用 `torch==2.8.0+cu126`、`transformers==4.57.3`、`qwen-tts==0.1.1`。
本工作区的 `envs/lm-tts` 与 `UltraEval-Audio/envs/tts` 这三项版本相同，且
Qwen-TTS modeling / inference 源文件的 SHA256 相同。

## 准备两个模型

在本仓库目录执行。原始模型保持只读，生成文件位于被 Git 忽略的 `artifacts/`。
训练 checkpoint 使用原仓库的导出器，校验 assembly identity，导出 FP32 权重以免
在导出时额外舍入；实际加载和推理仍统一为 BF16。

```bash
MODEL1="$(realpath ../UltraEval-Audio/init_model/emilia-v4-emilia2-cbinit-langid-step-00056384)"
bash scripts/export_step100.sh
MODEL2="$PWD/artifacts/step-00000100"

$TTS_PYTHON -m lm_tts_tools.prepare --model "$MODEL1" --output artifacts/backbone-56384
$TTS_PYTHON -m lm_tts_tools.prepare --model "$MODEL2" --output artifacts/backbone-100
```

`prepare` 提取 Talker 的 layers、norm、codec embedding，保留原权重 dtype；通过
`source.json` 记录来源。音频/文本的三个 MRoPE 位置轴相同，因此可以映射为普通
一维 RoPE；前端会检查此条件，拒绝其他位置布局。

## 推理与 BF16 对齐

```bash
CUDA_VISIBLE_DEVICES=6 $TTS_PYTHON -m lm_tts_tools.infer \
  --model "$MODEL1" --backbone artifacts/backbone-56384 \
  --vllm-python "$VLLM_PYTHON" --mode compare \
  --reference-audio ../UltraEval-Audio/raw_data/voice_clone_manifests/seed_tts_eval_zh/000000.wav \
  --reference-text '在此奉劝大家别乱打美白针。' \
  --text '简单地说，这相当于惠普把消费领域市场拱手相让了。' \
  --language Auto --greedy --max-new-tokens 256 \
  --output results/56384-greedy
```

第二个模型将 `--model` 换成 `$MODEL2`、`--backbone` 换成 `artifacts/backbone-100`。
输出目录必须是新目录。

- `--mode pytorch`：原 demo 基线、codec tokens、逐步输入/hidden trace。
- `--mode vllm`：通过插件生成音频。
- `--mode compare`：原 PyTorch 基线 → 同前缀 vLLM 数值比较 → vLLM 自由生成。
- `--greedy`：Talker 和 code predictor 都取 greedy，先排除采样器差异。
- 去掉 `--greedy`：使用 UltraEval 默认采样参数（两个阶段均 T=0.9、top-k=50；
  Talker repetition penalty=1.05）。相同随机种子不保证不同 logits 采到同一 token。
- demo 的低温 preset：`--temperature 0.3 --top-k 10 --repetition-penalty 2.0 --code-predictor-greedy`。
- `--precision-profile native`：对照 vLLM 原始融合算子。
- `--attention-backend FLASH_ATTN|FLASHINFER|TRITON_ATTN`：选择 vLLM 原生 attention backend，
  默认 `FLASH_ATTN`。可用性由当前 GPU 和 vLLM 校验；报告记录实际 backend、实现类和 FA 版本。
- **仅用于诊断** `--precision-profile sdpa --engine-mode reference`：在相同 vLLM 权重和 BF16 算术下
  使用 PyTorch SDPA，定位 attention 内核差异。它仅支持单请求完整前缀，是诊断模式，
  不能用来宣称 paged attention 或 CUDA graph 性能。
- `--engine-mode cached`：验证 KV cache 复用；teacher-forced 完成后先清空缓存，再测自由生成，
  避免参考 trace 预先填满缓存而美化速度结果。
- `--reference-run results/已有基线`：重放已保存的相同请求，避免重复基线推理。
- 回归门槛：`--require-identical-codes --greedy` 要求完整 codec 序列及长度一致；
  `--max-logit-error <容差>` 检查 teacher-forced 最大绝对误差。超出门槛时保留报告并返回非零退出码。

`report.json` 包含运行版本、逐步 hidden/logits 误差、有效 codec 范围内的原始 argmax
一致率、top-2 margin、自由生成的 token 一致率和首个不同帧。
argmax 比较尚未施加 repetition penalty，不等同于最终采样决策。
`reached_token_limit=true` 表示音频达到生成上限，不应当作自然 EOS 完成的样本。
`*-codes.npy` 是生成 codebook 序列，`*.wav` 是 24 kHz 音频，`trace.pt` 用于重放。
新运行同时记录 RTF、首个 codec frame 延迟、Talker engine 耗时和 prefix cache 命中量。
首个 codec frame 延迟不等于首个可播放音频的延迟；当前入口在完整解码后才输出 WAV。

## 沿用 UltraEval 的实际入口

step-56384 现有评测记录为 `speaker_icl`、`non_streaming=true`、`language_auto=true`、
`batch_size=1`。以下适配器直接执行原 `voice_clone_infer.py`，保留原 manifest、seed、
sampling、解码与 reference 裁剪，只在每次请求期间替换 Talker backbone：

```bash
CUDA_VISIBLE_DEVICES=6 $TTS_PYTHON -m lm_tts_tools.ultraeval \
  --ultraeval-root ../UltraEval-Audio \
  --backbone artifacts/backbone-56384 --vllm-python "$VLLM_PYTHON" \
  --worker-log "$PWD/results/eval-56384-vllm.log" -- \
  --model emilia-56384-vllm --model-path "$MODEL1" \
  --conditioning speaker_icl --non-streaming --language-auto --batch-size 1 \
  --manifest-dir raw_data/voice_clone_manifests \
  --run-dir "$PWD/results/ultraeval-56384" --limit 1
```

适配器另外写出 `*.plugin.json` 标识 vLLM 环境与插件参数；原评测配置中的 `qwen_tts`
仍描述前后处理入口。请使用全新的评测目录，不与纯 PyTorch 结果混用。
交错 streaming 的原评测路径为 `local_interleaved.InterleavedVoiceClone` →
`lm_tts.inference.codec.CodecSynthesizer`；当前适配器会拒绝该模式。

## 测试

```bash
PYTHONPATH=src:tools $TTS_PYTHON -m unittest discover -s tests -v
```

测试使用安装的 Qwen-TTS 实现作为 oracle，检查 BF16 RMSNorm/残差、MRoPE 和独立投影。
端到端 GPU 结果见本地 `results/*/report.json`，测量摘要见 [验证记录](docs/validation.md)。
权重、环境、trace 和音频不提交到 Git。

若需区分前缀重算与增量 KV 解码带来的误差，可在原 PyTorch 环境重放同一 trace：

```bash
CUDA_VISIBLE_DEVICES=6 $TTS_PYTHON -m lm_tts_tools.replay \
  --reference-run results/56384-greedy --output results/pytorch-prefix-audit.json
```

此对照不使用 vLLM，也不运行 ASR 或说话人评分。

接口参考：[vLLM 0.32.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)、
[plugin system](https://docs.vllm.ai/en/latest/design/plugin_system/)。

## 逐层定位原生 vLLM 的误差

正式计算路径使用 vLLM `FLASH_ATTN` backend；SDPA 替换只作为历史诊断对照，
不用于速度验收。以下工具始终保留 vLLM attention，比较同一首步 prefill 的
RMSNorm、Q/K/V 投影、Q/K norm、attention 输出、MLP 和逐层残差输出：

```bash
CUDA_VISIBLE_DEVICES=6 $TTS_PYTHON -m lm_tts_tools.layer_audit \
  --model "$MODEL1" --backbone artifacts/backbone-56384 \
  --reference-run results/56384-greedy --vllm-python "$VLLM_PYTHON" \
  --output results/56384-layer-audit
```

第一层保存所有 token，之后保存最后一个 token；`report.json` 给出首个不同的
操作和各处误差，两个 `*-layers.pt` 保留对照张量。诊断 hook 只在该工具发起
的单次 prefill 期间安装，不进入正常推理路径，也不用于测量速度。

## 原生 code predictor（当前只支持 greedy）

`--code-predictor-engine vllm-eager` / `vllm-graph` 使用插件内的 code predictor，
在 vLLM 环境执行；它调用 vLLM FlashAttention 2，使用 BF16 与帧内增量 KV cache，
不调用 SDPA、不依赖 Transformers。`vllm-graph` 把全部 15 个残差码本步骤捕获为
一个 CUDA graph。需同时指定 `--greedy` 或 `--code-predictor-greedy`，随机采样尚未实现，
工具会明确拒绝不支持的组合，不会悄悄回退到 PyTorch。

目前仍是单请求对齐路径：Talker 与 code predictor 通过工具层调用，额外 IPC 尚未消除，
也未接入原生 serving 调度；CUDA graph 加速的范围仅是 code predictor。其独立检查：

```bash
CUDA_VISIBLE_DEVICES=6 $TTS_PYTHON -m lm_tts_tools.code_predictor_audit reference \
  --model "$MODEL1" --reference-run results/56384-greedy --frames 0 1 2 7 \
  --output results/cp-reference.pt
CUDA_VISIBLE_DEVICES=6 $VLLM_PYTHON -m lm_tts_tools.code_predictor_audit native \
  --input results/cp-reference.pt --output results/cp-native.json
```

工具验证原 PyTorch logits/码本、原生 eager 与 graph 的逐元素一致性，以及帧间缓存重用。
`--require-identical-codes` 可额外要求原 PyTorch 全部码本一致，否则报告后返回失败。
独立的耗时是 CP 微基准，不等于端到端加速倍数。
