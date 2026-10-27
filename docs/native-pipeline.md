# 原生 TTS 接入（开发中）

可选 `.[omni]` 依赖固定 vLLM 0.32.0 与 vLLM-Omni commit
`ee37ec0620a99b3787dab5e5e3bd1f5cdfd0615f`。该提交的 requirements 支持
vLLM 0.32 / Torch 2.13 / Transformers 5.16.1–5.19；浅克隆安装生成的
`0.1.dev1` 版本号会触发上游版本提示，不能用这个生成版本号判断源码兼容性。
请使用独立环境，不升级原 PyTorch 基线环境。

`lm_tts_vllm.omni` 通过 `vllm_omni.general_plugins` 注册模型、HF config
与原生两阶段 pipeline。模型在标准 Omni worker 内运行；请求调度、KV cache
与音频输出由 Omni 管理，不使用本仓库开发工具的 client/worker。
Talker 使用 vLLM FlashAttention，残差码本使用本插件的 FA2 / CUDA graph。
目前仅支持 BF16、TP/PP=1、batch=1、greedy、显式 non_streaming_mode=True。
部署配置关闭 Talker graph、prefix cache 和 async scheduling，属于验证配置。

状态：两个 checkpoint 都已通过标准 Omni API 生成音频及 codec，但完整
codec 一致性未通过，不能视为已完成的 serving 接口。

| checkpoint | 原生 / PyTorch 帧数 | 音频 | 首个 codec 分歧（帧、码本，从 0 起） | 本地产物 |
| --- | --- | --- | --- | --- |
| 56384 | 79 / 79 | 24 kHz / 6.32 秒 | (1, 3)；第 0 帧全部 16 码本一致 | `results/native-omni-56384-v6` |
| 100 | 2 / 2 | 24 kHz / 0.16 秒 | (0, 13)；32 个值中 4 个不同 | `results/native-omni-100` |

两组分别使用与 `56384-greedy`、`100-native-cp-graph` 原始 PyTorch 基线相同
的请求，均 greedy、max_new_tokens=80。产物含 WAV、codec NPY、运行报告和
`codec-comparison.json`。原生单次请求分别为 8.67 / 5.16 秒，包含未预热的
前后处理，且运行在不同的共享 GPU 上；不作为性能结论。

首次对照发现首码本序列错位：上游 full-payload processor 在达到长度上限时用
`audio[-(len(output_token_ids)-1):]` 裁剪，丢失第一帧。插件新增 stage processor
保留前缀，仍由上游负责参考 codec 拼接和传输。完整复跑确认首帧恢复，修正后
第 1–78 帧与修正前第 0–77 帧完全相同，排除了用随机生成变化解释该修复。
原基线环境 9 项单测通过（可选 Omni 测试跳过）；Omni 环境的配置 roundtrip / 阶段覆盖
回归测试通过。主 wheel 已确认包含运行代码及 YAML，不包含 `lm_tts_tools`。
当前上游 SDK 在加载 general plugins 前解析 pipeline，直接构造 `Omni` 会因
未注册的 `lm_tts` model type 失败。冒烟工具先调用官方
`load_omni_general_plugins()`，再使用 `Omni(model=...)`；没有直接调用本插件
的注册函数。完全自动发现的 CLI 启动仍须单独验证。

准备工具只创建 config 副本和权重符号链接：

```bash
PYTHONPATH=tools python -m lm_tts_tools.prepare_native \
  --model /path/to/original-hf-checkpoint --output artifacts/native-model
```

`lm_tts_tools.native_smoke` 是标准 SDK 的测试调用者，输出 WAV、codec 数组和
运行报告。其 `--prompt-length` 暂时使用基线 trace 的准确 prefill 长度，因此
还不是面向任意文本的最终用户示例。后续需接入标准长度估计，继续定位两模型
原生链路的同前缀数值差异，再验证服务入口与性能配置。

本次 step-56384 的运行方式（独立 Omni 环境已安装主插件）：

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=src:tools python -m lm_tts_tools.native_smoke \
  --model artifacts/native-56384 \
  --ref-audio ../UltraEval-Audio/raw_data/voice_clone_manifests/seed_tts_eval_zh/000000.wav \
  --ref-text '在此奉劝大家别乱打美白针。' \
  --text '你好，这是语音合成测试。' --prompt-length 88 \
  --output results/native-new-run
```

当前 parent CP 权重仍用于原生 prompt / embedding 构建，本插件 CP 另外加载
自己的权重；有重复显存占用，后续需消除。完整 codec 一致性门槛不会因原生
启动成功而放宽。
