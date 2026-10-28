# 双模式 Gradio 试听

启动后的页面：`http://10.119.16.21:7861`。原 `tts-demo` 服务已停，新的界面与服务代码
全部位于本仓库 `tools/lm_tts_tools/demo/`。不进入纯插件 wheel。

- PyTorch：step-56384 使用原 demo Qwen-TTS；step-10000 使用 UltraEval 相同的
  LM-TTS CodecSynthesizer。均为 BF16 SDPA，使用 `envs/lm-tts` 原基线环境。
- vLLM：标准 Omni SDK 加载本插件，Talker 与 CP attention 均为 FA2；使用独立
  Omni 环境。CP greedy 保留 CUDA graph，CP sampling 暂用 eager FA2。
- 模型列表为 step-56384 / step-10000；step-10000 替换原 step-100。
  新模型来源为 `UltraEval-Audio/init_model/emilia-v4-emilia2-cbinit-langid-interleaved5x10-step-00010000`，
  原生模型视图为 `artifacts/native-10000`。协议实现见 [interleaved.md](interleaved.md)。
  原 step-100 的验证结果不能作为新模型的验证结果。
- 支持上传参考音频、语言、两个原 demo 预设、greedy 对齐预设，
  以及 Talker / CP 独立 temperature、top-k、top-p、重复惩罚和种子。
- 页面统一使用参考音频＋参考文本；仅 speaker embedding 克隆保留在后端 API，页面不提供此选项。
  支持 `non_streaming_mode` 文本条件切换。
  step-10000 关闭此选项使用训练的 5×10 布局；step-56384 使用 Qwen overlay。
- vLLM 音频通过 streaming Audio 组件边生成边播放，结束后提供完整 WAV。
  PyTorch 基线目前整段返回。两种模式都显示首块返回时间（不含模型加载）。
- 默认每次最多 512 帧，可调至 2048；参考音频最长 30 秒。两个后端各驻留一个模型，
  切换 checkpoint 会重新加载，vLLM 首次加载可能需要数分钟。
- 两种引擎的 RNG 算法及运算不同，相同 seed 不保证跨引擎生成相同 codec。
  当前完整 BF16 codec 对齐尚未通过；本页用于试听，不执行质量评分。

浏览器流式播放需要系统 PATH 中的 `ffmpeg` 和 `ffprobe`。也可以将两者放入
`artifacts/ffmpeg/bin/`，或用 `FFMPEG_BIN` 指定目录；启动脚本会在加载 GPU 前检查。

启动（前台管理三个服务，任一服务退出时清理其余服务）：

```bash
PYTORCH_GPU=0 VLLM_GPU=1 bash scripts/run_demo.sh
```

长期运行可在仓库根目录使用独立 tmux 会话：

```bash
tmux new-session -d -s lm-tts-demo 'bash scripts/run_demo.sh'
```

可用 `TTS_PYTHON`、`OMNI_PYTHON` 覆盖两个环境的 Python 路径，`PORT` 覆盖页面端口。
默认 PyTorch / vLLM 后端仅监听 `127.0.0.1:7862` / `127.0.0.1:7863`；这层 HTTP
用于保持旧 Transformers 4 与新 Transformers 5 环境隔离，插件本身不依赖它。
Gradio 页面监听 `0.0.0.0:7861`。

日志与 PID：`results/demo-service/`。每次合成保存 WAV 和请求参数 JSON 到
`results/demo/`，并保留逐块 WAV。可通过后端 `/health` 查看已加载模型。

CP 训练侧选择 SDPA 是为了避开大 frame batch 下 FA2 deterministic backward
的 workspace 失败；这里仅执行 forward，因此不受该训练限制。采样过滤行为
已与原 Transformers warpers 对照测试。推理内核差异仍须单独做数值验证。

2026-10-28 在线验证：step-10000 的 ICL/仅音色 × interleaved/非流式四种组合
均返回有效音频；另外验证了 Talker/CP 同时采样及相同 seed 的重复输出。
长文本 streaming 的逐块 WAV 拼接与完整 WAV 一致，Gradio 两种模式的上传、
AAC 转码和完整音频下载通过。当前持久会话为 `lm-tts-demo`，使用 GPU5/6。
