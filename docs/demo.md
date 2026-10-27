# 双模式 Gradio 试听

页面：`http://10.119.16.21:7861`。原 `tts-demo` 服务已停，新的界面与服务代码
全部位于本仓库 `tools/lm_tts_tools/demo/`。不进入纯插件 wheel。

- PyTorch：原 demo 的 Qwen-TTS / BF16 SDPA；使用 `envs/lm-tts` 原基线环境。
- vLLM：标准 Omni SDK 加载本插件，Talker 与 CP attention 均为 FA2；使用独立
  Omni 环境。CP greedy 保留 CUDA graph，CP sampling 暂用 eager FA2。
- 支持 56384 / step-100、上传参考音频、语言、两个原 demo 预设、greedy 对齐预设，
  以及 Talker / CP 独立 temperature、top-k、top-p、重复惩罚和种子。
- 开发分支新增仅 speaker embedding 克隆（参考文本可空），以及
  `non_streaming_mode` 文本条件切换。关闭该选项不代表音频分块返回。
  已构建并检查新版 UI schema；在线服务仍使用主分支版本。
- 默认每次最多 512 帧，开发版可调至 2048（在线旧版为 1024）；参考音频最长 30 秒。两个后端各驻留一个模型，
  切换 checkpoint 会重新加载，vLLM 首次加载可能需要数分钟。
- 两种引擎的 RNG 算法及运算不同，相同 seed 不保证跨引擎生成相同 codec。
  当前完整 BF16 codec 对齐尚未通过；本页用于试听，不执行质量评分。

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
`results/demo/`；vLLM 另存 codec NPY。可通过后端 `/health` 查看已加载模型。

CP 训练侧选择 SDPA 是为了避开大 frame batch 下 FA2 deterministic backward
的 workspace 失败；这里仅执行 forward，因此不受该训练限制。采样过滤行为
已与原 Transformers warpers 对照测试。推理内核差异仍须单独做数值验证。
