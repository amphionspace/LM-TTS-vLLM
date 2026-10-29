# Demo

页面：`http://10.119.16.21:7861`。代码在 `tools/lm_tts_tools/demo/`，不进入插件 wheel。

```bash
tmux new-session -d -s lmtts-demo 'PYTORCH_GPU=0 VLLM_GPU=5 bash scripts/run_demo.sh'
```

- `web.py`（`0.0.0.0:7861`）提供静态页面，并把 `/api/{pytorch,vllm}/...` 转发到两个后端
  （`127.0.0.1:7862` / `7863`）。两个后端分属 `envs/lm-tts`（Transformers 4）和
  `/tmp/lm-tts-omni`（vLLM-Omni），可用 `TTS_PYTHON`、`OMNI_PYTHON` 覆盖。任一服务退出时其余服务一并退出。
- 页面只选一个推理引擎。模型：step-20000（5×10 interleaved，支持流式输入，
  `artifacts/native-20000`）和 step-56384（Qwen overlay，仅整段合成）。
- vLLM 后端用 `native_input_streaming.yaml`，Talker 用 torch.compile + CUDA graph（`FULL_AND_PIECEWISE`：decode 整图，流式分段的小 prefill 走分段图），
  `stage_init_timeout=1200`、`init_timeout=1800`：机器负载高时启动会超过默认的 300 / 600 秒。
- 默认残差码本（CP）采样，temperature 0.9 / top-k 50。CP greedy 在长文本上会循环、停不下来，只用于对齐测试。
- 日志和 PID 在 `results/demo-service/`，每次合成的 WAV 与参数在 `results/demo/`。
  等 `/api/health` 两个后端都 `ready: true` 后再合成。
