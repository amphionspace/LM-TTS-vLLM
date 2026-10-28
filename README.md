# LM-TTS-vLLM

LM-TTS（Emilia Qwen3-TTS Base checkpoint）的 vLLM / vLLM-Omni 外部插件。
安装后由插件入口自动注册，不需要修改 vLLM。

当前优先级：**先能用**（整段合成 + 流式文本输入 + 试听 demo），
其次 BF16 对齐，最后性能。

## 现在能做什么

| 能力 | 状态 |
| --- | --- |
| step-10000（5×10 interleaved）整段合成，分块返回音频 | 可用 |
| step-10000 **流式文本输入**：同一请求分段追加文本、保留 KV，边输入边出音频 | 可用（vLLM 与 PyTorch 两个后端） |
| step-56384（Qwen overlay）整段合成 | 可用；不支持流式文本输入 |
| ICL（参考音频＋文本）/ 仅 x-vector 克隆，Talker / CP 采样，请求级 seed | 可用 |
| 标准 `vllm serve --omni` 与 `/v1/audio/speech` | 整段可用 |
| 与 PyTorch BF16 基线 codec 一致 | 未验收 |
| 性能（Talker CUDA graph、prefix cache、并发） | 未做，当前是 eager 单请求配置 |

### 流式文本输入怎么工作

使用标准 `AsyncOmni.generate(StreamingInput 生成器)` 续传同一请求：

1. 第一段需要覆盖参考帧对应的文本：`(参考帧数 // 10 + 1) × 5` 个 token（参考文本＋目标文本）。
   之后每凑够 5 个 token 送一段，模型生成 10 帧后停下等待下一段。
2. 结束输入时追加文本 EOS，最后一段生成到音频 EOS 或长度上限。
3. 分段缓冲在 `src/lm_tts_vllm/input_stream.py`，续传时的 prefill 在 `omni_model.py` / `interleaved.py`。
4. 部署配置 `native_input_streaming.yaml`：stage 0（Talker）用 MRv2，
   **stage 1（Code2Wav）必须是 `model_runner: v1`**，因为上游的续传不支持 MRv2 下游。

注意，首块音频要等文本覆盖完参考帧之后才能出来。默认参考音频是 8.3 秒（104 帧、27 个参考文本 token），
所以需要先收到 28 个目标文本 token。参考音频越短，首块越早。这是训练布局决定的，不是实现上的限制。

流式输入和整段输入在同一个引擎里不会 bit 级相同：续传的 5 个文本 token 走 prefill kernel，
整段输入是逐个 decode。`tools/lm_tts_tools/stream_parity.py` 用来做这项对照。

## Demo

```bash
PYTORCH_GPU=5 VLLM_GPU=6 bash scripts/run_demo.sh   # 页面 :7861，后端 127.0.0.1:7862 / 7863
curl --noproxy '*' http://127.0.0.1:7863/health     # ready: true 才算加载完
```

页面可选 PyTorch / vLLM 后端、整段 / 流式输入。流式模式下可以手动逐段发送，
也可以用"模拟流式输入"一键体验。说明见 [docs/demo.md](docs/demo.md)。

命令行验证流式会话（模拟 LLM 每 0.3 秒送 2 个字）：

```bash
PYTHONPATH=tools $TTS_PYTHON -m lm_tts_tools.input_session_smoke --port 7863
```

## 环境

- 原生环境 `/tmp/lm-tts-omni/bin/python`：vLLM 0.32.0（`/tmp/vllm-lm-tts-shared`），
  vLLM-Omni `ee37ec06` 以 editable 方式安装自 **`vendor-reference/vllm-omni`**，这个目录不能删。
  两者都在 `/tmp` 下，机器重启后需要重建。
- 基线环境 `../envs/lm-tts/bin/python`：Torch 2.8.0、Transformers 4.57.3、qwen-tts 0.1.1；
  step-10000 基线调用 `../LM-TTS-Training` 的 `CodecSynthesizer`。
- 共享机器上需要 `VLLM_PLUGINS=lm_tts,vllm_omni_register_models`（`run_demo.sh` 已设置）。

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/lm_tts_vllm/` | 插件：注册（`omni.py`）、Talker 适配（`omni_model.py`）、5×10 布局（`interleaved.py`）、流式输入（`input_stream.py`）、CP（`code_predictor.py`）、部署 YAML |
| `tools/lm_tts_tools/` | `prepare_native`（生成模型视图）、`training_reference`（PyTorch 基线导出）、各类 smoke、`demo/` |
| `tests/` | `PYTHONPATH=src:tools python -m unittest discover -s tests` |
| `docs/` | [5×10 布局](docs/interleaved.md)、[demo](docs/demo.md)、[标准服务](docs/native-serving.md)、[功能矩阵](docs/qwen-compatibility.md) |
| `artifacts/` | `native-10000` / `native-56384` 模型视图（config＋权重软链），ffmpeg |

旧的 pooling 对齐路径和 step-100 相关内容已删除，需要时可以从 git 历史（`6154819` 之前）找回。
