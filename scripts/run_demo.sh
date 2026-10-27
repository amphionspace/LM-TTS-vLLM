#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:$PWD/tools${PYTHONPATH:+:$PYTHONPATH}"
export GRADIO_ANALYTICS_ENABLED=False
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset all_proxy ALL_PROXY http_proxy HTTP_PROXY https_proxy HTTPS_PROXY
TTS_PYTHON=${TTS_PYTHON:-/workspace/workspace/yanglin/envs/lm-tts/bin/python}
OMNI_PYTHON=${OMNI_PYTHON:-/tmp/lm-tts-omni/bin/python}
mkdir -p results/demo-service
cleanup() { kill "$torch_pid" "$native_pid" "$ui_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
CUDA_VISIBLE_DEVICES=${PYTORCH_GPU:-0} "$TTS_PYTHON" -u -m lm_tts_tools.demo.backend --mode pytorch --port 7862 > results/demo-service/pytorch.log 2>&1 &
torch_pid=$!
CUDA_VISIBLE_DEVICES=${VLLM_GPU:-1} "$OMNI_PYTHON" -u -m lm_tts_tools.demo.backend --mode vllm --port 7863 > results/demo-service/vllm.log 2>&1 &
native_pid=$!
"$TTS_PYTHON" -u -m lm_tts_tools.demo.app --port "${PORT:-7861}" > results/demo-service/gradio.log 2>&1 &
ui_pid=$!
printf '%s\n' "$torch_pid" "$native_pid" "$ui_pid" > results/demo-service/pids
wait -n "$torch_pid" "$native_pid" "$ui_pid"
