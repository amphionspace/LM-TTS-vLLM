#!/usr/bin/env bash
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
workspace=$(dirname -- "$repo")
python=${TTS_PYTHON:-"$workspace/envs/lm-tts/bin/python"}
training="$workspace/LM-TTS-Training"
checkpoint="$workspace/LM-TTS-Training-Runs/supervised-tts-20261027-emilia-v4-emilia2-refmask-cbinit-langid-interleaved5x10-bf16-16gpu-acc1-lr3e-4-bblr1e-4-ep3-wsd/checkpoints/step-00000100"
PYTHONPATH="$training" "$python" "$training/scripts/export_checkpoint.py" \
    --checkpoint "$checkpoint" \
    --assembled-model "$training/assets/assembled/qwen3-tts-frozen-conditioning-codebook-codec-lang" \
    --output "$repo/artifacts/step-00000100" --precision fp32
