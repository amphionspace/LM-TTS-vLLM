"""Run UltraEval's original non-streaming evaluator with only its Talker replaced.

Pass existing voice_clone_infer.py arguments after `--`. The original script
retains manifests, seeds, conditioning, sampling, waveform trimming and metrics.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

from .client import TalkerWorker
from .infer import backbone_override


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ultraeval-root", type=Path, required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--vllm-python", required=True)
    parser.add_argument("--worker-log", type=Path, required=True)
    parser.add_argument("--precision-profile", choices=["bf16", "native", "sdpa"], default="bf16")
    parser.add_argument("--engine-mode", choices=["reference", "cached"], default="reference")
    parser.add_argument("--attention-backend", choices=["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"], default="FLASH_ATTN")
    parser.add_argument("evaluation_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.precision_profile == "sdpa" and args.engine_mode != "reference":
        parser.error("The SDPA diagnostic requires --engine-mode reference")
    forwarded = args.evaluation_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    # Reject modes requiring a different input protocol instead of silently
    # running the baseline evaluator without the plugin.
    check = argparse.ArgumentParser(add_help=False)
    check.add_argument("--non-streaming", action="store_true")
    check.add_argument("--batch-size", type=int, default=1)
    check.add_argument("--model-path", required=True)
    check.add_argument("--conditioning", required=True)
    selected, _ = check.parse_known_args(forwarded)
    if not selected.non_streaming or selected.batch_size != 1 or selected.conditioning != "speaker_icl":
        parser.error("This adapter requires --non-streaming --batch-size 1 --conditioning speaker_icl")
    source = json.loads((Path(args.backbone) / "source.json").read_text())["model"]
    chosen = Path(selected.model_path)
    if not chosen.is_absolute():
        chosen = args.ultraeval_root / chosen
    if chosen.resolve() != Path(source).resolve():
        parser.error("--backbone does not belong to the evaluation model")
    from qwen_tts import Qwen3TTSModel
    original = Qwen3TTSModel.generate_voice_clone
    args.worker_log.parent.mkdir(parents=True, exist_ok=True)
    if args.worker_log.exists():
        raise FileExistsError(args.worker_log)
    with TalkerWorker(args.vllm_python, args.backbone, args.worker_log, args.precision_profile,
                      engine_mode=args.engine_mode, attention_backend=args.attention_backend) as worker:
        def generate(self, *positional, **kwargs):
            with backbone_override(self, worker):
                return original(self, *positional, **kwargs)
        Qwen3TTSModel.generate_voice_clone = generate
        args.worker_log.with_suffix(".plugin.json").write_text(json.dumps({
            "worker": worker.info, "precision_profile": args.precision_profile,
            "engine_mode": args.engine_mode,
            "backbone": str(Path(args.backbone).resolve()), "evaluation_args": forwarded,
            "implementation": "lm-tts-vllm", "layout": "non_streaming"}, indent=2) + "\n")
        script = args.ultraeval_root / "scripts/voice_clone_infer.py"
        old_argv = sys.argv
        try:
            sys.argv = [str(script), *forwarded]
            sys.path.insert(0, str(script.parent.resolve()))
            runpy.run_path(str(script), run_name="__main__")
        finally:
            sys.argv = old_argv
            Qwen3TTSModel.generate_voice_clone = original


if __name__ == "__main__":
    main()
