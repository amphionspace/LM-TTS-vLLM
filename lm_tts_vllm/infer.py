"""Run the exact tts-demo frontend, with an optional vLLM Talker and parity audit."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import time

import numpy as np
import soundfile as sf
import torch
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from .client import TalkerWorker


class LengthCache(DynamicCache):
    def __init__(self, length):
        super().__init__()
        self.length = length

    def get_seq_length(self, layer_idx=0):
        return self.length


@contextmanager
def backbone_override(model, worker=None, trace=None):
    backbone = model.model.talker.model
    original = backbone.forward
    chunks = []

    def forward(*args, **kwargs):
        embeds = kwargs["inputs_embeds"]
        if embeds.shape[0] != 1 or not kwargs["attention_mask"].bool().all():
            raise ValueError("Only single, unpadded demo requests are supported")
        positions = kwargs["position_ids"]
        if positions.ndim != 3 or not torch.equal(positions[0], positions[1]) or not torch.equal(positions[0], positions[2]):
            raise ValueError("Only equal-axis audio/text MRoPE positions are supported")
        if worker is None:
            output = original(*args, **kwargs)
            if trace is not None:
                trace.append({"embeds": embeds.detach().cpu(),
                              "hidden": output.last_hidden_state[:, -1:].detach().cpu()})
            return output
        chunks.append(embeds)
        prefix = torch.cat(chunks, dim=1)
        hidden = worker.hidden(prefix)
        return BaseModelOutputWithPast(last_hidden_state=hidden,
                                       past_key_values=LengthCache(prefix.shape[1]),
                                       hidden_states=(hidden,))

    backbone.forward = forward
    try:
        yield
    finally:
        backbone.forward = original


def tensor_error(reference, actual):
    ref, out = reference.float(), actual.float()
    diff = out - ref
    return {"max_abs": diff.abs().max().item(), "mean_abs": diff.abs().mean().item(),
            "relative_l2": (diff.norm() / ref.norm().clamp_min(1e-12)).item(),
            "cosine": torch.nn.functional.cosine_similarity(ref.flatten(), out.flatten(), dim=0).item()}


@torch.inference_mode()
def audit(model, worker, trace):
    chunks, rows = [], []
    head = model.model.talker.codec_head
    for step, sample in enumerate(trace):
        chunks.append(sample["embeds"].to(device=head.weight.device))
        prefix = torch.cat(chunks, 1)
        actual = worker.hidden(prefix)
        expected = sample["hidden"].to(actual)
        a, b = head(expected).float(), head(actual).float()
        # Match official Qwen suppression, excluding special tokens except EOS.
        eos = model.model.config.talker_config.codec_eos_token_id
        valid = list(range(a.shape[-1] - 1024)) + [eos]
        av, bv = a[..., valid], b[..., valid]
        top = av.topk(2).values.flatten()
        rows.append({"step": step, "prefix_tokens": prefix.shape[1],
                     "hidden": tensor_error(expected, actual), "logits": tensor_error(a, b),
                     "argmax_equal": av.argmax().item() == bv.argmax().item(),
                     "reference_top2_margin": (top[0] - top[1]).item()})
    return {"steps": rows, "argmax_agreement": sum(x["argmax_equal"] for x in rows) / len(rows),
            "max_logit_abs_error": max(x["logits"]["max_abs"] for x in rows)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backbone")
    parser.add_argument("--vllm-python")
    parser.add_argument("--mode", choices=["pytorch", "vllm", "compare"], default="compare")
    parser.add_argument("--precision-profile", choices=["bf16", "native", "sdpa"], default="bf16")
    parser.add_argument("--engine-mode", choices=["reference", "cached"], default="reference")
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--code-predictor-greedy", action="store_true")
    parser.add_argument("--reference-run", help="Reuse an identical request's saved PyTorch trace and audio")
    parser.add_argument("--require-identical-codes", action="store_true",
                        help="Fail the comparison unless all greedy codec frames and lengths match")
    parser.add_argument("--max-logit-error", type=float,
                        help="Fail if teacher-forced maximum absolute logit error exceeds this value")
    args = parser.parse_args()
    if args.precision_profile == "sdpa" and args.engine_mode != "reference":
        parser.error("The SDPA diagnostic requires --engine-mode reference")
    if args.require_identical_codes and (args.mode != "compare" or not args.greedy):
        parser.error("--require-identical-codes requires --mode compare --greedy")
    if args.max_logit_error is not None and (args.mode != "compare" or not 0 <= args.max_logit_error < float("inf")):
        parser.error("--max-logit-error requires compare mode and a finite nonnegative tolerance")
    if args.mode != "pytorch" and not (args.backbone and args.vllm_python):
        parser.error("vLLM requires --backbone and --vllm-python")
    if args.mode != "pytorch":
        source = json.loads((Path(args.backbone) / "source.json").read_text())
        if Path(source["model"]).resolve() != Path(args.model).resolve():
            parser.error("--backbone does not belong to --model; prepare the matching checkpoint")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    from qwen_tts import Qwen3TTSModel
    import transformers
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    model = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0", dtype=torch.bfloat16,
                                        attn_implementation="sdpa")
    prompt = model.create_voice_clone_prompt(ref_audio=args.reference_audio,
                                             ref_text=args.reference_text.strip(), x_vector_only_mode=False)
    settings = dict(text=[args.text.strip()], language=[args.language], voice_clone_prompt=prompt,
                    non_streaming_mode=True, x_vector_only_mode=False,
                    max_new_tokens=args.max_new_tokens, do_sample=not args.greedy,
                    temperature=args.temperature, top_k=args.top_k, top_p=1.0,
                    repetition_penalty=args.repetition_penalty,
                    subtalker_dosample=not (args.greedy or args.code_predictor_greedy),
                    subtalker_top_k=50, subtalker_top_p=1.0, subtalker_temperature=0.9)
    report = {"arguments": vars(args), "torch": torch.__version__, "transformers": transformers.__version__,
              "dtype": "bfloat16", "attention": "sdpa", "layout": "non_streaming", "runs": {}}
    waves, codes = {}, {}
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()

    def generate(label, worker=None, trace=None):
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        collected = []
        first_frame_seconds = None
        def collect(module, inputs, output):
            nonlocal first_frame_seconds
            if output.hidden_states[-1] is not None:
                collected.append(output.hidden_states[-1].detach().cpu())
                if first_frame_seconds is None:
                    first_frame_seconds = time.monotonic() - start
        hook = model.model.talker.register_forward_hook(collect)
        original_generate = model.model.talker.generate
        finish = {}
        def capture_generate(*positional, **kwargs):
            result = original_generate(*positional, **kwargs)
            finish["last_token"] = result.sequences[0, -1].item()
            return result
        model.model.talker.generate = capture_generate
        start = time.monotonic()
        first_worker_step = len(worker.steps) if worker else 0
        try:
            with backbone_override(model, worker, trace):
                wavs, rate = model.generate_voice_clone(**settings)
        finally:
            hook.remove()
            model.model.talker.generate = original_generate
        wav = np.asarray(wavs[0], dtype=np.float32)
        if not len(wav) or not np.isfinite(wav).all():
            raise RuntimeError("Generated waveform is empty or non-finite")
        sf.write(out / f"{label}.wav", wav, rate, subtype="PCM_16")
        frame_codes = torch.cat(collected, 0).numpy()
        np.save(out / f"{label}-codes.npy", frame_codes)
        waves[label], codes[label] = sf.read(out / f"{label}.wav", dtype="float32")[0], frame_codes
        report["runs"][label] = {"seconds": time.monotonic() - start,
                                  "audio_seconds": len(wav) / rate, "sample_rate": rate,
                                  "codec_frames": len(frame_codes),
                                  "ended_with_eos": finish["last_token"] == model.model.config.talker_config.codec_eos_token_id,
                                  "reached_token_limit": finish["last_token"] != model.model.config.talker_config.codec_eos_token_id}
        report["runs"][label]["rtf"] = report["runs"][label]["seconds"] / (len(wav) / rate)
        report["runs"][label]["first_codec_frame_seconds"] = first_frame_seconds
        if worker:
            steps = worker.steps[first_worker_step:]
            report["runs"][label]["talker_engine_seconds"] = sum(s["seconds"] for s in steps)
            report["runs"][label]["submitted_tokens"] = sum(s["submitted_tokens"] for s in steps)
            report["runs"][label]["cached_tokens"] = sum(s["cached_tokens"] or 0 for s in steps)
        (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    trace = []
    if args.reference_run:
        if args.mode != "compare":
            parser.error("--reference-run requires compare mode")
        previous = Path(args.reference_run)
        saved = json.loads((previous / "report.json").read_text())
        for key in ["model", "reference_audio", "reference_text", "text", "language", "max_new_tokens",
                    "seed", "greedy", "temperature", "top_k", "repetition_penalty", "code_predictor_greedy"]:
            if saved["arguments"][key] != getattr(args, key):
                raise ValueError(f"Reference request differs at {key}")
        trace = torch.load(previous / "trace.pt", map_location="cpu", weights_only=True)
        report["runs"]["pytorch"] = saved["runs"]["pytorch"]
        report["reference_run"] = str(previous.resolve())
        codes["pytorch"] = np.load(previous / "pytorch-codes.npy")
        waves["pytorch"] = sf.read(previous / "pytorch.wav", dtype="float32")[0]
        for name in ["trace.pt", "pytorch.wav", "pytorch-codes.npy"]:
            shutil.copy2(previous / name, out / name)
    elif args.mode in {"pytorch", "compare"}:
        generate("pytorch", trace=trace)
        if trace:
            torch.save(trace, out / "trace.pt")
    if args.mode in {"vllm", "compare"}:
        with TalkerWorker(args.vllm_python, args.backbone, out / "vllm.log", args.precision_profile,
                          engine_mode=args.engine_mode) as worker:
            report["worker"] = worker.info
            if trace:
                report["teacher_forced"] = audit(model, worker, trace)
                (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
                worker.reset_cache()
            generate("vllm", worker=worker)
    if len(codes) == 2:
        a, b = codes["pytorch"], codes["vllm"]
        shared = min(len(a), len(b))
        different = np.flatnonzero(np.any(a[:shared] != b[:shared], axis=-1))
        report["free_running"] = {"identical_codes": np.array_equal(a, b),
                                  "shared_frames": shared,
                                  "token_agreement_on_shared_frames": float((a[:shared] == b[:shared]).mean()),
                                  "first_different_frame": int(different[0]) if len(different) else None,
                                  "identical_waveform": np.array_equal(waves["pytorch"], waves["vllm"])}
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "teacher_forced"}, indent=2, ensure_ascii=False))
    if args.require_identical_codes and not report["free_running"]["identical_codes"]:
        raise SystemExit("Greedy codec parity failed; inspect report.json")
    if args.max_logit_error is not None and report["teacher_forced"]["max_logit_abs_error"] > args.max_logit_error:
        raise SystemExit("Logit tolerance exceeded; inspect report.json")


if __name__ == "__main__":
    main()
