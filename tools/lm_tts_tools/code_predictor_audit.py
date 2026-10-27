"""Export original BF16 CP cases, then validate native KV cache and CUDA graphs."""
import argparse
import json
from pathlib import Path
import time

import torch


def timed(call):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start


@torch.inference_mode()
def reference(args):
    import numpy as np
    from qwen_tts import Qwen3TTSModel
    trace = torch.load(args.reference_run / "trace.pt", map_location="cpu", weights_only=True)
    codes = np.load(args.reference_run / "pytorch-codes.npy")
    saved = json.loads((args.reference_run / "report.json").read_text())
    if Path(saved["arguments"]["model"]).resolve() != Path(args.model).resolve():
        raise ValueError("Trace does not belong to this checkpoint")
    if any(i < 0 or i >= len(codes) for i in args.frames):
        raise ValueError("Requested frame is outside the saved generated codes")
    model = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0",
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    talker, cases = model.model.talker, []
    for index in args.frames:
        token = torch.tensor([[int(codes[index, 0])]], device="cuda")
        inputs = torch.cat([trace[index]["hidden"].cuda(), talker.get_input_embeddings()(token)], 1)
        def generate():
            return talker.code_predictor.generate(
                inputs_embeds=inputs, max_new_tokens=talker.config.num_code_groups - 1,
                do_sample=False, return_dict_in_generate=True, output_scores=True)
        generate()  # Warm up; excluded from measurements.
        seconds = []
        for _ in range(args.repeats):
            output, elapsed = timed(generate)
            seconds.append(elapsed)
        cases.append({"frame": index, "inputs": inputs.cpu(), "codes": output.sequences.cpu(),
                      "logits": torch.stack(output.scores, 1).cpu(), "seconds": seconds})
    with args.output.open("xb") as stream:
        torch.save({"model": str(Path(args.model).resolve()), "torch": str(torch.__version__),
                    "dtype": "bfloat16", "cases": cases}, stream)


@torch.inference_mode()
def native(args):
    from lm_tts_vllm.code_predictor import CodePredictor
    pack = torch.load(args.input, map_location="cpu", weights_only=True)
    predictor = CodePredictor(pack["model"])
    rows, first_graph = [], None
    for case in pack["cases"]:
        inputs = case["inputs"].cuda()
        (codes, scores), eager_time = timed(lambda: predictor.run(inputs, use_graph=False))
        forced, forced_scores = predictor.eager(inputs, case["codes"][0].cuda())
        graph_codes, graph_scores = predictor.run(inputs, use_graph=True)
        if first_graph is None:
            first_graph = graph_codes.clone(), graph_scores.clone()
        times = [timed(lambda: predictor.run(inputs, use_graph=True))[1] for _ in range(args.repeats)]
        different = (codes.cpu() != case["codes"]).nonzero()
        rows.append({"frame": case["frame"], "identical_codes": torch.equal(codes.cpu(), case["codes"]),
                     "first_different_codebook": int(different[0, 1]) + 1 if len(different) else None,
                     "teacher_forced_argmax_agreement": float((forced.cpu() == case["codes"]).float().mean()),
                     "max_logit_error": float((forced_scores.cpu() - case["logits"]).abs().max()),
                     "finite": bool(torch.isfinite(scores).all()),
                     "graph_matches_eager": torch.equal(codes, graph_codes) and torch.equal(scores, graph_scores),
                     "native_eager_seconds": eager_time, "graph_seconds": times,
                     "reference_seconds": case["seconds"]})
    again = predictor.run(pack["cases"][0]["inputs"].cuda(), use_graph=True)
    isolated = all(torch.equal(a, b) for a, b in zip(again, first_graph))
    report = {"model": pack["model"], "torch": str(torch.__version__), "reference_torch": pack["torch"],
              "backend": "vllm_flash_attention_2", "dtype": "bfloat16", "kv_cache": "frame_local",
              "graph_cache_reuse": isolated, "cases": rows}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "identical_cases": sum(r["identical_codes"] for r in rows),
                      "graph_cache_reuse": isolated, "max_logit_error": max(r["max_logit_error"] for r in rows)}))
    if not isolated or not all(r["finite"] and r["graph_matches_eager"] for r in rows):
        raise SystemExit("Native graph/cache validation failed; inspect the report")
    if args.require_identical_codes and not all(r["identical_codes"] for r in rows):
        raise SystemExit("Original PyTorch codec parity failed; inspect the report")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    ref = modes.add_parser("reference")
    ref.add_argument("--model", required=True)
    ref.add_argument("--reference-run", type=Path, required=True)
    ref.add_argument("--frames", type=int, nargs="+", default=[0])
    engine = modes.add_parser("native")
    engine.add_argument("--input", type=Path, required=True)
    engine.add_argument("--require-identical-codes", action="store_true")
    for mode in [ref, engine]:
        mode.add_argument("--output", type=Path, required=True)
        mode.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1:
        parser.error("Output must be new and repeats must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    (reference if args.mode == "reference" else native)(args)


if __name__ == "__main__":
    main()
