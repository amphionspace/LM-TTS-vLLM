"""A 2x2 ablation of Talker and code-predictor numerical differences.

Every case fixes the original Talker prefix and first codebook. Logit comparisons
also force the same original residual-codebook prefix at every predictor step.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

from .client import TalkerWorker
from .infer import tensor_error


def decisions(reference, actual):
    reference = reference.float().reshape(-1, reference.shape[-1])
    actual = actual.float().reshape_as(reference)
    if not torch.isfinite(reference).all() or not torch.isfinite(actual).all():
        raise ValueError("Non-finite predictor logits")
    rows = []
    for index, (a, b) in enumerate(zip(reference, actual)):
        ra, rb = a.argmax().item(), b.argmax().item()
        top = a.topk(2).values
        actual_top = b.topk(2).values
        rows.append({"codebook": index + 1, "reference_token": ra, "actual_token": rb,
                     "argmax_equal": ra == rb, "reference_top2_margin": (top[0] - top[1]).item(),
                     "actual_top2_margin": (actual_top[0] - actual_top[1]).item(),
                     "flip_is_reference_tie": ra != rb and a[ra].item() == a[rb].item(),
                     "flip_creates_tie": ra != rb and b[ra].item() == b[rb].item(),
                     "reference_pair": a[[ra, rb]].tolist(), "actual_pair": b[[ra, rb]].tolist(),
                     "max_abs_error": (a - b).abs().max().item()})
    return {"error": tensor_error(reference, actual), "steps": rows,
            "argmax_agreement": sum(r["argmax_equal"] for r in rows) / len(rows),
            "tie_flips": sum(r["flip_is_reference_tie"] for r in rows)}


class ForcePrefix(LogitsProcessor):
    def __init__(self, codes):
        self.codes, self.raw = codes, []

    def __call__(self, input_ids, scores):
        index = len(self.raw)
        self.raw.append(scores.detach().clone())
        forced = torch.full_like(scores, -float("inf"))
        forced[:, self.codes[index]] = 0
        return forced


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--vllm-python", required=True)
    parser.add_argument("--frames", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = json.loads((args.reference_run / "report.json").read_text())
    source = json.loads((Path(args.backbone) / "source.json").read_text())
    if any(Path(p).resolve() != Path(args.model).resolve()
           for p in [saved["arguments"]["model"], source["model"]]):
        parser.error("Trace, backbone and model must belong to the same checkpoint")
    if not saved["arguments"]["greedy"]:
        parser.error("This ablation requires a greedy PyTorch reference run")
    trace = torch.load(args.reference_run / "trace.pt", map_location="cpu", weights_only=True)
    codes = np.load(args.reference_run / "pytorch-codes.npy")
    if any(i < 0 or i >= len(codes) for i in args.frames):
        parser.error("Selected frame is outside the reference output")
    args.output.mkdir(parents=True, exist_ok=False)
    from qwen_tts import Qwen3TTSModel
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0",
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    talker = model.model.talker
    def reference_cp(inputs, forced=None):
        processor = ForcePrefix(forced) if forced is not None else None
        output = talker.code_predictor.generate(
            inputs_embeds=inputs, max_new_tokens=talker.config.num_code_groups - 1,
            do_sample=False, output_scores=True, return_dict_in_generate=True,
            logits_processor=LogitsProcessorList([processor]) if processor else None)
        scores = torch.stack(processor.raw if processor else output.scores, 1)
        return output.sequences, scores

    report = {"model": str(Path(args.model).resolve()), "reference_run": str(args.reference_run.resolve()),
              "conditioning": "same original Talker prefix, first codebook and residual prefixes",
              "frames": []}
    tensors = {}
    with TalkerWorker(args.vllm_python, args.backbone, args.output / "vllm.log",
                      engine_mode="cached", code_predictor_engine="vllm-graph") as worker:
        report["worker"] = worker.info
        chunks = []
        for index, sample in enumerate(trace[:max(args.frames) + 1]):
            chunks.append(sample["embeds"].cuda())
            native_hidden = worker.hidden(torch.cat(chunks, 1))
            if index not in args.frames:
                continue
            original_hidden = sample["hidden"].cuda()
            first = torch.tensor([[int(codes[index, 0])]], device="cuda")
            first_embed = talker.get_input_embeddings()(first)
            original_inputs = torch.cat([original_hidden, first_embed], 1)
            native_inputs = torch.cat([native_hidden, first_embed], 1)
            reference_codes, reference_logits = reference_cp(original_inputs)
            if not np.array_equal(reference_codes.cpu().numpy()[0], codes[index, 1:]):
                raise RuntimeError(f"Original CP did not reproduce saved frame {index}")
            perturbed_codes, _ = reference_cp(native_inputs)
            _, perturbed_logits = reference_cp(native_inputs, reference_codes[0])

            def native_cp(inputs, forced=None):
                request = {"predict_codes": inputs.float().cpu().numpy(), "return_logits": True}
                if forced is not None:
                    request["forced_codes"] = forced.cpu().numpy()
                worker.connection.send(request)
                result = worker.receive()
                return torch.from_numpy(result["codes"]).cuda(), torch.from_numpy(result["logits"]).cuda()

            native_codes, _ = native_cp(original_inputs)
            _, native_logits = native_cp(original_inputs, reference_codes[0])
            combined_codes, _ = native_cp(native_inputs)
            _, combined_logits = native_cp(native_inputs, reference_codes[0])
            row = {"frame": index, "first_codebook": int(first.item()),
                   "talker_hidden_error": tensor_error(original_hidden, native_hidden),
                   "codes": {"reference": reference_codes.tolist()[0], "native_cp_only": native_codes.tolist()[0],
                             "native_talker_only": perturbed_codes.tolist()[0], "both_native": combined_codes.tolist()[0]},
                   "comparisons": {
                       "native_cp_only": decisions(reference_logits, native_logits),
                       "native_talker_only": decisions(reference_logits, perturbed_logits),
                       "both_native": decisions(reference_logits, combined_logits),
                       "cp_given_native_hidden": decisions(perturbed_logits, combined_logits)}}
            report["frames"].append(row)
            tensors[index] = {"original_inputs": original_inputs.cpu(), "native_inputs": native_inputs.cpu(),
                              "reference_logits": reference_logits.cpu(), "native_cp_logits": native_logits.cpu(),
                              "native_talker_logits": perturbed_logits.cpu(), "combined_logits": combined_logits.cpu()}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(tensors, args.output / "logits.pt")
    print(json.dumps({"frames": len(report["frames"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
