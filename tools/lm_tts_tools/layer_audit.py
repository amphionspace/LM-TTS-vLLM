"""Locate the first differing prefill operation using vLLM's real attention backend."""
import argparse
import json
from pathlib import Path

import torch

from .client import TalkerWorker
from .infer import tensor_error
from .layer_trace import install


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--backbone", required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--vllm-python", required=True)
    parser.add_argument("--precision-profile", choices=["bf16", "native"], default="bf16")
    parser.add_argument("--attention-backend", choices=["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"], default="FLASH_ATTN")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    saved = json.loads((args.reference_run / "report.json").read_text())
    source = json.loads((Path(args.backbone) / "source.json").read_text())
    if any(Path(p).resolve() != Path(args.model).resolve()
           for p in [source["model"], saved["arguments"]["model"]]):
        parser.error("Model, backbone and trace must belong to the same checkpoint")
    args.output.mkdir(parents=True, exist_ok=False)
    from qwen_tts import Qwen3TTSModel
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Qwen3TTSModel.from_pretrained(args.model, device_map="cuda:0",
                                        dtype=torch.bfloat16, attn_implementation="sdpa")
    sample = torch.load(args.reference_run / "trace.pt", map_location="cpu", weights_only=True)[0]
    embeddings = sample["embeds"].cuda()
    positions = torch.arange(embeddings.shape[1], device=embeddings.device)
    reference, handles = install(model.model.talker.model, engine=False)
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    original_sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
    layer_ids = {id(layer.self_attn): i for i, layer in enumerate(model.model.talker.model.layers)}
    def record_attention(module, query, key, value, *positional, **kwargs):
        index = layer_ids[id(module)]
        for name, tensor in zip(["query", "key", "value"], [query, key, value]):
            tensor = tensor.transpose(1, 2).reshape(tensor.shape[2], -1)
            reference[f"layers.{index}.attention_{name}"] = (
                tensor if index == 0 else tensor[-1:]).detach().cpu().clone()
        return original_sdpa(module, query, key, value, *positional, **kwargs)
    ALL_ATTENTION_FUNCTIONS.register("sdpa", record_attention)
    try:
        model.model.talker.model(
            inputs_embeds=embeddings, use_cache=False, cache_position=positions,
            attention_mask=torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.long),
            position_ids=positions.reshape(1, 1, -1).expand(3, 1, -1))
    finally:
        ALL_ATTENTION_FUNCTIONS.register("sdpa", original_sdpa)
        for handle in handles:
            handle.remove()
    torch.save(reference, args.output / "pytorch-layers.pt")
    with TalkerWorker(args.vllm_python, args.backbone, args.output / "vllm.log",
                      args.precision_profile, attention_backend=args.attention_backend) as worker:
        worker.connection.send({"trace_embeds": embeddings[0].float().cpu().numpy()})
        response = worker.receive()
        actual = {k: torch.from_numpy(v).bfloat16() for k, v in response["layers"].items()}
        info = worker.info
    torch.save(actual, args.output / "vllm-layers.pt")
    if reference.keys() != actual.keys():
        raise RuntimeError("Incomplete or mismatched layer hooks")
    rows = []
    for name, expected in reference.items():
        value = actual[name]
        if expected.shape != value.shape:
            raise RuntimeError(f"Shape mismatch at {name}: {expected.shape} != {value.shape}")
        rows.append({"name": name, "exact": torch.equal(expected, value),
                     "error": tensor_error(expected, value)})
    report = {"worker": info, "profile": args.precision_profile,
              "reference_torch": torch.__version__, "scope": "first full prefill",
              "first_difference": next((r["name"] for r in rows if not r["exact"]), None),
              "layers": rows}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "layers"}, indent=2))


if __name__ == "__main__":
    main()
