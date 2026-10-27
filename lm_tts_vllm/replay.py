"""Isolate full-prefix versus incremental decoding in the original BF16 runtime."""
import argparse
import json
from pathlib import Path

import torch

from .infer import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists")
    saved = json.loads((args.reference_run / "report.json").read_text())
    from qwen_tts import Qwen3TTSModel
    import transformers
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Qwen3TTSModel.from_pretrained(
        saved["arguments"]["model"], device_map="cuda:0",
        dtype=torch.bfloat16, attn_implementation="sdpa")
    trace = torch.load(args.reference_run / "trace.pt", map_location="cpu", weights_only=True)

    class FullPrefix:
        def hidden(self, embeddings):
            positions = torch.arange(embeddings.shape[1], device=embeddings.device)
            output = model.model.talker.model(
                inputs_embeds=embeddings,
                attention_mask=torch.ones(embeddings.shape[:2], device=embeddings.device, dtype=torch.long),
                position_ids=positions.reshape(1, 1, -1).expand(3, 1, -1),
                cache_position=positions, use_cache=False)
            return output.last_hidden_state[:, -1:]

    result = audit(model, FullPrefix(), trace)
    result.update(torch=torch.__version__, transformers=transformers.__version__,
                  dtype="bfloat16", attention="sdpa",
                  reference_run=str(args.reference_run.resolve()),
                  comparison="PyTorch full-prefix vs saved incremental trace")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "steps"}, indent=2))


if __name__ == "__main__":
    main()
