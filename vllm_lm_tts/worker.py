"""Private local-pipe worker; runs in the isolated vLLM environment."""
import argparse
import json
from multiprocessing.connection import Connection
from pathlib import Path
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()
    connection = Connection(args.fd)
    try:
        import torch
        import vllm
        from vllm import LLM
        from vllm.config import PoolerConfig
        from . import register
        register()
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        # v0.32's pooling scoring processor requires an initialized tokenizer,
        # even though this worker only submits already constructed embeddings.
        source = json.loads((Path(args.model) / "source.json").read_text())["model"]
        llm = LLM(model=args.model, tokenizer=source, runner="pooling", dtype="bfloat16",
                  enable_prompt_embeds=True,
                  pooler_config=PoolerConfig(task="embed", seq_pooling_type="LAST", use_activation=False),
                  enforce_eager=True, enable_prefix_caching=False,
                  max_model_len=args.max_model_len, max_num_seqs=1,
                  max_num_batched_tokens=args.max_model_len,
                  kv_cache_memory_bytes=512 * 1024**2,
                  gpu_memory_utilization=0.15,
                  attention_config={"backend": "FLASH_ATTN"})
        connection.send({"ready": True, "vllm": vllm.__version__, "torch": torch.__version__,
                         "gpu": torch.cuda.get_device_name()})
        while True:
            value = connection.recv()
            if value is None:
                break
            embeds = torch.from_numpy(value).to(torch.bfloat16)
            output = llm.encode({"prompt_embeds": embeds}, pooling_task="embed", use_tqdm=False)[0]
            connection.send({"hidden": output.outputs.data.detach().float().cpu().numpy()})
    except BaseException:
        connection.send({"error": traceback.format_exc()})
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
