"""Private local-pipe worker; runs in the isolated vLLM environment."""
import argparse
from multiprocessing.connection import Connection
import os
import json
from pathlib import Path
import traceback
import time


def attention_info(model):
    attention = model.model.layers[0].self_attn.attn
    return {"attention_backend": attention.attn_backend.get_name(),
            "attention_implementation": type(attention.impl).__name__,
            "flash_attention_version": getattr(attention.impl, "vllm_flash_attn_version", None)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--engine-mode", choices=["reference", "cached"], default="reference")
    parser.add_argument("--attention-backend", choices=["FLASH_ATTN", "FLASHINFER", "TRITON_ATTN"], default="FLASH_ATTN")
    parser.add_argument("--code-predictor-engine", choices=["pytorch", "vllm-eager", "vllm-graph"], default="pytorch")
    args = parser.parse_args()
    connection = Connection(args.fd)
    try:
        import torch
        import vllm
        from vllm import LLM
        from vllm.config import PoolerConfig
        precision = os.environ.get("LM_TTS_PRECISION", "bf16")
        if precision == "sdpa":
            if args.engine_mode != "reference":
                raise ValueError("SDPA diagnosis requires full-prefix reference mode")
            os.environ["LM_TTS_PRECISION"] = "bf16"
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        # v0.32's pooling scoring processor requires an initialized tokenizer,
        # even though this worker only submits already constructed embeddings.
        llm = LLM(model=args.model, runner="pooling", dtype="bfloat16",
                  enable_prompt_embeds=True,
                  pooler_config=PoolerConfig(task="embed", seq_pooling_type="LAST", use_activation=False),
                  enforce_eager=True, enable_prefix_caching=args.engine_mode == "cached",
                  block_size=16,
                  max_model_len=args.max_model_len, max_num_seqs=1,
                  max_num_batched_tokens=args.max_model_len,
                  kv_cache_memory_bytes=512 * 1024**2,
                  gpu_memory_utilization=0.04,
                  attention_config={"backend": args.attention_backend})
        if precision == "sdpa":
            from .diagnostic_attention import patch_sdpa
            llm.apply_model(patch_sdpa)
        predictor = None
        if args.code_predictor_engine != "pytorch":
            from lm_tts_vllm.code_predictor import CodePredictor
            source = json.loads((Path(args.model) / "source.json").read_text())["model"]
            predictor = CodePredictor(source)
            if args.code_predictor_engine == "vllm-graph":
                predictor.capture()
        connection.send({"ready": True, "vllm": vllm.__version__, "torch": torch.__version__,
                         "gpu": torch.cuda.get_device_name(), "engine_mode": args.engine_mode,
                         "attention": "pytorch_sdpa_diagnostic" if precision == "sdpa" else "vllm",
                         "code_predictor_engine": args.code_predictor_engine,
                         "code_predictor_attention": "vllm_flash_attention_2" if predictor else "frontend_pytorch_sdpa",
                         **llm.apply_model(attention_info)[0]})
        while True:
            value = connection.recv()
            if value is None:
                break
            if isinstance(value, dict) and value.get("reset_cache"):
                connection.send({"reset": llm.reset_prefix_cache()})
                continue
            if isinstance(value, dict) and "predict_codes" in value:
                if predictor is None:
                    raise ValueError("Native code predictor was not enabled")
                inputs = torch.from_numpy(value["predict_codes"]).to(device="cuda", dtype=torch.bfloat16)
                started = time.perf_counter()
                codes, _ = predictor.run(inputs, use_graph=args.code_predictor_engine == "vllm-graph")
                codes = codes.cpu().numpy()
                connection.send({"codes": codes, "seconds": time.perf_counter() - started})
                continue
            layer_trace = isinstance(value, dict) and "trace_embeds" in value
            if layer_trace:
                from .layer_trace import begin_worker, end_worker
                if args.engine_mode != "reference":
                    raise ValueError("Layer tracing requires reference mode for a full prefill")
                llm.apply_model(begin_worker)
                value = value["trace_embeds"]
            embeds = torch.from_numpy(value).to(torch.bfloat16)
            started = time.perf_counter()
            output = llm.encode({"prompt_embeds": embeds}, pooling_task="embed", use_tqdm=False)[0]
            hidden = output.outputs.data.detach().float().cpu().numpy()
            response = {"hidden": hidden, "seconds": time.perf_counter() - started,
                             "submitted_tokens": len(embeds),
                             "cached_tokens": output.num_cached_tokens}
            if layer_trace:
                response["layers"] = llm.apply_model(end_worker)[0]
            connection.send(response)
    except BaseException:
        connection.send({"error": traceback.format_exc()})
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
