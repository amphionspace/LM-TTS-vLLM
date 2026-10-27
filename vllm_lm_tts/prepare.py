"""Extract a standard Qwen3 backbone without changing the original TTS checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(source, output):
    from safetensors import safe_open
    from safetensors.torch import save_file

    source, output = Path(source).resolve(), Path(output).resolve()
    raw = json.loads((source / "config.json").read_text())
    cfg = raw["talker_config"]
    if cfg["lm_tts_input_protocol"] != "qwen3_non_streaming":
        raise ValueError("Expected Qwen3-TTS protocol")
    weights = {}
    for shard in sorted(source.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith("talker.model.layers.") or key == "talker.model.norm.weight":
                    weights[key.removeprefix("talker.")] = f.get_tensor(key)
                elif key == "talker.model.codec_embedding.weight":
                    weights["model.embed_tokens.weight"] = f.get_tensor(key)
    expected = 2 + 11 * cfg["num_hidden_layers"]
    if len(weights) != expected:
        raise ValueError(f"Unexpected backbone weights: {len(weights)} != {expected}")
    keys = ["hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "hidden_act", "rms_norm_eps", "vocab_size",
            "max_position_embeddings", "attention_bias"]
    config = {key: cfg[key] for key in keys}
    # TTS assigns identical positions to all three MRoPE axes. In that case its
    # interleaved MRoPE is exactly the ordinary 1D (NeoX-style) rotary transform.
    config.update(model_type="qwen3", architectures=["LMTTSTalkerModel"],
                  rope_theta=cfg["rope_theta"],
                  rope_parameters={"rope_type": "default", "rope_theta": cfg["rope_theta"]},
                  tie_word_embeddings=False, is_causal=True, dtype="bfloat16",
                  layer_types=["full_attention"] * cfg["num_hidden_layers"])
    output.mkdir(parents=True, exist_ok=False)
    save_file(weights, str(output / "model.safetensors"))
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "source.json").write_text(json.dumps({"model": str(source),
        "config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        "tensor_count": len(weights)}, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(prepare(args.model, args.output))
