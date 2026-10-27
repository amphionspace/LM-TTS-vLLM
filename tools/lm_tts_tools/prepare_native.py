"""Create an HF config view for the optional native pipeline; weights stay read-only."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    raw = json.loads((source / "config.json").read_text())
    if raw["talker_config"].get("lm_tts_input_protocol") != "qwen3_non_streaming":
        raise ValueError("Expected the supported LM-TTS non-streaming checkpoint")
    if not (source / "speech_tokenizer/config.json").is_file():
        raise ValueError("Speech tokenizer is missing")
    output.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.name != "config.json":
            (output / path.name).symlink_to(path, target_is_directory=path.is_dir())
    raw["model_type"] = "lm_tts"
    raw["architectures"] = ["LMTTSForConditionalGeneration"]
    (output / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
    (output / "lm_tts_source.json").write_text(json.dumps({
        "model": str(source), "config_sha256": hashlib.sha256((source / "config.json").read_bytes()).hexdigest(),
        "layout": "non_streaming", "weights": "symlinks_to_original"}, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(prepare(args.model, args.output))


if __name__ == "__main__":
    main()
