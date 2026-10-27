"""Exercise the optional plugin through the standard Omni offline API."""
import argparse
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--prompt-length", type=int, required=True,
                        help="Exact prefill length from the reference trace")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from vllm_omni import Omni
    from vllm_omni.plugins import load_omni_general_plugins

    # This upstream revision resolves the pipeline before EngineArgs loads
    # plugins. Use its public discovery hook before constructing the SDK.
    load_omni_general_plugins()
    args.output.mkdir(parents=True, exist_ok=False)
    info = {"task_type": ["Base"], "ref_audio": [str(Path(args.ref_audio).resolve())],
            "ref_text": [args.ref_text], "text": [args.text], "language": [args.language],
            "x_vector_only_mode": [False], "non_streaming_mode": [True],
            "max_new_tokens": [80], "full_utterance_decode": [True]}
    engine = Omni(model=str(Path(args.model).resolve()))
    try:
        start = time.perf_counter()
        outputs = engine.generate([{"prompt_token_ids": [0] * args.prompt_length,
                                    "additional_information": info}])
        reports = []
        for index, output in enumerate(outputs):
            mm = output.outputs[0].multimodal_output
            audio = mm["audio"]
            audio = torch.cat(audio, dim=-1) if isinstance(audio, list) else audio
            sr = mm["sr"]
            sr = sr[-1] if isinstance(sr, list) else sr
            sr = int(sr.item() if hasattr(sr, "item") else sr)
            sf.write(args.output / f"{index}.wav", audio.float().cpu().numpy().flatten(), sr)
            codes = mm.get("codec_frames")
            if codes is None:
                raise RuntimeError("Native output omitted codec_frames required for parity validation")
            if isinstance(codes, list):
                codes = torch.cat(codes, dim=0)
            np.save(args.output / f"{index}-codes.npy", codes.cpu().numpy())
            reports.append({"request_id": output.request_id, "sample_rate": sr,
                            "audio_seconds": audio.numel() / sr,
                            "codec_shape": list(codes.shape)})
        if not reports:
            raise RuntimeError("Native generation returned no outputs")
        report = {"model": args.model, "runtime": "native_vllm_omni",
                  "seconds": time.perf_counter() - start, "outputs": reports}
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
