"""Preserve the reference frame order across the native stage connector."""
import torch


def reference_frames(audio, output_token_ids):
    if not isinstance(audio, torch.Tensor) or audio.ndim != 2 or not audio.numel():
        return audio
    valid = (audio >= 0).all(1) & audio.any(1) & (audio.max(1).values < 2048)
    audio = audio[valid]
    # The PyTorch baseline returns the residual frames preceding the final
    # sampled token. MRv2 eager MTP also computes that final frame at a cap.
    # Keep the prefix, not the tail: cropping the tail drops the first frame.
    count = max(len(output_token_ids) - 1, 0)
    return audio[:count]


def talker2code2wav_full_payload(transfer_manager, pooling_output, request):
    from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
        talker2code2wav_full_payload as upstream,
    )
    if isinstance(pooling_output, dict):
        audio = pooling_output.get("codes.audio")
        if audio is None and isinstance(pooling_output.get("codes"), dict):
            audio = pooling_output["codes"].get("audio")
        if isinstance(audio, torch.Tensor):
            pooling_output = dict(pooling_output)
            pooling_output["codes.audio"] = reference_frames(
                audio, list(getattr(request, "output_token_ids", None) or []))
    return upstream(transfer_manager, pooling_output, request)
