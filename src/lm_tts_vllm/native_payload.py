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


def reference_stream_frame(output, request, is_finished=False):
    """Apply the same final-token convention to incremental codec delivery."""
    ids = list(getattr(request, "output_token_ids", None) or [])
    finished = is_finished or request.is_finished()
    if isinstance(output, dict) and finished and ids and ids[-1] != 2150:
        output = dict(output)
        output["meta"] = dict(output.get("meta") or {}, codec_frame_valid=False)
    return output


def talker2code2wav_async_chunk(transfer_manager, multimodal_output, request, is_finished=False):
    from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
        talker2code2wav_async_chunk as upstream,
    )
    return upstream(transfer_manager, reference_stream_frame(multimodal_output, request, is_finished),
                    request, is_finished)


def talker2code2wav_async_chunk_batch(transfer_manager, pooling_outputs, requests, is_finished):
    from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
        talker2code2wav_async_chunk_batch as upstream,
    )
    if not len(pooling_outputs) == len(requests) == len(is_finished):
        raise ValueError("Streaming batch metadata lengths differ")
    outputs = [reference_stream_frame(output, request, finished)
               for output, request, finished in zip(pooling_outputs, requests, is_finished)]
    return upstream(transfer_manager, outputs, requests, is_finished)
