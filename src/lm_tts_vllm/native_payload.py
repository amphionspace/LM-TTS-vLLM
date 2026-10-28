"""Preserve the reference frame order across the native stage connector."""
import torch


def normalize_voice_prompt(info):
    """Restore JSON codec matrices before the upstream tensor-only ICL path."""
    nested = info.get("additional_information")
    if isinstance(nested, dict):
        # Match upstream preprocess: flattened runtime state takes precedence.
        info = {**nested, **{k: v for k, v in info.items() if k != "additional_information"}}
    raw = info.get("voice_clone_prompt")
    prompt = raw[0] if isinstance(raw, list) and raw else raw
    if not isinstance(prompt, dict) or not isinstance(prompt.get("ref_code"), list):
        return info
    codes = torch.as_tensor(prompt["ref_code"], dtype=torch.long)
    if codes.ndim == 3 and codes.shape[0] == 1:
        codes = codes[0]
    if codes.ndim != 2 or codes.shape[1] != 16 or codes.shape[0] == 0:
        raise ValueError("voice_clone_prompt.ref_code must have shape [frames, 16]")
    if (codes < 0).any() or (codes >= 2048).any():
        raise ValueError("Reference codec IDs must be in [0, 2048)")
    prompt = dict(prompt, ref_code=codes)
    return dict(info, voice_clone_prompt=[prompt] if isinstance(raw, list) else prompt)


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
    # Native MRv2 streaming snapshots omit the full history to avoid O(T²)
    # copies and expose just the latest token separately.
    last = ids[-1] if ids else getattr(request, "last_output_token_id", None)
    finished = is_finished or request.is_finished()
    # The native output worker publishes the capped token before the scheduler
    # ACK marks the request finished. Waiting for that ACK is too late: the
    # frame is already in the connector's history (or in a delivered chunk).
    limit = getattr(getattr(request, "sampling_params", None), "max_tokens", None)
    count = getattr(request, "output_token_count", len(ids))
    capped = limit is not None and count >= limit and not getattr(request, "resumable", False)
    if isinstance(output, dict) and (finished or capped) and last is not None and last != 2150:
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
