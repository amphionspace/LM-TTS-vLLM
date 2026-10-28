"""Training-compatible 5-text/10-frame layout and native decode bookkeeping.

Text slots use forced codec-pad tokens so ordinary vLLM scheduling allocates
one KV position per text token. They never enter the codec decoder.
"""
from dataclasses import dataclass


def first(value, default=None):
    return value[0] if isinstance(value, list) and value else (default if value is None else value)


def config_value(config, key, default=None):
    return config.get(key, default) if isinstance(config, dict) else getattr(config, key, default)


def uses_training_layout(config):
    return config_value(config, 'lm_tts_streaming_layout') == 'interleaved'


def layout(streamed, frames, chunk_text=5, chunk_frames=10):
    if streamed < 1 or frames < 0 or min(chunk_text, chunk_frames) < 1:
        raise ValueError('Invalid interleaved dimensions')
    segments, text, frame = [], 0, 0
    while True:
        end = min(text + chunk_text, streamed)
        segments.append(('text', text, end))
        text = end
        done = text == streamed
        take = frames - frame if done else min(chunk_frames, frames - frame)
        if take:
            segments.append(('frames', frame, frame + take))
        frame += take
        if done or take < chunk_frames:
            return segments, text


@dataclass(frozen=True)
class DecodePlan:
    # -1 denotes a sampled audio frame; >=0 indexes text after tts_bos.
    slots: tuple[int, ...]
    text_done: tuple[bool, ...]
    prefill_text: int


def decode_plan(streamed, reference_frames, maximum, chunk_text=5, chunk_frames=10):
    _, text = layout(streamed, reference_frames, chunk_text, chunk_frames)
    prefill_text = text
    slots, done = [], []
    for frame in range(maximum):
        slots.append(-1)
        done.append(text == streamed)
        # Text after the final sampled frame is never needed at a cap.
        if frame + 1 == maximum:
            break
        total_frames = reference_frames + frame + 1
        if text < streamed and total_frames % chunk_frames == 0:
            end = min(text + chunk_text, streamed)
            for index in range(text, end):
                slots.append(index)
                done.append(False)
            text = end
    return DecodePlan(tuple(slots), tuple(done), prefill_text)


def request_layout(info, config, tokenize, estimate_ref_code_len=None):
    """Return exact placeholder length and decode plan without loading weights."""
    prompt = first(info.get('voice_clone_prompt'), {}) or {}
    icl = bool(first(prompt.get('icl_mode'), not first(info.get('x_vector_only_mode'), False)))
    ref_text = (first(info.get('ref_text'), '') or first(prompt.get('ref_text'), '')) if icl else ''
    text = first(info.get('text'), '')
    ids = tokenize(ref_text or '') + tokenize(text)
    streamed = len(ids) + 1  # includes tts_eos
    refs = 0
    if icl:
        codes = prompt.get('ref_code')
        refs = len(codes) if codes is not None else first(info.get('ref_code_length'))
        if refs is None and estimate_ref_code_len is not None:
            refs = estimate_ref_code_len(info.get('ref_audio'))
        if refs is None:
            raise ValueError('ICL requires an exact reference codec length')
        refs = int(refs)
    streaming = not first(info.get('non_streaming_mode'), False)
    chunk = config_value(config, 'lm_tts_streaming_chunk', [5, 10])
    maximum = int(first(info.get('max_new_tokens'), 2048))
    if maximum < 1:
        raise ValueError('max_new_tokens must be positive')
    plan = decode_plan(streamed, refs, maximum, *chunk) if streaming else DecodePlan(
        (-1,) * maximum, (True,) * maximum, streamed)
    language = first(info.get('language'), 'Auto').lower()
    if language != 'auto' and language not in config_value(config, 'codec_language_id', {}):
        raise ValueError(f'Unsupported language: {language}')
    prefix = len(config_value(config, 'lm_tts_role_ids', [151644, 77091, 198]))
    prefix += (3 if language == 'auto' else 4) + 1 + 1  # controls, speaker, text BOS
    length = prefix + plan.prefill_text + refs + (0 if streaming else 1)
    return length, plan, ids, refs


class TrainingPromptBuilder:
    """Reuse native audio preprocessing, then assemble the training layout."""
    def __init__(self, base, owner):
        self.base = base
        self.owner = owner

    def __getattr__(self, name):
        return getattr(self.base, name)

    def build_prompt_embeds(self, *, task_type, info_dict):
        import torch
        if task_type != 'Base':
            raise ValueError('LM-TTS interleaved checkpoints require Base voice cloning')
        # The native builder owns speaker/codec caches and input formats. Its
        # fixed role/controls/speaker/BOS prefix also matches training.
        raw = dict(info_dict, non_streaming_mode=[True])
        full, _, _, ref_codes = self.base.build_prompt_embeds(task_type=task_type, info_dict=raw)
        refs = 0 if ref_codes is None else len(ref_codes)
        info = dict(info_dict, ref_code_length=[refs])
        tokenizer = self.base.get_text_tokenizer()
        length, plan, ids, expected_refs = request_layout(
            info, self.owner.talker_config, lambda text: tokenizer.encode(text, add_special_tokens=False))
        if expected_refs != refs:
            raise ValueError('Reference codec length disagrees with request')
        model = self.owner
        device = full.device
        # Keep BOS in the projection batch, as the training frontend does.
        # Projecting it separately changes BF16 GEMM rounding for some rows.
        ids = torch.tensor([model.config.tts_bos_token_id] + ids + [model.config.tts_eos_token_id],
                           device=device, dtype=torch.long)
        projected = model.text_projection(model.text_embedding(ids))
        text = projected[1:]
        codec_pad = model.embed_input_ids(torch.tensor([model.talker_config.codec_pad_id], device=device))
        text_rows = text + codec_pad
        pad = model._tts_pad_embed.reshape(1, -1)
        frames = full.new_empty((0, full.shape[-1]))
        if refs:
            codes = ref_codes.to(device=device, dtype=torch.long)
            frames = model.embed_input_ids(codes[:, 0])
            for group in range(1, codes.shape[1]):
                frames = frames + model._stacked_codec_embed[group - 1, codes[:, group]]
            frames = frames + pad
        streaming = not first(info.get('non_streaming_mode'), False)
        prefix_length = length - plan.prefill_text - refs - (0 if streaming else 1)
        pieces = [full[:prefix_length-1], projected[:1] + codec_pad]
        if streaming:
            segments, _ = layout(len(text), refs, *model.talker_config.lm_tts_streaming_chunk)
            pieces.extend(text_rows[a:b] if kind == 'text' else frames[a:b] for kind, a, b in segments)
        else:
            bos = model.embed_input_ids(torch.tensor([model.talker_config.codec_bos_id], device=device))
            pieces.extend([text_rows, bos + pad, frames])
        result = torch.cat(pieces)
        if len(result) != length:
            raise ValueError('Training prompt length mismatch')
        declared = info.get('_omni_prompt_len')
        if declared is not None and int(declared) != length:
            raise ValueError(f'Expected {length} prompt positions, received {declared}')
        # Audio slots add tts_pad. Forced text slots add their projected text
        # to the codec-pad embedding returned by MTP, with one BF16 addition.
        tail = torch.cat([pad if slot < 0 else text[slot:slot+1] for slot in plan.slots])
        model._training_plans[str(info_dict['req_id'])] = plan
        while len(model._training_plans) > 4096:
            model._training_plans.pop(next(iter(model._training_plans)))
        return result, tail, refs, ref_codes
