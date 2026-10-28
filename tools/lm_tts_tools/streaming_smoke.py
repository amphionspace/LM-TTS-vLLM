"""Check real incremental audio delivery through the standard AsyncOmni API."""
import argparse
import asyncio
import copy
import json
import math
from pathlib import Path
import time
from functools import partial


def check_reference(model, directory):
    """Trusted offline apply_model callback; never enabled by the demo/server."""
    import torch
    directory = Path(directory)
    reference = torch.load(directory / 'reference.pt', map_location='cpu', weights_only=True)
    info = json.loads((directory / 'prompt.json').read_text())
    info['req_id'] = 'offline-parity'
    model._validate_prompt(info)
    with torch.inference_mode():
        prefill, _, _, _ = model._prompt_builder.build_prompt_embeds(task_type='Base', info_dict=info)
    diff = (prefill.cpu().float() - reference['prefill'].float()).abs()
    torch.save(prefill.cpu(), directory / 'native-prefill.pt')
    (directory / 'native-first.pt').unlink(missing_ok=True)
    result = {'prefill_shape': list(prefill.shape), 'prefill_max_abs': float(diff.max()),
              'prefill_mismatch_elements': int((diff != 0).sum())}
    def capture(module, args, output):
        handle.remove()
        hidden = args[1][-1:].detach()
        logits = output[-1:].detach()
        torch.save({'hidden': hidden.cpu(), 'logits': logits.cpu()}, directory / 'native-first.pt')
    # The compiled backbone bypasses Module hooks. LogitsProcessor retains
    # normal Module dispatch and sees the actual sampled predictor hidden.
    handle = model.logits_processor.register_forward_hook(capture)
    return result


async def run(args):
    import numpy as np
    import soundfile as sf
    import torch
    from vllm.sampling_params import RequestOutputKind
    from transformers import AutoTokenizer
    from vllm_omni import AsyncOmni
    from vllm_omni.plugins import load_omni_general_plugins
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder
    import lm_tts_vllm

    load_omni_general_plugins()
    model = args.model.resolve()
    config = json.loads((model / 'config.json').read_text())['talker_config']
    codec = json.loads((model / 'speech_tokenizer/config.json').read_text())
    tok = AutoTokenizer.from_pretrained(model)
    ref = sf.info(args.ref_audio)
    info = {'task_type': ['Base'], 'ref_audio': [str(args.ref_audio.resolve())],
            'ref_text': ['在此奉劝大家别乱打美白针。'],
            'text': [args.text],
            'language': ['Chinese'], 'x_vector_only_mode': [False],
            'non_streaming_mode': [not args.interleaved], 'full_utterance_decode': [False],
            'max_new_tokens': [80], 'subtalker_dosample': [False],
            'ref_code_length': [math.ceil(ref.frames * codec['input_sample_rate'] /
                                          ref.samplerate / codec['encode_downsample_rate'])]}
    if args.reference:
        info = json.loads((args.reference / 'prompt.json').read_text())
        info['full_utterance_decode'] = [False]
    length = Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        info, task_type='Base', tokenize_prompt=lambda t: tok(t)['input_ids'],
        codec_language_id=config['codec_language_id'], spk_is_dialect=config.get('spk_is_dialect'))
    from lm_tts_vllm.interleaved import request_layout, uses_training_layout
    maximum = 80
    if uses_training_layout(config):
        length, plan, _, _ = request_layout(info, config, lambda s: tok.encode(s, add_special_tokens=False))
        maximum = len(plan.slots)
        print(json.dumps({'prompt_length': length, 'decode_steps': maximum,
                          'text_slots': sum(s >= 0 for s in plan.slots)}), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    engine = AsyncOmni(model=str(model), deploy_config=str(
        Path(lm_tts_vllm.__file__).parent / 'native_streaming.yaml'),
        stage_overrides={'0': {'kv_cache_memory_bytes': 536870912}})
    try:
        assert engine.async_chunk, 'Streaming deploy config was not applied'
        if args.reference:
            parity = await engine.collective_rpc('apply_model', timeout=120,
                args=(partial(check_reference, directory=str(args.reference.resolve())),), stage_ids=[0])
            (args.output / 'prefill-parity.json').write_text(json.dumps(parity, indent=2))
            print(json.dumps({'prefill_parity': parity}), flush=True)
        params = copy.deepcopy(engine.default_sampling_params_list)
        for param in params:
            param.output_kind = RequestOutputKind.DELTA
        params[0].max_tokens = maximum
        params[0].min_tokens = maximum  # Force the capped-generation boundary.
        params[0].seed = 42
        params[0].extra_args = {'tts_local_seed': 42}
        start, chunks, final = time.perf_counter(), [], None
        audio_parts = []
        async for output in engine.generate(
                {'prompt_token_ids': [0] * length, 'additional_information': info},
                request_id='stream-cap', sampling_params_list=params):
            mm = output.outputs[0].multimodal_output
            audio = mm.get('audio')
            if audio is None:
                continue
            audio = torch.cat(audio, dim=-1) if isinstance(audio, list) else audio
            assert audio.numel() > 0 and torch.isfinite(audio).all()
            audio_parts.append(audio.float().cpu().flatten())
            sr = mm['sr'][-1] if isinstance(mm['sr'], list) else mm['sr']
            row = {'finished': output.finished, 'samples': audio.numel(),
                   'elapsed_seconds': time.perf_counter() - start}
            print(json.dumps(row), flush=True)
            if output.finished:
                final = row
                codes = mm.get('codec_frames')
                if codes is not None:
                    codes = torch.cat(codes, dim=0) if isinstance(codes, list) else codes
                    np.save(args.output / 'codes.npy', codes.cpu().numpy())
                    final['codec_shape'] = list(codes.shape)
            else:
                sf.write(args.output / f'chunk-{len(chunks)}.wav',
                         audio.float().cpu().numpy().flatten(), int(sr))
                chunks.append(row)
        complete_audio = torch.cat(audio_parts)
        sf.write(args.output / 'final.wav', complete_audio.numpy(), int(sr))
        report = {'chunks': chunks, 'final': final, 'total_samples': complete_audio.numel()}
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        if args.reference:
            reference = torch.load(args.reference / 'reference.pt', map_location='cpu', weights_only=True)
            native = torch.load(args.reference / 'native-first.pt', map_location='cpu', weights_only=True)
            parity = {}
            for key in ('hidden', 'logits'):
                a, b = reference[key].float(), native[key].float()
                diff = (a - b).abs()
                parity[key] = {'max_abs': float(diff.max()), 'mean_abs': float(diff.mean()),
                    'cosine': float(torch.nn.functional.cosine_similarity(a.flatten()[None], b.flatten()[None]))}
            parity['argmax'] = {name: int(value['logits'].argmax())
                                for name, value in [('pytorch', reference), ('vllm', native)]}
            (args.output / 'first-step-parity.json').write_text(json.dumps(parity, indent=2))
        assert final is not None, 'Missing final output'
        assert len(chunks) > 1, 'No real incremental audio delivery'
        assert complete_audio.numel() == 79 * 1920, 'Capped generation must retain 79 codec frames'
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--ref-audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interleaved', action='store_true')
    parser.add_argument('--text', default='你好，这是语音合成测试。我们正在测试流式音频输出，检查每一段声音是否正常。')
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    if args.reference:
        import os
        if os.environ.get('VLLM_ALLOW_INSECURE_SERIALIZATION') != '1':
            parser.error('--reference requires trusted local RPC with VLLM_ALLOW_INSECURE_SERIALIZATION=1')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
