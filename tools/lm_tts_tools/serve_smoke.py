"""Exercise the stock /v1/audio/speech endpoint, without a custom backend."""
import argparse
import base64
import io
import json
from pathlib import Path
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:7864')
    parser.add_argument('--ref-audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(args.url + '/v1/models', timeout=30) as response:
        model = json.load(response)['data'][0]['id']
    reference = 'data:audio/wav;base64,' + base64.b64encode(args.ref_audio.read_bytes()).decode()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []

    def send(request):
        req = urllib.request.Request(args.url + '/v1/audio/speech',
            data=json.dumps(request).encode(), headers={'Content-Type': 'application/json'})
        try:
            with opener.open(req, timeout=180) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            raise RuntimeError(f'HTTP {error.code}: {error.read().decode()[:800]}') from None

    for xvec in [False, True]:
        for nonstream in [True, False]:
            request = {'model': model, 'input': '你好，这是语音合成测试。', 'task_type': 'Base',
                       'language': 'Chinese', 'ref_audio': reference, 'x_vector_only_mode': xvec,
                       'non_streaming_mode': nonstream, 'max_new_tokens': 8, 'seed': 42,
                       'response_format': 'wav', 'extra_params': {
                           'do_sample': False, 'subtalker_dosample': not nonstream,
                           'subtalker_temperature': 0.9, 'subtalker_top_k': 50,
                           'subtalker_top_p': 0.9, 'repetition_penalty': 1.05}}
            if not xvec:
                request['ref_text'] = '在此奉劝大家别乱打美白针。'
            data = send(request)
            audio, sr = sf.read(io.BytesIO(data), dtype='float32')
            assert audio.size > 0 and np.isfinite(audio).all()
            path = args.output / f'{len(reports)}.wav'
            path.write_bytes(data)
            row = {'x_vector_only': xvec, 'non_streaming_mode': nonstream,
                   'cp_sampling': not nonstream, 'sample_rate': sr, 'audio_seconds': len(audio) / sr}
            reports.append(row)
            print(json.dumps(row), flush=True)
    (args.output / 'report.json').write_text(json.dumps(reports, indent=2) + '\n')
    # Last case samples CP only. Check that HTTP actually forwards CP options
    # and preserves request-local randomness rather than silently using greedy.
    repeated = send(request)
    changed = send(dict(request, seed=43))
    checks = {'same_seed_equal': data == repeated, 'different_seed_changes_audio': data != changed}
    (args.output / 'sampling.json').write_text(json.dumps(checks, indent=2) + '\n')
    assert all(checks.values()), checks
    print(json.dumps(checks), flush=True)


if __name__ == '__main__':
    main()
