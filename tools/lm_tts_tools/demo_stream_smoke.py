"""Verify the running demo forwards incremental native audio and sampling."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    import numpy as np
    import soundfile as sf
    from .demo.common import REFERENCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7863)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f'http://127.0.0.1:{args.port}'
    deadline = time.monotonic() + 900
    while True:
        try:
            with opener.open(url + '/health', timeout=3) as response:
                if json.load(response)['ready']:
                    break
        except OSError:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError('Demo backend did not become ready')
        time.sleep(5)
    text = '你好，这是新模型的流式语音合成测试。我们需要验证长文本能够按照训练时的方式，每次输入五个文本符号，然后生成十帧声音。整个过程中，要确保文本顺序正确，音频可以连续播放，并且不会漏掉任何一个词语。接下来继续测试后面的内容，检查最后一段音频是否完整。'
    cases = [(False, False, True, 80), (False, False, False, 40), (False, False, False, 40),
             (False, True, True, 8), (True, False, False, 8), (True, True, True, 8)]
    rows, sampled = [], []
    for xvec, nonstream, greedy, cap in cases:
        request = dict(model='10000', ref_audio=REFERENCE, ref_text='' if xvec else '在此奉劝大家别乱打美白针。',
            text=text, x_vector_only_mode=xvec, non_streaming_mode=nonstream,
            temperature=0 if greedy else 0.9, greedy_cp=greedy, max_tokens=cap, seed=42)
        req = urllib.request.Request(url + '/synthesize_stream', data=json.dumps(request).encode(),
                                     headers={'Content-Type': 'application/json'})
        events, received = [], []
        started = time.perf_counter()
        with opener.open(req, timeout=600) as response:
            for line in response:
                event = json.loads(line)
                if event['type'] == 'error':
                    raise RuntimeError(event['detail'])
                events.append(event)
                received.append(time.perf_counter() - started)
        result = events[-1]
        assert result['type'] == 'result'
        chunks = [e for e in events if e['type'] == 'chunk']
        parts = [sf.read(e['audio'], dtype='int16')[0] for e in chunks]
        audio, rate = sf.read(result['audio'], dtype='int16')
        assert rate == 24000 and np.array_equal(audio, np.concatenate(parts))
        if not nonstream:
            assert len(audio) == (cap-1)*1920, (len(audio), cap)
        if cap >= 40:
            assert len(chunks) > 1 and received[0] < received[-1]
        if cap == 40:
            sampled.append(audio)
        row = dict(x_vector_only=xvec, non_streaming_mode=nonstream, greedy=greedy,
                   cap=cap, result=result, received_seconds=received)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    assert np.array_equal(*sampled), 'Repeated CP/Talker sampling with the same seed changed'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
