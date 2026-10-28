"""Drive a demo backend input session the way an LLM token stream would."""
import argparse
import json
import threading
import time
import urllib.request
from pathlib import Path

from .demo.common import REFERENCE, REFERENCE_TEXT

TEXT = '今天天气这么好，阳光照在身上暖洋洋的，我们一起去公园散散步，顺便买杯咖啡吧。'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=7863)
    p.add_argument('--text', default=TEXT)
    p.add_argument('--chars', type=int, default=2, help='characters per text delta')
    p.add_argument('--interval', type=float, default=0.3, help='seconds between deltas')
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f'http://127.0.0.1:{args.port}'

    def post(path, data):
        req = urllib.request.Request(base + path, data=json.dumps(data).encode(),
                                     headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=60) as r:
            return json.loads(r.read())

    sid = post('/input_sessions', {'model': '10000', 'ref_audio': REFERENCE, 'ref_text': REFERENCE_TEXT,
                                   'text': '-', 'seed': 42})['session_id']
    start = time.perf_counter()
    log = []

    def sender():
        for i in range(0, len(args.text), args.chars):
            post(f'/input_sessions/{sid}/text', {'text': args.text[i:i + args.chars]})
            log.append(('sent', args.text[:i + args.chars], round(time.perf_counter() - start, 2)))
            time.sleep(args.interval)
        post(f'/input_sessions/{sid}/text', {'text': '', 'final': True})
        log.append(('closed', None, round(time.perf_counter() - start, 2)))
    threading.Thread(target=sender, daemon=True).start()

    result = None
    with opener.open(base + f'/input_sessions/{sid}/events', timeout=600) as r:
        for line in r:
            event = json.loads(line)
            t = round(time.perf_counter() - start, 2)
            if event['type'] == 'chunk':
                print(f'{t:6.2f}s audio chunk {event["index"]}', flush=True)
            elif event['type'] == 'error':
                raise SystemExit(f'error: {event["detail"]}')
            elif event['type'] == 'result':
                result = event
    sent = [e for e in log if e[0] == 'sent']
    print(f'text deltas {len(sent)}, input closed at {log[-1][2]}s')
    print(json.dumps({k: result[k] for k in ('audio', 'chunks', 'first_audio_seconds', 'audio_seconds', 'seconds')},
                     ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps({'log': log, 'result': result}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
