"""Static listening page plus a thin proxy to the loopback PyTorch / vLLM backends."""
import argparse
import json
import uuid
from pathlib import Path

from .common import MODELS, REFERENCE, REFERENCE_TEXT, ROOT

STATIC = Path(__file__).with_name('static')
OUTPUT = ROOT / 'results/demo'
UPLOADS = OUTPUT / 'uploads'
AUDIO_SUFFIXES = {'.wav', '.flac', '.mp3', '.ogg', '.m4a', '.webm'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--pytorch-port', type=int, default=7862)
    parser.add_argument('--vllm-port', type=int, default=7863)
    args = parser.parse_args()
    import httpx
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, Response, StreamingResponse

    backends = {'pytorch': f'http://127.0.0.1:{args.pytorch_port}', 'vllm': f'http://127.0.0.1:{args.vllm_port}'}
    # Loopback backends must not go through an inherited HTTP/SOCKS proxy.
    client = httpx.AsyncClient(timeout=httpx.Timeout(10, read=1200), trust_env=False)
    app = FastAPI()
    UPLOADS.mkdir(parents=True, exist_ok=True)

    def backend(engine):
        if engine not in backends:
            raise HTTPException(404, 'Unknown engine')
        return backends[engine]

    @app.get('/')
    def index():
        return FileResponse(STATIC / 'index.html')

    @app.get('/api/config')
    def config():
        raw = json.loads((MODELS['10000']['source'] / 'config.json').read_text())
        languages = ['Auto'] + [s.capitalize() for s in raw['talker_config']['codec_language_id'] if 'dialect' not in s]
        return {'reference': {'path': REFERENCE, 'text': REFERENCE_TEXT}, 'languages': languages}

    @app.get('/api/health')
    async def health():
        status = {}
        for engine, url in backends.items():
            try:
                status[engine] = (await client.get(url + '/health', timeout=3)).json()
            except Exception:
                status[engine] = {'ready': False}
        return status

    @app.post('/api/upload')
    async def upload(file: UploadFile):
        suffix = Path(file.filename or '').suffix.lower()
        if suffix not in AUDIO_SUFFIXES:
            raise HTTPException(400, '不支持的音频格式')
        path = UPLOADS / f'{uuid.uuid4().hex}{suffix}'
        data = await file.read()
        if len(data) > 50 << 20:
            raise HTTPException(400, '文件过大')
        path.write_bytes(data)
        return {'path': str(path)}

    @app.get('/api/audio')
    def audio(path: str):
        target = Path(path).resolve()
        if target != Path(REFERENCE).resolve() and OUTPUT.resolve() not in target.parents:
            raise HTTPException(403, 'Forbidden')
        if not target.is_file():
            raise HTTPException(404, 'Not found')
        return FileResponse(target)

    @app.api_route('/api/{engine}/{path:path}', methods=['GET', 'POST', 'DELETE'])
    async def proxy(engine: str, path: str, request: Request):
        url = f'{backend(engine)}/{path}'
        body = await request.body()
        try:
            upstream = await client.send(client.build_request(
                request.method, url, content=body or None,
                headers={'Content-Type': request.headers.get('content-type', 'application/json')}), stream=True)
        except httpx.HTTPError as exc:
            raise HTTPException(503, '推理后端正在加载或暂不可用') from exc
        media = upstream.headers.get('content-type', 'application/json')
        if 'ndjson' in media:
            async def relay():
                try:
                    async for chunk in upstream.aiter_raw():
                        yield chunk
                finally:
                    await upstream.aclose()
            return StreamingResponse(relay(), status_code=upstream.status_code, media_type=media)
        content = await upstream.aread()
        await upstream.aclose()
        return Response(content, status_code=upstream.status_code, media_type=media)

    uvicorn.run(app, host='0.0.0.0', port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
