"""Verify native reference loading, including HTTP, without loading GPU models."""
import argparse
import base64
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref-audio', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder
    from .native_inputs import reference_audio_payload

    path = args.ref_audio.resolve()
    expected, expected_sr = sf.read(path, dtype='float32')
    if expected.ndim == 2:
        expected = expected.mean(axis=-1)
    builder = object.__new__(Qwen3TTSPromptEmbedsBuilder)
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(SimpleHTTPRequestHandler, directory=str(path.parent)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        from urllib.parse import quote
        inputs = {'path': str(path), 'file_uri': path.as_uri(),
                  'http': f'http://127.0.0.1:{server.server_port}/{quote(path.name)}',
                  'base64': base64.b64encode(path.read_bytes()).decode(),
                  'data_uri': 'data:audio/wav;base64,' + base64.b64encode(path.read_bytes()).decode(),
                  'waveform': reference_audio_payload((expected, expected_sr))}
        for name, value in inputs.items():
            # The explicit waveform payload is already normalized above.
            value = reference_audio_payload(value) if isinstance(value, str) else value
            wav, sr = builder.normalize_ref_audio(value)
            assert sr == expected_sr, (name, sr, expected_sr)
            np.testing.assert_array_equal(wav, expected, err_msg=name)
            print(f'{name}: exact waveform and sample rate', flush=True)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == '__main__':
    main()
