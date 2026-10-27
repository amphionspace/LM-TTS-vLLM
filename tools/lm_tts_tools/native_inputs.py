"""Optional SDK input helpers; the plugin does not depend on this module."""
from pathlib import Path


def reference_audio_payload(audio):
    """Make Qwen AudioLike safe for Omni's additional_information transport.

    Raw nested numpy arrays lose their external buffers in the pinned Omni
    metadata serializer. Use its prompt builder's explicit inline-array format.
    File/URL/base64 strings pass through unchanged.
    """
    if isinstance(audio, Path):
        return str(audio.resolve())
    if isinstance(audio, str):
        return audio
    if not isinstance(audio, (tuple, list)) or len(audio) != 2:
        raise ValueError('Reference audio must be a path/URL/base64 string or (waveform, sample_rate)')
    import numpy as np
    wav, sr = audio
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=-1)
    if wav.ndim != 1 or not wav.size or not np.isfinite(wav).all():
        raise ValueError('Reference waveform must contain finite audio samples')
    if not 1000 <= int(sr) <= 200000:
        raise ValueError('Invalid reference audio sample rate')
    wav = np.ascontiguousarray(wav)
    return ({'__ndarray__': True, 'data': wav.tobytes(),
             'dtype': wav.dtype.str, 'shape': list(wav.shape)}, int(sr))
