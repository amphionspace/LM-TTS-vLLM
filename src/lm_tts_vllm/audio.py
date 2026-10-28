"""Normalize reference strings before the native Qwen audio loader."""
import base64
import binascii


def canonical_audio_string(value):
    """Recognize encoded audio even when bare base64 contains path separators."""
    if not isinstance(value, str) or len(value) < 256:
        return value
    if value.startswith(('data:', 'http://', 'https://', 'file:')):
        return value
    try:
        data = base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error):
        return value
    is_audio = (data.startswith((b'RIFF', b'fLaC', b'OggS', b'ID3'))
                or (len(data) >= 2 and data[0] == 255 and data[1] & 224 == 224))
    return 'data:audio/wav;base64,' + value if is_audio else value
