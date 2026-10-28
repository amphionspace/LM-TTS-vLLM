import base64
import io
import unittest
import wave

from lm_tts_vllm.audio import canonical_audio_string


class AudioStringTests(unittest.TestCase):
    def test_bare_wav_base64_with_slashes_is_not_treated_as_path(self):
        output = io.BytesIO()
        with wave.open(output, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b'\xff\x7f' * 512)
        encoded = base64.b64encode(output.getvalue()).decode()
        self.assertIn('/', encoded)
        uri = canonical_audio_string(encoded)
        self.assertTrue(uri.startswith('data:audio/'))
        self.assertEqual(base64.b64decode(uri.split(',', 1)[1]), output.getvalue())
        self.assertEqual(canonical_audio_string(uri), uri)

    def test_paths_urls_and_non_audio_strings_are_preserved(self):
        values = ['/tmp/reference.wav', 'file:///tmp/reference.wav',
                  'https://example.com/' + 'a' * 400, 'a' * 400,
                  '/tmp/' + 'nested/' * 50 + 'reference.wav']
        for value in values:
            self.assertEqual(canonical_audio_string(value), value)


if __name__ == '__main__':
    unittest.main()
