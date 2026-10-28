import unittest
from lm_tts_vllm.input_stream import TextStreamBuffer


class InputStreamTest(unittest.TestCase):
    def test_waits_for_text_and_never_adds_eos_before_final(self):
        b = TextStreamBuffer(list(range(14)), 60, 999, 79)
        b.append([100]*20)
        self.assertEqual(list(b.drain()), [])
        b.append([101])
        first = list(b.drain())
        self.assertEqual(len(first), 1)
        self.assertEqual(len(first[0].token_ids), 35)
        self.assertEqual(first[0].audio_frames, 10)
        self.assertFalse(first[0].final)
        self.assertNotIn(999, first[0].token_ids)
        b.append([102]*4)
        self.assertEqual(list(b.drain()), [])
        b.append([], final=True)
        final = list(b.drain())[0]
        self.assertEqual(final.token_ids, (102,102,102,102,999))
        self.assertEqual(final.audio_frames, 69)
        self.assertTrue(final.final)
        with self.assertRaises(ValueError):
            b.append([3])

    def test_reference_partial_frame_block_and_early_close(self):
        b = TextStreamBuffer([1]*10, 63, 999, 79)
        b.append([2]*25)
        self.assertEqual(list(b.drain())[0].audio_frames, 7)
        b.append([3]*5)
        self.assertEqual(list(b.drain())[0].audio_frames, 10)
        b.append([], final=True)
        self.assertEqual(list(b.drain())[0].audio_frames, 62)
        short = TextStreamBuffer([1]*10, 63, 999, 79)
        short.append([2], final=True)
        chunk = list(short.drain())[0]
        self.assertEqual(len(chunk.token_ids),12)
        self.assertTrue(chunk.final)


if __name__ == '__main__':
    unittest.main()
