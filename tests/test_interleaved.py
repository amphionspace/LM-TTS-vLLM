import importlib.util
from pathlib import Path
import unittest

from lm_tts_vllm.interleaved import decode_plan, layout, request_layout


class InterleavedTest(unittest.TestCase):
    def test_generation_prefix_matches_training(self):
        # Independent expected ordering: five text positions then ten frames;
        # once the last text chunk is placed, all remaining frames follow.
        for text in (1, 5, 6, 35, 36, 103):
            complete = []
            for start in range(0, text, 5):
                complete.extend(('text', i) for i in range(start, min(start+5, text)))
                if start+5 < text:
                    complete.extend(('frames', i) for i in range(start//5*10, start//5*10+10))
                else:
                    complete.extend(('frames', i) for i in range(start//5*10, 300))
            for refs in (0, 1, 9, 10, 11, 60):
                segments, placed = layout(text, refs)
                prefix = [(kind, i) for kind, a, b in segments for i in range(a,b)]
                self.assertEqual(prefix, complete[:len(prefix)])
                plan = decode_plan(text, refs, 80)
                frames = refs
                future = []
                for slot, done in zip(plan.slots, plan.text_done):
                    if slot < 0:
                        self.assertEqual(done, placed == text)
                        future.append(('frames', frames))
                        frames += 1
                    else:
                        self.assertEqual(slot, placed)
                        future.append(('text', slot))
                        placed += 1
                self.assertEqual(prefix + future, complete[:len(prefix)+len(future)])
                self.assertEqual(sum(slot < 0 for slot in plan.slots), 80)

    def test_request_length_and_token_boundaries(self):
        calls = []
        def tokenize(s):
            calls.append(s)
            return list(range(len(s)))
        info = {'text': ['x'*40], 'ref_text': ['ref'], 'ref_code_length': [10],
                'max_new_tokens': [25], 'non_streaming_mode': [False]}
        length, plan, ids, refs = request_layout(info, {}, tokenize)
        self.assertEqual(calls, ['ref', 'x'*40])
        self.assertEqual(length, 8+10+10)
        self.assertEqual(refs, 10)
        self.assertEqual(len(plan.slots), 35)
        self.assertEqual(len(ids), 43)


if __name__ == '__main__':
    unittest.main()
