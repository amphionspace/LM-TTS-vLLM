import unittest

import torch

from lm_tts_vllm.native_payload import reference_frames


class NativePayloadTests(unittest.TestCase):
    def test_token_cap_keeps_first_frame_and_drops_final_eager_frame(self):
        frames = torch.tensor([[1174, 9], [444, 8], [3, 7]])
        self.assertTrue(torch.equal(reference_frames(frames, [1174, 444, 3]), frames[:2]))

    def test_eos_and_prefill_padding_do_not_shift_valid_frames(self):
        frames = torch.tensor([[0, 0], [1174, 9], [444, 8], [2150, 1]])
        self.assertTrue(torch.equal(reference_frames(frames, [1174, 444, 2150]), frames[1:3]))


if __name__ == "__main__":
    unittest.main()
