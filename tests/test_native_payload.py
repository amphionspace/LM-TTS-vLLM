import unittest
from types import SimpleNamespace

import torch

from lm_tts_vllm.native_payload import normalize_voice_prompt, reference_frames, reference_stream_frame


class NativePayloadTests(unittest.TestCase):
    def test_cached_prompt_restores_single_frame_without_unwrapping_codebooks(self):
        raw = {"voice_clone_prompt": [{"ref_code": [list(range(16))], "icl_mode": True}]}
        result = normalize_voice_prompt(raw)
        self.assertEqual(result["voice_clone_prompt"][0]["ref_code"].shape, (1, 16))
        self.assertIsInstance(raw["voice_clone_prompt"][0]["ref_code"], list)
        self.assertIs(normalize_voice_prompt(result), result)
        wrapped = normalize_voice_prompt({"additional_information": raw, "non_streaming_mode": [False]})
        self.assertEqual(wrapped["voice_clone_prompt"][0]["ref_code"].shape, (1, 16))
        self.assertEqual(wrapped["non_streaming_mode"], [False])
        with self.assertRaises(ValueError):
            normalize_voice_prompt({"voice_clone_prompt": {"ref_code": [[2150] * 16]}})

    def test_stream_cap_omits_only_final_frame_without_mutating_output(self):
        output = {"meta": {"codec_frame_valid": True}, "codes": {"audio": torch.tensor([[1174, 9]])}}
        request = SimpleNamespace(output_token_ids=[1174, 444], is_finished=lambda: True)
        result = reference_stream_frame(output, request)
        self.assertFalse(result["meta"]["codec_frame_valid"])
        self.assertTrue(output["meta"]["codec_frame_valid"])
        request.output_token_ids = [1174, 2150]
        self.assertIs(reference_stream_frame(output, request), output)
        request.output_token_ids = [1174, 444]
        request.is_finished = lambda: False
        self.assertIs(reference_stream_frame(output, request), output)
        snapshot = SimpleNamespace(output_token_ids=[], last_output_token_id=444, is_finished=lambda: True)
        self.assertFalse(reference_stream_frame(output, snapshot)["meta"]["codec_frame_valid"])
        snapshot.last_output_token_id = 2150
        self.assertIs(reference_stream_frame(output, snapshot), output)
        pending_ack = SimpleNamespace(output_token_ids=[], last_output_token_id=444,
            output_token_count=80, sampling_params=SimpleNamespace(max_tokens=80),
            is_finished=lambda: False)
        self.assertFalse(reference_stream_frame(output, pending_ack)["meta"]["codec_frame_valid"])
        pending_ack.output_token_count = 79
        self.assertIs(reference_stream_frame(output, pending_ack), output)

    def test_token_cap_keeps_first_frame_and_drops_final_eager_frame(self):
        frames = torch.tensor([[1174, 9], [444, 8], [3, 7]])
        self.assertTrue(torch.equal(reference_frames(frames, [1174, 444, 3]), frames[:2]))

    def test_eos_and_prefill_padding_do_not_shift_valid_frames(self):
        frames = torch.tensor([[0, 0], [1174, 9], [444, 8], [2150, 1]])
        self.assertTrue(torch.equal(reference_frames(frames, [1174, 444, 2150]), frames[1:3]))


if __name__ == "__main__":
    unittest.main()
