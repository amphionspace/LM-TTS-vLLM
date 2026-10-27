import unittest

import torch

from lm_tts_tools.pipeline_audit import decisions


class DecisionMarginTests(unittest.TestCase):
    def test_distinguishes_existing_tie_new_tie_and_clear_flip(self):
        original = torch.tensor([[[4., 4., 2.], [3., 4., 2.], [4., 3., 2.]]])
        actual = torch.tensor([[[3.5, 4., 2.], [4., 4., 2.], [3., 4., 2.]]])
        report = decisions(original, actual)
        self.assertEqual(report["argmax_agreement"], 0)
        self.assertEqual(report["tie_flips"], 1)
        self.assertEqual([r["flip_is_reference_tie"] for r in report["steps"]], [True, False, False])
        self.assertEqual([r["flip_creates_tie"] for r in report["steps"]], [False, True, False])
        self.assertEqual(report["error"]["max_abs"], 1)

    def test_same_tie_choice_is_not_a_flip(self):
        scores = torch.tensor([[[4., 4., 2.]]])
        row = decisions(scores, scores)["steps"][0]
        self.assertTrue(row["argmax_equal"])
        self.assertFalse(row["flip_is_reference_tie"])
        self.assertFalse(row["flip_creates_tie"])

    def test_nonfinite_logits_are_rejected(self):
        with self.assertRaises(ValueError):
            decisions(torch.tensor([[[float("nan"), 0.]]]), torch.zeros(1, 1, 2))


if __name__ == "__main__":
    unittest.main()
