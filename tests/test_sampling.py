import unittest
import torch
from lm_tts_vllm.code_predictor import CodePredictor


class SamplingTests(unittest.TestCase):
    def test_matches_reference_warpers_and_seed(self):
        from transformers import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
        scores = torch.randn(8, 37, generator=torch.Generator().manual_seed(7))
        ids = torch.zeros(8, 1, dtype=torch.long)
        expected = scores.clone()
        for warper in (TemperatureLogitsWarper(0.7), TopKLogitsWarper(9), TopPLogitsWarper(0.8)):
            expected = warper(ids, expected)
        expected = torch.multinomial(expected.softmax(-1), 1, generator=torch.Generator().manual_seed(42)).squeeze(-1)
        actual = CodePredictor.sample(scores, 0.7, 9, 0.8, torch.Generator().manual_seed(42))
        self.assertTrue(torch.equal(expected, actual))

    def test_topk_one_and_seed_reproducibility(self):
        scores = torch.tensor([[1., 2., 3.]]).repeat(64, 1)
        self.assertTrue((CodePredictor.sample(scores, top_k=1) == 2).all())
        a = CodePredictor.sample(scores, top_k=0, generator=torch.Generator().manual_seed(4))
        b = CodePredictor.sample(scores, top_k=0, generator=torch.Generator().manual_seed(4))
        c = CodePredictor.sample(scores, top_k=0, generator=torch.Generator().manual_seed(5))
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))

    def test_graph_pick_keeps_warper_support_and_distribution(self):
        from transformers import TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper
        scores = torch.randn(4, 37, generator=torch.Generator().manual_seed(7))
        ids = torch.zeros(4, 1, dtype=torch.long)
        expected = scores.clone()
        for warper in (TemperatureLogitsWarper(0.7), TopKLogitsWarper(9), TopPLogitsWarper(0.8)):
            expected = warper(ids, expected)
        expected = expected.softmax(-1)
        draws = 20000
        options = torch.tensor([[1 / 0.7, 9, 0.8, 1.0]]).repeat(4 * draws, 1)
        uniforms = torch.rand(4 * draws, 37, generator=torch.Generator().manual_seed(1)).clamp_min(1e-20)
        picked = CodePredictor.pick(scores.repeat(draws, 1), uniforms, options).view(draws, 4)
        for row in range(4):
            freq = torch.bincount(picked[:, row], minlength=37).float() / draws
            self.assertTrue(((freq > 0) <= (expected[row] > 0)).all())
            self.assertLess((freq - expected[row]).abs().max().item(), 0.02)
        greedy = torch.tensor([[1.0, 1, 2.0, 0.0]]).repeat(4, 1)
        self.assertTrue(torch.equal(CodePredictor.pick(scores, uniforms[:4], greedy), scores.argmax(-1)))
