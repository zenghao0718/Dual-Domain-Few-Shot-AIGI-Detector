import unittest
import importlib.util

if importlib.util.find_spec("torch") is None:
    raise unittest.SkipTest("NOT_RUN_ENV_MISSING: original DDFSD core tests require torch")

import torch

from datasets.ddfsd_datasets import validate_generator_name
from model.ddfsd_losses import compute_alpha, compute_prototypes, compute_query_logits
from test_ddfsd import resolve_checkpoint_step
from util.ddfsd_frequency import raw_rgb_to_frequency
from util.ddfsd_sampling import sample_support_query_indices


class DDFSDCoreTests(unittest.TestCase):
    def test_support_query_sampling(self):
        support, query = sample_support_query_indices(50, 10, seed=42)
        self.assertEqual(len(support), 10)
        self.assertTrue(set(support).isdisjoint(query))
        self.assertEqual((support, query), sample_support_query_indices(50, 10, seed=42))
        self.assertNotEqual(support, sample_support_query_indices(50, 10, seed=43)[0])

    def test_haar_dwt_shape_and_finiteness(self):
        output = raw_rgb_to_frequency(torch.rand(2, 3, 224, 224))
        self.assertEqual(tuple(output.shape), (2, 3, 112, 112))
        self.assertTrue(torch.isfinite(output).all())

    def test_alpha_is_bounded_and_detached(self):
        rgb = torch.tensor([[0.1, 0.4]], requires_grad=True)
        freq = torch.tensor([[0.4, 0.1]], requires_grad=True)
        alpha = compute_alpha(rgb, freq, tau_r=0.1)
        self.assertTrue(torch.all((alpha >= 0.25) & (alpha <= 0.75)))
        self.assertFalse(alpha.requires_grad)

    def test_prototype_and_query_shapes(self):
        support_rgb = torch.randn(2, 5, 3, 8)
        support_freq = torch.randn(2, 5, 3, 8)
        proto_rgb, proto_freq = compute_prototypes(support_rgb, support_freq)
        self.assertEqual(tuple(proto_rgb.shape), (2, 3, 8))
        self.assertEqual(tuple(proto_freq.shape), (2, 3, 8))
        query = torch.randn(2, 12, 8)
        alpha = torch.full((2, 3), 0.5)
        output = compute_query_logits(query, query, proto_rgb, proto_freq, alpha)
        self.assertEqual(tuple(output["logits"].shape), (2, 12, 3))

    def test_checkpoint_step_validation(self):
        self.assertEqual(resolve_checkpoint_step(15000, 15000, "model.pth"), 15000)
        with self.assertRaises(ValueError):
            resolve_checkpoint_step(15000, 12500, "model.pth")

    def test_invalid_exclude_class(self):
        with self.assertRaises(ValueError):
            validate_generator_name("invalid")


if __name__ == "__main__":
    unittest.main()
