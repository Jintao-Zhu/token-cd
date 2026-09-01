import unittest
import numpy as np

from research.semantic_token_cd.st_shr_policy import EntityHistory, centroid, harmonic_reconstruct, translated_prior


class STSHROperatorTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.h = self.rng.normal(size=(256, 12)).astype(np.float32)
        self.region = np.array([85, 86, 101, 102])

    def test_no_current_semantic_leakage(self):
        first = harmonic_reconstruct(self.h, self.region)
        changed = self.h.copy()
        changed[self.region] = self.rng.normal(size=(len(self.region), 12)) * 1000
        second = harmonic_reconstruct(changed, self.region)
        self.assertEqual(float(np.max(np.abs(first - second))), 0.0)

    def test_first_step_is_plain_shr(self):
        prior, meta = translated_prior(self.region, None)
        self.assertIsNone(prior)
        self.assertEqual(meta["fallback_reason"], "no_history")
        st = harmonic_reconstruct(self.h, self.region, 0.0, prior)
        shr = harmonic_reconstruct(self.h, self.region)
        np.testing.assert_array_equal(st, shr)

    def test_same_entity_translation(self):
        previous_region = self.region - 17
        reconstructed = self.rng.normal(size=(4, 12)).astype(np.float32)
        history = EntityHistory(previous_region, reconstructed, centroid(previous_region))
        prior, meta = translated_prior(self.region, history)
        self.assertIsNone(meta["fallback_reason"])
        np.testing.assert_array_equal(prior, reconstructed)

    def test_large_motion_falls_back(self):
        previous_region = np.array([0, 1, 16, 17])
        history = EntityHistory(previous_region, self.h[previous_region], centroid(previous_region))
        prior, meta = translated_prior(self.region, history)
        self.assertIsNone(prior)
        self.assertEqual(meta["fallback_reason"], "displacement")

    def test_temporal_term_pulls_toward_prior(self):
        shr = harmonic_reconstruct(self.h, self.region)
        prior = np.full_like(shr, 5.0)
        st = harmonic_reconstruct(self.h, self.region, beta=4.0, prior=prior)
        self.assertLess(np.linalg.norm(st - prior), np.linalg.norm(shr - prior))


if __name__ == "__main__":
    unittest.main()
