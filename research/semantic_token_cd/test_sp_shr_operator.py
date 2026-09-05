import unittest

import numpy as np

from research.semantic_token_cd.sp_shr_policy import (
    boundary_and_interior,
    deterministic_partial_region,
)


class SPSHROperatorTest(unittest.TestCase):
    def test_boundary_is_one_token_four_neighbor_ring(self):
        region = np.asarray([r * 16 + c for r in range(4, 8) for c in range(5, 9)])
        boundary, interior = boundary_and_interior(region)
        self.assertEqual(len(boundary), 12)
        self.assertEqual(set(interior), {5 * 16 + 6, 5 * 16 + 7, 6 * 16 + 6, 6 * 16 + 7})
        self.assertEqual(set(boundary) | set(interior), set(region))
        self.assertFalse(set(boundary) & set(interior))

    def test_thin_region_has_no_interior(self):
        boundary, interior = boundary_and_interior(np.asarray([17, 18, 33, 34]))
        self.assertEqual(len(boundary), 4)
        self.assertEqual(len(interior), 0)

    def test_partial_is_exact_deterministic_half(self):
        region = np.arange(30, 51)
        first, seed_a = deterministic_partial_region(region, 0.5, 2, 7, 3, 1)
        second, seed_b = deterministic_partial_region(region, 0.5, 2, 7, 3, 1)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(seed_a, seed_b)
        self.assertEqual(len(first), 11)
        self.assertTrue(set(first).issubset(set(region)))

    def test_partial_changes_across_replans(self):
        region = np.arange(40)
        first, _ = deterministic_partial_region(region, 0.5, 0, 9, 0, 0)
        second, _ = deterministic_partial_region(region, 0.5, 0, 9, 1, 0)
        self.assertFalse(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()
