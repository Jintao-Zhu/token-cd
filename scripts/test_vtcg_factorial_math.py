"""CPU-only standalone unittest; imports NumPy and the pure math helper only."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from research.semantic_token_cd import vtcg_factorial_math as m


class MathTests(unittest.TestCase):
    def test_attention_ties_and_mask(self):
        scores = np.ones(256)
        order = m.attention_ranking(scores)
        np.testing.assert_array_equal(order, np.arange(256))
        np.testing.assert_array_equal(m.select_top_count(order[::-1], 3), [253,254,255])
        self.assertEqual(m.top_p_count(scores, .5), 128)
        self.assertEqual(m.top_p_count(scores, .5, 1, 64), 64)
        self.assertEqual(m.top_p_count(scores*7, .5), 128)
        peaked = np.zeros(256); peaked[8] = 1
        self.assertEqual(m.top_p_count(peaked, .99), 1)

    def test_budget_validation(self):
        self.assertEqual(m.fixed_count(32), 32)
        self.assertEqual(m.matched_count([1, 7, 99]), 3)
        for bad in [0, 257, True, 1.5, '32']:
            with self.assertRaises(ValueError): m.fixed_count(bad)
        for bad in [[], [1,1], [-1], [256], [1.0]]:
            with self.assertRaises(ValueError): m.matched_count(bad)
        for bad in [0., 1., np.nan, True]:
            with self.assertRaises(ValueError): m.top_p_count(np.ones(256), bad)
        with self.assertRaises(ValueError): m.top_p_count(np.ones(256), .8, 64, 16)
        with self.assertRaises(ValueError): m.attention_ranking(np.zeros(256))
        with self.assertRaises(ValueError): m.attention_ranking(-np.ones(256))
        with self.assertRaises(ValueError): m.attention_ranking(np.full(256,np.nan))
        with self.assertRaises(ValueError): m.validate_ranking(np.arange(255))
        with self.assertRaises(ValueError): m.validate_ranking(np.zeros(256,dtype=int))

    def test_cluster_order_and_label_invariance(self):
        f = np.zeros((256,2)); f[:128,0]=1; f[128:,1]=1
        labels = np.repeat([0,1],128)
        order = m.kmeans_full_ranking(f,labels,[[0,1]])
        np.testing.assert_array_equal(order,np.r_[np.arange(128,256),np.arange(128)])
        np.testing.assert_array_equal(order,m.kmeans_full_ranking(f,9-labels,[[0,1]]))
        # Equal cluster scores resolve by min token id, not arbitrary cluster label.
        np.testing.assert_array_equal(m.kmeans_full_ranking(f,labels,[[1,1]]),np.arange(256))

    def test_within_cluster_cosine(self):
        f = np.tile([1.,0.],(256,1)); f[0]=[1,1]; f[1]=[1,-1]
        order=m.kmeans_full_ranking(f,np.zeros(256,dtype=int),[[1,0]])
        np.testing.assert_array_equal(order,np.r_[np.arange(2,256),[0,1]])
        self.assertEqual(len(np.unique(order)),256)

    def test_cosine_errors(self):
        f=np.ones((256,3)); labels=np.zeros(256,dtype=int)
        for features, labs, entities in [
            (f[:255],labels,[[1,1,1]]),(f,labels.astype(float),[[1,1,1]]),
            (f,labels,[[1,1]]),(f,labels,[[0,0,0]]),
            (np.zeros((256,3)),labels,[[1,1,1]])]:
            with self.assertRaises(ValueError): m.kmeans_full_ranking(features,labs,entities)

    def test_noise_norms_determinism_mask_and_zero(self):
        f=np.random.default_rng(1).normal(size=(256,16)).astype(np.float32)
        ids=np.array([0,8,77,255]); harmonic=f[ids].astype(float).copy()
        harmonic[1:]+=np.random.default_rng(2).normal(size=(3,16))
        seed=m.deterministic_seed('task',100,4,'norm_noise')
        a,audit=m.norm_matched_random_replacement(f,ids,harmonic,seed)
        b,_=m.norm_matched_random_replacement(f,ids,harmonic,seed)
        c,_=m.norm_matched_random_replacement(f,ids,harmonic,seed+1)
        np.testing.assert_array_equal(a,b)
        self.assertFalse(np.array_equal(a[1:],c[1:]))
        np.testing.assert_array_equal(a[0],f[ids[0]])
        np.testing.assert_allclose(np.linalg.norm(a-f[ids],axis=1),np.linalg.norm(harmonic-f[ids],axis=1),atol=1e-12)
        self.assertTrue(audit['non_target_bit_identical'])
        self.assertEqual(audit['zero_residual_count'],1)
        self.assertNotEqual(seed,m.deterministic_seed('task',100,5,'norm_noise'))
        np.testing.assert_array_equal(f,np.random.default_rng(1).normal(size=(256,16)).astype(np.float32))

    def test_noise_bad_input(self):
        f=np.ones((256,4))
        for ids,harmonic,seed in [([2,1],np.ones((2,4)),1),([1,1],np.ones((2,4)),1),
                                 ([1],np.ones((2,4)),1),([1],np.ones((1,3)),1),
                                 ([1],np.ones((1,4)),-1),([1],np.full((1,4),np.inf),1)]:
            with self.assertRaises(ValueError): m.norm_matched_random_replacement(f,ids,harmonic,seed)
        with self.assertRaises(ValueError): m.deterministic_seed(object())


if __name__ == '__main__':
    unittest.main(verbosity=2)
