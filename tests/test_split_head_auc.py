"""Numerical controls for the isolated frozen-readout experiment."""
import unittest
import numpy as np
from scipy.optimize import check_grad
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from analyze_split_head_auc import (pair_auc, match_objective, match_fit,
                                    linear_fit, screen)


class FrozenReadoutTests(unittest.TestCase):
    def test_pair_decomposition_with_ties(self):
        y = np.array([0, 1, 0, 1, 0, 1])
        z = np.array([0, 0, 1, 2, 2, 1.])
        r = np.array([0, 0, 0, 1, 1, 2])
        out = pair_auc(y, z, r)
        same, cross = [], []
        for i in np.flatnonzero(y == 1):
            for j in np.flatnonzero(y == 0):
                (same if r[i] == r[j] else cross).append(float(z[i] > z[j]) + .5*(z[i] == z[j]))
        self.assertEqual(out['same_pairs'], len(same))
        self.assertEqual(out['cross_pairs'], len(cross))
        self.assertAlmostEqual(out['same_auc'], np.mean(same))
        self.assertAlmostEqual(out['cross_auc'], np.mean(cross))
        self.assertAlmostEqual(out['auc'], roc_auc_score(y, z))
        self.assertIsNone(pair_auc(np.ones(3), np.arange(3), np.zeros(3))['auc'])
        self.assertIsNone(pair_auc(y, z, np.arange(len(y)))['same_auc'])

    def test_match_analytic_gradient(self):
        rng = np.random.default_rng(12)
        c, d = rng.normal(size=(20, 4)), rng.normal(size=(20, 4))
        y, t = rng.integers(0, 2, size=20), rng.normal(size=9)
        error = check_grad(lambda x: match_objective(x, c, d, y, .7)[0],
                           lambda x: match_objective(x, c, d, y, .7)[1], t)
        self.assertLess(error, 1e-5)

    def test_shared_is_feasible_with_identical_penalty(self):
        rng = np.random.default_rng(4)
        c, d = rng.normal(size=(80, 3)), rng.normal(size=(80, 3)) * .2
        y = rng.integers(0, 2, size=80)
        with threadpool_limits(limits=1):
            shared = linear_fit(c+d, y, 2., 1., 1000)
            free = linear_fit(np.c_[c, d], y, 2., .5, 1000)
            t = np.r_[shared['w'], shared['w'], shared['b']]
            self.assertAlmostEqual(match_objective(t, c, d, y, 2.)[0], shared['objective'], places=9)
            match = match_fit(c, d, y, 2., shared, free, 1000)
        self.assertLess(match['norm_error'], 1e-10)
        self.assertLessEqual(match['objective'], shared['objective'] + 1e-7)
        self.assertLessEqual(free['objective'], match['objective'] + 1e-6)

    def test_selection_does_not_use_test_labels(self):
        rng = np.random.default_rng(33)
        cache = dict(gamma=np.array(2.))
        for name in ('train', 'val', 'test'):
            c, d = rng.normal(size=(30, 3)), rng.normal(size=(30, 3))*.3
            cache.update({f'{name}_c': c, f'{name}_d': d, f'{name}_q': c+d,
                          f'{name}_y': np.tile([0, 1], 15), f'{name}_region': np.arange(30) % 3,
                          f'{name}_logits': c[:, 0]+d[:, 1], f'{name}_region_logits': c[:, 0]})
        with threadpool_limits(limits=1):
            a = screen(cache, grid=(.01, 1.), maxiter=500)
            cache['test_y'] = 1-cache['test_y']
            b = screen(cache, grid=(.01, 1.), maxiter=500)
        for arm in a['selected']:
            self.assertEqual(a['selected'][arm]['C'], b['selected'][arm]['C'])
            np.testing.assert_array_equal(a['selected'][arm]['w'], b['selected'][arm]['w'])


if __name__ == '__main__':
    unittest.main()
