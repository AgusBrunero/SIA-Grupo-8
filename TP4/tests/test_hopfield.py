"""Invariantes de la dinámica y casos exactos; ejecutar con unittest discover."""
import unittest

import numpy as np

from models.hopfield import Hopfield


class HopfieldTests(unittest.TestCase):
    def test_hebb_symmetric_zero_diagonal(self):
        memories = np.array([[1, 1, -1, -1], [1, -1, 1, -1]])
        w = Hopfield().fit(memories).weights
        np.testing.assert_allclose(w, w.T)
        np.testing.assert_array_equal(np.diag(w), 0)
        self.assertAlmostEqual(w[0, 3], -0.5)

    def test_single_memory_corrects_minority_errors(self):
        memory = np.array([1, -1, 1, -1, 1, -1, 1])
        model = Hopfield().fit([memory])
        noisy = memory.copy()
        noisy[[0, 4]] *= -1
        for seed in range(5):
            result = model.recall(noisy, seed=seed)
            np.testing.assert_array_equal(result.state, memory)
            self.assertTrue(result.converged)

    def test_energy_trace_matches_full_energy_and_never_increases(self):
        rng = np.random.default_rng(4)
        model = Hopfield().fit(rng.choice([-1, 1], (4, 25)))
        result = model.recall(rng.choice([-1, 1], 25), seed=8)
        self.assertTrue(np.all(np.diff(result.energies) <= 1e-10))
        for sweep, state in enumerate(result.states):
            self.assertAlmostEqual(result.energies[sweep * 25], model.energy(state))
        self.assertTrue(result.converged)
        self.assertTrue(model.is_fixed_point(result.state))
        self.assertEqual(len(result.energies), 1 + 25 * result.sweeps)

    def test_ties_preserve_state(self):
        model = Hopfield().fit([[1, 1], [1, -1]])
        result = model.recall([-1, 1])
        np.testing.assert_array_equal(result.state, [-1, 1])
        self.assertEqual(result.sweeps, 1)
        self.assertTrue(result.converged)

    def test_inverse_memory_is_also_fixed(self):
        pattern = np.array([1, -1, -1, 1, 1])
        model = Hopfield().fit([pattern])
        self.assertTrue(model.is_fixed_point(pattern))
        self.assertTrue(model.is_fixed_point(-pattern))

    def test_validation(self):
        for invalid in ([], [1, -1], [[0, 1]], [[np.nan, 1]]):
            with self.assertRaises(ValueError):
                Hopfield().fit(invalid)
        with self.assertRaises(RuntimeError):
            Hopfield().recall([1, -1])
        model = Hopfield().fit([[1, -1]])
        for invalid in ([1], [1, 0], [[1, -1]]):
            with self.assertRaises(ValueError):
                model.recall(invalid)
        with self.assertRaises(ValueError):
            model.recall([1, -1], max_sweeps=0)


if __name__ == "__main__":
    unittest.main()
