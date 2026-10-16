"""Integración de QR; requiere las dependencias opcionales de requirements-qr.txt."""
import importlib.util
import unittest

import numpy as np

from models.hopfield import Hopfield


@unittest.skipUnless(importlib.util.find_spec("zxingcpp"), "Instalar requirements-qr.txt")
class QRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from qr import cargar
        cls.names, cls.patterns = cargar()

    def test_originals_decode_after_padding_and_scaling(self):
        from qr import imagen_lector, leer
        for name, pattern in zip(self.names, self.patterns):
            image = imagen_lector(pattern)
            self.assertEqual(image.shape, (360, 360))
            self.assertTrue(np.all(image[:32] == 255))
            texts = leer(pattern)
            self.assertEqual(len(texts), 1)
            self.assertTrue(texts[0].endswith(f"/{name}_menu.png"))

    def test_exact_random_noise_and_reproducibility(self):
        from qr import deteriorar
        pattern = self.patterns[0]
        noisy, mask = deteriorar(pattern, "aleatorio", 0.30, 7)
        self.assertEqual(int(mask.sum()), 411)
        self.assertEqual(int((pattern != noisy).sum()), 411)
        np.testing.assert_array_equal(deteriorar(pattern, "aleatorio", 0.30, 7)[0], noisy)

    def test_structured_damage_distinguishes_area_from_changed_bits(self):
        from qr import deteriorar
        for kind, fill, area in [("esquina_blanca", 1, 81),
                                 ("mancha_negra", -1, 81), ("raya_blanca", 1, 333)]:
            noisy, mask = deteriorar(self.patterns[0], kind, 9, 4)
            self.assertEqual(int(mask.sum()), area)
            np.testing.assert_array_equal(noisy[~mask], self.patterns[0][~mask])
            self.assertTrue(np.all(noisy[mask] == fill))
            self.assertLess(int((noisy != self.patterns[0]).sum()), area)

    def test_wear_only_fades_black_and_is_reproducible(self):
        from qr import deteriorar
        pattern = self.patterns[0]
        black = pattern == -1
        for kind in ["desgaste", "desgaste_disperso"]:
            worn, mask = deteriorar(pattern, kind, 0.30, 11)
            self.assertTrue(np.all(black[mask]))
            self.assertTrue(np.all(worn[mask] == 1))
            np.testing.assert_array_equal(worn[~mask], pattern[~mask])
            self.assertEqual(int(mask.sum()), round(0.30 * black.sum()))
            self.assertFalse(np.any((pattern == 1) & (worn == -1)))
            np.testing.assert_array_equal(deteriorar(pattern, kind, 0.30, 11)[0], worn)

    def test_grouped_wear_is_spatially_clustered(self):
        from qr import desgaste

        def faded_neighbours(mask):
            grid = mask.reshape(37, 37)
            padded = np.pad(grid, 1)
            neighbours = (padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:])
            return (grid & neighbours).sum() / grid.sum()

        grouped = np.mean([faded_neighbours(desgaste(self.patterns[0], 0.3, s)[1]) for s in range(20)])
        scattered = np.mean([faded_neighbours(desgaste(self.patterns[0], 0.3, s, agrupado=False)[1])
                             for s in range(20)])
        self.assertGreater(grouped, scattered + 0.1)

    def test_correlated_qr_interference_and_projection_control(self):
        hebb = Hopfield().fit(self.patterns)
        projection = Hopfield().fit(self.patterns, rule="pseudoinverse")
        self.assertFalse(any(hebb.is_fixed_point(p) for p in self.patterns))
        self.assertTrue(all(projection.is_fixed_point(p) for p in self.patterns))
        self.assertTrue(all(Hopfield().fit([p]).is_fixed_point(p) for p in self.patterns))


if __name__ == "__main__":
    unittest.main()
