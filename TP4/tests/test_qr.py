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

    def test_correlated_qr_interference_and_projection_control(self):
        hebb = Hopfield().fit(self.patterns)
        projection = Hopfield().fit(self.patterns, rule="pseudoinverse")
        self.assertFalse(any(hebb.is_fixed_point(p) for p in self.patterns))
        self.assertTrue(all(projection.is_fixed_point(p) for p in self.patterns))
        self.assertTrue(all(Hopfield().fit([p]).is_fixed_point(p) for p in self.patterns))


if __name__ == "__main__":
    unittest.main()
