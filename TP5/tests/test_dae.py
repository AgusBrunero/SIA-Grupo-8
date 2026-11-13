"""Ruido salt-and-pepper y Denoising Autoencoder (Autoencoders diap. 22).

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import unittest

import numpy as np

from src.data.font import cargar_font
from src.data.ruido import sal_y_pimienta
from src.modelos.autoencoder import Autoencoder, entrenar
from src.modelos.dae import corruptor, evaluar_con_ruido
from src.nn.optimizadores import Adam


class RuidoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.X, _ = cargar_font()

    def test_extremos(self) -> None:
        rng = np.random.default_rng(0)
        np.testing.assert_array_equal(sal_y_pimienta(self.X, 0.0, rng), self.X)
        np.testing.assert_array_equal(sal_y_pimienta(self.X, 1.0, rng), 1.0 - self.X)

    def test_salida_binaria_y_entrada_intacta(self) -> None:
        original = self.X.copy()
        ruidosa = sal_y_pimienta(self.X, 0.3, np.random.default_rng(0))
        self.assertTrue(np.all((ruidosa == 0.0) | (ruidosa == 1.0)))
        np.testing.assert_array_equal(self.X, original)

    def test_fraccion_de_pixeles_invertidos(self) -> None:
        grande = np.tile(self.X, (200, 1))              # 224.000 píxeles
        for p in (0.05, 0.1, 0.2, 0.3):
            with self.subTest(p=p):
                invertidos = (sal_y_pimienta(grande, p, np.random.default_rng(1)) != grande).mean()
                self.assertAlmostEqual(invertidos, p, delta=0.005)

    def test_invierte_en_los_dos_sentidos(self) -> None:
        ruidosa = sal_y_pimienta(self.X, 0.3, np.random.default_rng(2))
        self.assertTrue(np.any((self.X == 0) & (ruidosa == 1)))      # sal
        self.assertTrue(np.any((self.X == 1) & (ruidosa == 0)))      # pimienta

    def test_semilla(self) -> None:
        a = sal_y_pimienta(self.X, 0.2, np.random.default_rng(5))
        b = sal_y_pimienta(self.X, 0.2, np.random.default_rng(5))
        c = sal_y_pimienta(self.X, 0.2, np.random.default_rng(6))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))

    def test_p_invalido(self) -> None:
        with self.assertRaises(ValueError):
            sal_y_pimienta(self.X, 1.5, np.random.default_rng(0))


class DaeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.X, _ = cargar_font()

    def autoencoder(self) -> Autoencoder:
        return Autoencoder([35, 20, 10], ['tanh', 'identidad'], 'logistica', semilla=0)

    def test_el_ruido_cambia_en_cada_epoca_y_la_salida_esperada_no(self) -> None:
        entradas = []

        def espia(X: np.ndarray) -> np.ndarray:
            ruidosa = corruptor(0.2, rng_ruido)(X)
            entradas.append(ruidosa)
            return ruidosa

        rng_ruido = np.random.default_rng(1)
        original = self.X.copy()
        entrenar(self.autoencoder(), self.X, self.X, Adam(), epocas=3, tamano_lote=32,
                 rng=np.random.default_rng(0), perdida='bce', corromper=espia)
        self.assertEqual(len(entradas), 3)
        self.assertFalse(np.array_equal(entradas[0], entradas[1]))
        self.assertFalse(np.array_equal(entradas[0], self.X))
        np.testing.assert_array_equal(self.X, original)

    def test_sin_corromper_es_el_autoencoder_comun(self) -> None:
        errores = []
        for corromper in (None, corruptor(0.0, np.random.default_rng(9))):
            historial = entrenar(self.autoencoder(), self.X, self.X, Adam(), epocas=20, tamano_lote=32,
                                 rng=np.random.default_rng(0), perdida='bce', corromper=corromper)
            errores.append(historial.error)
        self.assertEqual(errores[0], errores[1])

    def test_evaluacion(self) -> None:
        ae = self.autoencoder()
        tabla = evaluar_con_ruido(ae, self.X, 0.1, realizaciones=20, rng=np.random.default_rng(3))
        self.assertEqual(len(tabla), 32 * 20)
        self.assertEqual(tabla.groupby('patron').size().tolist(), [20] * 32)
        self.assertTrue(tabla.alterados.between(0, 35).all() and tabla.mal.between(0, 35).all())
        self.assertAlmostEqual(tabla.alterados.mean(), 3.5, delta=0.4)           # 35 píxeles × 0,1
        sin_ruido = evaluar_con_ruido(ae, self.X, 0.0, realizaciones=2, rng=np.random.default_rng(3))
        self.assertEqual(int(sin_ruido.alterados.sum()), 0)


if __name__ == '__main__':
    unittest.main()
