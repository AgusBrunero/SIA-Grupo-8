"""Activaciones (Clase 10.2) y pérdidas (Clase 11; Autoencoders diap. 51): valores y gradient check.

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import unittest

import numpy as np

from src.nn.activaciones import NOMBRES, Activacion
from src.nn.perdidas import PERDIDAS, bce, error_cuadratico
from tests.gradcheck import TOLERANCIA, error_relativo, gradiente_numerico

BETAS = (0.5, 1.0, 2.0)


class ActivacionesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.h = np.random.default_rng(0).uniform(-2.0, 2.0, size=(4, 3))

    def test_valores_conocidos(self) -> None:
        cero = np.zeros(1)
        self.assertEqual(Activacion('identidad')(np.array([1.7]))[0], 1.7)
        self.assertEqual(Activacion('tanh', 2.0)(cero)[0], 0.0)
        self.assertEqual(Activacion('logistica', 2.0)(cero)[0], 0.5)
        # La logística de la cátedra tiene exponente −2βh: θ(h) = (1 + tanh(βh)) / 2
        for beta in BETAS:
            np.testing.assert_allclose(Activacion('logistica', beta)(self.h),
                                       (1.0 + np.tanh(beta * self.h)) / 2.0, rtol=1e-12)

    def test_imagenes(self) -> None:
        h = np.linspace(-50.0, 50.0, 201)
        tanh = Activacion('tanh')(h)
        logistica = Activacion('logistica')(h)
        self.assertTrue(np.all((tanh >= -1.0) & (tanh <= 1.0)))
        # Estrictamente dentro de (0, 1), también saturada: lo necesita el log de la BCE
        self.assertTrue(np.all((logistica > 0.0) & (logistica < 1.0)))

    def test_gradient_check_de_cada_activacion(self) -> None:
        for nombre in NOMBRES:
            for beta in BETAS:
                with self.subTest(activacion=nombre, beta=beta):
                    activacion = Activacion(nombre, beta)
                    h = self.h.copy()
                    # d/dh de Σ θ(h) es θ'(h) elemento a elemento
                    numerico = gradiente_numerico(lambda: float(np.sum(activacion(h))), h)
                    self.assertLess(error_relativo(activacion.derivada(h), numerico), TOLERANCIA)

    def test_nombre_desconocido(self) -> None:
        with self.assertRaises(ValueError):
            Activacion('relu')


class PerdidasTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(1)
        self.O = rng.uniform(0.05, 0.95, size=(5, 4))
        self.zeta = rng.integers(0, 2, size=(5, 4)).astype(np.float64)

    def test_valores_conocidos(self) -> None:
        O = np.array([[0.5, 0.0], [1.0, 1.0]])
        zeta = np.array([[1.0, 0.0], [1.0, 0.0]])
        # ½ (0,5² + 0 + 0 + 1²) = 0,625
        self.assertAlmostEqual(error_cuadratico(O, zeta)[0], 0.625)
        self.assertAlmostEqual(error_cuadratico(O, zeta, 'media')[0], 0.625 / 2)
        # Un solo píxel con O = 0,5: −ln 0,5
        self.assertAlmostEqual(bce(np.array([[0.5]]), np.array([[1.0]]))[0], np.log(2.0))
        # Reconstrucción perfecta da pérdida 0
        self.assertEqual(error_cuadratico(zeta, zeta)[0], 0.0)

    def test_gradient_check_de_cada_perdida(self) -> None:
        for nombre, perdida in PERDIDAS.items():
            for reduccion in ('suma', 'media'):
                with self.subTest(perdida=nombre, reduccion=reduccion):
                    O = self.O.copy()
                    _, analitico = perdida(O, self.zeta, reduccion)
                    numerico = gradiente_numerico(lambda: perdida(O, self.zeta, reduccion)[0], O)
                    self.assertLess(error_relativo(analitico, numerico), TOLERANCIA)

    def test_media_es_suma_dividida_por_p(self) -> None:
        p = self.O.shape[0]
        for nombre, perdida in PERDIDAS.items():
            with self.subTest(perdida=nombre):
                valor_suma, grad_suma = perdida(self.O, self.zeta, 'suma')
                valor_media, grad_media = perdida(self.O, self.zeta, 'media')
                self.assertAlmostEqual(valor_media, valor_suma / p)
                np.testing.assert_allclose(grad_media, grad_suma / p)

    def test_delta_de_salida_con_bce_y_logistica(self) -> None:
        """Con BCE sumada y salida logística, ∂E/∂h = 2β(O − ζ): θ' se cancela.
        Es derivación nuestra (doc de cátedra §3), no está en las diapositivas."""
        h = np.random.default_rng(2).uniform(-2.0, 2.0, size=(5, 4))
        for beta in BETAS:
            with self.subTest(beta=beta):
                activacion = Activacion('logistica', beta)
                O = activacion(h)
                _, dE_dO = bce(O, self.zeta)
                np.testing.assert_allclose(dE_dO * activacion.derivada(h),
                                           2.0 * beta * (O - self.zeta), rtol=1e-9)

    def test_bce_saturada_es_finita(self) -> None:
        h = np.array([[80.0, -80.0]])
        O = Activacion('logistica')(h)
        for zeta in (np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]])):
            valor, gradiente = bce(O, zeta)
            self.assertTrue(np.isfinite(valor))
            self.assertTrue(np.all(np.isfinite(gradiente)))

    def test_reduccion_desconocida(self) -> None:
        with self.assertRaises(ValueError):
            error_cuadratico(self.O, self.zeta, 'promedio')


if __name__ == '__main__':
    unittest.main()
