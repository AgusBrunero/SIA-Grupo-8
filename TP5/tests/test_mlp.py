"""MLP (Clase 11): gradient check, XOR, semillas, lectura de capas y encadenado de redes.

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import unittest

import numpy as np

from src.nn.mlp import MLP
from src.nn.perdidas import PERDIDAS
from tests.gradcheck import TOLERANCIA, error_relativo, gradiente_numerico

X_XOR = np.array([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
ZETA_XOR = np.array([[-1.0], [1.0], [1.0], [-1.0]])
SEMILLAS_XOR = range(10)
MINIMO_SEMILLAS_XOR = 8          # de 10


def entrenar_xor(capas: list[int], semilla: int, eta: float = 0.1, epocas: int = 5000) -> bool:
    """Descenso por gradiente en batch (Clase 11, diap. 54), tanh y salidas en {−1, 1}."""
    red = MLP(capas, ['tanh'] * (len(capas) - 1), semilla=semilla, amplitud_inicial=0.5)
    for _ in range(epocas):
        _, gradientes = red.gradientes(X_XOR, ZETA_XOR)
        for W, g in zip(red.params, gradientes):
            W -= eta * g
    return bool(np.array_equal(np.sign(red.forward(X_XOR)), ZETA_XOR))


class GradientCheckTests(unittest.TestCase):
    """MLP de 3 capas con pesos, una activación distinta por capa."""

    def setUp(self) -> None:
        rng = np.random.default_rng(0)
        self.X = rng.uniform(-1.0, 1.0, size=(6, 5))
        self.zeta = rng.integers(0, 2, size=(6, 3)).astype(np.float64)

    def red(self, salida: str) -> MLP:
        return MLP([5, 4, 2, 3], ['tanh', 'identidad', salida], semilla=3, beta=0.8, amplitud_inicial=0.7)

    def test_pesos_con_cada_perdida(self) -> None:
        casos = [('error_cuadratico', 'logistica'), ('error_cuadratico', 'tanh'),
                 ('error_cuadratico', 'identidad'), ('bce', 'logistica')]
        for perdida, salida in casos:
            for reduccion in ('suma', 'media'):
                red = self.red(salida)
                _, analiticos = red.gradientes(self.X, self.zeta, perdida, reduccion)
                for m, W in enumerate(red.params):
                    with self.subTest(perdida=perdida, salida=salida, reduccion=reduccion, capa=m + 1):
                        numerico = gradiente_numerico(
                            lambda: PERDIDAS[perdida](red.forward(self.X), self.zeta, reduccion)[0], W)
                        self.assertLess(error_relativo(analiticos[m], numerico), TOLERANCIA)

    def test_gradiente_externo_sobre_la_salida(self) -> None:
        """backward desde un ∂E/∂O cualquiera: E = Σ c ⊙ O, con c fijo."""
        red = self.red('logistica')
        c = np.random.default_rng(5).normal(size=(6, 3))
        X = self.X.copy()
        red.forward(X)
        analiticos, dE_dX = red.backward(c)
        funcion = lambda: float(np.sum(c * red.forward(X)))
        for m, W in enumerate(red.params):
            with self.subTest(capa=m + 1):
                self.assertLess(error_relativo(analiticos[m], gradiente_numerico(funcion, W)), TOLERANCIA)
        with self.subTest(respecto_de='entrada'):
            self.assertLess(error_relativo(dE_dX, gradiente_numerico(funcion, X)), TOLERANCIA)

    def test_dos_redes_encadenadas(self) -> None:
        """Encoder y decoder como dos MLP: el ∂E/∂X del decoder es el ∂E/∂O del encoder."""
        encoder = MLP([5, 4, 2], ['tanh', 'identidad'], semilla=1, amplitud_inicial=0.7)
        decoder = MLP([2, 4, 5], ['tanh', 'logistica'], semilla=2, amplitud_inicial=0.7)
        zeta = (self.X > 0).astype(np.float64)

        def perdida() -> float:
            return PERDIDAS['bce'](decoder.forward(encoder.forward(self.X)), zeta, 'suma')[0]

        _, dE_dO = PERDIDAS['bce'](decoder.forward(encoder.forward(self.X)), zeta, 'suma')
        grad_decoder, dE_dZ = decoder.backward(dE_dO)
        grad_encoder, _ = encoder.backward(dE_dZ)
        for nombre, red, analiticos in (('encoder', encoder, grad_encoder), ('decoder', decoder, grad_decoder)):
            for m, W in enumerate(red.params):
                with self.subTest(red=nombre, capa=m + 1):
                    self.assertLess(error_relativo(analiticos[m], gradiente_numerico(perdida, W)), TOLERANCIA)


class XorTests(unittest.TestCase):
    """Ejercicio de validación del TP3 (CLAUDE.md §4)."""

    def test_xor_aprende(self) -> None:
        for capas in ([2, 2, 1], [2, 3, 2, 1]):
            with self.subTest(capas=capas):
                resueltas = sum(entrenar_xor(capas, semilla) for semilla in SEMILLAS_XOR)
                self.assertGreaterEqual(
                    resueltas, MINIMO_SEMILLAS_XOR,
                    f'{capas}: aprendió XOR en {resueltas}/{len(SEMILLAS_XOR)} semillas')


class EstructuraTests(unittest.TestCase):
    def test_misma_semilla_mismos_pesos(self) -> None:
        a = MLP([35, 10, 2], ['tanh', 'identidad'], semilla=7)
        b = MLP([35, 10, 2], ['tanh', 'identidad'], semilla=7)
        c = MLP([35, 10, 2], ['tanh', 'identidad'], semilla=8)
        for Wa, Wb in zip(a.params, b.params):
            np.testing.assert_array_equal(Wa, Wb)
        self.assertFalse(np.array_equal(a.params[0], c.params[0]))

    def test_pesos_iniciales_chicos_y_no_nulos(self) -> None:
        red = MLP([35, 10, 2], ['tanh', 'identidad'], semilla=0, amplitud_inicial=0.1)
        for W in red.params:
            self.assertEqual(W.dtype, np.float64)
            self.assertTrue(np.all(np.abs(W) < 0.1))
            self.assertTrue(np.all(W != 0.0))
        with self.assertRaises(ValueError):
            MLP([2, 2, 1], ['tanh', 'tanh'], semilla=0, amplitud_inicial=0.0)

    def test_bias_como_x0(self) -> None:
        """W^m tiene una fila más que neuronas la capa anterior; la fila 0 es el bias."""
        red = MLP([3, 4, 2], ['tanh', 'identidad'], semilla=0)
        self.assertEqual([W.shape for W in red.params], [(4, 4), (5, 2)])
        self.assertEqual(red.n_params(), 16 + 10)
        # Con entrada nula y pesos de entrada nulos, la primera capa da θ(w0)
        red.W[0][1:] = 0.0
        V1 = red.salidas_por_capa(np.zeros((1, 3)))[0]
        np.testing.assert_allclose(V1[0], np.tanh(red.W[0][0]))

    def test_lectura_de_cualquier_capa(self) -> None:
        red = MLP([35, 10, 2, 10, 35], ['tanh', 'identidad', 'tanh', 'logistica'], semilla=0)
        X = np.random.default_rng(0).integers(0, 2, size=(32, 35)).astype(np.float64)
        salidas = red.salidas_por_capa(X)
        self.assertEqual([V.shape for V in salidas], [(32, 10), (32, 2), (32, 10), (32, 35)])
        np.testing.assert_array_equal(salidas[-1], red.forward(X))

    def test_vector_ida_y_vuelta(self) -> None:
        red = MLP([3, 4, 2], ['tanh', 'identidad'], semilla=0)
        vector = red.vector()
        self.assertEqual(vector.shape, (red.n_params(),))
        X = np.ones((2, 3))
        salida = red.forward(X)
        red.cargar_vector(np.zeros_like(vector))
        self.assertTrue(np.all(red.forward(X) == 0.0))
        red.cargar_vector(vector)
        np.testing.assert_array_equal(red.forward(X), salida)
        with self.assertRaises(ValueError):
            red.cargar_vector(vector[:-1])

    def test_argumentos_invalidos(self) -> None:
        with self.assertRaises(ValueError):
            MLP([2, 2, 1], ['tanh'], semilla=0)
        with self.assertRaises(RuntimeError):
            MLP([2, 1], ['tanh'], semilla=0).backward(np.zeros((1, 1)))


if __name__ == '__main__':
    unittest.main()
