"""Optimizadores (Clase 12.1), modos de entrenamiento (Clase 11, diap. 53) y Powell.

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import unittest

import numpy as np

from src.nn.mlp import MLP
from src.nn.optimizadores import GD, Adam, EtaAdaptativo, Momentum, RMSProp, crear, lotes
from src.nn.perdidas import error_cuadratico
from src.nn.scipy_adapter import minimizar_powell
from tests.test_mlp import X_XOR, ZETA_XOR

# Cuadrática conocida: E(w) = ½ Σ c_i (w_i − w*_i)², con mínimo E = 0 en w = w*
CURVATURAS = np.array([[1.0, 4.0], [0.5, 2.0]])
W_OPTIMO = np.array([[1.0, -2.0], [0.5, 3.0]])

CONFIGS = [
    {'nombre': 'gd', 'eta': 0.1},
    {'nombre': 'momentum', 'eta': 0.05, 'alfa': 0.9},
    {'nombre': 'rmsprop', 'eta': 0.05, 'gamma': 0.9, 'epsilon': 1e-8},
    {'nombre': 'adam', 'eta': 0.05},
]


def cuadratica(w: np.ndarray) -> tuple[float, np.ndarray]:
    return 0.5 * float(np.sum(CURVATURAS * (w - W_OPTIMO) ** 2)), CURVATURAS * (w - W_OPTIMO)


class CuadraticaTests(unittest.TestCase):
    def test_cada_optimizador_baja_la_perdida(self) -> None:
        for config in CONFIGS:
            with self.subTest(optimizador=config['nombre']):
                optimizador = crear(**config)
                w = np.zeros((2, 2))
                inicial, _ = cuadratica(w)
                for _ in range(500):
                    _, gradiente = cuadratica(w)
                    optimizador.paso([w], [gradiente])
                final, _ = cuadratica(w)
                self.assertLess(final, 1e-3 * inicial)
                np.testing.assert_allclose(w, W_OPTIMO, atol=0.1)

    def test_optimizador_desconocido(self) -> None:
        with self.assertRaises(ValueError):
            crear('lbfgs', eta=0.1)


class FormulasTests(unittest.TestCase):
    """Primeros pasos calculados a mano con las fórmulas de las diapositivas."""

    def setUp(self) -> None:
        self.g = np.array([2.0, -0.5])

    def test_gd(self) -> None:
        w = np.zeros(2)
        GD(eta=0.1).paso([w], [self.g])
        np.testing.assert_allclose(w, -0.1 * self.g)

    def test_momentum(self) -> None:
        w = np.zeros(2)
        optimizador = Momentum(eta=0.1, alfa=0.9)
        optimizador.paso([w], [self.g])            # Δw(1) = −ηg
        optimizador.paso([w], [self.g])            # Δw(2) = −ηg + α Δw(1)
        np.testing.assert_allclose(w, -0.1 * self.g + (-0.1 * self.g + 0.9 * -0.1 * self.g))

    def test_rmsprop_epsilon_dentro_de_la_raiz(self) -> None:
        w = np.zeros(2)
        RMSProp(eta=0.1, gamma=0.9, epsilon=0.01).paso([w], [self.g])
        S = 0.1 * self.g ** 2
        np.testing.assert_allclose(w, -0.1 * self.g / np.sqrt(S + 0.01))
        self.assertFalse(np.allclose(w, -0.1 * self.g / (np.sqrt(S) + 0.01)))

    def test_adam_con_correccion_de_sesgo(self) -> None:
        # En t = 1: m̂ = g y v̂ = g², así que el paso es −α g / (|g| + ϵ)
        w = np.zeros(2)
        Adam(eta=0.001, epsilon=0.01).paso([w], [self.g])
        np.testing.assert_allclose(w, -0.001 * self.g / (np.abs(self.g) + 0.01))


class EtaAdaptativoTests(unittest.TestCase):
    def test_sube_despues_de_k_bajadas_y_baja_despues_de_k_prima_subidas(self) -> None:
        optimizador = GD(eta=0.1)
        adaptativo = EtaAdaptativo(optimizador, a=0.01, b=0.5, k=3, k_prima=2)
        etas = []
        for error in [10.0, 9.0, 8.0, 7.0, 6.0, 6.5, 7.0, 7.0, 8.0]:
            adaptativo.fin_de_epoca(error)
            etas.append(round(optimizador.eta, 6))
        #       1ª    baja  baja  baja→+a  baja  sube  sube→−bη  igual  sube
        self.assertEqual(etas, [0.1, 0.1, 0.1, 0.11, 0.11, 0.11, 0.055, 0.055, 0.055])

    def test_k_invalido(self) -> None:
        with self.assertRaises(ValueError):
            EtaAdaptativo(GD(eta=0.1), a=0.01, b=0.5, k=0, k_prima=1)


class LotesTests(unittest.TestCase):
    def test_batch_online_y_minibatch(self) -> None:
        p = 32
        batch = list(lotes(p, p, np.random.default_rng(0)))
        self.assertEqual(len(batch), 1)
        np.testing.assert_array_equal(batch[0], np.arange(p))
        online = list(lotes(p, 1, np.random.default_rng(0)))
        self.assertEqual([len(lote) for lote in online], [1] * p)
        minibatch = list(lotes(p, 10, np.random.default_rng(0)))
        self.assertEqual([len(lote) for lote in minibatch], [10, 10, 10, 2])
        # Cada patrón aparece exactamente una vez por época
        for particion in (online, minibatch):
            np.testing.assert_array_equal(np.sort(np.concatenate(particion)), np.arange(p))

    def test_misma_semilla_mismo_orden(self) -> None:
        a = np.concatenate(list(lotes(32, 4, np.random.default_rng(3))))
        b = np.concatenate(list(lotes(32, 4, np.random.default_rng(3))))
        np.testing.assert_array_equal(a, b)

    def test_tamano_invalido(self) -> None:
        with self.assertRaises(ValueError):
            list(lotes(32, 33, np.random.default_rng(0)))


class EnElMlpTests(unittest.TestCase):
    """Los optimizadores y Powell sobre una red real: XOR con [2, 2, 1]."""

    def red(self) -> MLP:
        return MLP([2, 2, 1], ['tanh', 'tanh'], semilla=0, amplitud_inicial=0.5)

    def test_cada_optimizador_baja_el_error_de_xor(self) -> None:
        for config in CONFIGS:
            with self.subTest(optimizador=config['nombre']):
                red, optimizador = self.red(), crear(**config)
                inicial, _ = red.gradientes(X_XOR, ZETA_XOR)
                for _ in range(1000):
                    _, gradientes = red.gradientes(X_XOR, ZETA_XOR)
                    optimizador.paso(red.params, gradientes)
                final, _ = red.gradientes(X_XOR, ZETA_XOR)
                self.assertLess(final, inicial)

    def test_powell_baja_el_error_y_deja_los_pesos_en_la_red(self) -> None:
        red = self.red()
        perdida = lambda: error_cuadratico(red.forward(X_XOR), ZETA_XOR)[0]
        resultado = minimizar_powell([red], perdida, max_evaluaciones=5000)
        self.assertLess(resultado.perdida_final, resultado.perdida_inicial)
        self.assertAlmostEqual(perdida(), resultado.perdida_final)
        self.assertGreater(resultado.evaluaciones, red.n_params())
        self.assertGreater(resultado.segundos, 0.0)

    def test_powell_con_dos_redes(self) -> None:
        encoder = MLP([2, 2], ['tanh'], semilla=1, amplitud_inicial=0.5)
        decoder = MLP([2, 1], ['tanh'], semilla=2, amplitud_inicial=0.5)
        perdida = lambda: error_cuadratico(decoder.forward(encoder.forward(X_XOR)), ZETA_XOR)[0]
        resultado = minimizar_powell([encoder, decoder], perdida, max_evaluaciones=5000)
        self.assertLess(resultado.perdida_final, resultado.perdida_inicial)
        self.assertAlmostEqual(perdida(), resultado.perdida_final)


if __name__ == '__main__':
    unittest.main()
