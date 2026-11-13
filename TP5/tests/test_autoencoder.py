"""Autoencoder (Autoencoders diap. 3–4): estructura, gradient check y ciclo de entrenamiento.

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import unittest

import numpy as np

from src.data.font import a_bipolar, cargar_font
from src.modelos.autoencoder import Autoencoder, entrenar, evaluar
from src.nn.optimizadores import Adam, EtaAdaptativo, GD
from src.nn.perdidas import PERDIDAS
from tests.gradcheck import TOLERANCIA, error_relativo, gradiente_numerico


def autoencoder(salida: str = 'logistica', semilla: int = 0, amplitud: float = 0.5) -> Autoencoder:
    return Autoencoder([35, 8, 2], ['tanh', 'identidad'], salida, semilla, amplitud_inicial=amplitud)


class EstructuraTests(unittest.TestCase):
    def test_decoder_es_el_espejo_del_encoder(self) -> None:
        ae = Autoencoder([35, 20, 10, 2], ['tanh', 'logistica', 'identidad'], 'logistica', semilla=0)
        self.assertEqual(ae.encoder.capas, [35, 20, 10, 2])
        self.assertEqual(ae.decoder.capas, [2, 10, 20, 35])
        self.assertEqual([a.nombre for a in ae.encoder.activaciones], ['tanh', 'logistica', 'identidad'])
        self.assertEqual([a.nombre for a in ae.decoder.activaciones], ['logistica', 'tanh', 'logistica'])
        self.assertEqual(ae.n_params(), ae.encoder.n_params() + ae.decoder.n_params())

    def test_latente_de_dos_dimensiones(self) -> None:
        X, _ = cargar_font()
        ae = autoencoder()
        self.assertEqual(ae.codificar(X).shape, (32, 2))
        self.assertEqual(ae.reconstruir(X).shape, (32, 35))
        np.testing.assert_array_equal(ae.decodificar(ae.codificar(X)), ae.reconstruir(X))
        # Generar desde un z que no viene de ningún patrón (diap. 34)
        self.assertEqual(ae.decodificar(np.array([[0.3, -1.2]])).shape, (1, 35))

    def test_semillas(self) -> None:
        a, b, c = autoencoder(semilla=4), autoencoder(semilla=4), autoencoder(semilla=5)
        for Wa, Wb in zip(a.params, b.params):
            np.testing.assert_array_equal(Wa, Wb)
        self.assertFalse(np.array_equal(a.params[0], c.params[0]))
        # Encoder y decoder no arrancan con los mismos números
        self.assertFalse(np.array_equal(a.encoder.W[1], a.decoder.W[0].T[:, :3]))


class GradientCheckTests(unittest.TestCase):
    def test_encoder_y_decoder(self) -> None:
        X, _ = cargar_font()
        casos = [('error_cuadratico', 'logistica', X), ('bce', 'logistica', X),
                 ('error_cuadratico', 'tanh', a_bipolar(X))]
        for perdida, salida, datos in casos:
            ae = autoencoder(salida, semilla=1)
            _, analiticos = ae.gradientes(datos, datos, perdida)
            funcion = lambda: PERDIDAS[perdida](ae.reconstruir(datos), datos, 'suma')[0]
            for indice, W in enumerate(ae.params):
                with self.subTest(perdida=perdida, salida=salida, matriz=indice):
                    self.assertLess(error_relativo(analiticos[indice], gradiente_numerico(funcion, W)),
                                    TOLERANCIA)

    def test_salida_esperada_distinta_de_la_entrada(self) -> None:
        """Lo que usa el DAE: entrada X̃ y salida esperada X (diap. 22)."""
        X, _ = cargar_font()
        ruidosa = np.abs(X - (np.random.default_rng(0).random(X.shape) < 0.1))
        ae = autoencoder(semilla=2)
        _, analiticos = ae.gradientes(ruidosa, X)
        funcion = lambda: PERDIDAS['error_cuadratico'](ae.reconstruir(ruidosa), X, 'suma')[0]
        for indice, W in enumerate(ae.params):
            with self.subTest(matriz=indice):
                self.assertLess(error_relativo(analiticos[indice], gradiente_numerico(funcion, W)),
                                TOLERANCIA)


class EntrenamientoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.X, _ = cargar_font()

    def test_el_error_baja_y_el_historial_tiene_una_fila_por_epoca(self) -> None:
        ae = autoencoder()
        inicial, _ = evaluar(ae, self.X, self.X, 'error_cuadratico', '01')
        historial = entrenar(ae, self.X, self.X, Adam(eta=0.01), epocas=50, tamano_lote=32,
                             rng=np.random.default_rng(0))
        for serie in (historial.error, historial.max_pixeles, historial.total_pixeles, historial.eta):
            self.assertEqual(len(serie), 50)
        self.assertLess(historial.error[-1], inicial)
        self.assertGreater(historial.segundos, 0.0)
        # La última fila es el estado final de la red
        final, incorrectos = evaluar(ae, self.X, self.X, 'error_cuadratico', '01')
        self.assertEqual(historial.error[-1], final)
        self.assertEqual(historial.max_pixeles[-1], incorrectos.max())
        self.assertEqual(historial.total_pixeles[-1], incorrectos.sum())

    def test_misma_semilla_misma_corrida(self) -> None:
        errores = []
        for _ in range(2):
            historial = entrenar(autoencoder(semilla=3), self.X, self.X, Adam(eta=0.01), epocas=20,
                                 tamano_lote=8, rng=np.random.default_rng(3))
            errores.append(historial.error)
        self.assertEqual(errores[0], errores[1])

    def test_modos_batch_minibatch_y_online(self) -> None:
        for tamano_lote in (32, 8, 1):
            with self.subTest(tamano_lote=tamano_lote):
                ae = autoencoder()
                inicial, _ = evaluar(ae, self.X, self.X, 'error_cuadratico', '01')
                historial = entrenar(ae, self.X, self.X, GD(eta=0.01), epocas=20,
                                     tamano_lote=tamano_lote, rng=np.random.default_rng(0))
                self.assertLess(historial.error[-1], inicial)

    def test_corte_en_el_objetivo(self) -> None:
        """Con 2 patrones y latente 2 se llega a ≤ 1 píxel; el corte frena en esa época."""
        X = self.X[:2]
        historial = entrenar(autoencoder(), X, X, Adam(eta=0.01), epocas=3000, tamano_lote=2,
                             rng=np.random.default_rng(0), cortar_en_objetivo=True)
        self.assertIsNotNone(historial.epoca_objetivo)
        self.assertEqual(len(historial.error), historial.epoca_objetivo)
        self.assertLessEqual(historial.max_pixeles[-1], 1)
        self.assertGreater(historial.max_pixeles[0], 1)

    def test_eta_adaptativo_cambia_eta(self) -> None:
        optimizador = GD(eta=0.001)
        entrenar(autoencoder(), self.X, self.X, optimizador, epocas=30, tamano_lote=32,
                 rng=np.random.default_rng(0),
                 eta_adaptativo=EtaAdaptativo(optimizador, a=0.0005, b=0.5, k=3, k_prima=1))
        self.assertNotEqual(optimizador.eta, 0.001)

    def test_codificacion_incoherente(self) -> None:
        with self.assertRaises(ValueError):
            entrenar(autoencoder('tanh'), self.X, self.X, GD(eta=0.01), epocas=1, tamano_lote=32,
                     rng=np.random.default_rng(0), codificacion='01')


if __name__ == '__main__':
    unittest.main()
