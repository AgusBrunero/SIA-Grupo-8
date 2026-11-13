"""AE contra VAE y generación (consigna 2b y 2c): config, distancia por píxel y medidas.

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import json
import unittest
from pathlib import Path

import numpy as np

from experiments.e2c_latente_generacion import Modelo, entrenar_ae, entrenar_vae_modelo, etiqueta, medir, mosaico
from src.data.emojis import generar_caritas
from src.modelos.generacion import vecino_por_pixel
from src.modelos.autoencoder import Autoencoder
from tests.gradcheck import TOLERANCIA, error_relativo, gradiente_numerico

RAIZ = Path(__file__).resolve().parents[1]


class VecinoPorPixelTests(unittest.TestCase):
    def test_distancia_y_vecino(self) -> None:
        X = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.0, 0.0]])
        muestras = np.array([[0.0, 0.0, 0.0, 0.0], [0.9, 1.0, 1.0, 1.0], [0.5, 0.5, 0.2, 0.0]])
        indices, distancias = vecino_por_pixel(muestras, X)
        np.testing.assert_array_equal(indices, [0, 1, 2])
        np.testing.assert_allclose(distancias, [0.0, 0.025, 0.05])

    def test_excluir_identicas(self) -> None:
        X = np.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.0]])
        indices, distancias = vecino_por_pixel(X, X, excluir_identicas=True)
        np.testing.assert_array_equal(indices, [2, 2, 0])
        np.testing.assert_allclose(distancias, [0.1, 0.9, 0.1])


class E2cTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((RAIZ / 'config' / 'e2c.json').read_text(encoding='utf-8'))
        cls.e2b = json.loads((RAIZ / 'config' / 'e2b.json').read_text(encoding='utf-8'))

    def test_config_mismo_protocolo_que_e2b(self) -> None:
        for clave in ('dataset', 'optimizador', 'tamano_lote', 'epocas', 'beta', 'amplitud_inicial'):
            self.assertEqual({k: v for k, v in self.config[clave].items() if k != 'origen'} if clave == 'dataset' else self.config[clave],
                             {k: v for k, v in self.e2b[clave].items() if k != 'origen'} if clave == 'dataset' else self.e2b[clave])
        for clave in ('capas_encoder', 'capas_decoder', 'activacion_oculta'):
            self.assertEqual(self.config['modelo'][clave], self.e2b['modelo'][clave])
        self.assertEqual(len(set(self.config['lambdas'])), 3)
        self.assertGreaterEqual(len(set(self.config['semillas'])), 5)
        self.assertIn(self.config['semilla_figuras'], self.config['semillas'])

    def test_gradient_check_del_ae_con_bce_y_media(self) -> None:
        """La combinación que entrena este experimento: AE con latente identidad, BCE, reducción 'media'."""
        X = np.random.default_rng(0).random((5, 12))
        ae = Autoencoder([12, 6, 2], ['tanh', 'identidad'], 'logistica', semilla=1, amplitud_inicial=0.6)
        _, analiticos = ae.gradientes(X, X, 'bce', 'media')
        for W, analitico in zip(ae.params, analiticos):
            numerico = gradiente_numerico(lambda: ae.gradientes(X, X, 'bce', 'media')[0], W)
            self.assertLess(error_relativo(analitico, numerico), TOLERANCIA)

    def test_corrida_corta(self) -> None:
        config = dict(self.config, epocas=3)
        X = generar_caritas(60, 0).X
        Z_prior = np.random.default_rng(0).standard_normal((15, 2))
        for lam in (None, 1.0):
            filas = []
            for _ in range(2):
                modelo = entrenar_ae(config, 0, X) if lam is None else entrenar_vae_modelo(config, 0, X, lam)
                fila, Z, generadas, vecino, distancia = medir(modelo, X, Z_prior)
                filas.append({k: v for k, v in fila.items() if k != 'segundos'})
            self.assertEqual(str(filas[0]), str(filas[1]))                 # misma semilla, misma corrida
            self.assertEqual((Z.shape, generadas.shape, vecino.shape, distancia.shape), ((60, 2), (15, 256), (15,), (15,)))
            self.assertEqual(np.isnan(fila['R']), lam is None)             # el AE no tiene R
            self.assertAlmostEqual(fila['distancia_vecino_media'], distancia.mean())
            for clave in ('fraccion_codigos_en_radio', 'fraccion_prior_en_rectangulo'):
                self.assertTrue(0.0 <= fila[clave] <= 1.0)

    def test_fracciones_con_un_modelo_conocido(self) -> None:
        """Códigos fijos: 2 de 4 dentro de ‖z‖ ≤ 3; el rectángulo de los códigos es [0, 10] × [0, 10]."""
        codigos = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 0.0], [0.0, 10.0]])
        X = np.eye(4)
        modelo = Modelo(lambda datos: codigos, lambda Z: np.full((len(Z), 4), 0.5), {'R': np.nan, 'sigma_media': np.nan, 'segundos': 0.0})
        fila, *_ = medir(modelo, X, np.array([[1.0, 1.0], [-1.0, 1.0], [5.0, 5.0]]))
        self.assertEqual(fila['fraccion_codigos_en_radio'], 0.5)
        self.assertAlmostEqual(fila['fraccion_prior_en_rectangulo'], 2 / 3)
        self.assertEqual(fila['distancia_vecino_media'], 0.5)

    def test_etiqueta_y_mosaico(self) -> None:
        self.assertEqual((etiqueta(None), etiqueta(0.1), etiqueta(10.0)), ('AE', 'VAE, λ = 0,1', 'VAE, λ = 10'))
        lienzo = mosaico(np.ones((6, 256)), 3)
        self.assertEqual(lienzo.shape, (2 * 17 - 1, 3 * 17 - 1))
        self.assertEqual(lienzo.sum(), 6 * 256)


if __name__ == '__main__':
    unittest.main()
