"""VAE (Autoencoders diap. 74–89, convención del doc de cátedra §2.1). Tests obligatorios
antes de entrenar (CLAUDE.md §4 y sesión 5): gradient check de J con ϵ fijo, R = 0 en
μ = 0 y s = 0, y con λ = 0 y ϵ = 0 el VAE es un autoencoder con los mismos pesos.

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import unittest

import numpy as np

from src.modelos.vae import VAE, entrenar_vae, kl_normal_estandar
from src.nn.mlp import MLP
from src.nn.optimizadores import Adam
from src.nn.perdidas import bce
from tests.gradcheck import TOLERANCIA, error_relativo, gradiente_numerico


def datos(p: int = 6, n: int = 12) -> np.ndarray:
    return np.random.default_rng(0).random((p, n))            # grises en [0, 1]


def vae(n: int = 12, k: int = 3) -> VAE:
    return VAE([n, 8, k], [k, 8, n], 'tanh', semilla=1, amplitud_inicial=0.6)


def J(modelo: VAE, X: np.ndarray, epsilon: np.ndarray, lam: float) -> float:
    """J = L + λR calculada desde cero, sin usar VAE.gradientes."""
    mu, s = modelo.codificar(X)
    L, _ = bce(modelo.decodificar(mu + epsilon * np.exp(s / 2.0)), X, 'media')
    R = 0.5 * np.sum(np.exp(s) + mu ** 2 - 1.0 - s) / len(X)
    return L + lam * R


class KlTests(unittest.TestCase):
    def test_vale_cero_en_la_normal_estandar(self) -> None:
        R, dR_dmu, dR_ds = kl_normal_estandar(np.zeros((4, 3)), np.zeros((4, 3)))
        self.assertEqual(R, 0.0)
        self.assertTrue(np.all(dR_dmu == 0.0) and np.all(dR_ds == 0.0))

    def test_es_positiva_fuera_de_ahi(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(20):
            R, _, _ = kl_normal_estandar(rng.normal(size=(4, 3)), rng.normal(size=(4, 3)))
            self.assertGreater(R, 0.0)

    def test_valor_conocido(self) -> None:
        # Un patrón, k = 1, μ = 2, s = 0: ½ (1 + 4 − 1 − 0) = 2
        self.assertAlmostEqual(kl_normal_estandar(np.array([[2.0]]), np.array([[0.0]]))[0], 2.0)

    def test_gradient_check(self) -> None:
        rng = np.random.default_rng(1)
        mu, s = rng.normal(size=(5, 3)), rng.normal(size=(5, 3))
        _, dR_dmu, dR_ds = kl_normal_estandar(mu, s)
        self.assertLess(error_relativo(dR_dmu, gradiente_numerico(lambda: kl_normal_estandar(mu, s)[0], mu)), TOLERANCIA)
        self.assertLess(error_relativo(dR_ds, gradiente_numerico(lambda: kl_normal_estandar(mu, s)[0], s)), TOLERANCIA)


class VaeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X = datos()
        self.epsilon = np.random.default_rng(2).standard_normal((6, 3))

    def test_gradient_check_de_J_con_epsilon_fijo(self) -> None:
        for lam in (0.0, 1.0, 2.5):
            modelo = vae()
            _, _, analiticos = modelo.gradientes(self.X, self.epsilon, lam)
            nombres = ['encoder'] * len(modelo.encoder.params) + ['decoder'] * len(modelo.decoder.params)
            for indice, (W, nombre) in enumerate(zip(modelo.params, nombres)):
                with self.subTest(lam=lam, red=nombre, matriz=indice):
                    numerico = gradiente_numerico(lambda: J(modelo, self.X, self.epsilon, lam), W)
                    self.assertLess(error_relativo(analiticos[indice], numerico), TOLERANCIA)

    def test_L_y_R_coinciden_con_la_definicion(self) -> None:
        modelo = vae()
        L, R, _ = modelo.gradientes(self.X, self.epsilon, 1.0)
        self.assertAlmostEqual(L + 1.0 * R, J(modelo, self.X, self.epsilon, 1.0))
        self.assertAlmostEqual(L, J(modelo, self.X, self.epsilon, 0.0))

    def test_el_decoder_solo_recibe_el_gradiente_de_reconstruccion(self) -> None:
        """Autoencoders diap. 88: los gradientes del decoder no dependen de λ; los del encoder sí."""
        modelo = vae()
        n_encoder = len(modelo.encoder.params)
        _, _, con_cero = modelo.gradientes(self.X, self.epsilon, 0.0)
        _, _, con_cinco = modelo.gradientes(self.X, self.epsilon, 5.0)
        for a, b in zip(con_cero[n_encoder:], con_cinco[n_encoder:]):
            np.testing.assert_array_equal(a, b)
        self.assertFalse(np.allclose(con_cero[n_encoder - 1], con_cinco[n_encoder - 1]))

    def test_con_lambda_cero_y_epsilon_cero_es_un_autoencoder(self) -> None:
        """Mismos pesos: z = μ, la salida es decoder(μ) y los gradientes son los del AE con BCE."""
        modelo = vae()
        k = modelo.k
        ceros = np.zeros((6, k))
        L, _, gradientes = modelo.gradientes(self.X, ceros, 0.0)
        # Autoencoder equivalente: el encoder sin las k neuronas de s
        encoder_ae = MLP([12, 8, k], ['tanh', 'identidad'], semilla=0)
        encoder_ae.W[0][...] = modelo.encoder.W[0]
        encoder_ae.W[1][...] = modelo.encoder.W[1][:, :k]
        salida = modelo.decoder.forward(encoder_ae.forward(self.X))
        L_ae, dL_dsalida = bce(salida, self.X, 'media')
        gradientes_decoder, dL_dz = modelo.decoder.backward(dL_dsalida)
        gradientes_encoder, _ = encoder_ae.backward(dL_dz)
        self.assertAlmostEqual(L, L_ae)
        np.testing.assert_allclose(gradientes[0], gradientes_encoder[0])
        np.testing.assert_allclose(gradientes[1][:, :k], gradientes_encoder[1])
        np.testing.assert_array_equal(gradientes[1][:, k:], 0.0)        # s no recibe gradiente
        for a, b in zip(gradientes[2:], gradientes_decoder):
            np.testing.assert_allclose(a, b)

    def test_estructura(self) -> None:
        modelo = vae()
        mu, s = modelo.codificar(self.X)
        self.assertEqual((mu.shape, s.shape), ((6, 3), (6, 3)))
        self.assertEqual(modelo.encoder.capas, [12, 8, 6])               # 2k salidas lineales
        self.assertEqual(modelo.encoder.activaciones[-1].nombre, 'identidad')
        self.assertEqual(modelo.decoder.activaciones[-1].nombre, 'logistica')
        generadas = modelo.decodificar(np.random.default_rng(0).standard_normal((10, 3)))
        self.assertEqual(generadas.shape, (10, 12))
        self.assertTrue(np.all((generadas > 0.0) & (generadas < 1.0)))
        with self.assertRaises(ValueError):
            VAE([12, 8, 3], [2, 8, 12], 'tanh', semilla=0)

    def test_entrenar_baja_J_y_es_reproducible(self) -> None:
        historiales = []
        for _ in range(2):
            modelo = vae()
            historiales.append(entrenar_vae(modelo, self.X, Adam(eta=0.01), epocas=200, tamano_lote=3, lam=1.0,
                                            rng_lotes=np.random.default_rng(0), rng_epsilon=np.random.default_rng(1)))
        h = historiales[0]
        self.assertEqual(len(h.reconstruccion), 200)
        self.assertLess(h.reconstruccion[-1] + h.regularizacion[-1], h.reconstruccion[0] + h.regularizacion[0])
        self.assertEqual(h.reconstruccion, historiales[1].reconstruccion)
        self.assertEqual(h.regularizacion, historiales[1].regularizacion)


if __name__ == '__main__':
    unittest.main()
