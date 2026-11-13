"""PCA con SVD (Autoencoders diap. 7–16) y generación desde el latente (diap. 33–34).

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import unittest

import numpy as np

from src.data.font import cargar_font
from src.modelos.generacion import grilla_latente, interpolar, vecino_mas_cercano
from src.modelos.pca import ajustar_pca


class PcaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.X, _ = cargar_font()

    def test_componentes_ortonormales_y_ordenadas(self) -> None:
        pca = ajustar_pca(self.X, 5)
        np.testing.assert_allclose(pca.componentes @ pca.componentes.T, np.eye(5), atol=1e-12)
        self.assertTrue(np.all(np.diff(pca.varianza_explicada) <= 0))
        self.assertLess(pca.varianza_explicada.sum(), 1.0)

    def test_coincide_con_los_autovectores_de_la_covarianza(self) -> None:
        pca = ajustar_pca(self.X, 2)
        centrada = self.X - self.X.mean(axis=0)
        autovalores, autovectores = np.linalg.eigh(centrada.T @ centrada)
        principales = autovectores[:, ::-1][:, :2].T
        # Cada componente es igual al autovector salvo el signo
        for componente, autovector in zip(pca.componentes, principales):
            self.assertAlmostEqual(abs(float(componente @ autovector)), 1.0, places=10)
        np.testing.assert_allclose(pca.varianza_explicada, autovalores[::-1][:2] / autovalores.sum())

    def test_proyecciones_de_media_cero_y_no_correlacionadas(self) -> None:
        Z = ajustar_pca(self.X, 2).proyectar(self.X)
        np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
        self.assertAlmostEqual(float(Z[:, 0] @ Z[:, 1]), 0.0, places=9)

    def test_con_todas_las_componentes_reconstruye_exacto(self) -> None:
        pca = ajustar_pca(self.X, 31)          # 32 patrones centrados: rango ≤ 31
        np.testing.assert_allclose(pca.reconstruir(pca.proyectar(self.X)), self.X, atol=1e-10)
        self.assertAlmostEqual(float(pca.varianza_explicada.sum()), 1.0)

    def test_dos_componentes_no_reconstruyen_exacto(self) -> None:
        pca = ajustar_pca(self.X, 2)
        reconstruida = pca.reconstruir(pca.proyectar(self.X))
        self.assertEqual(reconstruida.shape, (32, 35))
        self.assertGreater(np.abs(reconstruida - self.X).max(), 0.1)


class GeneracionTests(unittest.TestCase):
    def test_grilla_cubre_el_rectangulo_de_los_datos(self) -> None:
        Z = np.array([[-1.0, 2.0], [3.0, -4.0], [0.5, 0.0]])
        grilla = grilla_latente(Z, 5)
        self.assertEqual(grilla.shape, (25, 2))
        np.testing.assert_array_equal(grilla[0], [-1.0, 2.0])       # arriba a la izquierda
        np.testing.assert_array_equal(grilla[4], [3.0, 2.0])        # arriba a la derecha
        np.testing.assert_array_equal(grilla[-1], [3.0, -4.0])      # abajo a la derecha
        np.testing.assert_allclose(np.diff(grilla[:5, 0]), 1.0)     # paso regular en z1

    def test_interpolacion_incluye_los_extremos(self) -> None:
        a, b = np.array([0.0, 0.0]), np.array([2.0, -4.0])
        camino = interpolar(a, b, 5)
        self.assertEqual(camino.shape, (5, 2))
        np.testing.assert_array_equal(camino[0], a)
        np.testing.assert_array_equal(camino[-1], b)
        np.testing.assert_allclose(camino[2], [1.0, -2.0])

    def test_vecino_mas_cercano_por_hamming(self) -> None:
        X, etiquetas = cargar_font()
        binario = X > 0.5
        letra_l = binario[etiquetas.index('l')].copy()
        indices, distancias = vecino_mas_cercano(binario, X)
        np.testing.assert_array_equal(indices, np.arange(32))       # cada letra es su propio vecino
        np.testing.assert_array_equal(distancias, 0)
        # Una 'l' con un píxel cambiado: sigue siendo la 'l', a distancia 1
        letra_l[34] = ~letra_l[34]
        indice, distancia = vecino_mas_cercano(letra_l[None, :], X)
        self.assertEqual((etiquetas[indice[0]], distancia[0]), ('l', 1))


if __name__ == '__main__':
    unittest.main()
