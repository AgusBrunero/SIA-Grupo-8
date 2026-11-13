"""Dataset de caritas (consigna 2a): rango, reproducibilidad y que el dibujo sea el esperado.

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import unittest

import numpy as np

from src.data.emojis import LADO, PARAMETROS, RANGOS, a_ascii, dibujar, generar_caritas


def carita(**cambios: float) -> np.ndarray:
    """Carita con los parámetros en el centro de su rango, salvo los indicados."""
    valores = {nombre: (minimo + maximo) / 2 for nombre, (minimo, maximo) in RANGOS.items()}
    valores.update(cambios)
    return dibujar(np.array([valores[nombre] for nombre in PARAMETROS]))


def fila_de_la_boca(imagen: np.ndarray, columna: int) -> float:
    """Fila media de la tinta en una columna, mirando solo la mitad de abajo de la cara."""
    filas = np.arange(9, 14)
    pesos = imagen[9:14, columna]
    return float((filas * pesos).sum() / pesos.sum())


class CaritasTests(unittest.TestCase):
    def test_forma_rango_y_semilla(self) -> None:
        a, b, c = generar_caritas(50, 0), generar_caritas(50, 0), generar_caritas(50, 1)
        self.assertEqual(a.X.shape, (50, LADO * LADO))
        self.assertEqual(a.parametros.shape, (50, len(PARAMETROS)))
        self.assertEqual(a.X.dtype, np.float64)
        self.assertTrue(a.X.min() >= 0.0 and a.X.max() <= 1.0)
        self.assertTrue(((a.X > 0.0) & (a.X < 1.0)).any())          # hay grises, no es binaria
        np.testing.assert_array_equal(a.X, b.X)
        self.assertFalse(np.array_equal(a.X, c.X))
        self.assertEqual(len(np.unique(a.X, axis=0)), 50)           # no hay caritas repetidas

    def test_parametros_dentro_de_su_rango(self) -> None:
        parametros = generar_caritas(200, 3).parametros
        for columna, nombre in enumerate(PARAMETROS):
            minimo, maximo = RANGOS[nombre]
            self.assertTrue(np.all((parametros[:, columna] >= minimo) & (parametros[:, columna] <= maximo)), nombre)

    def test_dibujo_de_una_carita_centrada(self) -> None:
        """El dibujo: contorno, dos ojos, dos cejas y una boca, simétrico respecto del eje vertical."""
        imagen = carita(centro_x=0.0, centro_y=0.0, curvatura_boca=1.0, inclinacion_cejas=0.0)
        print('\n' + a_ascii(imagen))
        np.testing.assert_allclose(imagen, imagen[:, ::-1], atol=1e-12)        # simetría izquierda-derecha
        self.assertEqual(imagen[0, 0], 0.0)                                    # esquinas vacías
        self.assertGreater(imagen[8, 1], 0.5)                                  # contorno a la izquierda
        self.assertGreater(imagen[6, 5], 0.5)                                  # ojo izquierdo
        self.assertGreater(imagen[6, 10], 0.5)                                 # ojo derecho
        self.assertEqual(imagen[8, 8], 0.0)                                    # nada entre los ojos y la boca

    def test_la_curvatura_decide_sonrisa_o_boca_triste(self) -> None:
        sonrisa = carita(centro_x=0.0, centro_y=0.0, curvatura_boca=1.0)
        triste = carita(centro_x=0.0, centro_y=0.0, curvatura_boca=-1.0)
        # Sonrisa: las puntas de la boca están más arriba (fila menor) que el centro
        self.assertLess(fila_de_la_boca(sonrisa, 5), fila_de_la_boca(sonrisa, 8))
        self.assertGreater(fila_de_la_boca(triste, 5), fila_de_la_boca(triste, 8))

    def test_la_apertura_cambia_el_tamano_de_los_ojos(self) -> None:
        abiertos = carita(apertura_ojos=RANGOS['apertura_ojos'][1])
        cerrados = carita(apertura_ojos=RANGOS['apertura_ojos'][0])
        self.assertGreater(abiertos[4:9, 3:13].sum(), cerrados[4:9, 3:13].sum())


if __name__ == '__main__':
    unittest.main()
