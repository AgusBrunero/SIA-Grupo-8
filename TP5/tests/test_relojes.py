"""Dataset de relojes (consigna 2a): rango, reproducibilidad y que el dibujo sea el esperado.

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import unittest

import numpy as np

from src.data.relojes import LADO, PARAMETROS, RANGOS, a_ascii, a_texto, angulos, dibujar, generar_relojes


def reloj(hora: float) -> np.ndarray:
    return dibujar(np.array([hora]))


class RelojesTests(unittest.TestCase):
    def test_forma_rango_y_semilla(self) -> None:
        a, b, c = generar_relojes(50, 0), generar_relojes(50, 0), generar_relojes(50, 1)
        self.assertEqual(a.X.shape, (50, LADO * LADO))
        self.assertEqual(a.parametros.shape, (50, len(PARAMETROS)))
        self.assertEqual(a.X.dtype, np.float64)
        self.assertTrue(a.X.min() >= 0.0 and a.X.max() <= 1.0)
        self.assertTrue(((a.X > 0.0) & (a.X < 1.0)).any())          # hay grises, no es binaria
        np.testing.assert_array_equal(a.X, b.X)
        self.assertFalse(np.array_equal(a.X, c.X))

    def test_dos_relojes_iguales_marcan_el_mismo_minuto(self) -> None:
        """Una imagen repetida solo puede salir de dos horas a menos de un minuto (1/60 h) de distancia."""
        relojes = generar_relojes(500, 2026)
        _, grupo = np.unique(relojes.X, axis=0, return_inverse=True)
        horas = relojes.parametros[:, 0]
        for g in np.unique(grupo):
            mismas = horas[grupo.ravel() == g]
            self.assertLess(mismas.max() - mismas.min(), 1 / 60)

    def test_parametros_dentro_de_su_rango(self) -> None:
        parametros = generar_relojes(200, 3).parametros
        for columna, nombre in enumerate(PARAMETROS):
            minimo, maximo = RANGOS[nombre]
            self.assertTrue(np.all((parametros[:, columna] >= minimo) & (parametros[:, columna] <= maximo)), nombre)

    def test_las_agujas_apuntan_donde_dice_la_hora(self) -> None:
        """A las 3:00 la de los minutos apunta arriba y la de las horas a la derecha; a las 9:00, arriba
        e izquierda; a las 6:30, las dos abajo. Se mira la tinta dentro de la esfera (radio < 4,5)."""
        eje = np.arange(LADO) + 0.5 - LADO / 2
        x, y = np.meshgrid(eje, eje)
        adentro = np.hypot(x, y) < 4.5
        def tinta(imagen, zona):
            return float(imagen[adentro & zona].sum())
        tres, nueve, seis_y_media = reloj(3.0), reloj(9.0), reloj(6.5)
        print('\n' + a_ascii(tres))
        self.assertGreater(tinta(tres, x > 1), 3 * tinta(tres, x < -1))
        self.assertGreater(tinta(tres, y < -1), 3 * tinta(tres, y > 1))
        self.assertGreater(tinta(nueve, x < -1), 3 * tinta(nueve, x > 1))
        self.assertGreater(tinta(seis_y_media, y > 1), 3 * tinta(seis_y_media, y < -1))
        np.testing.assert_allclose(tres, nueve[:, ::-1], atol=1e-12)     # espejo izquierda-derecha
        self.assertEqual(tres[0, 0], 0.0)                                # esquinas vacías

    def test_angulos_de_un_reloj_real(self) -> None:
        np.testing.assert_allclose(angulos(3.0), (np.pi / 2, 0.0), atol=1e-12)
        np.testing.assert_allclose(angulos(6.5), (6.5 / 12 * 2 * np.pi, np.pi), atol=1e-12)   # 6:30

    def test_la_hora_es_ciclica(self) -> None:
        """Las 12:00 y las 11:59 son casi la misma imagen; las 12:00 y las 6:30, no."""
        doce = reloj(0.0)
        self.assertLess(np.abs(doce - reloj(12.0 - 1 / 60)).mean(), np.abs(doce - reloj(6.5)).mean() / 3)
        np.testing.assert_allclose(doce, reloj(12.0), atol=1e-12)

    def test_a_texto(self) -> None:
        self.assertEqual(a_texto(0.0), '12:00')
        self.assertEqual(a_texto(3.5), '3:30')
        self.assertEqual(a_texto(11.999), '12:00')


if __name__ == '__main__':
    unittest.main()
