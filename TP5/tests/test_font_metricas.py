"""font.h parseado desde el archivo (CLAUDE.md §4) y métrica de píxeles incorrectos.

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import unittest

import numpy as np

from src.data.font import a_ascii, a_bipolar, cargar_font, distancias_hamming
from src.modelos.metricas import binarizar, pixeles_incorrectos, resumir, validar_codificacion

# Arte ASCII escrito a mano mirando el carácter, no copiado de la salida del parser.
# 0x10 es la columna izquierda.
ARTE = {
    'a': '\n'.join(['.....',
                    '.###.',
                    '....#',
                    '.##.#',
                    '#..##',
                    '#..##',
                    '.##.#']),
    'l': '\n'.join(['.##..',
                    '..#..',
                    '..#..',
                    '..#..',
                    '..#..',
                    '..#..',
                    '..#..']),
    '|': '\n'.join(['..#..',
                    '..#..',
                    '..#..',
                    '.....',
                    '..#..',
                    '..#..',
                    '..#..']),
}


class FontTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.X, cls.etiquetas = cargar_font()

    def indice(self, etiqueta: str) -> int:
        return self.etiquetas.index(etiqueta)

    def test_forma_y_valores(self) -> None:
        self.assertEqual(self.X.shape, (32, 35))
        self.assertEqual(self.X.dtype, np.float64)
        self.assertTrue(np.all((self.X == 0.0) | (self.X == 1.0)))

    def test_etiquetas(self) -> None:
        self.assertEqual(self.etiquetas[0], '`')
        self.assertEqual(self.etiquetas[1:27], list('abcdefghijklmnopqrstuvwxyz'))
        self.assertEqual(self.etiquetas[27:], ['{', '|', '}', '~', 'DEL'])
        self.assertEqual(len(set(self.etiquetas)), 32)

    def test_arte_ascii(self) -> None:
        for etiqueta, arte in ARTE.items():
            with self.subTest(caracter=etiqueta):
                self.assertEqual(a_ascii(self.X[self.indice(etiqueta)]), arte)

    def test_hechos_del_dataset(self) -> None:
        """Doc de cátedra §0.1: pares más parecidos y cantidad de píxeles encendidos."""
        distancias = distancias_hamming(self.X)
        self.assertEqual(distancias[self.indice('l'), self.indice('|')], 2)
        self.assertEqual(distancias[self.indice('n'), self.indice('r')], 3)
        self.assertEqual(distancias[self.indice('`'), self.indice('~')], 4)
        fuera_de_la_diagonal = distancias[~np.eye(32, dtype=bool)]
        self.assertEqual(fuera_de_la_diagonal.min(), 2)          # no hay patrones repetidos
        encendidos = self.X.sum(axis=1)
        self.assertEqual(encendidos[self.indice('DEL')], 35)
        self.assertEqual(encendidos[self.indice('`')], 3)
        self.assertEqual(encendidos[self.indice('~')], 5)

    def test_bipolar(self) -> None:
        bipolar = a_bipolar(self.X)
        self.assertTrue(np.all((bipolar == -1.0) | (bipolar == 1.0)))
        np.testing.assert_array_equal(bipolar > 0, self.X > 0.5)


class MetricaTests(unittest.TestCase):
    def test_conteo_por_patron(self) -> None:
        esperado = np.array([[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        salida = np.array([[0.9, 0.1, 0.6, 0.4],        # 0 mal
                           [0.4, 0.9, 0.1, 0.2],        # 1 mal
                           [0.9, 0.8, 0.7, 0.1]])       # 3 mal
        incorrectos = pixeles_incorrectos(salida, esperado, '01')
        np.testing.assert_array_equal(incorrectos, [0, 1, 3])
        resumen = resumir(incorrectos)
        self.assertEqual((resumen.maximo, resumen.total, resumen.patrones_con_error, resumen.cumple),
                         (3, 4, 1, False))
        self.assertTrue(resumir(incorrectos[:2]).cumple)

    def test_umbral_segun_la_codificacion(self) -> None:
        valores = np.array([[-0.2, 0.2, 0.6]])
        np.testing.assert_array_equal(binarizar(valores, '01'), [[False, False, True]])
        np.testing.assert_array_equal(binarizar(valores, 'bipolar'), [[False, True, True]])
        # La misma reconstrucción, en las dos convenciones, da el mismo conteo
        esperado = np.array([[1.0, 0.0, 1.0]])
        salida = np.array([[0.8, 0.7, 0.3]])
        self.assertEqual(pixeles_incorrectos(salida, esperado, '01')[0], 2)
        self.assertEqual(pixeles_incorrectos(a_bipolar(salida), a_bipolar(esperado), 'bipolar')[0], 2)

    def test_no_se_mezclan_convenciones(self) -> None:
        validar_codificacion('logistica', '01')
        validar_codificacion('tanh', 'bipolar')
        validar_codificacion('identidad', '01')
        validar_codificacion('identidad', 'bipolar')
        for salida, codificacion in (('logistica', 'bipolar'), ('tanh', '01'), ('tanh', 'gris')):
            with self.assertRaises(ValueError):
                validar_codificacion(salida, codificacion)


if __name__ == '__main__':
    unittest.main()
