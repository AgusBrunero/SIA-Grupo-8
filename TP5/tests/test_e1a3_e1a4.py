"""Config y medidas del experimento de latente y generación (consigna 1a.3 y 1a.4).

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import json
import unittest
from pathlib import Path

import numpy as np

from experiments.e1a2_escalera import CLAVES_COMPARABLES
from experiments.e1a3_e1a4_latente import distancias_entre_pares, vecino_en_latente
from src.data.font import cargar_font, distancias_hamming

RAIZ = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((RAIZ / 'config' / 'e1a3_e1a4.json').read_text(encoding='utf-8'))
        cls.e1a2 = json.loads((RAIZ / 'config' / 'e1a2.json').read_text(encoding='utf-8'))

    def test_el_modelo_es_el_escalon_e9(self) -> None:
        e9 = next(e for e in self.e1a2['escalones'] if e['id'] == 'E9')
        for clave in CLAVES_COMPARABLES:
            self.assertEqual(self.config['modelo'][clave], e9[clave], clave)
        for clave in ('epocas', 'codificacion', 'reduccion', 'beta', 'amplitud_inicial'):
            self.assertEqual(self.config[clave], self.e1a2[clave], clave)

    def test_al_menos_cinco_semillas(self) -> None:
        self.assertGreaterEqual(len(set(self.config['semillas'])), 5)

    def test_el_par_lejano_es_el_de_mayor_hamming_entre_letras(self) -> None:
        X, etiquetas = cargar_font()
        letras = [i for i, e in enumerate(etiquetas) if len(e) == 1 and e.isalpha()]
        distancias = distancias_hamming(X)[np.ix_(letras, letras)]
        i, j = np.unravel_index(distancias.argmax(), distancias.shape)
        self.assertEqual({etiquetas[letras[i]], etiquetas[letras[j]]}, set(self.config['pares_interpolacion'][-1]))
        self.assertEqual(int((np.triu(distancias) == distancias.max()).sum()), 1)      # sin empates


class MedidasTests(unittest.TestCase):
    def test_no_cambian_con_rotacion_reflejo_ni_traslacion(self) -> None:
        Z = np.random.default_rng(0).normal(size=(32, 2))
        angulo = 0.7
        rotacion = np.array([[np.cos(angulo), -np.sin(angulo)], [np.sin(angulo), np.cos(angulo)]])
        transformado = (Z @ rotacion) * np.array([1.0, -1.0]) + np.array([3.0, -5.0])
        self.assertEqual(distancias_entre_pares(Z).shape, (496,))
        np.testing.assert_allclose(distancias_entre_pares(transformado), distancias_entre_pares(Z))
        np.testing.assert_array_equal(vecino_en_latente(transformado), vecino_en_latente(Z))


if __name__ == '__main__':
    unittest.main()
