"""Config y agregación del experimento del DAE (consigna 1b).

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import json
import unittest
from pathlib import Path

import pandas as pd

from experiments.e1b_dae import resumir_celdas

RAIZ = Path(__file__).resolve().parents[1]


class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((RAIZ / 'config' / 'e1b.json').read_text(encoding='utf-8'))

    def test_protocolo(self) -> None:
        self.assertEqual(self.config['niveles'], [0.0, 0.05, 0.1, 0.2, 0.3])
        self.assertGreaterEqual(len(set(self.config['semillas'])), 5)
        self.assertGreaterEqual(self.config['realizaciones'], 20)
        self.assertIn(len(self.config['arquitecturas']), (2, 3))

    def test_receta_de_e9_y_solo_cambia_la_arquitectura(self) -> None:
        e1a2 = json.loads((RAIZ / 'config' / 'e1a2.json').read_text(encoding='utf-8'))
        e9 = next(e for e in e1a2['escalones'] if e['id'] == 'E9')
        for clave, valor in self.config['modelo'].items():
            self.assertEqual(valor, e9[clave], clave)
        self.assertEqual(self.config['arquitecturas'][0]['capas_encoder'], e9['capas_encoder'])   # A es la red de 1a
        for arquitectura in self.config['arquitecturas']:
            self.assertEqual(arquitectura['capas_encoder'][0], 35)
            self.assertTrue(arquitectura['sosten'])


class AgregacionTests(unittest.TestCase):
    def test_media_y_desvio_entre_semillas_y_conteo(self) -> None:
        config = {'semillas': [0, 1], 'niveles': [0.0], 'arquitecturas': [{'id': 'A'}], 'realizaciones': 20}
        celdas = pd.DataFrame({
            'arquitectura': 'A', 'p_entrenamiento': 0.0, 'p_prueba': 0.0, 'semilla': [0, 1], 'n': 640,
            'alterados_media': [1.0, 3.0], 'mal_media': [2.0, 4.0], 'exactos': [0.5, 0.7],
            'hasta_1px': [0.6, 0.8], 'identidad_correcta': [0.9, 1.0]})
        fila = resumir_celdas(config, celdas).iloc[0]
        self.assertEqual(fila.semillas, 2)
        self.assertAlmostEqual(fila.mal_media, 3.0)
        self.assertAlmostEqual(fila.mal_desvio, 2 ** 0.5)
        self.assertAlmostEqual(fila.hasta_1px_media, 0.7)
        # Una semilla repetida (doble conteo) tiene que hacer fallar la agregación
        with self.assertRaises(AssertionError):
            resumir_celdas(config, pd.concat([celdas, celdas.iloc[:1]]))


if __name__ == '__main__':
    unittest.main()
