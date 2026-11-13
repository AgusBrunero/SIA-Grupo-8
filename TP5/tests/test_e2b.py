"""Config y medidas del entrenamiento del VAE (consigna 2a y 2b).

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import json
import unittest
from pathlib import Path

import numpy as np

from experiments.e2b_vae import construir_vae, entrenar_semilla, evaluar_vae
from src.data.emojis import N_PIXELES, generar_caritas

RAIZ = Path(__file__).resolve().parents[1]


class E2bTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads((RAIZ / 'config' / 'e2b.json').read_text(encoding='utf-8'))

    def test_config(self) -> None:
        modelo = self.config['modelo']
        self.assertEqual(modelo['capas_encoder'][0], N_PIXELES)
        self.assertEqual(modelo['capas_decoder'][-1], N_PIXELES)
        self.assertEqual(modelo['capas_encoder'][-1], modelo['capas_decoder'][0])
        self.assertGreaterEqual(len(set(self.config['semillas'])), 5)
        self.assertEqual(self.config['lambda'], 1.0)

    def test_corrida_corta(self) -> None:
        config = dict(self.config, epocas=3)
        X = generar_caritas(60, 0).X
        resultados = []
        for _ in range(2):
            vae, historial = entrenar_semilla(config, 0, X, config['lambda'])
            resultados.append((historial.reconstruccion, historial.regularizacion, evaluar_vae(vae, X)))
        self.assertEqual(len(resultados[0][0]), 3)
        self.assertEqual(resultados[0][0], resultados[1][0])            # misma semilla, misma corrida
        self.assertEqual(resultados[0][2], resultados[1][2])
        self.assertLess(resultados[0][0][-1], resultados[0][0][0])      # L baja

    def test_medidas_sin_ruido_de_muestreo(self) -> None:
        X = generar_caritas(20, 0).X
        vae = construir_vae(self.config, 0)
        a, b = evaluar_vae(vae, X), evaluar_vae(vae, X)
        self.assertEqual(a, b)
        self.assertGreater(a['R'], 0.0)
        self.assertTrue(0.0 <= a['error_absoluto_por_pixel'] <= 1.0)
        self.assertTrue(np.isfinite(a['L_con_z_igual_mu']))


if __name__ == '__main__':
    unittest.main()
