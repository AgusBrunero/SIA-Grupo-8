"""AE contra VAE en generación, comparación justa (experiments/e2c_generacion_justa.py).

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e2c_generacion_justa import banco_sin_repetidas, muestrear, resumir
from src.data.relojes import generar_relojes

RAIZ = Path(__file__).resolve().parents[1]


class MuestreoTests(unittest.TestCase):
    def test_el_ae_se_muestrea_de_la_normal_ajustada_a_sus_codigos(self) -> None:
        rng = np.random.default_rng(0)
        Z = rng.multivariate_normal([10.0, -20.0], [[9.0, 3.0], [3.0, 4.0]], size=500)
        z = muestrear('AE', Z, 200_000, np.random.default_rng(1))
        np.testing.assert_allclose(z.mean(axis=0), Z.mean(axis=0), atol=0.05)
        np.testing.assert_allclose(np.cov(z, rowvar=False), np.cov(Z, rowvar=False), rtol=0.03)

    def test_el_vae_se_muestrea_del_prior(self) -> None:
        Z = np.full((500, 2), 50.0)         # los códigos no importan: el VAE usa N(0, I)
        z = muestrear('VAE', Z, 200_000, np.random.default_rng(1))
        np.testing.assert_allclose(z.mean(axis=0), [0.0, 0.0], atol=0.01)
        np.testing.assert_allclose(z.std(axis=0), [1.0, 1.0], atol=0.01)


class BancoTests(unittest.TestCase):
    def test_banco_y_referencia_sin_imagenes_identicas_a_las_500(self) -> None:
        config = json.loads((RAIZ / 'config' / 'e2c_justa.json').read_text(encoding='utf-8'))
        self.assertNotEqual(config['banco']['semilla'], config['dataset']['semilla'])
        self.assertNotEqual(config['referencia']['semilla'], config['banco']['semilla'])
        chico = dict(config, banco={'n': 300, 'semilla': config['banco']['semilla']},
                     referencia={'n': 50, 'semilla': config['referencia']['semilla']})
        X = generar_relojes(config['dataset']['n'], config['dataset']['semilla']).X
        banco, referencia = banco_sin_repetidas(chico, X)
        self.assertTrue(250 <= len(banco) <= 300 and 40 <= len(referencia) <= 50)
        # Ninguna imagen del banco ni de la referencia es idéntica a una de las 500
        from src.modelos.generacion import vecino_por_pixel
        for nuevas in (banco, referencia):
            self.assertTrue((vecino_por_pixel(nuevas, X)[1] > 0).all())


class ResumenTests(unittest.TestCase):
    def tabla(self, n_muestras: int) -> pd.DataFrame:
        filas = [{'modelo': m, 'semilla': s, 'd_banco': 0.03, 'd_500': 0.04}
                 for m in ('AE', 'VAE') for s in range(2) for _ in range(n_muestras)]
        filas += [{'modelo': 'Relojes reales', 'semilla': np.nan, 'd_banco': 0.02, 'd_500': 0.035}] * 3
        return pd.DataFrame(filas)

    def test_conteo_por_celda(self) -> None:
        config = {'generacion': {'n_muestras': 4}, 'referencia': {'n': 3}}
        resumen = resumir(self.tabla(4), config).set_index('modelo')
        self.assertEqual(resumen.loc['AE', 'muestras'], 8)
        self.assertEqual(resumen.loc['AE', 'd_banco_ae_vs_vae'], 'empatan')
        with self.assertRaises(AssertionError):
            resumir(self.tabla(3), config)


if __name__ == '__main__':
    unittest.main()


class FallasTests(unittest.TestCase):
    def test_el_umbral_sale_de_las_reconstrucciones_y_no_de_las_muestras(self) -> None:
        from experiments.e2c_generacion_justa import fallas
        config = {'generacion': {'n_muestras': 4}, 'semillas': [0]}
        # Reconstrucciones a distancia 0,01 a 0,02; muestras generadas a 0,005, 0,015, 0,03 y 0,05
        rec = pd.DataFrame({'modelo': 'AE', 'semilla': 0, 'd_banco': np.linspace(0.01, 0.02, 101)})
        tabla = pd.DataFrame({'modelo': 'AE', 'semilla': 0, 'd_banco': [0.005, 0.015, 0.03, 0.05]})
        f = fallas(tabla, rec, config).set_index('umbral')
        self.assertAlmostEqual(f.loc['max', 'fallan'], 0.5)       # 0,03 y 0,05 superan el máximo (0,02)
        self.assertAlmostEqual(f.loc['p95', 'fallan'], 0.5)
        self.assertAlmostEqual(f.loc['max', 'valor_umbral'], 0.02)
