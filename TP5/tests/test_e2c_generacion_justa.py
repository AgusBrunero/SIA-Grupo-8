"""AE contra VAE en generación, comparación justa (experiments/e2c_generacion_justa.py).

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e2c_generacion_justa import banco_sin_repetidas, muestrear, resumir
from src.data.emojis import generar_caritas

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
    def test_banco_y_referencia_no_repiten_las_500(self) -> None:
        config = json.loads((RAIZ / 'config' / 'e2c_justa.json').read_text(encoding='utf-8'))
        self.assertNotEqual(config['banco']['semilla'], config['dataset']['semilla'])
        self.assertNotEqual(config['referencia']['semilla'], config['banco']['semilla'])
        chico = dict(config, banco={'n': 300, 'semilla': config['banco']['semilla']},
                     referencia={'n': 50, 'semilla': config['referencia']['semilla']})
        X = generar_caritas(config['dataset']['n'], config['dataset']['semilla']).X
        banco, referencia = banco_sin_repetidas(chico, X)
        self.assertEqual(banco.shape, (300, 256))
        self.assertEqual(referencia.shape, (50, 256))


class ResumenTests(unittest.TestCase):
    def tabla(self, n_muestras: int) -> pd.DataFrame:
        filas = [{'modelo': m, 'semilla': s, 'd_banco': 0.03, 'd_500': 0.04}
                 for m in ('AE', 'VAE') for s in range(2) for _ in range(n_muestras)]
        filas += [{'modelo': 'Caritas reales', 'semilla': np.nan, 'd_banco': 0.02, 'd_500': 0.035}] * 3
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
