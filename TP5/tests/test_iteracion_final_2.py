"""Experimentos de la iteración final 2 de la presentación: réplicas del DAE, η adaptativo con
entropía cruzada, grilla fina del latente del AE y parámetros codificados en el latente del VAE.

Correr desde TP5/:  .venv/bin/python -m unittest discover -s tests -t . -v
"""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.e1a4_grilla import minima_entre_letras
from experiments.e2c_latente_parametros import celda_de, varianza_explicada
from src.data.font import cargar_font

RAIZ = Path(__file__).resolve().parents[1]
RESULTADOS = RAIZ / 'results'


class ReplicasTests(unittest.TestCase):
    def test_la_variante_actual_reproduce_e1b(self) -> None:
        resumen = pd.read_csv(RESULTADOS / 'e1b_replicas' / 'resumen.csv')
        actual = resumen[(resumen.replicas == 1) & (resumen.epocas == 10000)].set_index('p_prueba')
        e1b = pd.read_csv(RESULTADOS / 'e1b' / 'resumen.csv')
        e1b = e1b[(e1b.arquitectura == 'B') & np.isclose(e1b.p_entrenamiento, 0.1)].set_index('p_prueba')
        np.testing.assert_allclose(actual.exactos_media, e1b.loc[actual.index, 'exactos_media'], rtol=1e-12)

    def test_conteo_por_celda(self) -> None:
        celdas = pd.read_csv(RESULTADOS / 'e1b_replicas' / 'celdas.csv')
        self.assertTrue((celdas.groupby(['variante', 'p_prueba']).size() == 5).all())
        self.assertTrue((celdas.n == 640).all())


class EtaAdaptativoTests(unittest.TestCase):
    def test_la_reproduccion_coincide_con_las_corridas(self) -> None:
        saturacion = pd.read_csv(RESULTADOS / 'e1a2_eta_adaptativo' / 'saturacion.csv')
        corridas = pd.read_csv(RESULTADOS / 'e1a2_opt_bce' / 'corridas.csv').set_index(['escalon', 'semilla'])
        for fila in saturacion.itertuples():
            self.assertEqual(fila.total_pixeles, corridas.loc[(fila.escalon, fila.semilla), 'total_pixeles'])

    def test_eta_adaptativo_no_se_va_a_cero(self) -> None:
        saturacion = pd.read_csv(RESULTADOS / 'e1a2_eta_adaptativo' / 'saturacion.csv')
        adaptativo = saturacion[saturacion.escalon.str.startswith('eta_adaptativo')]
        self.assertTrue((adaptativo.eta_min > 0.0005).all())


class GrillaTests(unittest.TestCase):
    def test_la_distancia_minima_entre_letras_es_2(self) -> None:
        X, etiquetas = cargar_font()
        minima = minima_entre_letras(X, etiquetas).iloc[0]
        self.assertEqual(minima.distancia_minima, 2)
        self.assertEqual(minima.pares, 'l y |')

    def test_las_salidas_unicas_no_son_letras_ni_se_repiten(self) -> None:
        unicas = pd.read_csv(RESULTADOS / 'e1a4_grilla' / 'unicas.csv', dtype={'pixeles': str})
        X, _ = cargar_font()
        letras = {''.join(str(int(v)) for v in fila) for fila in X}
        self.assertTrue(unicas.pixeles.is_unique)
        self.assertTrue((unicas.distancia > 0).all())
        self.assertFalse(set(unicas.pixeles) & letras)
        self.assertTrue((unicas.pixeles.str.len() == 35).all())
        self.assertLessEqual(unicas.puntos_de_grilla.sum(), 50 * 50)


class VarianzaExplicadaTests(unittest.TestCase):
    def test_vale_uno_si_el_parametro_depende_solo_de_la_celda(self) -> None:
        rng = np.random.default_rng(0)
        mu = rng.uniform(-1, 1, size=(2000, 2))
        celdas = celda_de(mu, 8)
        valores = np.sin(celdas.astype(float))           # una función no lineal de la celda
        self.assertAlmostEqual(varianza_explicada(valores, celdas), 1.0, places=12)

    def test_con_ruido_queda_cerca_del_azar(self) -> None:
        rng = np.random.default_rng(1)
        mu = rng.uniform(-1, 1, size=(500, 2))
        valores = rng.normal(size=500)
        explicada = varianza_explicada(valores, celda_de(mu, 8))
        self.assertLess(explicada, 0.25)                  # 64 celdas y 500 puntos: ≈ 63/499 por azar

    def test_es_el_mismo_vae_que_e2c_justa(self) -> None:
        """Las correlaciones lineales coinciden con las de e2c_justa: es el mismo modelo."""
        celdas = pd.read_csv(RESULTADOS / 'e2c_parametros' / 'celdas.csv')
        justa = pd.read_csv(RESULTADOS / 'e2c_justa' / 'correlaciones.csv')
        r1 = justa[justa.eje == 'z1'].set_index(['semilla', 'parametro']).r
        for fila in celdas.itertuples():
            self.assertAlmostEqual(fila.r_mu1, r1.loc[(fila.semilla, fila.parametro)], places=9)


class PresentacionTests(unittest.TestCase):
    def test_reparto_en_bloques_contiguos_y_en_orden(self) -> None:
        laminas = json.loads((RAIZ / 'docs' / 'presentacion' / 'laminas.json').read_text(encoding='utf-8'))
        quienes = [l['presenta'] for l in laminas]
        orden = [q for i, q in enumerate(quienes) if i == 0 or q != quienes[i - 1]]
        self.assertEqual(orden, ['Louis', 'Gago', 'Agus', 'Canzo', 'Bruno'])   # cada uno aparece en un solo bloque

    def test_las_letras_elegidas_existen_y_estan_lejos(self) -> None:
        config = json.loads((RAIZ / 'config' / 'elegidas.json').read_text(encoding='utf-8'))['letras_grilla']
        unicas = pd.read_csv(RAIZ / config['origen']).set_index('numero')
        self.assertTrue(1 <= len(config['numeros']) <= 3)
        self.assertTrue((unicas.loc[config['numeros'], 'distancia'] >= 3).all())


if __name__ == '__main__':
    unittest.main()
