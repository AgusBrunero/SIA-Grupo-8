"""Consigna 1b · Línea de referencia del DAE: el decodificador "letra más cercana".

Pregunta: con el mismo ruido con que se probó el DAE, ¿cuántas letras recupera exactamente
un decodificador que no aprende nada y devuelve la letra del dataset a menor distancia de
Hamming de la entrada ruidosa? ¿Cuántas entradas ruidosas quedan estrictamente más cerca de
otra letra que de la original (y entonces ningún decodificador que mire solo la entrada
puede recuperarlas sin equivocarse)?

  (a) % de recuperaciones exactas de la letra más cercana; si varias letras empatan en
      distancia se elige una al azar (se reporta el valor esperado: 1/k si la original
      está entre las k empatadas);
  (b) % de entradas ruidosas estrictamente más cerca de otra letra que de la original;
  (c) % de reconstrucciones exactas del DAE elegido (B, entrenado con p = 0,1), leído de
      results/e1b/resumen.csv; no se recalcula.

El ruido es exactamente el de la evaluación de e1b: para cada semilla y nivel de prueba,
el generador SeedSequence([semilla, 3, índice del nivel]) y `realizaciones` llamadas a
sal_y_pimienta, en el mismo orden que evaluar_con_ruido (src/modelos/dae.py).
Sostén: Autoencoders diap. 22 (ruido salt-and-pepper y DAE). La asignación por distancia de
Hamming a la letra más cercana es la misma que usa la matriz de confusión de e1b.

Uso, desde TP5/ y en primer plano:
    .venv/bin/python -m experiments.e1b_referencia config/e1b.json

Salida en results/e1b/: referencia.csv, referencia_semillas.csv, fig_referencia.png.
"""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.comun import COLOR, RAIZ, TINTA_SECUNDARIA, aplicar_estilo
from src.data.font import cargar_font
from src.data.ruido import sal_y_pimienta

ARQUITECTURA_ELEGIDA, P_ENTRENAMIENTO_ELEGIDO = 'B', 0.1


def medir_entradas(ruidosas: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Para cada entrada ruidosa i (versión de la letra i):
    devuelve (probabilidad de que la letra más cercana, con empate al azar, sea la original;
              si queda estrictamente más cerca de otra letra que de la original)."""
    distancias = (ruidosas[:, None, :] != X[None, :, :]).sum(axis=2)
    propia = distancias[np.arange(len(X)), np.arange(len(X))]
    minima = distancias.min(axis=1)
    empatadas = (distancias == minima[:, None]).sum(axis=1)
    acierto = np.where(propia == minima, 1.0 / empatadas, 0.0)
    return acierto, propia > minima


def correr(config: dict, X: np.ndarray) -> pd.DataFrame:
    filas = []
    for indice, p in enumerate(config['niveles']):
        for semilla in config['semillas']:
            rng = np.random.default_rng(np.random.SeedSequence([semilla, 3, indice]))
            aciertos, otra, alterados = [], [], []
            for _ in range(config['realizaciones']):
                ruidosa = sal_y_pimienta(X, p, rng)
                acierto, mas_cerca_de_otra = medir_entradas(ruidosa, X)
                aciertos.append(acierto)
                otra.append(mas_cerca_de_otra)
                alterados.append((ruidosa != X).sum(axis=1))
            filas.append({'p_prueba': p, 'semilla': semilla, 'n': len(X) * config['realizaciones'],
                          'alterados_media': float(np.concatenate(alterados).mean()),
                          'letra_cercana_exacta': float(np.concatenate(aciertos).mean()),
                          'mas_cerca_de_otra': float(np.concatenate(otra).mean())})
    return pd.DataFrame(filas)


def resumir(config: dict, semillas: pd.DataFrame, resumen_dae: pd.DataFrame) -> pd.DataFrame:
    grupos = semillas.groupby('p_prueba')
    # Conteo: una fila por semilla en cada nivel, cada una con 32 letras × realizaciones
    assert (grupos.size() == len(config['semillas'])).all()
    assert (semillas.n == 32 * config['realizaciones']).all()
    tabla = grupos.agg(semillas=('semilla', 'size'), entradas=('n', 'sum'),
                       letra_cercana_media=('letra_cercana_exacta', 'mean'),
                       letra_cercana_desvio=('letra_cercana_exacta', 'std'),
                       mas_cerca_de_otra_media=('mas_cerca_de_otra', 'mean'),
                       mas_cerca_de_otra_desvio=('mas_cerca_de_otra', 'std')).reset_index()
    assert (tabla.entradas == len(config['semillas']) * 32 * config['realizaciones']).all()
    dae = resumen_dae[(resumen_dae.arquitectura == ARQUITECTURA_ELEGIDA)
                      & np.isclose(resumen_dae.p_entrenamiento, P_ENTRENAMIENTO_ELEGIDO)].set_index('p_prueba')
    assert len(dae) == len(config['niveles'])
    tabla['dae_B_exactos_media'] = dae.loc[tabla.p_prueba, 'exactos_media'].to_numpy()
    tabla['dae_B_exactos_desvio'] = dae.loc[tabla.p_prueba, 'exactos_desvio'].to_numpy()
    tabla['dae_B_hasta_1px_media'] = dae.loc[tabla.p_prueba, 'hasta_1px_media'].to_numpy()
    return tabla


def figura(tabla: pd.DataFrame, ruta) -> None:
    p = tabla.p_prueba.to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.errorbar(p, 100 * tabla.dae_B_exactos_media, yerr=100 * tabla.dae_B_exactos_desvio, color=COLOR['DAE'],
                marker='o', linewidth=2, capsize=3, label='DAE elegido (B, entrenado con p = 0,1)')
    ax.errorbar(p, 100 * tabla.letra_cercana_media, yerr=100 * tabla.letra_cercana_desvio, color=TINTA_SECUNDARIA,
                marker='s', linewidth=2, linestyle='--', capsize=3, label='Letra más cercana a la entrada ruidosa')
    ax.bar(p, 100 * tabla.mas_cerca_de_otra_media, width=0.018, color=TINTA_SECUNDARIA, alpha=0.25,
           label='Entradas más cerca de otra letra que de la original')
    ax.set_xticks(p, [f'{v:g}'.replace('.', ',') for v in p])
    ax.set_xlabel('p de prueba (probabilidad de invertir cada píxel)')
    ax.set_ylabel('% de las 3200 entradas de cada nivel\n(media ± desvío entre semillas)')
    ax.set_ylim(0, 102)
    ax.legend(fontsize=9, loc='lower left')
    fig.savefig(ruta)
    plt.close(fig)


def main(ruta_config: str) -> None:
    aplicar_estilo()
    config = json.loads(open(ruta_config, encoding='utf-8').read())
    salida = RAIZ / 'results' / config['experimento']
    X, _ = cargar_font()
    semillas = correr(config, X)
    tabla = resumir(config, semillas, pd.read_csv(salida / 'resumen.csv'))
    semillas.to_csv(salida / 'referencia_semillas.csv', index=False)
    tabla.to_csv(salida / 'referencia.csv', index=False)
    figura(tabla, salida / 'fig_referencia.png')
    print(tabla.to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1])
