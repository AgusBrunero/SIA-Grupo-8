"""Consigna 1a.4 · Letra nueva: grilla fina del latente (punto 3 de la iteración final 2).

Preguntas:
  (a) ¿Cuál es la distancia de Hamming mínima entre dos letras distintas del dataset? Es la
      escala con la que se juzga si una salida es "otra letra" o "la misma con ruido".
  (b) Recorriendo una grilla de 50 × 50 sobre el rectángulo que cubre los códigos del modelo
      elegido en 1a.1 (config/e1a3_e1a4.json; semilla 0), ¿qué salidas distintas que no son letras del
      dataset produce el decoder, y a qué distancia de la letra más cercana está cada una?
  (c) Figura con todas las que están a 2 px o más, ordenadas por distancia, con su z.
El grupo elige de 1 a 3 candidatas; este script no elige.

Para cada salida única se guarda cuántos puntos de la grilla la producen y un z
representativo: el punto de la grilla, de los que la producen, más cercano al promedio de
esos puntos.
Sostén: Autoencoders diap. 33–34 (moverse en el latente y decodificar z).

Uso, desde TP5/ y en primer plano:
    .venv/bin/python -m experiments.e1a4_grilla config/e1a4_grilla.json

Salida en results/e1a4_grilla/: minima.csv, unicas.csv, fig_candidatas_*.png, config.json.
"""
import json
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.comun import RAIZ, aplicar_estilo
from experiments.e1a3_e1a4_latente import como_texto, dibujar_muestra, entrenar_modelo
from src.data.font import cargar_font, distancias_hamming
from src.modelos.generacion import grilla_latente, vecino_mas_cercano
from src.modelos.metricas import binarizar

COLUMNAS = 15


def minima_entre_letras(X: np.ndarray, etiquetas: list[str]) -> pd.DataFrame:
    D = distancias_hamming(X).astype(float)
    np.fill_diagonal(D, np.inf)
    minima = D.min()
    pares = [(etiquetas[i], etiquetas[j]) for i, j in zip(*np.where(D == minima)) if i < j]
    return pd.DataFrame([{'distancia_minima': int(minima), 'pares': ' ; '.join(f'{a} y {b}' for a, b in pares)}])


def salidas_unicas(puntos: np.ndarray, salidas: np.ndarray, X: np.ndarray, etiquetas: list[str]) -> pd.DataFrame:
    """Una fila por salida binaria distinta que no es una letra del dataset."""
    claves = [como_texto(s) for s in salidas]
    vecinos, distancias = vecino_mas_cercano(salidas, X)
    tabla = pd.DataFrame({'pixeles': claves, 'z1': puntos[:, 0], 'z2': puntos[:, 1],
                          'vecino': [etiquetas[v] for v in vecinos], 'distancia': distancias})
    filas = []
    for clave, g in tabla[tabla.distancia > 0].groupby('pixeles'):
        centro = g[['z1', 'z2']].mean().to_numpy()
        representante = g.iloc[np.argmin(np.linalg.norm(g[['z1', 'z2']].to_numpy() - centro, axis=1))]
        filas.append({'pixeles': clave, 'distancia': int(g.distancia.iloc[0]), 'vecino': g.vecino.iloc[0],
                      'puntos_de_grilla': len(g), 'z1': representante.z1, 'z2': representante.z2})
    unicas = pd.DataFrame(filas).sort_values(['distancia', 'puntos_de_grilla'], ascending=[False, False]).reset_index(drop=True)
    unicas.insert(0, 'numero', np.arange(1, len(unicas) + 1))
    return unicas


def figuras(unicas: pd.DataFrame, minima: int, por_figura: int, salida) -> list[str]:
    elegibles = unicas[unicas.distancia >= minima]
    rutas = []
    for parte, inicio in enumerate(range(0, len(elegibles), por_figura), start=1):
        bloque = elegibles.iloc[inicio:inicio + por_figura]
        filas = int(np.ceil(len(bloque) / COLUMNAS))
        fig, ejes = plt.subplots(filas, COLUMNAS, figsize=(COLUMNAS * 1.05, filas * 1.75), squeeze=False)
        for ax in ejes.ravel():
            ax.axis('off')
        for ax, fila in zip(ejes.ravel(), bloque.itertuples()):
            ax.axis('on')
            muestra = np.array(list(fila.pixeles), dtype=int)
            dibujar_muestra(ax, muestra, f'#{fila.numero}: {fila.distancia} px ({fila.vecino})\n'
                                         f'z = ({fila.z1:.1f}; {fila.z2:.1f})'.replace('.', ','))
        fig.supxlabel(f'Salidas del decoder en una grilla de 50 × 50 del latente que no son letras del dataset, a ≥ {minima} px de toda '
                      'letra, ordenadas por distancia. Debajo: número, distancia, letra más cercana y un z que la produce.', fontsize=10)
        fig.tight_layout(h_pad=0.6, w_pad=0.3)
        ruta = salida / f'fig_candidatas_{parte}.png'
        fig.savefig(ruta)
        plt.close(fig)
        rutas.append(ruta.name)
    return rutas


def main(ruta_config: str) -> None:
    aplicar_estilo()
    config = json.loads(open(ruta_config, encoding='utf-8').read())
    modelo = json.loads((RAIZ / config['config_modelo']).read_text(encoding='utf-8'))
    salida = RAIZ / 'results' / config['experimento']
    salida.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ruta_config, salida / 'config.json')
    X, etiquetas = cargar_font()
    minima = minima_entre_letras(X, etiquetas)
    minima.to_csv(salida / 'minima.csv', index=False)
    ae, fila = entrenar_modelo(modelo, config['semilla'], X)
    assert fila['logro'], fila
    Z = ae.codificar(X)
    previo = pd.read_csv(RAIZ / 'results' / 'e1a3_e1a4' / 'latente.csv')
    previo = previo[(previo.modelo == 'AE') & (previo.semilla == config['semilla'])].set_index('etiqueta')
    np.testing.assert_allclose(Z, previo.loc[etiquetas, ['z1', 'z2']].to_numpy(), rtol=1e-9, atol=1e-9)
    puntos = grilla_latente(Z, config['n_grilla'])
    assert len(puntos) == config['n_grilla'] ** 2
    salidas = binarizar(ae.decodificar(puntos), modelo['codificacion'])
    unicas = salidas_unicas(puntos, salidas, X, etiquetas)
    # Ninguna salida única es una letra del dataset ni se repite
    assert (unicas.distancia > 0).all() and unicas.pixeles.is_unique
    unicas.to_csv(salida / 'unicas.csv', index=False)
    rutas = figuras(unicas, config['distancia_minima_figura'], config['por_figura'], salida)
    print(minima.to_string(index=False))
    print(f"{len(unicas)} salidas únicas que no son letras; por distancia:\n{unicas.distancia.value_counts().sort_index().to_string()}")
    print('figuras:', rutas)


if __name__ == '__main__':
    main(sys.argv[1])
