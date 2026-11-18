"""Consigna 1a.4 · Interpolación fina de la "o" a la "e" (punto 10 de la última iteración de
la presentación).

Pregunta: recorriendo en el latente el segmento que une el código de la "o" con el de la
"e", ¿a qué distancia de Hamming de la letra más cercana del dataset queda la salida en cada
paso, y dónde cae ese z respecto de las 32 letras? ¿Hay algún paso a 3 píxeles o más de toda
letra del dataset? (Si hay, el grupo juzga si se lee como una letra.)

El modelo es el autoencoder de 1a.3/1a.4 (la red elegida en 1a.1; config/e1a3_e1a4.json) reentrenado con
la misma semilla: el latente es el mismo que en results/e1a3_e1a4/latente.csv (se verifica).
La interpolación de e1a3_e1a4 tenía 9 pasos; acá son los del config.
Sostén: Autoencoders diap. 33 (moverse en el latente entre dos patrones) y 34 (decodificar
un z que no es el código de ningún patrón).

Uso, desde TP5/ y en primer plano:
    .venv/bin/python -m experiments.e1a4_interpolacion config/e1a4_interpolacion.json

Salida en results/e1a4_interpolacion/: pasos.csv, candidatas.csv, fig_interpolacion.png, config.json.
"""
import json
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.comun import COLOR, RAIZ, TINTA, TINTA_SECUNDARIA, aplicar_estilo
from experiments.e1a3_e1a4_latente import como_texto, dibujar_muestra, entrenar_modelo
from src.data.font import cargar_font
from src.modelos.generacion import interpolar, vecino_mas_cercano
from src.modelos.metricas import binarizar

COLUMNAS_TIRA = 14


def recorrer(ae, X: np.ndarray, etiquetas: list[str], par: list[str], pasos: int, codificacion: str):
    """Decodifica `pasos` puntos del segmento entre los códigos del par. Devuelve (tabla, salidas, Z)."""
    Z = ae.codificar(X)
    camino = interpolar(Z[etiquetas.index(par[0])], Z[etiquetas.index(par[1])], pasos)
    salidas = binarizar(ae.decodificar(camino), codificacion)
    vecinos, distancias = vecino_mas_cercano(salidas, X)
    tabla = pd.DataFrame({'paso': np.arange(pasos), 't': np.linspace(0.0, 1.0, pasos), 'z1': camino[:, 0],
                          'z2': camino[:, 1], 'vecino': [etiquetas[v] for v in vecinos], 'distancia': distancias,
                          'pixeles': [como_texto(s) for s in salidas]})
    return tabla, salidas, Z


def figura(tabla: pd.DataFrame, salidas: np.ndarray, Z: np.ndarray, etiquetas: list[str], par: list[str],
           minima: int, ruta) -> None:
    filas_tira = int(np.ceil(len(tabla) / COLUMNAS_TIRA))
    fig = plt.figure(figsize=(15, 6.2))
    grilla = fig.add_gridspec(filas_tira, COLUMNAS_TIRA + 9, wspace=0.25, hspace=0.6)
    for i, fila in tabla.iterrows():
        ax = fig.add_subplot(grilla[i // COLUMNAS_TIRA, i % COLUMNAS_TIRA])
        dibujar_muestra(ax, salidas[i], f"{fila.paso}: {fila.distancia} px\n({fila.vecino})")
        if fila.distancia >= minima:
            for borde in ax.spines.values():
                borde.set_color(COLOR['AE'])
                borde.set_linewidth(2.2)
    ax = fig.add_subplot(grilla[:, COLUMNAS_TIRA + 1:])
    ax.scatter(Z[:, 0], Z[:, 1], s=14, color=TINTA_SECUNDARIA, zorder=2)
    for (z1, z2), etiqueta in zip(Z, etiquetas):
        ax.annotate(etiqueta, (z1, z2), xytext=(3, 3), textcoords='offset points', fontsize=9, color=TINTA)
    tope = int(max(minima, tabla.distancia.max()))
    puntos = ax.scatter(tabla.z1, tabla.z2, c=tabla.distancia, cmap=plt.get_cmap('Blues', tope + 1), vmin=-0.5,
                        vmax=tope + 0.5, s=26, edgecolors=COLOR['AE'], linewidths=0.6, zorder=3)
    lejanos = tabla[tabla.distancia >= minima]
    ax.scatter(lejanos.z1, lejanos.z2, s=80, facecolors='none', edgecolors=COLOR['AE'], linewidths=1.6, zorder=4)
    barra = fig.colorbar(puntos, ax=ax, fraction=0.04, pad=0.02, ticks=range(tope + 1))
    barra.set_label('Píxeles distintos de la letra más cercana')
    ax.set_xlabel('z1 (unidades del latente)')
    ax.set_ylabel('z2 (unidades del latente)')
    fig.supxlabel(f'Izquierda: salida decodificada en cada paso de “{par[0]}” a “{par[1]}”; debajo, el paso, la distancia de Hamming '
                  f'a la letra más cercana y esa letra. Recuadro azul: a {minima} px o más de toda letra. Derecha: los mismos pasos en el latente.',
                  fontsize=10)
    fig.savefig(ruta)
    plt.close(fig)


def main(ruta_config: str) -> None:
    aplicar_estilo()
    config = json.loads(open(ruta_config, encoding='utf-8').read())
    modelo = json.loads((RAIZ / config['config_modelo']).read_text(encoding='utf-8'))
    salida = RAIZ / 'results' / config['experimento']
    salida.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ruta_config, salida / 'config.json')
    X, etiquetas = cargar_font()
    ae, fila = entrenar_modelo(modelo, config['semilla'], X)
    assert fila['logro'], fila
    tabla, salidas, Z = recorrer(ae, X, etiquetas, config['par'], config['pasos'], modelo['codificacion'])
    # El latente es el mismo que el de e1a3_e1a4 (misma receta y misma semilla)
    previo = pd.read_csv(RAIZ / 'results' / 'e1a3_e1a4' / 'latente.csv')
    previo = previo[(previo.modelo == 'AE') & (previo.semilla == config['semilla'])].set_index('etiqueta')
    np.testing.assert_allclose(Z, previo.loc[etiquetas, ['z1', 'z2']].to_numpy(), rtol=1e-9, atol=1e-9)
    assert len(tabla) == config['pasos']
    candidatas = tabla[tabla.distancia >= config['distancia_minima_candidata']]
    tabla.to_csv(salida / 'pasos.csv', index=False)
    candidatas.to_csv(salida / 'candidatas.csv', index=False)
    figura(tabla, salidas, Z, etiquetas, config['par'], config['distancia_minima_candidata'], salida / 'fig_interpolacion.png')
    print(tabla[['paso', 't', 'z1', 'z2', 'vecino', 'distancia']].to_string(index=False))
    print(f"{len(candidatas)} pasos a ≥ {config['distancia_minima_candidata']} px de toda letra")


if __name__ == '__main__':
    main(sys.argv[1])
