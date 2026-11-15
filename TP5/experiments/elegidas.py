"""Consigna 1a.4 y 2c: las muestras generadas que eligió el grupo, cada una junto a lo más
cercano del dataset.

1a.4  ¿Qué letra nueva se muestra y entre qué letras del dataset está?
2c    ¿Qué relojes generados se muestran y cuál es la imagen del dataset más cercana a cada uno?

No genera candidatas nuevas: lee las elegidas de los CSV de e1a3_e1a4 y de e2c. Para dibujar
los relojes vuelve a entrenar el VAE de e2c con la misma semilla y verifica que la distancia
a la imagen más cercana coincide con la guardada.
Sostén: Autoencoders diap. 33–34 (moverse en el latente y generar especificando z) y
diap. 74 y 90 (decodificar z ~ N(0, I)).

Uso, desde TP5/ y en primer plano:
    .venv/bin/python -m experiments.elegidas config/elegidas.json

Salida en results/elegidas/: elegidas.csv, fig_letra.png, fig_relojes.png y config.json.
"""
import json
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.comun import RAIZ, TINTA_SECUNDARIA, aplicar_estilo
from experiments.e2b_vae import dibujar, entrenar_semilla, num
from src.data.relojes import a_texto, generar_relojes
from src.modelos.generacion import vecino_por_pixel


def letra_elegida(config: dict) -> pd.DataFrame:
    """Las filas del origen, de la elegida y del destino de la interpolación, en ese orden."""
    tabla = pd.read_csv(RAIZ / config['origen'], dtype={'pixeles': str})
    tramo = tabla[(tabla.semilla == config['semilla']) & (tabla.par == config['par'])]
    filas = tramo[tramo.paso.isin([tramo.paso.min(), config['paso'], tramo.paso.max()])].sort_values('paso')
    assert len(filas) == 3 and filas.distancia.iloc[1] > 0, 'la letra elegida tiene que ser nueva (distancia > 0)'
    return filas


def relojes_elegidos(config: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """(filas de candidatas.csv, imágenes generadas, imágenes más cercanas del dataset, hora que marcan esas más cercanas)."""
    tabla = pd.read_csv(RAIZ / config['origen'])
    filas = tabla[(tabla['lambda'] == config['lambda']) & tabla.numero.isin(config['numeros'])]
    filas = filas.set_index('numero').loc[config['numeros']].reset_index()
    e2c = json.loads((RAIZ / config['config']).read_text(encoding='utf-8'))
    relojes = generar_relojes(e2c['dataset']['n'], e2c['dataset']['semilla'])
    X = relojes.X
    vae, _ = entrenar_semilla(e2c, int(filas.semilla.iloc[0]), X, config['lambda'])
    generadas = vae.decodificar(filas[['z1', 'z2']].to_numpy())
    vecino, distancia = vecino_por_pixel(generadas, X)
    assert np.array_equal(vecino, filas.vecino.to_numpy()) and np.allclose(distancia, filas.distancia.to_numpy())
    return filas, generadas, X[vecino], relojes.parametros[vecino, 0]


def figura_letra(filas: pd.DataFrame, ruta) -> None:
    fig, ejes = plt.subplots(1, 3, figsize=(5.4, 2.9))
    nombres = (f'"{filas.vecino.iloc[0]}" del dataset', f'generada: a {filas.distancia.iloc[1]} píxeles\nde la "{filas.vecino.iloc[1]}"',
               f'"{filas.vecino.iloc[2]}" del dataset')
    for ax, pixeles, nombre in zip(ejes, filas.pixeles, nombres):
        ax.imshow(np.array(list(pixeles), dtype=float).reshape(7, 5), cmap='Greys', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(visible=False)
        for borde in ax.spines.values():
            borde.set_visible(True)
        ax.set_xlabel(nombre, fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_relojes(filas: pd.DataFrame, generadas: np.ndarray, vecinas: np.ndarray, horas_vecinas: np.ndarray, ruta) -> None:
    fig, ejes = plt.subplots(len(filas), 2, figsize=(4.4, 2.4 * len(filas)), squeeze=False)
    for (ax_g, ax_v), (_, fila), generada, vecina, hora in zip(ejes, filas.iterrows(), generadas, vecinas, horas_vecinas):
        dibujar(ax_g, generada)
        dibujar(ax_v, vecina)
        ax_g.set_xlabel(f'generada (candidata {fila.numero})\nz = ({num(fila.z1)}; {num(fila.z2)})', fontsize=9)
        ax_v.set_xlabel(f'el más cercano del dataset ({a_texto(hora)})\ndistancia media por píxel: {num(fila.distancia, 3)}', fontsize=9,
                        color=TINTA_SECUNDARIA)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def main(ruta_config: str) -> None:
    aplicar_estilo()
    config = json.loads(open(ruta_config, encoding='utf-8').read())
    salida = RAIZ / 'results' / config['experimento']
    salida.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ruta_config, salida / 'config.json')
    letra = letra_elegida(config['letra'])
    relojes, generadas, vecinas, horas_vecinas = relojes_elegidos(config['relojes'])
    figura_letra(letra, salida / 'fig_letra.png')
    figura_relojes(relojes, generadas, vecinas, horas_vecinas, salida / 'fig_relojes.png')
    elegida = letra.iloc[1]
    tabla = pd.DataFrame(
        [{'item': '1a.4', 'identificador': f"{elegida.par}, semilla {elegida.semilla}, paso {elegida.paso}", 'z1': elegida.z1,
          'z2': elegida.z2, 'mas_cercano': elegida.vecino, 'distancia': elegida.distancia, 'unidad': 'píxeles distintos'}]
        + [{'item': '2c', 'identificador': f"candidata {f.numero}, λ = {f['lambda']:g}, semilla {f.semilla}", 'z1': f.z1, 'z2': f.z2,
            'mas_cercano': f.vecino, 'distancia': f.distancia, 'unidad': 'diferencia absoluta media por píxel'}
           for _, f in relojes.iterrows()])
    tabla.to_csv(salida / 'elegidas.csv', index=False)
    print(tabla.to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1])
