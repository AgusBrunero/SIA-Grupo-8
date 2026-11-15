"""Consigna 2b · Qué codifica el latente del VAE (punto 5 de la iteración final 2).

Pregunta: ¿qué guarda el latente del VAE de la hora que marca cada reloj? La hora es cíclica,
así que se mira como ángulo: su seno y su coseno. Pearson (results/e2c_justa/correlaciones.csv)
solo ve relaciones lineales; la media por celda detecta también las no lineales.

  - fig_parametros.png: el scatter de μ(x) de los 500 relojes, coloreado por la hora y por su seno y coseno
    (un panel por parámetro; semilla de figuras).
  - fig_celdas.png: la media de cada parámetro por celda de una grilla de 8 × 8 sobre el
    rectángulo de μ (celdas vacías en blanco).
  - celdas.csv: por semilla y parámetro, la fracción de la varianza del parámetro que
    explican las medias por celda (varianza entre celdas / varianza total). Vale 1 si el
    parámetro queda determinado por la celda y no depende de que la relación sea lineal. Con
    64 celdas y 500 relojes el azar ya explica algo: la referencia es la misma fracción con el
    parámetro mezclado al azar (media de `mezclas` permutaciones). Medida nuestra.

VAE con λ = 1, la misma receta que e2c y e2c_justa (config/e2c_justa.json), 5 semillas.
Sostén: Autoencoders diap. 74–78 (VAE, μ(x)); el dataset y sus parámetros son nuestros (D en
src/data/relojes.py).

Uso, desde TP5/ y en primer plano:
    .venv/bin/python -m experiments.e2c_latente_parametros config/e2c_parametros.json

Salida en results/e2c_parametros/: mu.csv, celdas.csv, fig_parametros.png, fig_celdas.png, config.json.
"""
import json
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.comun import RAIZ, aplicar_estilo
from experiments.e2c_generacion_justa import ROTULOS, factores
from experiments.e2c_latente_generacion import entrenar_vae_modelo
from src.data.relojes import generar_relojes


def celda_de(mu: np.ndarray, n: int) -> np.ndarray:
    """Índice de celda (0 … n² − 1) de cada punto en una grilla de n × n sobre el rectángulo de μ."""
    minimo, maximo = mu.min(axis=0), mu.max(axis=0)
    indices = np.clip(((mu - minimo) / (maximo - minimo) * n).astype(int), 0, n - 1)
    return indices[:, 0] * n + indices[:, 1]


def varianza_explicada(valores: np.ndarray, celdas: np.ndarray) -> float:
    """Varianza entre celdas sobre varianza total: 1 − (suma de cuadrados dentro de celdas) / (total)."""
    total = ((valores - valores.mean()) ** 2).sum()
    medias = pd.Series(valores).groupby(celdas).transform('mean').to_numpy()
    return float(1.0 - ((valores - medias) ** 2).sum() / total)


def medir(mu: np.ndarray, parametros: np.ndarray, n: int, mezclas: int, rng: np.random.Generator) -> list[dict]:
    celdas = celda_de(mu, n)
    filas = []
    for nombre, valores in factores(parametros).items():
        azar = [varianza_explicada(rng.permutation(valores), celdas) for _ in range(mezclas)]
        filas.append({'parametro': nombre, 'explicada': varianza_explicada(valores, celdas),
                      'azar_media': float(np.mean(azar)), 'azar_max': float(np.max(azar)),
                      'r_mu1': float(np.corrcoef(mu[:, 0], valores)[0, 1]), 'r_mu2': float(np.corrcoef(mu[:, 1], valores)[0, 1]),
                      'celdas_ocupadas': int(len(np.unique(celdas)))})
    return filas


def figura_scatter(mu: np.ndarray, parametros: np.ndarray, ruta) -> None:
    """μ(x) de los 500 relojes coloreado por la hora (mapa cíclico) y por su seno y coseno."""
    paneles = {'hora': (parametros[:, 0], 'twilight', 'Hora que marca el reloj'), **{
        nombre: (valores, 'viridis', ROTULOS[nombre].replace(chr(10), ' ')) for nombre, valores in factores(parametros).items()}}
    fig, ejes = plt.subplots(1, len(paneles), figsize=(5.2 * len(paneles), 4.6))
    for ax, (valores, mapa, rotulo) in zip(ejes, paneles.values()):
        puntos = ax.scatter(mu[:, 0], mu[:, 1], c=valores, cmap=mapa, s=9, linewidths=0)
        fig.colorbar(puntos, ax=ax, fraction=0.046, pad=0.02)
        ax.set_xlabel(rotulo, fontsize=10)
        ax.set_aspect('equal', adjustable='datalim')
    fig.supxlabel('μ(x) de los 500 relojes en el latente del VAE (λ = 1). Ejes: μ₁ horizontal, μ₂ vertical.', fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_celdas(mu: np.ndarray, parametros: np.ndarray, n: int, explicada: dict, ruta) -> None:
    celdas = celda_de(mu, n)
    valores_por_factor = factores(parametros)
    fig, ejes = plt.subplots(1, len(valores_por_factor), figsize=(5.2 * len(valores_por_factor), 4.6))
    for ax, (nombre, valores) in zip(ejes, valores_por_factor.items()):
        medias = np.full(n * n, np.nan)
        serie = pd.Series(valores).groupby(celdas).mean()
        medias[serie.index] = serie.to_numpy()
        # Fila de la imagen = índice de μ₂ de mayor a menor, columna = índice de μ₁
        imagen = ax.imshow(medias.reshape(n, n).T[::-1], cmap='viridis', interpolation='nearest')
        fig.colorbar(imagen, ax=ax, fraction=0.046, pad=0.02)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(visible=False)
        ax.set_xlabel(f"{ROTULOS[nombre].replace(chr(10), ' ')}\nvarianza explicada {explicada[nombre]:.2f}".replace('.', ','),
                      fontsize=10)
    fig.supxlabel(f'Media por celda de una grilla de {n} × {n} sobre μ (μ₁ horizontal, μ₂ vertical; blanco: celda vacía)',
                  fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def main(ruta_config: str) -> None:
    aplicar_estilo()
    config = json.loads(open(ruta_config, encoding='utf-8').read())
    base = json.loads((RAIZ / config['config_base']).read_text(encoding='utf-8'))
    salida = RAIZ / 'results' / config['experimento']
    salida.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ruta_config, salida / 'config.json')
    relojes = generar_relojes(base['dataset']['n'], base['dataset']['semilla'])
    filas, mus = [], []
    for semilla in base['semillas']:
        modelo = entrenar_vae_modelo(base, semilla, relojes.X, base['lambda'])
        mu = modelo.codificar(relojes.X)
        mus.append(pd.DataFrame({'semilla': semilla, 'reloj': np.arange(len(mu)), 'mu1': mu[:, 0], 'mu2': mu[:, 1]}))
        rng = np.random.default_rng(np.random.SeedSequence([config['semilla_mezclas'], semilla]))
        filas += [{'semilla': semilla, **f} for f in medir(mu, relojes.parametros, config['n_celdas'], config['mezclas'], rng)]
        if semilla == base['semilla_figuras']:
            explicada = {f['parametro']: f['explicada'] for f in filas if f['semilla'] == semilla}
            figura_scatter(mu, relojes.parametros, salida / 'fig_parametros.png')
            figura_celdas(mu, relojes.parametros, config['n_celdas'], explicada, salida / 'fig_celdas.png')
        print(f'semilla {semilla} lista', flush=True)
    tabla = pd.DataFrame(filas)
    assert (tabla.groupby('parametro').size() == len(base['semillas'])).all()
    tabla.to_csv(salida / 'celdas.csv', index=False)
    pd.concat(mus).to_csv(salida / 'mu.csv', index=False)
    resumen = tabla.groupby('parametro', sort=False).agg(explicada_media=('explicada', 'mean'), explicada_desvio=('explicada', 'std'),
                                                          azar_media=('azar_media', 'mean'), azar_max=('azar_max', 'max'))
    print(resumen.round(3).to_string())


if __name__ == '__main__':
    main(sys.argv[1])
