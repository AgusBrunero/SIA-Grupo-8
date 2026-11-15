"""Consigna 1b · DAE con réplicas ruidosas y más épocas (punto 1 de la iteración final 2).

Pregunta: el DAE elegido (B, entrenado con p = 0,1) reconstruye exactas menos letras que el
decodificador "letra más cercana" (results/e1b/referencia.csv). ¿Cambia eso con más
entrenamiento? Se prueban 1 o 5 réplicas ruidosas de cada letra por época (data
augmentation, Clase 13: cada réplica recibe ruido nuevo en cada época) y las épocas actuales
o 4 veces más. Con 5 réplicas y lote de 32 hay 5 actualizaciones por época en vez de 1.

Evaluación: la misma de e1b y de e1b_referencia: para cada semilla y nivel de prueba,
SeedSequence([semilla, 3, índice del nivel]) y evaluar_con_ruido con las mismas realizaciones.
Control: la variante ×1 con las épocas de e1b reproduce exactamente B con p = 0,1 de
results/e1b/resumen.csv.
Sostén: Autoencoders diap. 22 (DAE); Clase 13 (data augmentation).

Uso, desde TP5/ y en primer plano:
    .venv/bin/python -m experiments.e1b_replicas config/e1b_replicas.json            (entrena y analiza)
    .venv/bin/python -m experiments.e1b_replicas config/e1b_replicas.json analisis   (solo tablas y figura)

Salida en results/e1b_replicas/: celdas.csv, resumen.csv, fig_replicas.png, config.json.
"""
import json
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.comun import COLOR, RAIZ, TINTA_SECUNDARIA, aplicar_estilo
from experiments.e1a2_escalera import construir
from src.data.font import cargar_font
from src.modelos.autoencoder import entrenar
from src.modelos.dae import corruptor, evaluar_con_ruido
from src.nn.optimizadores import crear

RAMPA = ['#f6b294', '#eb6834', '#c44f20', '#6b290e']


def entrenar_variante(base: dict, capas: list[int], semilla: int, X: np.ndarray, p: float, replicas: int, epocas: int):
    """Igual que una corrida de e1b, pero con `replicas` copias de cada letra por época."""
    modelo = dict(base['modelo'], capas_encoder=capas)
    ae = construir(base, modelo, semilla)
    hiper = dict(modelo['optimizador'])
    optimizador = crear(hiper.pop('nombre'), **hiper)
    rng_lotes = np.random.default_rng(np.random.SeedSequence([semilla, 1]))
    rng_ruido = np.random.default_rng(np.random.SeedSequence([semilla, 2]))
    datos = np.tile(X, (replicas, 1))
    historial = entrenar(ae, datos, datos, optimizador, epocas, modelo['tamano_lote'], rng_lotes,
                         perdida=modelo['perdida'], codificacion='01', reduccion=base['reduccion'],
                         corromper=corruptor(p, rng_ruido))
    return ae, historial.segundos


def correr(config: dict, base: dict, salida) -> None:
    X, _ = cargar_font()
    capas = next(a for a in base['arquitecturas'] if a['id'] == config['arquitectura'])['capas_encoder']
    filas = []
    for variante in config['variantes']:
        for semilla in base['semillas']:
            ae, segundos = entrenar_variante(base, capas, semilla, X, config['p_entrenamiento'],
                                             variante['replicas'], variante['epocas'])
            for indice, p_prueba in enumerate(base['niveles']):
                rng_prueba = np.random.default_rng(np.random.SeedSequence([semilla, 3, indice]))
                tabla = evaluar_con_ruido(ae, X, p_prueba, base['realizaciones'], rng_prueba)
                filas.append({'variante': variante['id'], 'replicas': variante['replicas'], 'epocas': variante['epocas'],
                              'semilla': semilla, 'p_prueba': p_prueba, 'n': len(tabla),
                              'alterados_media': tabla.alterados.mean(), 'mal_media': tabla.mal.mean(),
                              'exactos': (tabla.mal == 0).mean(), 'hasta_1px': (tabla.mal <= 1).mean(),
                              'segundos': segundos})
            print(f"{variante['id']:<18} semilla {semilla}  {segundos:.1f} s  "
                  f"exactos con p = 0,1: {filas[-3]['exactos']:.3f}", flush=True)
    pd.DataFrame(filas).to_csv(salida / 'celdas.csv', index=False)


def resumir(config: dict, base: dict, celdas: pd.DataFrame) -> pd.DataFrame:
    grupos = celdas.groupby(['variante', 'replicas', 'epocas', 'p_prueba'], sort=False)
    # Conteo: una fila por semilla en cada celda, cada una con 32 letras × realizaciones
    assert (grupos.size() == len(base['semillas'])).all()
    assert len(grupos) == len(config['variantes']) * len(base['niveles'])
    assert (celdas.n == 32 * base['realizaciones']).all()
    resumen = grupos.agg(semillas=('semilla', 'size'), exactos_media=('exactos', 'mean'), exactos_desvio=('exactos', 'std'),
                         hasta_1px_media=('hasta_1px', 'mean'), hasta_1px_desvio=('hasta_1px', 'std'),
                         mal_media=('mal_media', 'mean'), segundos_media=('segundos', 'mean')).reset_index()
    referencia = pd.read_csv(RAIZ / 'results' / 'e1b' / 'referencia.csv').set_index('p_prueba')
    resumen['letra_cercana_media'] = referencia.loc[resumen.p_prueba, 'letra_cercana_media'].to_numpy()
    return resumen


def controlar(config: dict, base: dict, resumen: pd.DataFrame) -> None:
    """La variante ×1 con las épocas de e1b tiene que dar exactamente lo de e1b/resumen.csv."""
    original = pd.read_csv(RAIZ / 'results' / 'e1b' / 'resumen.csv')
    original = original[(original.arquitectura == config['arquitectura'])
                        & np.isclose(original.p_entrenamiento, config['p_entrenamiento'])].set_index('p_prueba')
    igual = resumen[(resumen.replicas == 1) & (resumen.epocas == base['epocas'])].set_index('p_prueba')
    assert len(igual) == len(base['niveles'])
    for columna in ('exactos_media', 'hasta_1px_media', 'mal_media'):
        np.testing.assert_allclose(igual[columna], original.loc[igual.index, columna], rtol=1e-12)


def figura(resumen: pd.DataFrame, ruta) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, (columna, nombre) in zip(ejes, (('exactos', 'exactas (0 píxeles incorrectos)'), ('hasta_1px', 'con ≤ 1 píxel incorrecto'))):
        for (variante, g), color in zip(resumen.groupby('variante', sort=False), RAMPA):
            ax.errorbar(g.p_prueba, 100 * g[f'{columna}_media'], yerr=100 * g[f'{columna}_desvio'], color=color, marker='o',
                        linewidth=2, capsize=3, label=f'DAE B, {variante.replace("x", "×", 1).replace("epocas", "épocas")}')
        ref = resumen.drop_duplicates('p_prueba')
        ax.plot(ref.p_prueba, 100 * ref.letra_cercana_media, color=TINTA_SECUNDARIA, linestyle='--', marker='s', linewidth=2,
                label='Letra más cercana a la entrada ruidosa')
        ax.set_xticks(ref.p_prueba, [f'{v:g}'.replace('.', ',') for v in ref.p_prueba])
        ax.set_xlabel('p de prueba (probabilidad de invertir cada píxel)')
        ax.set_ylabel(f'% de reconstrucciones {nombre}\n(media ± desvío entre 5 semillas)')
        ax.set_ylim(0, 102)
    ejes[0].legend(fontsize=9, loc='lower left')
    fig.supxlabel('DAE B entrenado con p = 0,1; ×1 o ×5 réplicas ruidosas de cada letra por época', fontsize=10, color=COLOR['DAE'])
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def analizar(config: dict, base: dict, salida) -> None:
    aplicar_estilo()
    resumen = resumir(config, base, pd.read_csv(salida / 'celdas.csv'))
    controlar(config, base, resumen)
    resumen.to_csv(salida / 'resumen.csv', index=False)
    figura(resumen, salida / 'fig_replicas.png')
    print(resumen[['variante', 'p_prueba', 'exactos_media', 'exactos_desvio', 'hasta_1px_media', 'letra_cercana_media',
                   'segundos_media']].round(4).to_string(index=False))


def main(ruta_config: str, modo: str = 'todo') -> None:
    config = json.loads(open(ruta_config, encoding='utf-8').read())
    base = json.loads((RAIZ / config['config_base']).read_text(encoding='utf-8'))
    salida = RAIZ / 'results' / config['experimento']
    salida.mkdir(parents=True, exist_ok=True)
    if modo == 'todo':
        shutil.copyfile(ruta_config, salida / 'config.json')
        correr(config, base, salida)
    analizar(config, base, salida)


if __name__ == '__main__':
    main(*sys.argv[1:])
