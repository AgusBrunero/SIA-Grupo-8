"""Consigna 1a.2 · Comparación justa de optimizadores (punto 2 de la última iteración de la presentación).

Pregunta: con el mismo modelo (35-20-2-20-35, ocultas tanh, latente identidad, salida
logística), dándole a cada optimizador su mejor η entre tres valores, ¿cuál aprende las 32
letras con a lo sumo 1 píxel incorrecto por letra, con error cuadrático y con entropía cruzada?
Control: error cuadrático + Adam en una red más profunda (35-30-20-10-2-10-20-30-35).

No entrena: lee los corridas.csv que escribe experiments.e1a2_escalera con
config/e1a2_opt_ec.json, config/e1a2_opt_bce.json y config/e1a2_profunda.json. Powell se
toma como está de results/e1a2/resumen.csv (escalón E10, red 35-2-35, sin η).

Mejor η de un optimizador: el que más semillas lleva a ≤ 1 px por letra; si empatan en eso,
el de menos píxeles incorrectos en total (media entre semillas). Si la diferencia de píxeles
con otro η no supera el desvío entre semillas, se lo lista en `empata_con` (CLAUDE.md §4).
Sostén: Clase 12.1 diap. 5–13 (momentum, η adaptativo, RMSProp, Adam); Autoencoders diap. 51
(entropía cruzada); Clase 11 diap. 40–43 (factor θ′ en el delta de salida).

Uso, desde TP5/:
    .venv/bin/python -m experiments.e1a2_optimizadores

Lee también results/e1a2_opt_borde_{ec,bce} (un η más allá del borde, iteración final 2) y
results/e1a2_profunda_larga (40.000 épocas).

Salida en results/e1a2_optimizadores/: todas.csv, mejor_eta.csv, profunda.csv,
fig_optimizador_costo.png.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.comun import RAIZ, RAMPA_AE, TINTA_SECUNDARIA, aplicar_estilo

RESULTADOS = RAIZ / 'results'
SALIDA = RESULTADOS / 'e1a2_optimizadores'
COSTOS = {'ec': 'Error cuadrático', 'bce': 'Entropía cruzada'}
NOMBRES = {'gd': 'Descenso por\ngradiente', 'momentum': 'Momentum', 'eta_adaptativo': 'η adaptativo',
           'rmsprop': 'RMSProp', 'adam': 'Adam'}
COLOR_COSTO = {'ec': RAMPA_AE[3], 'bce': RAMPA_AE[6]}
SEMILLAS = 10


def agregar(corridas: pd.DataFrame, costo: str) -> pd.DataFrame:
    """Una fila por (optimizador, η): semillas, logradas y píxeles totales en media ± desvío."""
    partes = corridas.escalon.str.extract(r'^(?P<optimizador>\w+) eta=(?P<eta>[\d.e-]+)$')
    assert partes.notna().all().all(), 'escalón con un id inesperado'
    datos = corridas.assign(optimizador=partes.optimizador, eta=partes.eta.astype(float))
    tabla = datos.groupby(['optimizador', 'eta']).agg(
        semillas=('semilla', 'size'), logradas=('logro', 'sum'),
        total_px_media=('total_pixeles', 'mean'), total_px_desvio=('total_pixeles', 'std'),
        max_px_media=('max_pixeles', 'mean'), error_media=('error_final', 'mean')).reset_index()
    # Conteo: exactamente SEMILLAS corridas por celda, sin repetir semilla
    assert (tabla.semillas == SEMILLAS).all()
    assert not datos.duplicated(['optimizador', 'eta', 'semilla']).any()
    return tabla.assign(costo=costo)


def mejor_eta(tabla: pd.DataFrame) -> pd.DataFrame:
    """Por (costo, optimizador): el η con más semillas logradas y, a igualdad, menos píxeles."""
    filas = []
    for (costo, optimizador), g in tabla.groupby(['costo', 'optimizador'], sort=False):
        orden = g.sort_values(['logradas', 'total_px_media'], ascending=[False, True])
        mejor = orden.iloc[0]
        otros = orden.iloc[1:]
        empatan = otros[(otros.logradas == mejor.logradas)
                        & ((otros.total_px_media - mejor.total_px_media).abs()
                           <= np.maximum(otros.total_px_desvio, mejor.total_px_desvio))]
        # En el borde: el mejor η es el menor o el mayor de los probados y el resultado todavía puede
        # mejorar (no es 10 de 10 con 0 píxeles), así que el óptimo podría estar afuera
        perfecto = mejor.logradas == SEMILLAS and mejor.total_px_media == 0
        en_borde = mejor.eta in (g.eta.min(), g.eta.max()) and not perfecto
        filas.append({**mejor.to_dict(), 'etas_probados': ' / '.join(f'{v:g}' for v in sorted(g.eta)),
                      'empata_con': ' / '.join(f'{v:g}' for v in sorted(empatan.eta)), 'en_borde': bool(en_borde)})
    return pd.DataFrame(filas)


def powell() -> dict:
    fila = pd.read_csv(RESULTADOS / 'e1a2' / 'resumen.csv').set_index('escalon').loc['E10']
    assert int(fila.semillas) == SEMILLAS
    return {'logradas': int(fila.logradas), 'total_px_media': fila.total_px_media, 'total_px_desvio': fila.total_px_desvio}


def figura(mejores: pd.DataFrame, ref_powell: dict, ruta) -> None:
    """Dos paneles con el mejor η de cada optimizador: semillas que llegan y píxeles mal."""
    optimizadores = list(NOMBRES)
    x = np.arange(len(optimizadores) + 1)
    ancho = 0.38
    fig, ejes = plt.subplots(1, 2, figsize=(13, 4.6))
    for k, costo in enumerate(COSTOS):
        m = mejores[mejores.costo == costo].set_index('optimizador').loc[optimizadores]
        desplazamiento = (k - 0.5) * ancho
        barras = ejes[0].bar(x[:-1] + desplazamiento, m.logradas, ancho, color=COLOR_COSTO[costo], label=COSTOS[costo])
        for barra, eta in zip(barras, m.eta):
            ejes[0].text(barra.get_x() + barra.get_width() / 2, barra.get_height() + 0.15, f'η = {eta:g}'.replace('.', ','),
                         ha='center', va='bottom', fontsize=7.5, color=TINTA_SECUNDARIA, rotation=90)
        # La barra de error no baja de 0 píxeles
        abajo = np.minimum(m.total_px_desvio, m.total_px_media)
        ejes[1].bar(x[:-1] + desplazamiento, m.total_px_media, ancho, yerr=[abajo, m.total_px_desvio], capsize=3,
                    color=COLOR_COSTO[costo], label=COSTOS[costo])
    ejes[0].bar(x[-1] - ancho / 2, ref_powell['logradas'], ancho, color=TINTA_SECUNDARIA, alpha=0.5)
    ejes[1].bar(x[-1] - ancho / 2, ref_powell['total_px_media'], ancho, yerr=ref_powell['total_px_desvio'], capsize=3,
                color=TINTA_SECUNDARIA, alpha=0.5, label='Powell (red 35–2–35, sin η)')
    rotulos = [NOMBRES[o] for o in optimizadores] + ['Powell']
    for ax in ejes:
        ax.set_xticks(x, rotulos, fontsize=9)
    ejes[0].set_ylim(0, SEMILLAS + 2.5)
    ejes[0].set_ylabel(f'Semillas que aprenden las 32 letras\ncon ≤ 1 píxel incorrecto (de {SEMILLAS})')
    ejes[1].set_yscale('symlog', linthresh=1)
    ejes[1].set_ylim(0, 2000)
    ejes[1].set_ylabel('Píxeles incorrectos en total, de 1120\n(media ± desvío; lineal hasta 1, log arriba)')
    fig.legend(*ejes[1].get_legend_handles_labels(), loc='upper center', ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.06))
    fig.supxlabel('Cada barra es el mejor η de ese optimizador entre tres valores, con 10 semillas', fontsize=10)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def leer_corridas(costo: str) -> pd.DataFrame:
    """Las corridas de la grilla original y, si existen, las de los η más allá del borde."""
    partes = [pd.read_csv(RESULTADOS / f'e1a2_opt_{costo}' / 'corridas.csv')]
    borde = RESULTADOS / f'e1a2_opt_borde_{costo}' / 'corridas.csv'
    if borde.exists():
        partes.append(pd.read_csv(borde))
    return pd.concat(partes, ignore_index=True)


def resumen_profunda(carpeta: str, epocas: int) -> dict:
    profunda = pd.read_csv(RESULTADOS / carpeta / 'corridas.csv')
    assert len(profunda) == SEMILLAS and profunda.escalon.nunique() == 1
    return {'red': '35-30-20-10-2-10-20-30-35', 'costo': 'ec', 'optimizador': 'adam', 'eta': 0.001, 'epocas': epocas,
            'lote': '32 (batch completo)', 'semillas': len(profunda), 'logradas': int(profunda.logro.sum()),
            'total_px_media': profunda.total_pixeles.mean(), 'total_px_desvio': profunda.total_pixeles.std(),
            'max_px_media': profunda.max_pixeles.mean()}


def main() -> None:
    aplicar_estilo()
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla = pd.concat([agregar(leer_corridas(costo), costo) for costo in COSTOS], ignore_index=True)
    # 3 η por optimizador y costo, más uno por cada caso que quedó en el borde
    assert len(tabla) >= len(COSTOS) * len(NOMBRES) * 3
    assert (tabla.groupby(['costo', 'optimizador']).size().between(3, 4)).all()
    mejores = mejor_eta(tabla)
    tabla.to_csv(SALIDA / 'todas.csv', index=False)
    mejores.to_csv(SALIDA / 'mejor_eta.csv', index=False)
    filas = [resumen_profunda(carpeta, epocas) for carpeta, epocas in (('e1a2_profunda', 10000), ('e1a2_profunda_larga', 40000))
             if (RESULTADOS / carpeta / 'corridas.csv').exists()]
    if filas:
        pd.DataFrame(filas).to_csv(SALIDA / 'profunda.csv', index=False)
        print(pd.DataFrame(filas).to_string(index=False))
    figura(mejores, powell(), SALIDA / 'fig_optimizador_costo.png')
    print(tabla.to_string(index=False))
    print(mejores[['costo', 'optimizador', 'eta', 'etas_probados', 'logradas', 'total_px_media', 'total_px_desvio', 'empata_con',
                   'en_borde']].to_string(index=False))


if __name__ == '__main__':
    main()
