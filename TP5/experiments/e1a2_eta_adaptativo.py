"""Consigna 1a.2 · Por qué falla η adaptativo con entropía cruzada (punto 2b de la iteración final 2).

Pregunta: con entropía cruzada, η adaptativo no lleva ninguna semilla a ≤ 1 px por letra
(0 de 10 con cualquier η inicial, results/e1a2_optimizadores/todas.csv), mientras que GD
con η fijo y bajo sí llega. ¿Diverge? ¿η se va a 0? ¿O se queda trabado?

  - Curvas de error, píxeles mal y η por época de las semillas 0 y 1, leídas de
    results/e1a2_opt_bce/curvas.csv (η adaptativo con η inicial 0,003), junto a GD con
    η = 0,003 fijo, las mismas semillas.
  - Saturación al final del entrenamiento (mismas corridas, reentrenadas): % de salidas de
    las neuronas tanh ocultas con |V| > 0,99, y píxeles incorrectos con la salida pegada al
    valor opuesto (el criterio de la lámina de saturación, Clase 11 diap. 40–43). Se comparan
    η adaptativo, GD con η fijo y Adam (10 de 10 con entropía cruzada).

η adaptativo (Clase 12.1 diap. 6–7): Δη = +a si el error baja k épocas seguidas, −bη si
sube k′ seguidas; a = 0,002, b = 0,5, k = 5, k′ = 3 (config/e1a2.json, E6).

Uso, desde TP5/ y en primer plano:
    .venv/bin/python -m experiments.e1a2_eta_adaptativo

Salida en results/e1a2_eta_adaptativo/: saturacion.csv, eta.csv, fig_curvas.png.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.comun import COLOR, RAIZ, TINTA_SECUNDARIA, aplicar_estilo
from experiments.e1a2_escalera import UMBRAL_SATURACION, cargar_datos, construir
from src.modelos.autoencoder import entrenar, evaluar
from src.nn.optimizadores import EtaAdaptativo, crear

SALIDA = RAIZ / 'results' / 'e1a2_eta_adaptativo'
CONFIG = RAIZ / 'config' / 'e1a2_opt_bce.json'
SEMILLAS = (0, 1)
CASOS = {'eta_adaptativo eta=0.003': ('η adaptativo (η inicial 0,003)', COLOR['DAE']),
         'gd eta=0.003': ('GD, η = 0,003 fijo', TINTA_SECUNDARIA),
         'adam eta=0.001': ('Adam, η = 0,001', COLOR['AE'])}
UMBRAL_TANH = 0.99


def saturacion(config: dict, escalon: dict, semilla: int, X: np.ndarray) -> dict:
    """Reentrena una corrida igual que e1a2_escalera y mide la saturación al final."""
    ae = construir(config, escalon, semilla)
    hiper = dict(escalon['optimizador'])
    optimizador = crear(hiper.pop('nombre'), **hiper)
    adaptativo = EtaAdaptativo(optimizador, **escalon['eta_adaptativo']) if escalon.get('eta_adaptativo') else None
    rng = np.random.default_rng(np.random.SeedSequence([semilla, 1]))
    historial = entrenar(ae, X, X, optimizador, config['epocas'], escalon['tamano_lote'], rng, perdida=escalon['perdida'],
                         codificacion=config['codificacion'], reduccion=config['reduccion'], eta_adaptativo=adaptativo)
    _, incorrectos = evaluar(ae, X, X, escalon['perdida'], config['codificacion'])
    # Salidas de las capas ocultas tanh: la primera del encoder y la primera del decoder
    V_encoder = ae.encoder.salidas_por_capa(X)[0]
    V_decoder = ae.decoder.salidas_por_capa(ae.codificar(X))[0]
    ocultas = np.concatenate([V_encoder.ravel(), V_decoder.ravel()])
    salida = ae.reconstruir(X)
    mal = (salida > 0.5) != (X > 0.5)
    return {'escalon': escalon['id'], 'semilla': semilla, 'total_pixeles': int(incorrectos.sum()),
            'ocultas_saturadas': float((np.abs(ocultas) > UMBRAL_TANH).mean()),
            'pixeles_mal': int(mal.sum()),
            'pixeles_mal_saturados': int((mal & (np.abs(salida - X) > UMBRAL_SATURACION)).sum()),
            'eta_final': float(historial.eta[-1]), 'eta_max': float(max(historial.eta)), 'eta_min': float(min(historial.eta))}


def figura(curvas: pd.DataFrame, ruta) -> None:
    fig, ejes = plt.subplots(2, len(SEMILLAS), figsize=(12, 6.4), sharex=True)
    for columna, semilla in enumerate(SEMILLAS):
        for escalon, (nombre, color) in CASOS.items():
            g = curvas[(curvas.escalon == escalon) & (curvas.semilla == semilla)]
            ejes[0, columna].plot(g.epoca, g.error, color=color, linewidth=1.6, label=nombre)
            ejes[1, columna].plot(g.epoca, g.eta, color=color, linewidth=1.2, label=nombre)
        ejes[0, columna].set_yscale('log')
        ejes[0, columna].set_title(f'Semilla {semilla}', fontsize=11)
        ejes[1, columna].set_xlabel('Época')
    ejes[0, 0].set_ylabel('Error (entropía cruzada sumada,\nescala log)')
    ejes[1, 0].set_ylabel('η al final de la época')
    ejes[0, 0].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def main() -> None:
    aplicar_estilo()
    SALIDA.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    escalones = {e['id']: e for e in config['escalones']}
    X, _ = cargar_datos(config['codificacion'])
    curvas = pd.read_csv(RAIZ / 'results' / 'e1a2_opt_bce' / 'curvas.csv')
    curvas = curvas[curvas.escalon.isin(CASOS) & curvas.semilla.isin(SEMILLAS)]
    assert curvas.groupby(['escalon', 'semilla']).ngroups == len(CASOS) * len(SEMILLAS)
    figura(curvas, SALIDA / 'fig_curvas.png')
    # η por tramos de 1000 épocas, para leer la figura con números
    curvas.assign(tramo=(curvas.epoca - 1) // 1000 * 1000).groupby(['escalon', 'semilla', 'tramo']).agg(
        error_media=('error', 'mean'), eta_min=('eta', 'min'), eta_max=('eta', 'max')).reset_index().to_csv(
        SALIDA / 'eta.csv', index=False)
    filas = [saturacion(config, escalones[escalon], semilla, X) for escalon in CASOS for semilla in SEMILLAS]
    tabla = pd.DataFrame(filas)
    # La reproducción coincide con la corrida original
    originales = pd.read_csv(RAIZ / 'results' / 'e1a2_opt_bce' / 'corridas.csv').set_index(['escalon', 'semilla'])
    for fila in tabla.itertuples():
        assert fila.total_pixeles == originales.loc[(fila.escalon, fila.semilla), 'total_pixeles']
    tabla.to_csv(SALIDA / 'saturacion.csv', index=False)
    print(tabla.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
