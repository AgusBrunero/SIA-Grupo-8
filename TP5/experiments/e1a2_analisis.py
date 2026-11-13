"""Consigna 1a.2: tablas, figuras y observaciones de la escalera e1a2 (y de su piloto y diagnóstico).

Pregunta: por escalón, ¿qué fracción de semillas aprende los 32 patrones con ≤ 1 píxel
incorrecto, con qué error, y qué patrones fallan cuando no se llega? (punto 5 de la sesión:
vecinos en el latente y por distancia de Hamming).

Solo lee los CSV que generó experiments/e1a2_escalera.py; no entrena nada.
Sostén: el mismo de e1a2_escalera.py. Gráficos del tipo que muestra la cátedra
(Análisis de Datos diap. 14–17: scatter y barras de error).

Uso, desde TP5/:
    .venv/bin/python -m experiments.e1a2_analisis

Salida: results/e1a2/resumen.csv, fallas.csv, fig_*.png y el bloque e1a2 de docs/RESULTADOS.md.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from experiments.comun import COLOR, RAIZ, RAMPA_AE, TINTA_SECUNDARIA, aplicar_estilo, escribir_seccion
from src.data.font import cargar_font, distancias_hamming
from src.modelos.metricas import OBJETIVO_PIXELES

RESULTADOS = RAIZ / 'results'


def leer(experimento: str) -> tuple[dict, pd.DataFrame]:
    carpeta = RESULTADOS / experimento
    config = json.loads((carpeta / 'config.json').read_text(encoding='utf-8'))
    corridas = pd.read_csv(carpeta / 'corridas.csv')
    # Conteo: cada escalón tiene exactamente una corrida por semilla (evita doble conteo)
    conteo = corridas.groupby('escalon').size()
    assert set(conteo.index) == {e['id'] for e in config['escalones']}
    assert (conteo == len(config['semillas'])).all(), conteo
    assert not corridas.duplicated(['escalon', 'semilla']).any()
    return config, corridas


def resumir(config: dict, corridas: pd.DataFrame) -> pd.DataFrame:
    """Una fila por escalón, en el orden del config. Media ± desvío entre semillas."""
    filas = []
    for escalon in config['escalones']:
        c = corridas[corridas.escalon == escalon['id']]
        logradas = c[c.logro]
        filas.append({
            'escalon': escalon['id'], 'semillas': len(c), 'logradas': int(c.logro.sum()),
            'max_px_media': c.max_pixeles.mean(), 'max_px_desvio': c.max_pixeles.std(),
            'total_px_media': c.total_pixeles.mean(), 'total_px_desvio': c.total_pixeles.std(),
            'error_media': c.error_final.mean(), 'error_desvio': c.error_final.std(),
            'epoca_objetivo_mediana': logradas.epoca_objetivo.median() if len(logradas) else np.nan,
            'llegaron_y_lo_perdieron': int((c.epoca_objetivo.notna() & ~c.logro).sum()),
            'px_mal_total': int(c.total_pixeles.sum()), 'px_mal_saturados': int(c.pixeles_mal_saturados.sum()),
            'theta_prima_max': c.theta_prima_max_en_mal.max(),
            'segundos_media': c.segundos.mean(), 'n_pesos': int(c.n_pesos.iloc[0]),
        })
    return pd.DataFrame(filas)


def tabla_fallas(config: dict) -> pd.DataFrame:
    """Por escalón y patrón que falla (> 1 px): en cuántas semillas, y quiénes son sus vecinos."""
    patrones = pd.read_csv(RESULTADOS / config['experimento'] / 'patrones.csv', keep_default_na=False)
    assert (patrones.groupby(['escalon', 'semilla']).size() == 32).all()
    X, etiquetas = cargar_font()
    hamming = distancias_hamming(X).astype(float)
    np.fill_diagonal(hamming, np.inf)
    filas = []
    for (escalon, semilla), corrida in patrones.groupby(['escalon', 'semilla'], sort=False):
        Z = corrida[['z1', 'z2']].to_numpy()
        latente = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
        np.fill_diagonal(latente, np.inf)
        for i in np.flatnonzero(corrida.pixeles_mal.to_numpy() > OBJETIVO_PIXELES):
            filas.append({'escalon': escalon, 'semilla': semilla, 'etiqueta': etiquetas[i],
                          'pixeles_mal': int(corrida.pixeles_mal.iloc[i]),
                          'vecino_latente': etiquetas[int(latente[i].argmin())],
                          'vecino_hamming': etiquetas[int(hamming[i].argmin())],
                          'distancia_hamming': int(hamming[i].min())})
    detalle = pd.DataFrame(filas)
    if detalle.empty:
        return detalle
    agregada = detalle.groupby(['escalon', 'etiqueta'], sort=False).agg(
        semillas_con_falla=('semilla', 'nunique'), pixeles_mal_media=('pixeles_mal', 'mean'),
        vecino_latente_mas_frecuente=('vecino_latente', lambda s: s.mode().iloc[0]),
        veces_ese_vecino=('vecino_latente', lambda s: int(s.value_counts().iloc[0])),
        vecino_hamming=('vecino_hamming', 'first'), distancia_hamming=('distancia_hamming', 'first'),
    ).reset_index()
    return agregada


# ---------- figuras ----------

def figura_logro(resumen: pd.DataFrame, ruta) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    fraccion = resumen.logradas / resumen.semillas
    ax.bar(resumen.escalon, fraccion, color=COLOR['AE'], width=0.6)
    for x, (logradas, semillas, y) in enumerate(zip(resumen.logradas, resumen.semillas, fraccion)):
        ax.text(x, y + 0.03, f'{logradas}/{semillas}', ha='center', va='bottom')
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(axis='x', visible=False)
    ax.set_xlabel('Escalón de la escalera')
    ax.set_ylabel('Fracción de semillas\ncon ≤ 1 píxel incorrecto por patrón')
    fig.savefig(ruta)
    plt.close(fig)


def figura_pixeles(config: dict, corridas: pd.DataFrame, ruta) -> None:
    """Un punto por corrida y la mediana por escalón. Eje simétrico-logarítmico por el rango 0–363."""
    fig, ax = plt.subplots(figsize=(8, 4))
    rng = np.random.default_rng(0)                       # solo para separar puntos superpuestos
    for x, escalon in enumerate(e['id'] for e in config['escalones']):
        valores = corridas[corridas.escalon == escalon].total_pixeles.to_numpy()
        ax.scatter(x + rng.uniform(-0.18, 0.18, len(valores)), valores, s=28, color=COLOR['AE'],
                   alpha=0.75, edgecolor='white', linewidth=0.6, zorder=3)
        ax.hlines(np.median(valores), x - 0.3, x + 0.3, color=TINTA_SECUNDARIA, linewidth=1.5, zorder=4)
    ax.set_yscale('symlog', linthresh=1)
    ax.set_ylim(-0.3, 600)
    ax.set_yticks([0, 1, 10, 100])
    ax.set_yticklabels(['0', '1', '10', '100'])
    ax.set_xticks(range(len(config['escalones'])))
    ax.set_xticklabels([e['id'] for e in config['escalones']])
    ax.grid(axis='x', visible=False)
    ax.set_xlabel('Escalón de la escalera (un punto por semilla; raya = mediana)')
    ax.set_ylabel('Píxeles incorrectos en total\n(suma sobre los 32 patrones)')
    fig.savefig(ruta)
    plt.close(fig)


def figura_curvas(config: dict, ruta) -> None:
    """Error por época, un panel por escalón entrenado por épocas y una línea por semilla."""
    curvas = pd.read_csv(RESULTADOS / config['experimento'] / 'curvas.csv')
    escalones = [e['id'] for e in config['escalones'] if e['id'] in set(curvas.escalon)]
    perdidas = {e['id']: e['perdida'] for e in config['escalones']}
    columnas = 3
    filas = int(np.ceil(len(escalones) / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(10, 2.6 * filas), sharex=True, sharey=True)
    for ax, escalon in zip(ejes.ravel(), escalones):
        for _, curva in curvas[curvas.escalon == escalon].groupby('semilla'):
            ax.plot(curva.epoca, curva.error, color=COLOR['AE'], linewidth=1, alpha=0.6)
        ax.set_yscale('log')
        nombre_perdida = {'error_cuadratico': 'error cuadrático', 'bce': 'BCE'}[perdidas[escalon]]
        ax.text(0.97, 0.93, f'{escalon} · {nombre_perdida}', transform=ax.transAxes, ha='right', va='top',
                fontweight='bold')
    for ax in ejes.ravel()[len(escalones):]:
        ax.set_visible(False)
    fig.supxlabel('Época')
    fig.supylabel('Error de la época (pérdida sumada sobre los 32 patrones, escala log)')
    fig.tight_layout()
    fig.savefig(ruta)
    plt.close(fig)


def figura_fallas(config: dict, fallas: pd.DataFrame, ruta) -> None:
    """Patrón × escalón: en cuántas semillas ese patrón queda con más de 1 píxel mal."""
    _, etiquetas = cargar_font()
    escalones = [e['id'] for e in config['escalones']]
    matriz = np.zeros((len(etiquetas), len(escalones)))
    for fila in fallas.itertuples():
        matriz[etiquetas.index(fila.etiqueta), escalones.index(fila.escalon)] = fila.semillas_con_falla
    fig, ax = plt.subplots(figsize=(6.5, 9))
    imagen = ax.imshow(matriz, cmap=LinearSegmentedColormap.from_list('ae', RAMPA_AE), aspect='auto',
                       vmin=0, vmax=len(config['semillas']))
    for i in range(matriz.shape[0]):
        for j in range(matriz.shape[1]):
            if matriz[i, j] > 0:
                ax.text(j, i, int(matriz[i, j]), ha='center', va='center', fontsize=8,
                        color='white' if matriz[i, j] > 5 else '#0b0b0b')
    ax.set_xticks(range(len(escalones)))
    ax.set_xticklabels(escalones)
    ax.set_yticks(range(len(etiquetas)))
    ax.set_yticklabels(etiquetas)
    ax.grid(visible=False)
    ax.set_xlabel('Escalón de la escalera')
    ax.set_ylabel('Patrón de font.h')
    barra = fig.colorbar(imagen, ax=ax, fraction=0.04, pad=0.03)
    barra.set_label(f'Semillas (de {len(config["semillas"])}) en que el patrón queda con más de 1 píxel mal')
    fig.savefig(ruta)
    plt.close(fig)


# ---------- texto generado desde los datos ----------

def pm(media: float, desvio: float, decimales: int = 1) -> str:
    return f'{media:.{decimales}f} ± {desvio:.{decimales}f}'.replace('.', ',')


def num(valor: float, decimales: int = 0) -> str:
    return '—' if pd.isna(valor) else f'{valor:.{decimales}f}'.replace('.', ',')


def tabla_md(config: dict, resumen: pd.DataFrame, con_cambio: bool = True) -> str:
    cambios = {e['id']: e.get('cambia', '') for e in config['escalones']}
    bases = {e['id']: e.get('respecto_de') or '—' for e in config['escalones']}
    lineas = ['| Escalón | ' + ('Respecto de | Qué cambia | ' if con_cambio else '') +
              'Semillas con ≤ 1 px | Máx. px por patrón | Px mal en total | Error final | Época en que llega (mediana) | Tiempo (s) |',
              '|---|' + ('---|---|' if con_cambio else '') + '---|---|---|---|---|---|']
    for f in resumen.itertuples():
        medio = f'{bases[f.escalon]} | {cambios[f.escalon]} | ' if con_cambio else ''
        lineas.append(f'| {f.escalon} | {medio}{f.logradas} de {f.semillas} | {pm(f.max_px_media, f.max_px_desvio)} | '
                      f'{pm(f.total_px_media, f.total_px_desvio)} | {pm(f.error_media, f.error_desvio, 2)} | '
                      f'{num(f.epoca_objetivo_mediana)} | {num(f.segundos_media, 1)} |')
    return '\n'.join(lineas)


def texto_piloto() -> str:
    config, corridas = leer('e1a2_piloto')
    _, corridas_eta = leer('e1a2_piloto_eta')
    con_cuadratico = corridas          # todo el piloto usa error cuadrático y salida logística
    lentos = con_cuadratico[con_cuadratico.escalon.str.contains('rmsprop|adam')]
    razon = lentos.error_final / lentos.total_pixeles
    saturados = lentos.pixeles_mal_saturados.sum() / lentos.total_pixeles.sum()
    mejor = corridas.loc[corridas.total_pixeles.idxmin()]
    return f"""### Hallazgos del piloto (`results/e1a2_piloto/`, `results/e1a2_piloto_eta/`)

Piloto para elegir η por optimizador: {corridas.escalon.nunique()} variantes × {len(config['semillas'])} semillas (las {' y '.join(map(str, config['semillas']))}, distintas de las oficiales), {config['epocas']} épocas, error cuadrático y salida logística. Más {corridas_eta.escalon.nunique()} combinaciones de las constantes del η adaptativo.

- **Ninguna de las {len(corridas)} corridas del piloto de η llega a ≤ 1 píxel por patrón** ({int(corridas.logro.sum())} de {len(corridas)}). El mínimo de máximo-por-patrón es {int(corridas.max_pixeles.min())} px; la corrida con menos píxeles mal en total tiene {int(mejor.total_pixeles)} ({mejor.escalon}, semilla {int(mejor.semilla)}).
- En las {len(lentos)} corridas con RMSProp o Adam, el error final dividido por la cantidad de píxeles incorrectos vale entre {num(razon.min(), 3)} y {num(razon.max(), 3)} (mediana {num(razon.median(), 3)}). Un píxel con la salida en el valor opuesto aporta ½·1² = 0,5 al error cuadrático.
- En esas corridas, el {num(100 * saturados, 1)} % de los píxeles incorrectos ({int(lentos.pixeles_mal_saturados.sum())} de {int(lentos.total_pixeles.sum())}) tiene |O − X| > 0,99, y la mediana entre corridas de la θ′ de salida más grande entre sus píxeles incorrectos es {lentos.theta_prima_max_en_mal.median():.1e}. Con error cuadrático el delta de salida es (ζ − O)·θ′(h) (Clase 11, diap. 40–43).
- En el piloto del η adaptativo, {int(corridas_eta.logro.sum())} de {len(corridas_eta)} corridas llegan a ≤ 1 px; las {int(corridas_eta.logro.sum())} son de combinaciones con k′ = 3.
"""


def texto_diagnostico() -> str:
    config, corridas = leer('e1a2_diagnostico')
    resumen = resumir(config, corridas)
    descripcion = {
        'D0_referencia_E8': 'Referencia: 35–20–2–20–35, Adam η = 0,001, error cuadrático',
        'D1_bce': 'BCE', 'D2_salida_identidad': 'Salida identidad',
        'D3_35-20-10-2': 'Más profunda: 35–20–10–2', 'D4_35-30-2': 'Más ancha: 35–30–2',
        'D5_bce_35-20-10-2': 'BCE y 35–20–10–2', 'D6_bce_gd_0.01': 'BCE con GD η = 0,01',
        'D7_bce_adam_0.003': 'BCE con Adam η = 0,003', 'D8_bce_35-30-2': 'BCE y 35–30–2',
    }
    lineas = ['| Variante (cambio sobre la referencia) | Semillas con ≤ 1 px | Px mal en total, por semilla | Px mal saturados |',
              '|---|---|---|---|']
    for f in resumen.itertuples():
        por_semilla = ' y '.join(str(v) for v in corridas[corridas.escalon == f.escalon].total_pixeles)
        lineas.append(f'| {descripcion[f.escalon]} | {f.logradas} de {f.semillas} | {por_semilla} | {f.px_mal_saturados} de {f.px_mal_total} |')
    con_bce = corridas[corridas.escalon.str.contains('bce') & corridas.escalon.str.contains('adam|D1|D5|D8')]
    sin_bce = corridas[~corridas.escalon.str.contains('bce')]
    return f"""### Hallazgos del diagnóstico (`results/e1a2_diagnostico/`)

Mismas {len(config['semillas'])} semillas del piloto, {config['epocas']} épocas. Cada variante cambia lo indicado sobre la referencia.

{chr(10).join(lineas)}

- Con error cuadrático, ninguna de las {len(sin_bce)} corridas llega a ≤ 1 px ({int(sin_bce.logro.sum())} de {len(sin_bce)}), tampoco con más capas, más neuronas o salida identidad.
- Con BCE y Adam, {int(con_bce.logro.sum())} de {len(con_bce)} corridas llegan, todas con {int(con_bce.total_pixeles.max())} píxeles incorrectos. Con BCE el delta de salida es 2β(O − ζ) (derivación nuestra, verificada en `tests/test_activaciones_perdidas.py`).
- BCE con GD y η = 0,01 no llega ({int(corridas[corridas.escalon == 'D6_bce_gd_0.01'].logro.sum())} de 2) y termina con error {' y '.join(num(v, 0) for v in corridas[corridas.escalon == 'D6_bce_gd_0.01'].error_final)}.
- Son {len(config['semillas'])} semillas por variante: sirvió para decidir la escalera, no es un resultado.
"""


def texto_escalera(config: dict, corridas: pd.DataFrame, resumen: pd.DataFrame, fallas: pd.DataFrame) -> str:
    n = len(config['semillas'])
    por_id = resumen.set_index('escalon')
    logran = [f'{e} ({int(por_id.loc[e, "logradas"])} de {n})' for e in por_id.index if por_id.loc[e, 'logradas'] > 0]
    cuadratico = [e['id'] for e in config['escalones'] if e['perdida'] == 'error_cuadratico' and e['optimizador']['nombre'] != 'powell']
    c_cuad = corridas[corridas.escalon.isin(cuadratico)]
    e9 = corridas[corridas.escalon == 'E9']
    powell = corridas[corridas.escalon == 'E10']
    perdieron = por_id[por_id.llegaron_y_lo_perdieron > 0].llegaron_y_lo_perdieron
    lineas_fallas = []
    for escalon in ['E4', 'E5', 'E6', 'E7', 'E8']:
        f = fallas[fallas.escalon == escalon].sort_values('semillas_con_falla', ascending=False).head(3)
        if len(f):
            detalle = '; '.join(f"'{r.etiqueta}' en {r.semillas_con_falla} de {n} (vecino en el latente más frecuente: '{r.vecino_latente_mas_frecuente}'; "
                                f"más cercano por Hamming: '{r.vecino_hamming}', a {r.distancia_hamming} px)" for r in f.itertuples())
            lineas_fallas.append(f'- **{escalon}:** {detalle}.')
    return f"""### Escalera oficial (`results/e1a2/`)

{len(config['escalones'])} escalones × {n} semillas ({config['semillas'][0]} a {config['semillas'][-1]}), {config['epocas']} épocas en los escalones entrenados por épocas, datos en {{0, 1}}, salida logística, latente de 2 dimensiones. Media ± desvío entre semillas. "Época en que llega" es la mediana entre las corridas que terminan con ≤ 1 px.

{tabla_md(config, resumen)}

Figuras: `fig_logro.png` (fracción de semillas por escalón), `fig_pixeles.png` (píxeles incorrectos por corrida), `fig_curvas.png` (error por época; el panel de E9 es BCE y no es comparable en valor con los de error cuadrático), `fig_fallas.png` (qué patrones fallan en cada escalón). Tablas: `resumen.csv`, `fallas.csv`.

Observaciones:

- Escalones con alguna semilla que termina con ≤ 1 px por patrón: {', '.join(logran)}. El resto, 0 de {n}.
- E9 (BCE) termina con {int(e9.total_pixeles.max())} píxeles incorrectos en las {n} semillas; llega a ≤ 1 px entre las épocas {int(e9.epoca_objetivo.min())} y {int(e9.epoca_objetivo.max())} (mediana {num(e9.epoca_objetivo.median())}).
- En los escalones con error cuadrático entrenados por épocas ({cuadratico[0]} a {cuadratico[-1]}), {int(c_cuad.pixeles_mal_saturados.sum())} de {int(c_cuad.total_pixeles.sum())} píxeles incorrectos ({num(100 * c_cuad.pixeles_mal_saturados.sum() / c_cuad.total_pixeles.sum(), 1)} %) tienen |O − X| > 0,99. De E4 a E8: {int(corridas[corridas.escalon.isin(['E4', 'E5', 'E6', 'E7', 'E8'])].pixeles_mal_saturados.sum())} de {int(corridas[corridas.escalon.isin(['E4', 'E5', 'E6', 'E7', 'E8'])].total_pixeles.sum())}.
- Píxeles incorrectos en total: E1 {pm(por_id.loc['E1', 'total_px_media'], por_id.loc['E1', 'total_px_desvio'])}; E2 {pm(por_id.loc['E2', 'total_px_media'], por_id.loc['E2', 'total_px_desvio'])}; E3 {pm(por_id.loc['E3', 'total_px_media'], por_id.loc['E3', 'total_px_desvio'])}; E4 {pm(por_id.loc['E4', 'total_px_media'], por_id.loc['E4', 'total_px_desvio'])}.
- Entre E4, E5, E6, E7 y E8 (mismo modelo, distinto optimizador, error cuadrático) el total de píxeles incorrectos va de {pm(por_id.loc[['E4', 'E5', 'E6', 'E7', 'E8'], 'total_px_media'].min(), por_id.loc[por_id.loc[['E4', 'E5', 'E6', 'E7', 'E8'], 'total_px_media'].idxmin(), 'total_px_desvio'])} ({por_id.loc[['E4', 'E5', 'E6', 'E7', 'E8'], 'total_px_media'].idxmin()}) a {pm(por_id.loc[['E4', 'E5', 'E6', 'E7', 'E8'], 'total_px_media'].max(), por_id.loc[por_id.loc[['E4', 'E5', 'E6', 'E7', 'E8'], 'total_px_media'].idxmax(), 'total_px_desvio'])} ({por_id.loc[['E4', 'E5', 'E6', 'E7', 'E8'], 'total_px_media'].idxmax()}).
- Corridas que en alguna época tuvieron ≤ 1 px y terminaron con más: {', '.join(f'{e}: {int(v)}' for e, v in perdieron.items()) if len(perdieron) else 'ninguna'}.
- E10 (Powell sobre 35–2–35, {int(powell.n_pesos.iloc[0])} pesos): 0 de {n}. scipy informa convergencia en las {int(powell.powell_convergio.sum())} corridas, con entre {int(powell.powell_iteraciones.min())} y {int(powell.powell_iteraciones.max())} iteraciones y entre {int(powell.evaluaciones.min())} y {int(powell.evaluaciones.max())} evaluaciones de la pérdida (tope: {config['escalones'][-1]['optimizador']['max_evaluaciones']}); tarda entre {num(powell.segundos.min(), 2)} y {num(powell.segundos.max(), 2)} s. Error final {pm(powell.error_final.mean(), powell.error_final.std(), 2)}, contra {pm(por_id.loc['E1', 'error_media'], por_id.loc['E1', 'error_desvio'], 2)} de E1 (misma red, GD). {int((powell.powell_iteraciones == powell.powell_iteraciones.min()).sum())} de {n} corridas se detienen en {int(powell.powell_iteraciones.min())} iteraciones con error {num(powell.error_final.max(), 2)}.

Patrones que más fallan (más de 1 px mal) y sus vecinos, en los escalones con error cuadrático y capa oculta tanh:

{chr(10).join(lineas_fallas)}
"""


def main() -> None:
    aplicar_estilo()
    config, corridas = leer('e1a2')
    carpeta = RESULTADOS / 'e1a2'
    resumen = resumir(config, corridas)
    resumen.to_csv(carpeta / 'resumen.csv', index=False)
    fallas = tabla_fallas(config)
    fallas.to_csv(carpeta / 'fallas.csv', index=False)
    figura_logro(resumen, carpeta / 'fig_logro.png')
    figura_pixeles(config, corridas, carpeta / 'fig_pixeles.png')
    figura_curvas(config, carpeta / 'fig_curvas.png')
    figura_fallas(config, fallas, carpeta / 'fig_fallas.png')
    texto = ('## 1a.2 · Arquitecturas y técnicas de optimización (e1a2)\n\n'
             + texto_escalera(config, corridas, resumen, fallas) + '\n'
             + texto_piloto() + '\n' + texto_diagnostico())
    escribir_seccion('e1a2', texto)
    print(texto)


if __name__ == '__main__':
    main()
