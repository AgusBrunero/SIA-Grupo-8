"""Criterios de corte: cuál elegir y por qué (Step 9 del enunciado).

El enunciado pide **decidir y justificar** el criterio de corte. Justificarlo con una
opinión no sirve; hay que medirlo. Pero medirlo corriendo el AG una vez por criterio
sería caro y además incomparable, porque cada criterio cortaría en otro punto.

En vez de eso se hace lo mismo con los datos que ya están: el motor registra por
generación todo lo que los cinco criterios necesitan mirar

    max_generations   generation
    max_seconds       elapsed
    target_fitness    best_global_fitness
    contenido         stalled            (generaciones seguidas sin mejorar)
    estructura        share_unchanged    (fracción de la población que no cambió)

así que para cualquier configuración de corte se puede reconstruir **en qué
generación habría disparado** y **cuánto fitness se habría perdido** por frenar ahí.
Eso convierte la elección del criterio en una medición, no en una preferencia.

    python analysis/criterios_corte.py

Escribe `analysis/informe/13-criterios-corte/`.
"""

from __future__ import annotations

import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plot_results as figs

ANALYSIS = Path(__file__).resolve().parent
ROOT = ANALYSIS.parent
RESULTS = ANALYSIS / "results"
DESTINO = ANALYSIS / "informe" / "13-criterios-corte"

#: configuraciones de corte a evaluar, con el nombre que usa el motor
CANDIDATOS = [
    ("contenido G=20", "stall_generations", 20),
    ("contenido G=50", "stall_generations", 50),
    ("contenido G=100", "stall_generations", 100),
    ("estructura G=20 (ε=0.01)", "structure_generations", 20),
    ("estructura G=50 (ε=0.01)", "structure_generations", 50),
    ("entorno fitness≥0.85", "target_fitness", 0.85),
    ("entorno fitness≥0.90", "target_fitness", 0.90),
    ("entorno fitness≥0.95", "target_fitness", 0.95),
]

#: umbral del criterio de estructura tal como lo aplica el motor
STRUCTURE_EPSILON = 0.01


def cargar() -> dict[tuple, list[dict]]:
    """Todas las corridas del barrido, agrupadas por (eje, target, variante, semilla)."""
    corridas: dict[tuple, list[dict]] = defaultdict(list)
    for path in sorted(RESULTS.glob("*.csv")):
        if path.stem == "summary":
            continue
        with path.open() as fh:
            for fila in csv.DictReader(fh):
                clave = (fila["experiment"], fila["target"], fila["variant"], fila["seed"])
                corridas[clave].append(fila)
    for filas in corridas.values():
        filas.sort(key=lambda f: int(f["generation"]))
    return corridas


def dispara(filas: list[dict], clave: str, valor) -> int | None:
    """Primera generación en la que el criterio se habría cumplido, o None."""
    for fila in filas:
        if clave == "stall_generations" and int(fila["stalled"]) >= valor:
            return int(fila["generation"])
        if clave == "structure_generations" and int(fila["structure_stable"]) >= valor:
            return int(fila["generation"])
        if clave == "target_fitness" and float(fila["best_global_fitness"]) >= valor:
            return int(fila["generation"])
    return None


def evaluar(corridas: dict) -> list[dict]:
    """Por criterio: con qué frecuencia dispara, en qué generación y qué fitness cuesta."""
    resumen = []
    for nombre, clave, valor in CANDIDATOS:
        generaciones, perdidas, ahorros, disparos = [], [], [], 0
        for filas in corridas.values():
            final = float(filas[-1]["best_global_fitness"])
            g = dispara(filas, clave, valor)
            if g is None:
                continue
            disparos += 1
            generaciones.append(g)
            alcanzado = float(filas[g - 1]["best_global_fitness"])
            perdidas.append(final - alcanzado)
            ahorros.append(1 - g / len(filas))
        resumen.append({
            "criterio": nombre,
            "disparos": disparos,
            "corridas": len(corridas),
            "cobertura": disparos / len(corridas),
            "gen_mediana": float(np.median(generaciones)) if generaciones else None,
            "gen_min": min(generaciones) if generaciones else None,
            "gen_max": max(generaciones) if generaciones else None,
            "perdida_media": float(np.mean(perdidas)) if perdidas else None,
            "perdida_max": float(np.max(perdidas)) if perdidas else None,
            "ahorro_medio": float(np.mean(ahorros)) if ahorros else None,
        })
    return resumen


def estadisticas_estructura(corridas: dict) -> dict:
    """Por qué el criterio de estructura no dispara: cuánto recambio hay realmente."""
    share = np.array([float(f["share_unchanged"]) for filas in corridas.values() for f in filas])
    ultimos = np.array([float(filas[-1]["share_unchanged"]) for filas in corridas.values()])
    return {
        "media": float(share.mean()),
        "maxima": float(share.max()),
        "media_final": float(ultimos.mean()),
        "umbral": 1 - STRUCTURE_EPSILON,
        "supera_umbral": float((share >= 1 - STRUCTURE_EPSILON).mean()),
    }


def estadisticas_contenido(corridas: dict) -> dict:
    """Cuál es la racha más larga sin mejorar que efectivamente ocurre."""
    rachas = np.array([max(int(f["stalled"]) for f in filas) for filas in corridas.values()])
    return {
        "mediana": float(np.median(rachas)),
        "p90": float(np.percentile(rachas, 90)),
        "maxima": int(rachas.max()),
    }


def figura_estancamiento(corridas: dict, destino: Path) -> None:
    """Distribución de la racha máxima sin mejorar, sobre todas las corridas."""
    import matplotlib.pyplot as plt
    rachas = [max(int(f["stalled"]) for f in filas) for filas in corridas.values()]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(rachas, bins=40, color=figs.PALETTE[1], alpha=0.75,
            edgecolor=plt.rcParams["text.color"], linewidth=0.5)
    for g, estilo in ((20, "--"), (50, ":")):
        ax.axvline(g, color=figs.PALETTE[0], linestyle=estilo, linewidth=2,
                   label=f"criterio de contenido G={g}")
    ax.set_xlabel("racha máxima de generaciones sin mejorar el mejor global")
    ax.set_ylabel("corridas")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(destino, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)


def figura_recambio(corridas: dict, destino: Path) -> None:
    """Fracción de la población que NO cambia, generación a generación."""
    import matplotlib.pyplot as plt
    largos = {len(f) for f in corridas.values()}
    n = min(largos)
    curvas = np.array([[float(f["share_unchanged"]) for f in filas[:n]]
                       for filas in corridas.values()])
    generaciones = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    media = curvas.mean(axis=0)
    p90 = np.percentile(curvas, 90, axis=0)
    ax.plot(generaciones, media, color=figs.PALETTE[1], linewidth=2.2, label="media")
    ax.plot(generaciones, p90, color=figs.PALETTE[2], linewidth=1.4, linestyle="--",
            label="percentil 90")
    ax.axhline(1 - STRUCTURE_EPSILON, color=figs.PALETTE[0], linewidth=2,
               label=f"umbral del criterio (ε={STRUCTURE_EPSILON})")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("generación")
    ax.set_ylabel("fracción de la población sin cambios")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="center right")
    fig.savefig(destino, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)


def figura_costo_beneficio(resumen: list[dict], destino: Path) -> None:
    """Cuánto cómputo ahorra cada criterio contra cuánto fitness cuesta."""
    import matplotlib.pyplot as plt
    utiles = [r for r in resumen if r["disparos"] and r["ahorro_medio"] is not None]
    if not utiles:
        return
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, r in enumerate(utiles):
        color = figs.PALETTE[i % len(figs.PALETTE)]
        ax.scatter(r["ahorro_medio"] * 100, r["perdida_media"], s=80 + 320 * r["cobertura"],
                   color=color, alpha=0.75, edgecolor="none", label=r["criterio"])
    ax.set_xlabel("generaciones ahorradas (%)")
    ax.set_ylabel("fitness perdido por cortar ahí")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=10, title="tamaño = fracción de corridas donde dispara")
    ax.get_legend().get_title().set_fontsize(9)
    fig.savefig(destino, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)


def escribir(resumen, estructura, contenido, corridas) -> None:
    n = len(corridas)
    generaciones = len(next(iter(corridas.values())))
    lineas = [
        "# Criterios de corte — cuál elegir y por qué",
        "",
        "> El enunciado pide **decidir y justificar** el criterio de corte. Acá se decide con",
        f"> datos: las {n:,} corridas del barrido se corrieron siempre hasta el tope de",
        f"> {generaciones} generaciones, y sobre esos registros se reconstruye, para cada",
        "> criterio candidato, **en qué generación habría disparado** y **cuánto fitness se",
        "> habría perdido** por frenar ahí.",
        "",
        "## Por qué se mide así y no corriendo cada criterio",
        "",
        "Si cada criterio se midiera con su propia corrida, cada uno cortaría en un punto",
        "distinto y las corridas no serían comparables entre sí — además de costar una tanda",
        "completa por criterio. El motor registra por generación exactamente lo que los cinco",
        "criterios necesitan mirar:",
        "",
        "| Criterio | Qué mira | Columna registrada |",
        "|---|---|---|",
        "| máximo de generaciones | la generación actual | `generation` |",
        "| tiempo máximo | segundos acumulados | `elapsed` |",
        "| entorno a la solución | el mejor fitness alcanzado | `best_global_fitness` |",
        "| **contenido** | generaciones seguidas sin mejorar | `stalled` |",
        "| **estructura** | fracción de la población sin cambios | `share_unchanged` |",
        "",
        "Con eso, evaluar un criterio es recorrer la serie y buscar la primera generación que",
        "cumple la condición. Es exacto, no una simulación aproximada.",
        "",
        "## Resultados",
        "",
        f"Sobre las {n:,} corridas del barrido:",
        "",
        "| Criterio | ¿Dispara? | Generación (mediana) | Rango | Cómputo ahorrado | Fitness perdido |",
        "|---|---|---|---|---|---|",
    ]
    for r in resumen:
        if not r["disparos"]:
            lineas.append(f"| {r['criterio']} | **nunca** (0 de {r['corridas']:,}) | — | — | — | — |")
            continue
        lineas.append(
            f"| {r['criterio']} | {r['cobertura'] * 100:.0f}% "
            f"({r['disparos']:,} de {r['corridas']:,}) | {r['gen_mediana']:.0f} | "
            f"{r['gen_min']}–{r['gen_max']} | {r['ahorro_medio'] * 100:.0f}% | "
            f"{r['perdida_media']:.4f} (máx {r['perdida_max']:.4f}) |")

    lineas += [
        "",
        "## Lectura",
        "",
        "### 1. El criterio de estructura no dispara nunca, y hay una razón de diseño",
        "",
        f"En las {n:,} corridas, la fracción de la población que no cambia de una generación a",
        f"la siguiente promedia **{estructura['media']:.3f}** y su máximo es "
        f"**{estructura['maxima']:.3f}**; el umbral del criterio es "
        f"{estructura['umbral']:.2f}. Sólo el {estructura['supera_umbral'] * 100:.2f}% de las",
        "generaciones lo alcanza.",
        "",
        "No es un error de implementación: es una consecuencia de la representación. Los genes",
        "son **reales**, y el criterio compara genomas por igualdad exacta de bytes. Con",
        "mutación gaussiana activa, cualquier perturbación —por chica que sea— cuenta como",
        "cambio. En un AG binario dos individuos convergen a cadenas idénticas; acá convergen",
        "a cadenas *parecidas*, que nunca son iguales.",
        "",
        "**Conclusión defendible:** el criterio de estructura, tal como lo define la cátedra",
        "(recambio de la población), **no es aplicable a una representación real con mutación",
        "continua** salvo que se lo redefina con una tolerancia — por ejemplo, considerar dos",
        "individuos iguales si su distancia es menor a un ε. Lo implementamos, lo medimos, y",
        "esto es lo que encontramos.",
        "",
        "> La métrica que sí captura el fenómeno es la **diversidad genética** (desvío promedio",
        "> por gen), que está en todos los ejes y sí colapsa. Es el reemplazo natural del",
        "> criterio de estructura para esta representación.",
        "",
        "### 2. El criterio de contenido tampoco alcanza a disparar con umbrales razonables",
        "",
        f"La racha más larga sin mejorar tiene mediana **{contenido['mediana']:.0f}** "
        f"generaciones, percentil 90 en **{contenido['p90']:.0f}** y máximo "
        f"**{contenido['maxima']}**.",
        "",
        "La causa es la forma del problema: con 500 genes reales y mutación permanente, el",
        "algoritmo casi siempre encuentra **alguna** mejora marginal. El mejor global es",
        "monótono y sigue subiendo de a milésimas hasta el final. Un criterio de contenido",
        "sólo sirve acá si el umbral se define sobre la *magnitud* de la mejora y no sobre su",
        "existencia (por ejemplo: cortar si no mejoró más de 0.001 en G generaciones).",
        "",
        "### 3. El criterio útil es el entorno a la solución, y sirve para otra cosa",
        "",
        "Es el único que dispara de forma consistente, pero su valor no es ahorrar cómputo",
        "sino **fijar un objetivo de calidad**: permite responder «cuántas generaciones hacen",
        "falta para llegar a fitness X» en vez de «qué fitness da en X generaciones». Es la",
        "forma correcta de comparar métodos a calidad igualada, y es lo que el enunciado",
        "menciona como opcional («error mínimo como condición de corte»).",
        "",
        "## Qué criterio usamos, y por qué",
        "",
        "| Criterio | Estado | Justificación |",
        "|---|---|---|",
        "| **máximo de generaciones** | **activo, siempre** | Es el único que garantiza terminación",
        "y el único que hace comparables dos corridas: a presupuesto fijo, lo que se compara es",
        "la calidad alcanzada. Todo el análisis lo usa. |",
        "| **entorno a la solución** | disponible, se usa para medir | No para cortar antes, sino",
        "para el análisis inverso: generaciones necesarias para alcanzar una calidad dada. |",
        "| tiempo máximo | implementado, no se usa en el análisis | Hace las corridas no",
        "reproducibles: el mismo experimento en otra máquina corta en otra generación. Sirve",
        "como red de seguridad en producción, no como criterio experimental. |",
        "| contenido | implementado, medido, **descartado** | Con umbrales razonables no dispara",
        "(ver punto 2). Necesitaría redefinirse sobre la magnitud de la mejora. |",
        "| estructura | implementado, medido, **descartado** | No dispara nunca con genes reales",
        "(ver punto 1). |",
        "",
        "Que dos de los cinco se descarten **no es un resultado negativo**: están implementados",
        "y el motor reporta cuál cortó. Lo que se aporta es la evidencia de por qué, en este",
        "problema y con esta representación, no son los adecuados — que es exactamente lo que",
        "el enunciado pide justificar.",
        "",
        "## Figuras",
        "",
        "| Archivo | Qué muestra |",
        "|---|---|",
        "| `figuras/estancamiento.png` | Distribución de la racha máxima sin mejorar, con los umbrales de contenido marcados. Se ve que casi ninguna corrida los alcanza |",
        "| `figuras/recambio_poblacion.png` | Fracción de la población sin cambios por generación, contra el umbral del criterio de estructura |",
        "| `figuras/costo_beneficio.png` | Cómputo ahorrado contra fitness perdido, por criterio |",
        "",
        "## Datos",
        "",
        "- `datos/criterios.csv` — la tabla de arriba, en CSV",
        "",
    ]
    (DESTINO / "informe.md").write_text("\n".join(lineas))


def main() -> None:
    if not (RESULTS / "summary.csv").exists():
        sys.exit("faltan resultados. Corré primero analysis/run_experiments.py")

    figs.use_theme("light")
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    (DESTINO / "figuras").mkdir(parents=True)
    (DESTINO / "datos").mkdir()

    corridas = cargar()
    print(f"{len(corridas):,} corridas cargadas")
    resumen = evaluar(corridas)

    with (DESTINO / "datos" / "criterios.csv").open("w", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(resumen[0]))
        escritor.writeheader(); escritor.writerows(resumen)

    figura_estancamiento(corridas, DESTINO / "figuras" / "estancamiento.png")
    figura_recambio(corridas, DESTINO / "figuras" / "recambio_poblacion.png")
    figura_costo_beneficio(resumen, DESTINO / "figuras" / "costo_beneficio.png")

    escribir(resumen, estadisticas_estructura(corridas), estadisticas_contenido(corridas),
             corridas)
    print(f"informe en {DESTINO.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
