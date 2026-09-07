"""¿Los ganadores de cada eje componen? (experimento OFAT)

Todo el análisis varía **un factor por vez** sobre una base fija. Eso aísla el efecto
de cada operador, pero deja abierta una pregunta que sí importa para elegir una
configuración: **¿armar la config con el ganador de cada eje da la mejor config?**

Sólo daría si los efectos fueran aditivos, o sea si ningún operador cambiara de
conveniencia según con qué otros se combine. Este experimento lo mide:

  1. Se corren tres configuraciones completas: la **base** del barrido, la
     **compuesta** (el ganador de cada eje, tomado del summary) y la **inversa**
     (el peor de cada eje), como cota inferior.
  2. Partiendo de la base, se reemplaza **un operador por vez** por el de la
     configuración compuesta (one-factor-at-a-time), y se mide la mejora de cada
     reemplazo por separado.
  3. Se compara la **suma de las mejoras individuales** con la **mejora real** de
     aplicarlas todas juntas.

Si la suma no da la mejora total, hay interacción entre operadores — y entonces
"optimizar cada componente por separado" no alcanza, que es la conclusión
metodológica que el trabajo puede aportar.

    python analysis/interaccion.py

Escribe `analysis/informe/14-interaccion/`.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plot_results as figs
from ga import engine
from ga.render import load_target

ANALYSIS = Path(__file__).resolve().parent
ROOT = ANALYSIS.parent
RESULTS = ANALYSIS / "results"
DESTINO = ANALYSIS / "informe" / "14-interaccion"

#: ejes que aportan un operador a la configuración compuesta. Se excluyen los que
#: son parámetros del problema (triangulos) o que no son decisiones de operador.
EJES_OPERADORES = ["seleccion", "supervivencia", "cruza", "granularidad", "tasa_cruza",
                   "mutacion", "tasa_mutacion", "sigma", "poblacion", "inicializacion"]


def cargar_spec() -> dict:
    return json.loads((ANALYSIS / "experiments.json").read_text())


def ganadores(summary: list[dict], targets: list[str], mejor: bool) -> dict[str, dict]:
    """Para cada eje, la variante con mejor (o peor) fitness PROMEDIADO entre targets.

    Se promedia entre imágenes a propósito: elegir por una sola imagen sería otra
    forma de sobreajustar, y el enunciado pide justificar qué conviene *en qué
    circunstancia*.
    """
    por_variante: dict[tuple, list[float]] = {}
    for fila in summary:
        if fila["experiment"] not in EJES_OPERADORES:
            continue
        clave = (fila["experiment"], fila["variant"])
        por_variante.setdefault(clave, []).append(float(fila["best_fitness_mean"]))

    elegidos: dict[str, dict] = {}
    for eje in EJES_OPERADORES:
        candidatas = {v: np.mean(f) for (e, v), f in por_variante.items() if e == eje}
        if not candidatas:
            continue
        elegida = (max if mejor else min)(candidatas, key=candidatas.get)
        elegidos[eje] = {"variante": elegida, "fitness_eje": float(candidatas[elegida])}
    return elegidos


def overrides_de(spec: dict, eje: str, variante: str) -> dict:
    return spec["experiments"][eje][variante]


def _correr(job):
    nombre, target_name, config, seed = job
    target = load_target(str(ROOT / config["image"]), config["canvas_size"],
                         config["background"], config.get("preserve_aspect", False))
    inicio = time.perf_counter()
    result = engine.run({**config, "seed": seed}, target)
    return nombre, target_name, seed, float(result.best.fitness), time.perf_counter() - inicio


def construir(spec, mejores, peores):
    """Las configuraciones a comparar: base, compuesta, inversa y los OFAT."""
    base = {**engine.DEFAULTS, **spec["base"]}
    compuesta = dict(base)
    for eje, dato in mejores.items():
        compuesta.update(overrides_de(spec, eje, dato["variante"]))
    inversa = dict(base)
    for eje, dato in peores.items():
        inversa.update(overrides_de(spec, eje, dato["variante"]))

    configs = {"base": base, "compuesta (mejor de cada eje)": compuesta,
               "inversa (peor de cada eje)": inversa}
    # un reemplazo por vez, desde la base hacia la compuesta
    for eje, dato in mejores.items():
        configs[f"base + {eje}"] = {**base, **overrides_de(spec, eje, dato["variante"])}
    return configs


def figura_ofat(deltas, total_real, total_suma, destino: Path) -> None:
    import matplotlib.pyplot as plt
    ejes = [e for e, _ in deltas]
    valores = [d for _, d in deltas]
    colores = [figs.PALETTE[0] if v < 0 else figs.PALETTE[3] for v in valores]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(range(len(ejes)), valores, color=colores, alpha=0.8)
    ax.set_yticks(range(len(ejes)))
    ax.set_yticklabels(ejes)
    ax.invert_yaxis()
    ax.axvline(0, color=plt.rcParams["text.color"], linewidth=1)
    ax.axvline(total_real, color=figs.PALETTE[1], linewidth=2, linestyle="--",
               label=f"mejora real de aplicar todo junto ({total_real:+.4f})")
    ax.axvline(total_suma, color=figs.PALETTE[2], linewidth=2, linestyle=":",
               label=f"suma de las mejoras individuales ({total_suma:+.4f})")
    ax.set_xlabel("cambio de fitness respecto de la base")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(destino, bbox_inches="tight", pad_inches=0.05, transparent=True)
    plt.close(fig)


def escribir(spec, mejores, peores, medias, deltas, total_real, total_suma, targets) -> None:
    lineas = [
        "# ¿Los ganadores de cada eje componen?",
        "",
        "> Todo el análisis varía **un factor por vez**. Eso aísla bien el efecto de cada",
        "> operador, pero no responde si armar la configuración con el ganador de cada eje da",
        "> la mejor configuración. Este experimento lo mide.",
        "",
        "## Método",
        "",
        "1. Se toma de `summary.csv` la variante ganadora de cada eje, **promediando entre las",
        f"   {len(targets)} imágenes** (elegir por una sola sería otra forma de sobreajustar).",
        "2. Se corren tres configuraciones completas: la **base** del barrido, la **compuesta**",
        "   (todos los ganadores juntos) y la **inversa** (todos los perdedores), como cota.",
        "3. Desde la base se reemplaza **un operador por vez** por el de la compuesta, y se mide",
        "   la mejora de cada reemplazo aislado.",
        "4. Se compara la **suma de las mejoras individuales** contra la **mejora real** de",
        "   aplicarlas todas juntas.",
        "",
        "Si los efectos fueran aditivos, las dos cantidades coincidirían.",
        "",
        "## La configuración compuesta",
        "",
        "| Eje | Ganador (promedio entre imágenes) | Perdedor |",
        "|---|---|---|",
    ]
    for eje in mejores:
        lineas.append(f"| `{eje}` | **{mejores[eje]['variante']}** | {peores[eje]['variante']} |")

    lineas += [
        "",
        "## Resultados",
        "",
        "| Configuración | Fitness (promedio entre imágenes) | Δ vs. base |",
        "|---|---|---|",
    ]
    base_fit = medias["base"]
    for nombre in ("base", "compuesta (mejor de cada eje)", "inversa (peor de cada eje)"):
        delta = medias[nombre] - base_fit
        marca = "—" if nombre == "base" else f"{delta:+.4f}"
        lineas.append(f"| {nombre} | {medias[nombre]:.4f} | {marca} |")

    lineas += [
        "",
        "### Reemplazando un operador por vez",
        "",
        "| Operador reemplazado | Fitness | Δ vs. base |",
        "|---|---|---|",
    ]
    for eje, delta in deltas:
        lineas.append(f"| {eje} | {medias[f'base + {eje}']:.4f} | {delta:+.4f} |")

    lineas += [
        "",
        f"| **Suma de las mejoras individuales** | | **{total_suma:+.4f}** |",
        f"| **Mejora real de aplicarlas todas juntas** | | **{total_real:+.4f}** |",
        "",
        "## Lectura",
        "",
    ]
    brecha = total_real - total_suma
    if abs(brecha) < 0.002:
        lineas += [
            f"La suma de los efectos individuales ({total_suma:+.4f}) y el efecto conjunto",
            f"({total_real:+.4f}) coinciden dentro de {abs(brecha):.4f}. **A esta escala los",
            "operadores se comportan de forma aproximadamente aditiva**, y componer los",
            "ganadores por eje es una estrategia razonable. Es un resultado tranquilizador",
            "para la metodología de un-factor-por-vez, no un resultado nulo: sin medirlo, no",
            "se podía afirmar.",
        ]
    elif total_real < total_suma:
        lineas += [
            f"**Los efectos no se suman.** Individualmente los reemplazos aportan",
            f"{total_suma:+.4f}, pero aplicados juntos rinden {total_real:+.4f}: se pierde",
            f"{abs(brecha):.4f} en el camino. Hay **interacción** entre operadores — parte de",
            "lo que cada uno aporta por separado es la misma mejora, o directamente se estorban.",
            "",
            "Consecuencia metodológica, y es la conclusión que vale la pena defender:",
            "**optimizar cada componente por separado y juntar los ganadores no da la mejor",
            "configuración.** Un barrido de un factor por vez sirve para *entender* qué hace",
            "cada operador; para *elegir* una configuración hay que medir configuraciones",
            "completas.",
        ]
    else:
        lineas += [
            f"**Los efectos se potencian.** Individualmente los reemplazos aportan",
            f"{total_suma:+.4f} y juntos rinden {total_real:+.4f}: hay {brecha:+.4f} de más.",
            "Hay interacción, pero a favor: algunos operadores se habilitan mutuamente.",
        ]

    inversa_delta = medias["inversa (peor de cada eje)"] - base_fit
    lineas += [
        "",
        f"La configuración **inversa** (el peor de cada eje) queda {inversa_delta:+.4f} respecto",
        "de la base. La distancia entre la compuesta y la inversa es el rango total que abarcan",
        f"las decisiones de operador: **{medias['compuesta (mejor de cada eje)'] - medias['inversa (peor de cada eje)']:.4f}",
        "de fitness**. Sirve para dimensionar cuánto importa realmente afinar los operadores",
        "comparado con, por ejemplo, cambiar la cantidad de triángulos o el presupuesto.",
        "",
        "## Figuras",
        "",
        "| Archivo | Qué muestra |",
        "|---|---|",
        "| `figuras/ofat.png` | La mejora de cada reemplazo aislado, con la suma y el efecto conjunto marcados como líneas verticales |",
        "",
        "## Datos",
        "",
        "- `datos/configuraciones.csv` — fitness por configuración, imagen y semilla",
        "- `datos/resumen.json` — los ganadores elegidos y los totales",
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

    spec = cargar_spec()
    summary = list(csv.DictReader((RESULTS / "summary.csv").open()))
    targets = list(spec["targets"])
    seeds = spec["seeds"]

    mejores = ganadores(summary, targets, mejor=True)
    peores = ganadores(summary, targets, mejor=False)
    configs = construir(spec, mejores, peores)

    jobs = [(nombre, tn, {**cfg, "image": img}, seed)
            for nombre, cfg in configs.items()
            for tn, img in spec["targets"].items()
            for seed in seeds]
    print(f"{len(jobs)} corridas ({len(configs)} configuraciones × {len(targets)} imágenes "
          f"× {len(seeds)} semillas)")

    filas = []
    inicio = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        for hecho, (nombre, tn, seed, fitness, elapsed) in enumerate(pool.map(_correr, jobs), 1):
            filas.append({"configuracion": nombre, "target": tn, "seed": seed,
                          "fitness": fitness, "segundos": round(elapsed, 2)})
            print(f"\r  {hecho}/{len(jobs)} | {time.perf_counter() - inicio:.0f}s",
                  end="", flush=True)
    print()

    with (DESTINO / "datos" / "configuraciones.csv").open("w", newline="") as fh:
        escritor = csv.DictWriter(fh, fieldnames=list(filas[0]))
        escritor.writeheader(); escritor.writerows(filas)

    medias = {}
    for nombre in configs:
        valores = [f["fitness"] for f in filas if f["configuracion"] == nombre]
        medias[nombre] = float(np.mean(valores))

    base_fit = medias["base"]
    deltas = [(eje, medias[f"base + {eje}"] - base_fit) for eje in mejores]
    total_suma = sum(d for _, d in deltas)
    total_real = medias["compuesta (mejor de cada eje)"] - base_fit

    figura_ofat(deltas, total_real, total_suma, DESTINO / "figuras" / "ofat.png")
    (DESTINO / "datos" / "resumen.json").write_text(json.dumps(
        {"mejores": mejores, "peores": peores, "medias": medias,
         "deltas": dict(deltas), "suma_individuales": total_suma, "efecto_conjunto": total_real},
        indent=2, ensure_ascii=False))

    escribir(spec, mejores, peores, medias, deltas, total_real, total_suma, targets)
    print(f"suma individual {total_suma:+.4f} vs conjunto {total_real:+.4f}")
    print(f"informe en {DESTINO.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
