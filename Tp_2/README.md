# TP 2 — Algoritmos Genéticos: aproximación de imágenes con triángulos

Motor de Algoritmos Genéticos implementado desde cero (sin librerías de AG) que
aproxima una imagen usando N triángulos traslúcidos sobre un canvas blanco.

Enunciado y plan de trabajo: [`docs/TP2.md`](docs/TP2.md)

> **Estado: Steps 0-11 del plan.** Están implementados los 6 métodos de selección
> (más la selección combinada A%/B%), las 4 cruzas, las 4 mutaciones, ambas
> estrategias de supervivencia, los 5 criterios de corte y el pipeline de
> experimentos. Falta la presentación (Step 12).

## Setup

```bash
cd Tp_2
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Uso

```bash
python images/generate_samples.py           # genera las imágenes target de ejemplo
python main.py                              # corre con config.json
python main.py --triangles 50 --generations 2000
python main.py --image images/germany.png --tag prueba --gif
```

| Flag | Qué hace |
|---|---|
| `--config` | Archivo de configuración (default `config.json`) |
| `--image` | Imagen target |
| `--triangles` | Cantidad de triángulos (el 2º parámetro del problema) |
| `--canvas` | Resolución de trabajo para evaluar el fitness |
| `--population` / `--offspring` | Tamaño de población N y de descendencia K |
| `--generations` | Cota de generaciones |
| `--seed` | Semilla, para corridas reproducibles |
| `--tag` | Sufijo para no pisar corridas anteriores |
| `--render-size` | Resolución de la imagen final (default 512) |
| `--gif` | Además guarda un gif de la evolución |
| `--rebuild` | Reconstruye la imagen desde un `triangles.json` y termina |

## Salida

Cada corrida escribe en `output/<imagen>[-tag]/`:

| Archivo | Contenido |
|---|---|
| `best.png` | Imagen generada (el genotipo es independiente de la resolución: se puede renderizar a cualquier tamaño) |
| `comparison.png` | Target vs. resultado, lado a lado |
| `triangles.json` | Enumeración de los triángulos — la "compresión" de la imagen (ver abajo) |
| `metrics.csv` | Una fila por generación: mejor fitness, promedio, desvío, diversidad, evaluaciones, tiempo |
| `run.json` | Config completa + fitness final, RMSE, generaciones, evaluaciones, tiempo y motivo de corte |
| `evolution.gif` | Sólo con `--gif` |

## El archivo de triángulos

`triangles.json` es el output que pide el enunciado como enumeración de triángulos.
Está pensado como un **formato autosuficiente**: con el archivo solo —sin el código
que lo generó— se puede reconstruir la imagen, porque declara el tamaño del canvas,
el color de fondo, la regla de composición y el orden de pintado.

```jsonc
{
  "format_version": 1,
  "canvas": { "width": 512, "height": 512, "background": [255, 255, 255] },
  "compositing": "rgba-source-over",
  "paint_order": "El array `triangles` está en orden de pintado: ...",
  "source_image": "images/japan.png",
  "fitness": 0.9269,
  "triangles": [
    { "vertices": [[341.5, 213.97], [178.17, 121.14], [88.4, 402.8]], "color": [188, 0, 45, 120] }
  ]
}
```

El PNG entregado se renderiza **desde el documento**, no desde el individuo en
memoria, así que la imagen y el archivo describen la misma cosa por construcción.
Para verificarlo:

```bash
python main.py --rebuild output/japan/triangles.json
# reconstruido desde output/japan/triangles.json -> output/japan/rebuilt.png
#   20 triángulos, canvas 512x512
#   diferencia máxima contra best.png: 0 (0 = idénticas)
```

`test_ga.py` incluye el test de ida y vuelta (`TestArtifact`): individuo → documento
→ disco → documento → imagen, comparando píxel a píxel.

## Configuración

`config.json` — todo lo que no esté definido toma el default de `ga/engine.py`.

```jsonc
{
  "image": "images/japan.png",
  "triangles": 20,              // parámetro del problema
  "canvas_size": 64,            // resolución a la que se evalúa el fitness
  "background": [255, 255, 255],

  "population_size": 50,        // N
  "offspring_size": 50,         // K

  // un nombre, o la selección combinada A% / (1-A)% que pide la cátedra
  "selection_parents": { "method_a": "elite", "method_b": "tournament_det", "a_ratio": 0.5 },
  "selection_survivors": { "method_a": "elite", "method_b": "universal", "a_ratio": 0.5 },
  "tournament": { "m": 4, "threshold": 0.75 },
  "boltzmann": { "t0": 100.0, "tmin": 1.0, "k": 0.01 },

  "crossover": "uniform",
  "crossover_rate": 0.85,
  "crossover_granularity": "triangle",   // "gene" | "triangle"
  "crossover_uniform_p": 0.5,

  "mutation": "non_uniform",
  "mutation_rate": 0.1,
  "mutation_sigma": 0.15,
  "mutation_genes": null,                // M para multigen (null = al azar)
  "mutation_decay_floor": 0.1,           // piso del decaimiento en no uniforme

  "replacement": "additive",

  "stop": {                              // se evalúan por OR; null = desactivado
    "max_generations": 500,
    "max_seconds": null,
    "target_fitness": null,
    "stall_generations": null,           // contenido: sin mejorar en G generaciones
    "structure_generations": null,       // estructura: sin recambio en G generaciones
    "structure_epsilon": 0.01
  },
  "seed": 42
}
```

### Métodos disponibles

| Operador | Valores |
|---|---|
| `selection_parents` / `selection_survivors` | `elite`, `roulette`, `universal`, `boltzmann`, `ranking`, `tournament_det`, `tournament_prob`, o `{method_a, method_b, a_ratio}` |
| `crossover` | `one_point`, `two_point`, `uniform`, `annular` |
| `mutation` | `gene`, `multigene`, `uniform`, `non_uniform` |
| `replacement` | `additive`, `exclusive` |
| `stop` | `max_generations`, `max_seconds`, `target_fitness`, `stall_generations` (contenido), `structure_generations` (estructura) |

## Diseño

- **Individuo**: lista ordenada de N triángulos; el orden define el z-order al pintar.
- **Genotipo**: vector plano de `N*10` floats en `[0,1]` —
  `[x1,y1,x2,y2,x3,y3,r,g,b,a]` por triángulo. Cruza y mutación operan sobre el
  `np.ndarray` sin conocer la semántica.
- **Fitness**: `1 - RMSE/255` contra el target, en `[0,1]` y a maximizar (ruleta,
  universal y Boltzmann necesitan valores positivos). Se cachea en el individuo.
- **Motor**: `ga/engine.py` no conoce ningún método concreto; los toma por nombre del
  `METHODS` de cada módulo. Agregar un método nuevo es registrarlo ahí.

```
ga/
├── individual.py    # representación y genotipo
├── render.py        # genotipo -> bitmap (Pillow)
├── fitness.py       # RMSE + caché
├── context.py       # contrato compartido por los operadores
├── selection.py     # 6 métodos + selección combinada
├── crossover.py     # 4 métodos, con granularidad gen/triángulo
├── mutation.py      # 4 métodos
├── replacement.py   # aditiva y exclusiva
├── stopping.py      # 5 criterios de corte, evaluados por OR
├── artifact.py      # formato de salida autosuficiente + decodificador
└── engine.py        # loop generacional + métricas
```

> (!) El enunciado no define cuál de las dos supervivencias es la "aditiva" y cuál
> la "exclusiva". Asumimos: **aditiva** = compiten los N padres con los K hijos y
> sobreviven N del pool N+K; **exclusiva** = los K hijos desplazan a los padres
> (si K < N se completa con padres). Consultado a la cátedra; si fuera al revés,
> se intercambian las claves de `METHODS` en `ga/replacement.py`.

## Experimentos

```bash
python analysis/run_experiments.py            # barrido completo
python analysis/run_experiments.py selection  # un solo eje
python analysis/run_experiments.py --quick    # grilla reducida, para verificar
python analysis/plot_results.py               # figuras a partir de los CSV
```

- Grilla: `analysis/experiments.json` (config base + un eje por experimento + semillas)
- Resultados: `analysis/results/<eje>.csv` (una fila por generación, variante y semilla)
  y `analysis/results/summary.csv` (una fila por variante)
- Figuras: `analysis/figures/<eje>.png` (convergencia y diversidad, promedio ± desvío)

Cada experimento varía **un solo eje** sobre la misma base y repite con varias
semillas, para reportar promedio y desvío en vez de una corrida suelta.

## Tests

```bash
python -m unittest test_ga.py -v
```

## Resultados

Con la configuración de `config.json` (torneo determinístico M=4, cruza uniforme,
mutación no uniforme pm=0.1, supervivencia aditiva, N=K=50, 500 generaciones,
canvas 64px):

| Target | Triángulos | Fitness | RMSE | Tiempo |
|---|---|---|---|---|
| `images/japan.png` | 20 | 0.927 | 18.6 | 13 s |
| `images/germany.png` | 10 | 0.927 | 18.6 | 11 s |
| `images/cross.png` | 15 | 0.900 | 25.6 | 11 s |

### Qué salió de los experimentos

Cada eje se corrió con 3 semillas, 300 generaciones, canvas 48px y N=K=40
(`analysis/results/summary.csv`). Variando **un eje a la vez** sobre esa base:

- **Selección**: torneo determinístico (0.904) > ranking (0.897) > Boltzmann (0.887)
  > torneo probabilístico (0.882) > ruleta (0.880) > universal (0.876) > elite (0.869).
  Elite es el peor: con `k = N` devuelve a toda la población y no agrega ninguna
  presión de selección, así que la mejora queda sólo a cargo de la mutación.
- **Supervivencia**: el resultado depende de K, no de la estrategia.
  `exclusiva K=2N` (0.904) ≈ `aditiva K=2N` (0.901) > `aditiva K=N` (0.869) >>
  `exclusiva K=N` (0.772). El último caso es degenerado: con K=N la exclusiva
  reemplaza toda la población por los hijos, y si además los padres se eligen con
  elite el algoritmo pierde toda presión de selección y se vuelve una caminata
  aleatoria (se ve clarísimo en `analysis/figures/replacement.png`: el fitness
  oscila y la diversidad se queda clavada en su valor inicial).
- **Cruza**: uniforme (0.896) > anular (0.889) > dos puntos (0.881) > un punto (0.869).
- **Granularidad de la cruza**: cortar por gen (0.896) y por triángulo (0.893) empatan
  dentro del desvío. La hipótesis de que cortar a nivel gen sería destructivo por el
  z-order **no se verifica** a esta escala.
- **Mutación**, igualando la intensidad esperada en ~4 genes por hijo para comparar el
  mecanismo y no la tasa: multigen (0.903) ≈ gen (0.901) ≈ uniforme (0.901) >
  no uniforme (0.884). La no uniforme pierde cuando la tasa base ya es baja, porque el
  decaimiento la deja sin mutación; con una tasa base alta (pm=0.1) es la que mejor
  anda, que es para lo que sirve.
- **Tasa de mutación** (uniforme): pm=0.02 (0.901) ≈ pm=0.05 (0.900) ≈ pm=0.005 (0.898)
  >> pm=0.2 (0.866). Importa más la tasa que el método.
- **Cantidad de triángulos**: a presupuesto fijo de generaciones, más triángulos da
  peor fitness (10 → 0.891, 100 → 0.775) y más tiempo por generación. El espacio de
  búsqueda crece con `10 * N_triángulos` dimensiones.

**Los ganadores por eje no componen.** Combinar el mejor de cada experimento
(torneo + uniforme + multigen + exclusiva K=2N) da 0.885, peor que la configuración
final (0.919 con el mismo presupuesto de evaluaciones). Hay interacción fuerte entre
la tasa de mutación y el resto: conviene elegir la configuración completa midiendo, no
armándola por partes.
