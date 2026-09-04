# TP 2 — Algoritmos Genéticos: aproximación de imágenes con triángulos

Motor de Algoritmos Genéticos implementado desde cero (sin librerías de AG) que
aproxima una imagen usando N triángulos traslúcidos sobre un canvas blanco.

Enunciado y plan de trabajo: [`docs/TP2.md`](docs/TP2.md)

> **Estado: MVP (Steps 0-4 del plan).** Ya corre de punta a punta con un método de
> cada tipo (elite / un punto / mutación de gen / supervivencia aditiva). Faltan el
> resto de los métodos de selección, cruza, mutación y la supervivencia exclusiva
> (Steps 5-8).

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

## Salida

Cada corrida escribe en `output/<imagen>[-tag]/`:

| Archivo | Contenido |
|---|---|
| `best.png` | Imagen generada (el genotipo es independiente de la resolución: se puede renderizar a cualquier tamaño) |
| `comparison.png` | Target vs. resultado, lado a lado |
| `triangles.json` | Enumeración de los triángulos (vértices + RGBA) — la "compresión" de la imagen |
| `metrics.csv` | Una fila por generación: mejor fitness, promedio, desvío, diversidad, evaluaciones, tiempo |
| `run.json` | Config completa + fitness final, RMSE, generaciones, evaluaciones, tiempo y motivo de corte |
| `evolution.gif` | Sólo con `--gif` |

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

  "selection_parents": "elite",     // elite (Step 5: + ruleta, universal, boltzmann, torneos, ranking)
  "selection_survivors": "elite",
  "crossover": "one_point",         // one_point (Step 6: + two_point, uniform, annular)
  "crossover_rate": 0.85,
  "crossover_granularity": "gene",  // "gene" | "triangle"
  "mutation": "gene",               // gene (Step 7: + multigene, uniform, non_uniform)
  "mutation_rate": 0.5,
  "mutation_sigma": 0.1,
  "replacement": "additive",        // additive (Step 8: + exclusive)

  "stop": {
    "max_generations": 500,
    "max_seconds": null,
    "target_fitness": null,
    "stall_generations": null
  },
  "seed": 42
}
```

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
├── selection.py     # Step 5
├── crossover.py     # Step 6
├── mutation.py      # Step 7
├── replacement.py   # Step 8
└── engine.py        # loop generacional + métricas
```

## Tests

```bash
python -m unittest test_ga.py -v
```

## Resultados del MVP

| Target | Triángulos | N/K | Generaciones | Fitness | RMSE | Tiempo |
|---|---|---|---|---|---|---|
| `images/japan.png` | 20 | 50/50 | 500 | 0.911 | 22.8 | 11 s |
| `images/germany.png` | 10 | 50/50 | 400 | 0.880 | 30.7 | 9 s |

Observación del MVP: con `elite` en padres **y** en sobrevivientes la diversidad se
desploma (de 0.28 a ~0.003 en 100 generaciones) y la mejora se vuelve casi puramente
por mutación. Es exactamente el argumento para los métodos de selección del Step 5.
