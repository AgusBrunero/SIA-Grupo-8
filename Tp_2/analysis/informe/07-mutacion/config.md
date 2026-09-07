# Configuración — Método de mutación

Barrido `mutacion` · batch `20260906T233756Z` · commit `5b59273` (árbol sucio)

- **Semillas**: [1, 2, 3, 4, 5]
- **Targets**: `plana` = `images/japan.png`, `detallada` = `images/pika.png`, `compleja` = `images/noche_estrellada.jpg`
- **Varía**: `mutation`, `mutation_genes`, `mutation_rate`

## Configuración fija

Todo lo que **no** cambia entre variantes de este eje.

| Parámetro | Valor |
|---|---|
| color de fondo (`background`) | `[255, 255, 255]` |
| Boltzmann (`boltzmann`) | `{"t0": 100.0, "tmin": 1.0, "k": 0.01}` |
| canvas (px) (`canvas_size`) | `64` |
| cruza (`crossover`) | `one_point` |
| granularidad de cruza (`crossover_granularity`) | `gene` |
| probabilidad de cruza pc (`crossover_rate`) | `0.85` |
| p de cruza uniforme (`crossover_uniform_p`) | `0.5` |
| inicialización (`initialization`) | `random` |
| piso de decaimiento (`mutation_decay_floor`) | `0.1` |
| σ de mutación (`mutation_sigma`) | `0.1` |
| mutation_zorder_rate (`mutation_zorder_rate`) | `0.0` |
| descendencia K (`offspring_size`) | `60` |
| población N (`population_size`) | `60` |
| preserve_aspect (`preserve_aspect`) | `True` |
| supervivencia (`replacement`) | `additive` |
| selección de padres (`selection_parents`) | `tournament_det` |
| selección de sobrevivientes (`selection_survivors`) | `elite` |
| criterios de corte (`stop`) | `{"max_generations": 800}` |
| torneo (`tournament`) | `{"m": 4, "threshold": 0.75}` |
| triángulos (`triangles`) | `50` |

## Variantes

| Variante | Qué cambia |
|---|---|
| **gen (carga 1)** | `mutation` = `gene` · `mutation_rate` = `1.0` |
| **multigen M=8 (carga 4)** | `mutation` = `multigene` · `mutation_genes` = `8` · `mutation_rate` = `0.5` |
| **uniforme (carga 4)** | `mutation` = `uniform` · `mutation_rate` = `0.008` |
| **no uniforme (carga 4 -> 0.4)** | `mutation` = `non_uniform` · `mutation_rate` = `0.008` |

## Reproducir

```bash
python analysis/run_experiments.py mutacion --clean
python analysis/build_report.py
```

La configuración completa de cada variante, ya resuelta contra los defaults del
motor, está en `analysis/results/manifest.json`.

<details><summary>Configuración completa de cada variante (JSON)</summary>

```json
{
  "gen (carga 1)": {
    "triangles": 50,
    "initialization": "random",
    "canvas_size": 64,
    "preserve_aspect": true,
    "background": [
      255,
      255,
      255
    ],
    "population_size": 60,
    "offspring_size": 60,
    "selection_parents": "tournament_det",
    "selection_survivors": "elite",
    "tournament": {
      "m": 4,
      "threshold": 0.75
    },
    "boltzmann": {
      "t0": 100.0,
      "tmin": 1.0,
      "k": 0.01
    },
    "crossover": "one_point",
    "crossover_rate": 0.85,
    "crossover_granularity": "gene",
    "crossover_uniform_p": 0.5,
    "mutation": "gene",
    "mutation_rate": 1.0,
    "mutation_sigma": 0.1,
    "mutation_genes": null,
    "mutation_decay_floor": 0.1,
    "mutation_zorder_rate": 0.0,
    "replacement": "additive",
    "stop": {
      "max_generations": 800
    }
  },
  "multigen M=8 (carga 4)": {
    "triangles": 50,
    "initialization": "random",
    "canvas_size": 64,
    "preserve_aspect": true,
    "background": [
      255,
      255,
      255
    ],
    "population_size": 60,
    "offspring_size": 60,
    "selection_parents": "tournament_det",
    "selection_survivors": "elite",
    "tournament": {
      "m": 4,
      "threshold": 0.75
    },
    "boltzmann": {
      "t0": 100.0,
      "tmin": 1.0,
      "k": 0.01
    },
    "crossover": "one_point",
    "crossover_rate": 0.85,
    "crossover_granularity": "gene",
    "crossover_uniform_p": 0.5,
    "mutation": "multigene",
    "mutation_rate": 0.5,
    "mutation_sigma": 0.1,
    "mutation_genes": 8,
    "mutation_decay_floor": 0.1,
    "mutation_zorder_rate": 0.0,
    "replacement": "additive",
    "stop": {
      "max_generations": 800
    }
  },
  "uniforme (carga 4)": {
    "triangles": 50,
    "initialization": "random",
    "canvas_size": 64,
    "preserve_aspect": true,
    "background": [
      255,
      255,
      255
    ],
    "population_size": 60,
    "offspring_size": 60,
    "selection_parents": "tournament_det",
    "selection_survivors": "elite",
    "tournament": {
      "m": 4,
      "threshold": 0.75
    },
    "boltzmann": {
      "t0": 100.0,
      "tmin": 1.0,
      "k": 0.01
    },
    "crossover": "one_point",
    "crossover_rate": 0.85,
    "crossover_granularity": "gene",
    "crossover_uniform_p": 0.5,
    "mutation": "uniform",
    "mutation_rate": 0.008,
    "mutation_sigma": 0.1,
    "mutation_genes": null,
    "mutation_decay_floor": 0.1,
    "mutation_zorder_rate": 0.0,
    "replacement": "additive",
    "stop": {
      "max_generations": 800
    }
  },
  "no uniforme (carga 4 -> 0.4)": {
    "triangles": 50,
    "initialization": "random",
    "canvas_size": 64,
    "preserve_aspect": true,
    "background": [
      255,
      255,
      255
    ],
    "population_size": 60,
    "offspring_size": 60,
    "selection_parents": "tournament_det",
    "selection_survivors": "elite",
    "tournament": {
      "m": 4,
      "threshold": 0.75
    },
    "boltzmann": {
      "t0": 100.0,
      "tmin": 1.0,
      "k": 0.01
    },
    "crossover": "one_point",
    "crossover_rate": 0.85,
    "crossover_granularity": "gene",
    "crossover_uniform_p": 0.5,
    "mutation": "non_uniform",
    "mutation_rate": 0.008,
    "mutation_sigma": 0.1,
    "mutation_genes": null,
    "mutation_decay_floor": 0.1,
    "mutation_zorder_rate": 0.0,
    "replacement": "additive",
    "stop": {
      "max_generations": 800
    }
  }
}
```

</details>
