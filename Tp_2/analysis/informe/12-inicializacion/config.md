# Configuración — Inicialización de la población

Barrido `inicializacion` · batch `20260906T233756Z` · commit `5b59273` (árbol sucio)

- **Semillas**: [1, 2, 3, 4, 5]
- **Targets**: `plana` = `images/japan.png`, `detallada` = `images/pika.png`, `compleja` = `images/noche_estrellada.jpg`
- **Varía**: `initialization`

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
| mutación (`mutation`) | `uniform` |
| piso de decaimiento (`mutation_decay_floor`) | `0.1` |
| M (multigen) (`mutation_genes`) | `None` |
| probabilidad de mutación pm (`mutation_rate`) | `0.008` |
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
| **al azar** | `initialization` = `random` |
| **grilla informada** | `initialization` = `grid` |

## Reproducir

```bash
python analysis/run_experiments.py inicializacion --clean
python analysis/build_report.py
```

La configuración completa de cada variante, ya resuelta contra los defaults del
motor, está en `analysis/results/manifest.json`.

<details><summary>Configuración completa de cada variante (JSON)</summary>

```json
{
  "al azar": {
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
  "grilla informada": {
    "triangles": 50,
    "initialization": "grid",
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
  }
}
```

</details>
