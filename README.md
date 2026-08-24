# SIA-Grupo-8

Repositorio de los trabajos prácticos del grupo 8 de la materia Sistemas de Inteligencia Artificial (72.27).

## Integrantes

* Agustín Julián Brunero
* Bruno Enzo Baumgart
* Juan Diego Gago
* Louis Bellet
* Nicolás Canzonieri

## TP 1 — Sokoban (métodos de búsqueda)

### Setup

```bash
cd "TP 1"
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Jugar / ver el motor

```bash
python sokoban_game.py
```

| Tecla | Acción |
|---|---|
| Flechas | Mover |
| `R` | Reset |
| `U` | Undo |
| `N` | Siguiente nivel |
| `A` / `B` / `G` / `D` | Resolver con A* / BFS / Greedy / DFS (heurística Hungarian) |
| `Shift` + la misma tecla | Misma búsqueda con heurística Manhattan simple |
| `w` | A* con heurística weighted (no admisible) |
| `W` | Greedy con heurística weighted (no admisible) |

Al resolver, una ventana muestra costo, nodos expandidos, frontera y tiempo; después se anima el camino.

### Benchmarks y gráficos

```bash
python analysis/run_benchmarks.py
python analysis/plot_results.py
python analysis/informedness.py
```

- Configuración: `analysis/config.json` (niveles, métodos, repeticiones, timeout)
- Resultados: `analysis/results/benchmarks.csv` (una fila por corrida)
- Figuras: `analysis/figures/`

Para re-correr solo A* del nivel 3 (el más pesado):

```bash
python analysis/rerun_astar_level3.py
```

### Tests

```bash
python -m unittest test_sokoban_solver.py
```
