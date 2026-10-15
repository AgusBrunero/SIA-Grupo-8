"""Ej. 2.2: Hopfield con letras de 5×5. Ejecutar: python letras.py.

Compara extremos de solapamiento entre cuartetos, estabilidad con 1–6 memorias,
recuperación con ruido y un punto fijo espurio. Salida: resultados/hopfield/letras/.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from graficos import AZUL, DIVERGENTE, NARANJA, guardar, plt
from models.hopfield import Hopfield

SALIDA = Path(__file__).parent / "resultados" / "hopfield" / "letras"
# # = +1 (trazo), . = −1 (fondo). J reproduce exactamente el ejemplo de TAREAS.md.
LETRAS = {
    "A": [".###.", "#...#", "#####", "#...#", "#...#"],
    "B": ["####.", "#...#", "####.", "#...#", "####."],
    "C": [".####", "#....", "#....", "#....", ".####"],
    "E": ["#####", "#....", "####.", "#....", "#####"],
    "F": ["#####", "#....", "####.", "#....", "#...."],
    "H": ["#...#", "#...#", "#####", "#...#", "#...#"],
    "I": ["#####", "..#..", "..#..", "..#..", "#####"],
    "J": ["#####", "...#.", "...#.", "#..#.", "###.."],
    "L": ["#....", "#....", "#....", "#....", "#####"],
    "O": [".###.", "#...#", "#...#", "#...#", ".###."],
    "P": ["####.", "#...#", "####.", "#....", "#...."],
    "T": ["#####", "..#..", "..#..", "..#..", "..#.."],
    "U": ["#...#", "#...#", "#...#", "#...#", ".###."],
    "X": ["#...#", ".#.#.", "..#..", ".#.#.", "#...#"],
    "Y": ["#...#", ".#.#.", "..#..", "..#..", "..#.."],
    "Z": ["#####", "...#.", "..#..", ".#...", "#####"],
}
SEMILLA = 0
REPETICIONES = 100
BITS_RUIDO = [0, 1, 2, 3, 5, 8, 10, 12]


def patrones():
    return np.array([[1 if c == "#" else -1 for row in rows for c in row]
                     for rows in LETRAS.values()], dtype=float)


def perturbar(pattern, bits, rng):
    state = pattern.copy()
    state[rng.choice(len(state), bits, replace=False)] *= -1
    return state


def clasificar(state, memories, target):
    matches = np.flatnonzero(np.all(memories == state, axis=1))
    if len(matches):
        return "correcto" if matches[0] == target else "otra_memoria"
    if np.any(np.all(-memories == state, axis=1)):
        return "inverso_memoria"
    return "espurio"


def ortogonalidad(patterns, names):
    products = patterns @ patterns.T
    pd.DataFrame(products.astype(int), index=names, columns=names).to_csv(
        SALIDA / "productos_internos.csv")
    overlaps = products / patterns.shape[1]
    pd.DataFrame(overlaps, index=names, columns=names).to_csv(SALIDA / "solapamientos.csv")
    rows = []
    for ids in combinations(range(len(patterns)), 4):
        pairs = np.abs(overlaps[np.ix_(ids, ids)][np.triu_indices(4, 1)])
        rows.append({"letras": "".join(names[i] for i in ids),
                     "solapamiento_abs_medio": pairs.mean(),
                     "solapamiento_abs_maximo": pairs.max()})
    table = pd.DataFrame(rows).sort_values(["solapamiento_abs_medio", "letras"])
    table.to_csv(SALIDA / "cuartetos.csv", index=False)
    groups = {}
    for label, row in [("menor_solapamiento", table.iloc[0]),
                       ("mayor_solapamiento", table.iloc[-1])]:
        ids = [names.index(name) for name in row.letras]
        groups[label] = (list(row.letras), patterns[ids])
    return groups


def estabilidad(groups):
    rows = []
    for group, (names, memories) in groups.items():
        model = Hopfield().fit(memories)
        for name, memory in zip(names, memories):
            result = model.recall(memory, seed=SEMILLA)
            rows.append({"grupo": group, "letra": name,
                         "punto_fijo": model.is_fixed_point(memory),
                         "bits_modificados": int((memory != result.state).sum()),
                         "energia_inicial": model.energy(memory),
                         "energia_final": result.energies[-1]})
    table = pd.DataFrame(rows)
    table.to_csv(SALIDA / "estabilidad.csv", index=False)
    return table


def ruido(groups):
    rows = []
    for group, (names, memories) in groups.items():
        model = Hopfield().fit(memories)
        for bits in BITS_RUIDO:
            for target, (name, memory) in enumerate(zip(names, memories)):
                for trial in range(REPETICIONES):
                    # Mismas posiciones de ruido/orden de actualización para comparar grupos.
                    seed = bits * 10000 + target * REPETICIONES + trial
                    noisy = perturbar(memory, bits, np.random.default_rng(seed))
                    result = model.recall(noisy, seed=seed)
                    assert np.all(np.diff(result.energies) <= 1e-9)
                    status = clasificar(result.state, memories, target) if result.converged else "limite"
                    rows.append({"grupo": group, "letra": name, "bits_ruido": bits,
                                 "fraccion_ruido": bits / 25, "semilla": seed,
                                 "resultado": status, "barridos": result.sweeps,
                                 "distancia_final": int((result.state != memory).sum()),
                                 "convergio": result.converged})
    table = pd.DataFrame(rows)
    table.to_csv(SALIDA / "recuperacion.csv", index=False)
    summary = table.assign(acierto=table.resultado == "correcto").groupby(
        ["grupo", "bits_ruido", "fraccion_ruido"], as_index=False).agg(
        ensayos=("acierto", "size"), recuperacion_exacta=("acierto", "mean"),
        barridos_medios=("barridos", "mean"), distancia_media=("distancia_final", "mean"))
    summary.to_csv(SALIDA / "recuperacion_resumen.csv", index=False)
    return summary


def capacidad(patterns, names):
    """Estabilidad exacta de letras sin ruido en hasta 100 subconjuntos por tamaño.

    No elige solo conjuntos favorables: muestrea sin reemplazo de todos los posibles.
    No estima una capacidad universal ni equivale a recuperación con ruido.
    """
    rng = np.random.default_rng(SEMILLA)
    rows = []
    for size in range(1, 7):
        subsets = list(combinations(range(len(patterns)), size))
        chosen = rng.choice(len(subsets), min(100, len(subsets)), replace=False)
        for idx in chosen:
            ids = subsets[idx]
            memories = patterns[list(ids)]
            model = Hopfield().fit(memories)
            stable = sum(model.is_fixed_point(memory) for memory in memories)
            rows.append({"patrones": size, "letras": "".join(names[i] for i in ids),
                         "estables": stable, "fraccion_estable": stable / size,
                         "todos_estables": stable == size})
    table = pd.DataFrame(rows)
    table.to_csv(SALIDA / "capacidad.csv", index=False)
    summary = table.groupby("patrones", as_index=False).agg(
        subconjuntos=("letras", "size"), fraccion_estable_media=("fraccion_estable", "mean"),
        proporcion_todos_estables=("todos_estables", "mean"))
    summary.to_csv(SALIDA / "capacidad_resumen.csv", index=False)
    return summary


def ejemplo_espurio(groups):
    # Buscar en el grupo de menor solapamiento: incluso este admite atractores adicionales.
    names, memories = groups["menor_solapamiento"]
    model = Hopfield().fit(memories)
    for trial in range(10000):
        seed = 1000000 + trial
        target = trial % len(memories)
        initial = perturbar(memories[target], 10, np.random.default_rng(seed))
        result = model.recall(initial, seed=seed)
        if result.converged and clasificar(result.state, memories, target) == "espurio":
            assert model.is_fixed_point(result.state)
            np.savez(SALIDA / "espurio.npz", memorias=memories, letras=names,
                     objetivo=names[target], entrada=initial, salida=result.state,
                     estados=result.states, energias=result.energies, weights=model.weights,
                     semilla=seed, bits_ruido=10, barridos=result.sweeps)
            pd.DataFrame({"actualizacion": np.arange(len(result.energies)),
                          "energia": result.energies}).to_csv(SALIDA / "energia_espurio.csv", index=False)
            return names[target], initial, result, memories, names, seed
    raise RuntimeError("No se encontró un estado espurio en el límite de búsqueda")


def dibujar(ax, pattern, title):
    ax.imshow(np.asarray(pattern).reshape(5, 5), cmap="binary", vmin=-1, vmax=1,
              interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def graficos(patterns, names, groups, recovery, capacity, example):
    fig, axes = plt.subplots(2, 8, figsize=(12, 4))
    for ax, name, pattern in zip(axes.flat, names, patterns):
        dibujar(ax, pattern, name)
    fig.suptitle("Patrones candidatos (trazo = +1, fondo = −1)")
    fig.tight_layout()
    guardar(fig, SALIDA / "patrones.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (group, (letters, memories)) in zip(axes, groups.items()):
        overlap = memories @ memories.T / 25
        im = ax.imshow(overlap, cmap=DIVERGENTE, vmin=-1, vmax=1)
        ax.set_xticks(range(4), letters)
        ax.set_yticks(range(4), letters)
        ax.grid(False)
        for (i, j), value in np.ndenumerate(overlap):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(group.replace("_", " "))
    fig.colorbar(im, ax=list(axes), shrink=0.7, label="producto interno / 25")
    guardar(fig, SALIDA / "ortogonalidad.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for (group, data), color in zip(recovery.groupby("grupo", sort=False), [NARANJA, AZUL]):
        axes[0].plot(data.fraccion_ruido * 100, data.recuperacion_exacta * 100,
                     "o-", color=color, label=group.replace("_", " "))
    axes[0].set(xlabel="bits invertidos (%)", ylabel="recuperación exacta (%)", ylim=(-2, 102))
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(capacity.patrones, capacity.fraccion_estable_media * 100, "o-",
                 color=AZUL, label="fracción media de memorias estables")
    axes[1].plot(capacity.patrones, capacity.proporcion_todos_estables * 100, "s-",
                 color=NARANJA, label="subconjuntos con todas estables")
    axes[1].axvline(0.138 * 25, color="gray", ls="--", label="referencia 0,138 N ≈ 3,45")
    axes[1].set(xlabel="cantidad de patrones almacenados", ylabel="porcentaje", ylim=(-2, 102))
    axes[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Hopfield: ruido en dos cuartetos y estabilidad con 1–6 letras")
    fig.tight_layout()
    guardar(fig, SALIDA / "recuperacion_capacidad.png")

    target, initial, result, memories, letters, seed = example
    fig, axes = plt.subplots(1, len(result.states) + 1, figsize=(2 * (len(result.states) + 1), 2.5))
    dibujar(axes[0], memories[letters.index(target)], f"objetivo {target}")
    for i, (ax, state) in enumerate(zip(axes[1:], result.states)):
        dibujar(ax, state, "entrada: 40 % ruido" if i == 0 else f"barrido {i}")
    fig.suptitle(f"Punto fijo espurio: semilla {seed}, no coincide con ninguna memoria ni su inversa")
    fig.tight_layout()
    guardar(fig, SALIDA / "espurio.png")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(result.energies, color=AZUL)
    ax.set(xlabel="actualización individual (0 = entrada)", ylabel="energía",
           title="Energía no creciente hasta el estado espurio")
    guardar(fig, SALIDA / "energia_espurio.png")


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    names, patterns = list(LETRAS), patrones()
    np.savez(SALIDA / "patrones.npz", letras=names, patrones=patterns)
    groups = ortogonalidad(patterns, names)
    stable = estabilidad(groups)
    recovery = ruido(groups)
    capacity = capacidad(patterns, names)
    example = ejemplo_espurio(groups)
    graficos(patterns, names, groups, recovery, capacity, example)
    print("Cuartetos elegidos por solapamiento absoluto medio:")
    for group, (letters, _) in groups.items():
        print(group, " ".join(letters))
    print(stable.to_string(index=False))
    print(recovery.round(3).to_string(index=False))
    print(capacity.round(3).to_string(index=False))
    print(f"Espurio: objetivo {example[0]}, semilla {example[-1]}, "
          f"{example[2].sweeps} barridos, energía {example[2].energies[-1]:.2f}")


if __name__ == "__main__":
    main()
