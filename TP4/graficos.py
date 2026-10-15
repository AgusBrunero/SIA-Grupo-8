"""Estilo y helpers de gráficos compartidos por los scripts del TP4."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Paleta categórica validada (orden fijo) + forma de marcador como codificación secundaria.
ESTILO_TEMA = {
    "fútbol": ("#2a78d6", "o"),
    "maquillaje": ("#eb6834", "s"),
    "perros": ("#1baf7a", "^"),
    "arte": ("#eda100", "D"),
}
AZUL, NARANJA, ROJO = "#2a78d6", "#eb6834", "#e34948"
DIVERGENTE = LinearSegmentedColormap.from_list("azul_rojo", [AZUL, "#f0efec", ROJO])
SECUENCIAL = LinearSegmentedColormap.from_list("azul", ["#f0efec", "#9ec5f4", AZUL, "#104281"])
TINTA, TINTA_SUAVE, GRILLA, FONDO = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
    "axes.edgecolor": GRILLA, "axes.labelcolor": TINTA_SUAVE, "text.color": TINTA,
    "xtick.color": TINTA_SUAVE, "ytick.color": TINTA_SUAVE, "axes.grid": True,
    "grid.color": GRILLA, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 9,
})


def guardar(fig, ruta):
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
