# -*- coding: utf-8 -*-
"""
Genera un esquema metodologico (no UML) de la estructura supervisada para el
pronostico day-ahead del capitulo 1: la ventana de predictoras reune toda la
informacion disponible hasta el dia base D, un corte cronologico en D garantiza
que ninguna predictora use informacion posterior, y las variables objetivo
describen el dia siguiente D+1 (24 objetivos horarios + 1 agregado).

Es un diagrama (cajas, flechas, texto); no lee datos ni usa pandas.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch

RAIZ = Path(__file__).resolve().parents[1]
FIGURAS_DIR = RAIZ / "reports" / "figures"
MEDIA_LATEX = Path(r"D:\Claude\Paper Latex\workspace\media")

# Paleta Okabe-Ito: azul para predictoras, verde para objetivos, bermellon para
# el corte cronologico.
MORADO = "#0072B2"
TEAL = "#009E73"
MORADO_CLARO = "#dcebf5"
TEAL_CLARO = "#d7efe7"
GRIS = "#444444"
GRIS_CLARO = "#8a8a8a"
BERMELLON = "#D55E00"


def _caja(ax, x, y, w, h, color_borde, color_relleno):
    caja = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.6, edgecolor=color_borde, facecolor=color_relleno, zorder=2)
    ax.add_patch(caja)
    return caja


def generar() -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # --- Bloque de predictoras (dia base D) ---
    _caja(ax, 0.3, 1.5, 4.7, 3.7, MORADO, MORADO_CLARO)
    ax.text(2.65, 4.85, "Predictoras (día base D)", ha="center", va="center",
            fontsize=12, fontweight="bold", color=MORADO)
    ax.text(2.65, 4.45, "toda la información disponible hasta D",
            ha="center", va="center", fontsize=9, style="italic", color=GRIS)

    grupos = [
        "Calendario", "Cíclicas (seno-coseno)",
        "Rezagos del precio", "Estadísticas móviles",
        "Medias exponenciales (EWMA)", "Exógenas (ONI, embalses)",
        "Perfil horario", "Horas pico y valle",
        "Volatilidad", "",
    ]
    x_izq, x_der = 0.65, 2.75
    y0, dy = 3.95, 0.42
    for i, g in enumerate(grupos):
        if not g:
            continue
        col_x = x_izq if i % 2 == 0 else x_der
        fila = i // 2
        ax.text(col_x, y0 - fila * dy, "•  " + g, ha="left", va="center",
                fontsize=8.6, color="#2b2b2b")

    # --- Corte cronologico en D ---
    ax.plot([5.5, 5.5], [0.75, 5.55], ls="--", lw=1.8, color=BERMELLON, zorder=3)
    ax.text(5.5, 5.75, "Corte cronológico", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color=BERMELLON)
    ax.text(5.0, 0.18, "ninguna predictora usa información posterior a D",
            ha="center", va="center", fontsize=8.2, color=BERMELLON,
            style="italic")

    # --- Bloque de objetivos (dia siguiente D+1) ---
    _caja(ax, 6.0, 1.5, 3.7, 3.7, TEAL, TEAL_CLARO)
    ax.text(7.85, 4.85, "Objetivos (día siguiente D+1)", ha="center", va="center",
            fontsize=12, fontweight="bold", color=TEAL)

    # 24 celdas horarias (2 filas de 12).
    cw, ch, gap = 0.26, 0.3, 0.025
    x_start = 6.25
    y_fila1, y_fila2 = 3.95, 3.55
    for j in range(12):
        ax.add_patch(Rectangle((x_start + j * (cw + gap), y_fila1), cw, ch,
                     facecolor="white", edgecolor=TEAL, lw=0.8, zorder=3))
        ax.add_patch(Rectangle((x_start + j * (cw + gap), y_fila2), cw, ch,
                     facecolor="white", edgecolor=TEAL, lw=0.8, zorder=3))
    ax.text(6.25, 4.3, "h00", ha="left", va="center", fontsize=7.5, color=GRIS)
    ax.text(9.6, 3.4, "h23", ha="right", va="center", fontsize=7.5, color=GRIS)
    ax.text(7.85, 3.12, "24 objetivos horarios (precio de cada hora)",
            ha="center", va="center", fontsize=8.6, color="#2b2b2b")

    # Objetivo agregado.
    ax.add_patch(Rectangle((6.45, 2.25), 2.8, 0.42, facecolor="white",
                 edgecolor=TEAL, lw=1.2, zorder=3))
    ax.text(7.85, 2.46, "+ 1 objetivo agregado: precio promedio diario",
            ha="center", va="center", fontsize=8.6, color="#2b2b2b")

    # --- Flecha predictoras -> objetivos (modelo) ---
    flecha = FancyArrowPatch((5.05, 3.35), (5.95, 3.35),
                             arrowstyle="-|>", mutation_scale=22,
                             lw=2.2, color=GRIS, zorder=4)
    ax.add_patch(flecha)
    ax.text(5.5, 3.05, "modelo", ha="center", va="center", fontsize=8.2,
            color=GRIS, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    # --- Linea de tiempo ---
    ax.annotate("", xy=(9.9, 1.0), xytext=(0.3, 1.0),
                arrowprops=dict(arrowstyle="-|>", color=GRIS_CLARO, lw=1.4))
    ax.text(9.9, 1.22, "tiempo", ha="right", va="bottom", fontsize=8,
            color=GRIS_CLARO, style="italic")
    for xt, lab in [(5.5, "D\n(día base)"), (7.85, "D+1\n(día siguiente)")]:
        ax.plot([xt, xt], [0.9, 1.1], color=GRIS, lw=1.2)
        ax.text(xt, 0.78, lab, ha="center", va="top", fontsize=8, color=GRIS)

    fig.tight_layout()

    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    ruta = FIGURAS_DIR / "cap1_estructura_dayahead.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    if MEDIA_LATEX.exists():
        fig.savefig(MEDIA_LATEX / "cap1_estructura_dayahead.png",
                    dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Generada:", ruta)
    return ruta


if __name__ == "__main__":
    generar()
