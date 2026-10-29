# -*- coding: utf-8 -*-
"""
Distribucion del precio de bolsa por hora del dia (diagrama de caja), figura del
capitulo 2. Sustenta la asimetria (sesgo a la derecha) y la mayor dispersion en
la tarde y la noche. Reproduce la vista "Explorar patrones -> Distribucion por
hora" del dashboard: bigotes hasta 1.5 veces el rango intercuartilico y sin los
valores atipicos extremos (para no distorsionar la escala).

Datos: data/processed/features_day_ahead.csv (columnas h00..h23, la serie horaria
del precio), la misma fuente que consume el dashboard.
Paleta Okabe-Ito (config.py).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import (PROCESSED_DIR, FIGURES_DIR, MEDIA_LATEX_DIR,
                    COLOR_PRIMARIO, COLOR_RESALTADO, BANDA_PICO)

HOUR_COLS = [f"h{h:02d}" for h in range(24)]
PICO_INI, PICO_FIN = 17.5, 21.5


def generar():
    df = pd.read_csv(PROCESSED_DIR / "features_day_ahead.csv", usecols=HOUR_COLS)
    datos = [df[c].dropna().to_numpy() for c in HOUR_COLS]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axvspan(PICO_INI, PICO_FIN, color=BANDA_PICO, alpha=0.15, zorder=0,
               label="Horas pico (18 a 21 h)")

    bp = ax.boxplot(datos, positions=list(range(24)), widths=0.6, whis=1.5,
                    showfliers=False, patch_artist=True, zorder=3)
    for caja in bp["boxes"]:
        caja.set(facecolor=COLOR_PRIMARIO, alpha=0.45, edgecolor=COLOR_PRIMARIO,
                 linewidth=1.1)
    for med in bp["medians"]:
        med.set(color=COLOR_RESALTADO, linewidth=1.8)
    for elem in bp["whiskers"] + bp["caps"]:
        elem.set(color=COLOR_PRIMARIO, linewidth=1.1)

    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_xlim(-0.8, 23.8)
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Precio de bolsa ($/kWh)")
    ax.set_title("Distribución del precio por hora del día")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = FIGURES_DIR / "cap2_boxplot_hora.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    if MEDIA_LATEX_DIR.exists():
        fig.savefig(MEDIA_LATEX_DIR / "cap2_boxplot_hora.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Generada:", ruta.name)
    return ruta


if __name__ == "__main__":
    generar()
