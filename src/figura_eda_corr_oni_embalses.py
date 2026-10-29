# -*- coding: utf-8 -*-
"""
Correlacion entre el Indice ONI y el nivel de embalses segun el rezago en dias,
figura del capitulo 2. Lee el output del EDA y solo grafica; no recalcula.

Datos: data/outputs/eda_correlacion_oni_rezagado_embalses.csv (02_eda.ipynb).
Paleta Okabe-Ito (config.py).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import (OUTPUTS_DIR, FIGURES_DIR, MEDIA_LATEX_DIR,
                    COLOR_PRIMARIO, COLOR_NEUTRO)


def generar():
    df = pd.read_csv(OUTPUTS_DIR / "eda_correlacion_oni_rezagado_embalses.csv")
    df = df.sort_values("rezago_dias")
    x = df["rezago_dias"].to_numpy()
    y = df["correlacion_oni_embalses"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color=COLOR_NEUTRO, lw=1, ls=":")
    ax.plot(x, y, color=COLOR_PRIMARIO, lw=2.4, marker="o", ms=6)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=8, color=COLOR_NEUTRO)

    ax.set_xticks(x)
    ax.set_xlabel("Rezago (días)")
    ax.set_ylabel("Correlación ONI - nivel de embalses")
    ax.set_title("El ONI anticipa el nivel de embalses")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = FIGURES_DIR / "eda_corr_oni_embalses.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    if MEDIA_LATEX_DIR.exists():
        fig.savefig(MEDIA_LATEX_DIR / "eda_corr_oni_embalses.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Generada:", ruta.name)
    return ruta


if __name__ == "__main__":
    generar()
