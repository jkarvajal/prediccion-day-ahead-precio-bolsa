# -*- coding: utf-8 -*-
"""
Perfil horario del precio de bolsa (promedio, mediana y percentil 95 por hora),
figura del capitulo 2. Lee el output del EDA y solo grafica; no recalcula.

Datos: data/outputs/eda_perfil_horario.csv (Bloque 10 de 02_eda.ipynb).
Paleta Okabe-Ito (config.py), con estilos de linea y marcadores.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import (OUTPUTS_DIR, FIGURES_DIR, MEDIA_LATEX_DIR,
                    COLOR_PRIMARIO, COLOR_SECUNDARIO, COLOR_TERCIARIO, BANDA_PICO)


def generar():
    df = pd.read_csv(OUTPUTS_DIR / "eda_perfil_horario.csv").sort_values("hora")
    h = df["hora"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    picos = df.loc[df["es_hora_pico"] == 1, "hora"]
    if not picos.empty:
        ax.axvspan(picos.min() - 0.5, picos.max() + 0.5, color=BANDA_PICO,
                   alpha=0.15, zorder=0, label="Horas pico (18 a 21 h)")

    ax.plot(h, df["precio_promedio"], color=COLOR_PRIMARIO, lw=2.4, marker="o",
            ms=4, label="Promedio")
    ax.plot(h, df["precio_mediana"], color=COLOR_SECUNDARIO, lw=2.0, marker="s",
            ms=4, label="Mediana")
    ax.plot(h, df["precio_p95"], color=COLOR_TERCIARIO, lw=2.0, ls="--",
            marker="^", ms=4, label="Percentil 95")

    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hora del día")
    ax.set_ylabel("Precio de bolsa ($/kWh)")
    ax.set_title("Perfil horario del precio de bolsa")
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", fontsize=9)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = FIGURES_DIR / "eda_perfil_horario.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    if MEDIA_LATEX_DIR.exists():
        fig.savefig(MEDIA_LATEX_DIR / "eda_perfil_horario.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Generada:", ruta.name)
    return ruta


if __name__ == "__main__":
    generar()
