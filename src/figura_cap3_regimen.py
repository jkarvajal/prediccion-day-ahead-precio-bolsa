# -*- coding: utf-8 -*-
"""
Desempeno por regimen de precio (MAE calmo vs volatil) por estrategia, figura del
capitulo 3. Lee el consolidado por regimen y solo grafica; no recalcula.

Datos: data/outputs/consolidado_por_regimen.csv.
Paleta Okabe-Ito (config.py): calmo en azul, volatil en bermellon.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import (OUTPUTS_DIR, FIGURES_DIR, MEDIA_LATEX_DIR,
                    COLOR_PRIMARIO, COLOR_RESALTADO)

COL_CALMO = "calmo (<300)"
COL_VOLATIL = "volatil (>=300)"


def generar():
    df = pd.read_csv(OUTPUTS_DIR / "consolidado_por_regimen.csv")
    pivote = df.pivot(index="estrategia", columns="regimen", values="MAE")
    pivote = pivote[pivote.index != "Prophet"].sort_values(COL_VOLATIL)

    etiquetas = list(pivote.index)
    y = range(len(pivote))
    alto = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh([i + alto / 2 for i in y], pivote[COL_CALMO], alto,
            label="Régimen calmo (menor a 300)", color=COLOR_PRIMARIO)
    ax.barh([i - alto / 2 for i in y], pivote[COL_VOLATIL], alto,
            label="Régimen volátil (300 o más)", color=COLOR_RESALTADO)
    ax.set_yticks(list(y))
    ax.set_yticklabels(etiquetas, fontsize=9)
    ax.set_xlabel("MAE ($/kWh)")
    ax.set_title("Desempeño por régimen de precio (MAE)")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = FIGURES_DIR / "cap3_regimen.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight", facecolor="white")
    if MEDIA_LATEX_DIR.exists():
        fig.savefig(MEDIA_LATEX_DIR / "cap3_regimen.png", dpi=150,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Generada:", ruta.name)
    return ruta


if __name__ == "__main__":
    generar()
